//! The align pass: every frame registered to its neighbour in focus order, chained to a reference.
//!
//! - **Work images** (≤ `work_edge`, 2048 px by default): grey `(0.25R + 0.6G + 0.15B)^(1/2.2)`,
//!   low-passed (σ = 1 px against noise and aliasing), in a pyramid down to 128–256 px. Only the previous frame's work image is kept, so memory doesn't grow with the
//!   number of frames.
//! - **Pairs**: inverse-compositional Lucas–Kanade ([`super::lk`]) from the identity; the
//!   sharper frame of the pair is the template. When the fit correlates poorly (NCC < 0.6),
//!   features + RANSAC ([`crate::features`], [`crate::ransac`]) give a seed for a second fit. A
//!   fit that correlates worse than doing nothing, or moves more than neighbouring frames of a
//!   bracket plausibly do, is replaced by the identity; a frame that correlates poorly with its
//!   neighbour (NCC < 0.5: another scene, a black frame; neighbours correlate > 0.9) is skipped
//!   and the chain continues from the previous good frame.
//! - **Brightness**: per channel, the median log ratio of mid-tone pixels between neighbours
//!   (a small, blurred colour thumbnail), chained like the geometry.
//! - **Chain**: `P_ref = I`, `P_k = M_k · P_{k−1}` after the reference and
//!   `P_k = M_{k+1}⁻¹ · P_{k+1}` before it (`M_k` maps frame `k−1` to frame `k`). Pairwise errors
//!   random-walk, which is why the reference sits in the middle by default.
//! - **Common area**: the largest rectangle every frame covers ([`crate::pano::finish::auto_crop`]).

use std::borrow::Cow;

use lightcraft_geom::{Homography, Point};
use lightcraft_raster::blur::gaussian;
use lightcraft_raster::resample::{Filter, fit, half};
use lightcraft_raster::{Plane, Rgb32f};
use rayon::prelude::*;

use super::lk::{self, Dof};
use super::{Reference, order};
use crate::align::{plausible, scale_homography};
use crate::{MergeError, Result};

/// Pyramids stop above this long edge.
const MIN_LEVEL_EDGE: usize = 128;
/// Below this correlation a second fit from a feature-based seed is tried.
const SEED_NCC: f32 = 0.6;
/// Below this correlation a frame is skipped (neighbours in a focus bracket correlate > 0.9).
pub const SKIP_NCC: f32 = 0.5;
/// Below this correlation a frame is reported as poorly aligned.
const WARN_NCC: f32 = 0.8;
/// Sharpness grid cells along the long edge.
pub const GRID: usize = 64;

/// What the align pass keeps of a frame.
pub struct WorkFrame {
    /// Grey pyramid, level 0 = the work image.
    pub pyr: Vec<Plane>,
    /// A small blurred colour thumbnail (brightness gains).
    pub small: Rgb32f,
    pub clip: f32,
    /// Mean focus measure (sum-modified Laplacian) at ≤ 640 px.
    pub sharpness: f64,
    /// The focus measure per cell of a [`GRID`]-wide grid.
    pub grid: Vec<f32>,
}

impl WorkFrame {
    pub fn size(&self) -> (usize, usize) {
        self.pyr.first().map_or((0, 0), |p| (p.width, p.height))
    }
    pub fn bytes(&self) -> usize {
        self.pyr.iter().map(|p| p.data.len() * 4).sum::<usize>() + self.small.data.len() * 12 + self.grid.len() * 4
    }
}

/// The modified Laplacian of a grey plane (borders 0).
fn ml_plane(p: &Plane) -> Plane {
    let (w, h) = (p.width, p.height);
    let mut out = Plane::new(w, h);
    if w < 3 || h < 3 {
        return out;
    }
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w - 1 {
            let i = y * w + x;
            let c = p.data[i];
            row[x] = (2.0 * c - p.data[i - 1] - p.data[i + 1]).abs() + (2.0 * c - p.data[i - w] - p.data[i + w]).abs();
        }
    });
    out
}

/// Grid dimensions for a `w × h` image.
pub fn grid_dims(w: usize, h: usize) -> (usize, usize) {
    if w >= h { (GRID, (GRID * h).div_ceil(w.max(1)).max(1)) } else { ((GRID * w).div_ceil(h.max(1)).max(1), GRID) }
}

/// Mean focus measure and its per-cell grid.
pub fn sharpness_grid(p: &Plane) -> (f64, Vec<f32>) {
    let ml = ml_plane(p);
    let (w, h) = (ml.width, ml.height);
    let (gx, gy) = grid_dims(w, h);
    let mut sum = vec![0f64; gx * gy];
    let mut cnt = vec![0u32; gx * gy];
    for y in 0..h {
        let cy = (y * gy / h.max(1)).min(gy - 1);
        for x in 0..w {
            let cx = (x * gx / w.max(1)).min(gx - 1);
            sum[cy * gx + cx] += ml.data[y * w + x] as f64;
            cnt[cy * gx + cx] += 1;
        }
    }
    let total: f64 = sum.iter().sum::<f64>() / (w * h).max(1) as f64;
    (total, sum.iter().zip(&cnt).map(|(s, &c)| (s / c.max(1) as f64) as f32).collect())
}

/// Grey value for alignment.
#[inline]
fn grey(c: [f32; 3]) -> f32 {
    (0.25 * c[0] + 0.6 * c[1] + 0.15 * c[2]).clamp(0.0, 1.0).powf(1.0 / 2.2)
}

/// The work image of a frame (scaled to ≤ `work_edge`).
pub fn work_frame(img: &Rgb32f, clip: f32, work_edge: usize) -> WorkFrame {
    let edge = work_edge.max(MIN_LEVEL_EDGE);
    let img: Cow<Rgb32f> = if img.width.max(img.height) > edge { Cow::Owned(fit(img, edge, edge, Filter::Box)) } else { Cow::Borrowed(img) };
    let inv_clip = if clip > 0.0 && clip.is_finite() { 1.0 / clip } else { 1.0 };
    let g = gaussian(&img.map(|c| grey(c.map(|v| v * inv_clip))), 1.0);
    let mut pyr = vec![g];
    while let Some(l) = pyr.last() {
        if l.width.max(l.height) < 2 * MIN_LEVEL_EDGE || l.width.min(l.height) < 16 {
            break;
        }
        let next = half(l);
        pyr.push(next);
    }
    let lvl = pyr.iter().position(|p| p.width.max(p.height) <= 640).unwrap_or(pyr.len().saturating_sub(1));
    let (sharpness, grid) = pyr.get(lvl).map(sharpness_grid).unwrap_or_default();
    let small = gaussian(&fit(&img, 384, 384, Filter::Box), 1.0);
    WorkFrame { pyr, small, clip, sharpness, grid }
}

/// The registration of one frame to the previous good frame of the sequence.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    /// Maps the previous good frame's work coordinates to this frame's.
    pub m: Homography,
    /// This frame's brightness relative to the previous good frame, per channel.
    pub gain: [f32; 3],
    pub ncc: f32,
    pub skipped: bool,
    /// The fit came from a feature-based seed.
    pub seeded: bool,
}

impl Pair {
    fn first() -> Pair {
        Pair { m: Homography::IDENTITY, gain: [1.0; 3], ncc: 1.0, skipped: false, seeded: false }
    }
}

/// Registers frames one at a time, in focus order.
pub struct Aligner {
    dof: Dof,
    align: bool,
    brightness: bool,
    work_edge: usize,
    last: Option<WorkFrame>,
    pub pairs: Vec<Pair>,
    pub grids: Vec<Vec<f32>>,
    pub sharpness: Vec<f64>,
    size: Option<(usize, usize)>,
}

impl Aligner {
    pub fn new(dof: Dof, align: bool, brightness: bool, work_edge: usize) -> Aligner {
        Aligner { dof, align, brightness, work_edge, last: None, pairs: Vec::new(), grids: Vec::new(), sharpness: Vec::new(), size: None }
    }

    pub fn bytes(&self) -> usize {
        self.last.as_ref().map_or(0, WorkFrame::bytes)
    }

    /// Register the next frame of the sequence (`index` names it in errors).
    pub fn push(&mut self, index: usize, img: &Rgb32f, clip: f32) -> Result<()> {
        let wf = work_frame(img, clip, self.work_edge);
        match self.size {
            None => self.size = Some(wf.size()),
            Some(s) if s != wf.size() => {
                return Err(MergeError::Mismatch(format!(
                    "photo {} is {}×{} (scaled), the others {}×{}; focus stacking needs photos of the same size",
                    index + 1,
                    wf.size().0,
                    wf.size().1,
                    s.0,
                    s.1
                )));
            }
            Some(_) => {}
        }
        self.grids.push(wf.grid.clone());
        self.sharpness.push(wf.sharpness);
        let Some(prev) = &self.last else {
            self.pairs.push(Pair::first());
            self.last = Some(wf);
            return Ok(());
        };
        let mut pair = Pair { m: Homography::IDENTITY, gain: [1.0; 3], ncc: 1.0, skipped: false, seeded: false };
        if self.align && self.dof.any() {
            let prev_is_template = prev.sharpness >= wf.sharpness;
            let (t, i) = if prev_is_template { (prev, &wf) } else { (&wf, prev) };
            let ident = lk::ncc(&t.pyr[0], &i.pyr[0], &Homography::IDENTITY);
            let mut best = lk::align(&t.pyr, &i.pyr, &Homography::IDENTITY, self.dof);
            if (!best.ncc.is_finite() || best.ncc < SEED_NCC)
                && let Some(seed) = feature_seed(t, i)
            {
                let f = lk::align(&t.pyr, &i.pyr, &seed, self.dof);
                if f.ncc > best.ncc || !best.ncc.is_finite() {
                    best = f;
                    pair.seeded = true;
                }
            }
            let (w, h) = t.size();
            if !best.ncc.is_finite() || best.ncc < ident || !neighbourly(&best.h, w as f64, h as f64) {
                best = lk::Fit { h: Homography::IDENTITY, ncc: ident };
                pair.seeded = false;
            }
            pair.ncc = best.ncc;
            // the fit maps template → image coordinates; `m` maps previous → this frame
            pair.m = if prev_is_template { best.h } else { best.h.inverse().unwrap_or(Homography::IDENTITY) };
            if best.ncc < SKIP_NCC {
                pair.skipped = true;
                self.pairs.push(pair);
                return Ok(());
            }
        }
        if self.brightness {
            pair.gain = gains(prev, &wf, &pair.m);
        }
        self.pairs.push(pair);
        self.last = Some(wf);
        Ok(())
    }

    /// The work image size (of the first frame).
    pub fn size(&self) -> Option<(usize, usize)> {
        self.size
    }

    /// Drop the last work image (end of the pass).
    pub fn release(&mut self) {
        self.last = None;
    }
}

/// Whether `h` is a plausible motion between neighbouring frames of a bracket: corners move
/// less than 10 % of the diagonal, the area changes by less than 12 %.
fn neighbourly(h: &Homography, w: f64, ht: f64) -> bool {
    let m = h.0;
    let det = m[0] * m[4] - m[1] * m[3];
    plausible(h, w, ht, 0.1) && (det - 1.0).abs() < 0.12
}

/// A seed from features + RANSAC (similarity) at ≤ 1100 px: template → image coordinates at
/// level 0. `None` when too few features agree.
fn feature_seed(t: &WorkFrame, i: &WorkFrame) -> Option<Homography> {
    let l = t.pyr.iter().position(|p| p.width.max(p.height) <= 1100)?;
    let (tp, ip) = (t.pyr.get(l)?, i.pyr.get(l)?);
    let ft = crate::features::detect(tp, 1500);
    let fi = crate::features::detect(ip, 1500);
    let m = crate::features::match_features(&ft, &fi, 0.8);
    let a: Vec<Point> = m.iter().map(|(p, _)| Point::new(ft.points[*p].x as f64, ft.points[*p].y as f64)).collect();
    let b: Vec<Point> = m.iter().map(|(_, q)| Point::new(fi.points[*q].x as f64, fi.points[*q].y as f64)).collect();
    let f = crate::ransac::ransac(crate::ransac::Model::Similarity, &a, &b, 1.5, 8, 7)?;
    if !neighbourly(&f.h, tp.width as f64, tp.height as f64) {
        return None;
    }
    Some(scale_homography(&f.h, 1.0 / (1u64 << l) as f64))
}

/// Brightness of `cur` relative to `prev` per channel: exp(median ln(cur / prev)) over mid-tone
/// pixels both frames show; 1 when fewer than 200 pixels qualify. Clamped to 0.5–2.
fn gains(prev: &WorkFrame, cur: &WorkFrame, m: &Homography) -> [f32; 3] {
    let (pw, _) = prev.size();
    if pw == 0 || prev.small.width == 0 {
        return [1.0; 3];
    }
    let s2 = prev.small.width as f64 / pw as f64;
    let ms = scale_homography(m, 1.0 / s2);
    let (sw, sh) = (cur.small.width as f64, cur.small.height as f64);
    let (lo_p, hi_p, lo_c, hi_c) = (0.02 * prev.clip, 0.85 * prev.clip, 0.02 * cur.clip, 0.85 * cur.clip);
    let mut ratios: [Vec<f32>; 3] = Default::default();
    let ps = &prev.small;
    for y in 0..ps.height {
        for x in 0..ps.width {
            let (cx, cy) = ms.apply_real(x as f64 + 0.5, y as f64 + 0.5);
            if !(cx >= 1.0 && cy >= 1.0 && cx <= sw - 1.0 && cy <= sh - 1.0) {
                continue;
            }
            let a = ps.data[y * ps.width + x];
            let b = cur.small.sample_bilinear(cx as f32, cy as f32);
            if a.iter().all(|v| *v > lo_p && *v < hi_p) && b.iter().all(|v| *v > lo_c && *v < hi_c) {
                for c in 0..3 {
                    ratios[c].push((b[c] / a[c]).ln());
                }
            }
        }
    }
    ratios.map(|mut r| {
        if r.len() < 200 {
            return 1.0;
        }
        let mid = r.len() / 2;
        r.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        r[mid].exp().clamp(0.5, 2.0)
    })
}

/// The registration of a whole sequence to its reference.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Per sequence position: reference → frame work coordinates (identity for skipped frames).
    pub p: Vec<Homography>,
    /// Per position: brightness relative to the reference (frames are divided by it).
    pub gain: Vec<[f32; 3]>,
    pub pairs: Vec<Pair>,
    /// Sequence position of the reference.
    pub reference: usize,
    pub work_size: (usize, usize),
    /// The area every used frame covers, in reference work pixels (x, y, w, h); `None` = all.
    pub common: Option<[f64; 4]>,
    pub warnings: Vec<String>,
}

impl Chain {
    pub fn used(&self, pos: usize) -> bool {
        self.pairs.get(pos).is_some_and(|p| !p.skipped)
    }
}

/// Chain the pairs of `a` to a reference.
pub fn chain(a: &Aligner, reference: Reference, common: bool) -> Result<Chain> {
    let n = a.pairs.len();
    let used: Vec<usize> = (0..n).filter(|&k| !a.pairs[k].skipped).collect();
    let Some(&first) = used.first() else { return Err(MergeError::TooFew(0, 2)) };
    let r = match reference {
        Reference::Middle => used.get(used.len() / 2).copied().unwrap_or(first),
        Reference::First => first,
        Reference::Sharpest => used.iter().copied().max_by(|&x, &y| a.sharpness[x].total_cmp(&a.sharpness[y])).unwrap_or(first),
    };
    let ri = used.iter().position(|&k| k == r).unwrap_or(0);
    let mut p = vec![Homography::IDENTITY; n];
    let mut gain = vec![[1f32; 3]; n];
    for w in ri + 1..used.len() {
        let (prev, k) = (used[w - 1], used[w]);
        p[k] = a.pairs[k].m.mul(&p[prev]);
        let g = a.pairs[k].gain;
        gain[k] = [gain[prev][0] * g[0], gain[prev][1] * g[1], gain[prev][2] * g[2]];
    }
    for w in (0..ri).rev() {
        let (k, next) = (used[w], used[w + 1]);
        let inv = a.pairs[next].m.inverse().unwrap_or(Homography::IDENTITY);
        p[k] = inv.mul(&p[next]);
        let g = a.pairs[next].gain;
        gain[k] = [gain[next][0] / g[0], gain[next][1] / g[1], gain[next][2] / g[2]];
    }
    let work_size = a.size().unwrap_or((0, 0));
    let mut warnings = Vec::new();
    let common = if common { common_area(&p, &used, work_size) } else { None };
    if let Some(c) = common
        && c[2] * c[3] < 0.6 * (work_size.0 * work_size.1) as f64
    {
        warnings.push(format!(
            "the frames only share {:.0} % of the picture (the result is cropped to it); check the alignment",
            100.0 * c[2] * c[3] / (work_size.0 * work_size.1).max(1) as f64
        ));
    }
    let poor: Vec<String> = used.iter().filter(|&&k| a.pairs[k].ncc < WARN_NCC).map(|&k| (k + 1).to_string()).collect();
    if !poor.is_empty() {
        warnings.push(format!("photo(s) {} in focus order aligned poorly; check them in the result", poor.join(", ")));
    }
    let skipped = n - used.len();
    if skipped > 0 {
        warnings.push(format!("{skipped} photo(s) didn't match their neighbours and were left out"));
    }
    let v = order::violations(&used.iter().map(|&k| a.grids[k].as_slice()).collect::<Vec<_>>());
    if v > 0.2 && used.len() >= 6 {
        warnings.push(format!("{:.0} % of the photos look out of focus order; try order = auto", v * 100.0));
    }
    Ok(Chain { p, gain, pairs: a.pairs.clone(), reference: r, work_size, common, warnings })
}

/// The largest rectangle (reference work pixels) inside every used frame.
fn common_area(p: &[Homography], used: &[usize], (w, h): (usize, usize)) -> Option<[f64; 4]> {
    if w == 0 || h == 0 {
        return None;
    }
    if used.iter().all(|&k| p[k] == Homography::IDENTITY) {
        return None;
    }
    let f = (w.max(h) as f64 / 512.0).max(1.0);
    let (aw, ah) = (((w as f64 / f).ceil() as usize).max(1), ((h as f64 / f).ceil() as usize).max(1));
    let margin = 2.0;
    let mut alpha = Plane::new(aw, ah);
    alpha.data.par_chunks_mut(aw).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            // the alpha pixel's footprint corners, in work pixels
            let (x0, y0, x1, y1) = (x as f64 * f, y as f64 * f, ((x + 1) as f64 * f).min(w as f64), ((y + 1) as f64 * f).min(h as f64));
            let inside = used.iter().all(|&k| {
                [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].iter().all(|&(px, py)| {
                    let (u, v) = p[k].apply_real(px, py);
                    u >= margin && v >= margin && u <= w as f64 - margin && v <= h as f64 - margin
                })
            });
            *o = if inside { 1.0 } else { 0.0 };
        }
    });
    let (x, y, cw, ch) = crate::pano::finish::auto_crop(&alpha)?;
    let r = [x as f64 * f, y as f64 * f, (cw as f64 * f).min(w as f64 - x as f64 * f), (ch as f64 * f).min(h as f64 - y as f64 * f)];
    (r[2] > 0.0 && r[3] > 0.0).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize) -> Rgb32f {
        Rgb32f::from_fn(w, h, |x, y| {
            let (x, y) = (x as f32, y as f32);
            let v = 0.3 + 0.15 * (x * 0.13).sin() * (y * 0.11).cos() + 0.1 * ((x + y) * 0.05).sin() + 0.08 * ((x * 0.31).sin() * (y * 0.29).sin());
            [v, v * 0.9, v * 0.8]
        })
    }

    fn shifted(src: &Rgb32f, dx: f32, dy: f32, gain: f32) -> Rgb32f {
        Rgb32f::from_fn(src.width, src.height, |x, y| src.sample_bilinear(x as f32 + 0.5 + dx, y as f32 + 0.5 + dy).map(|v| v * gain))
    }

    #[test]
    fn chains_shifts_and_gains_to_the_middle_frame() {
        let base = scene(360, 240);
        let frames = [shifted(&base, 0.0, 0.0, 1.0), shifted(&base, 1.5, -0.5, 1.1), shifted(&base, 3.0, -1.0, 1.21)];
        let mut a = Aligner::new(Dof::SIMILARITY, true, true, 2048);
        for (i, f) in frames.iter().enumerate() {
            a.push(i, f, 1.0).unwrap();
        }
        let c = chain(&a, Reference::Middle, true).unwrap();
        assert_eq!(c.reference, 1);
        // frame k(x) = base(x + d_k): P_k maps reference coordinates to frame k's: x − d_k + d_ref
        let p0 = c.p[0].apply(Point::new(100.0, 100.0));
        assert!((p0.x - 101.5).abs() < 0.05 && (p0.y - 99.5).abs() < 0.05, "{p0:?}");
        let p2 = c.p[2].apply(Point::new(100.0, 100.0));
        assert!((p2.x - 98.5).abs() < 0.05 && (p2.y - 100.5).abs() < 0.05, "{p2:?}");
        assert!((c.gain[0][0] - 1.0 / 1.1).abs() < 0.01 && (c.gain[2][1] - 1.1).abs() < 0.01, "{:?}", c.gain);
        let common = c.common.unwrap();
        assert!(common[2] > 340.0 && common[2] < 360.0 && common[3] > 225.0, "{common:?}");
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    }

    #[test]
    fn a_foreign_frame_is_skipped_and_the_chain_continues() {
        let base = scene(320, 200);
        let other = Rgb32f::from_fn(320, 200, |x, y| {
            let v = if ((x / 7) + (y / 5)) % 2 == 0 { 0.7 } else { 0.1 };
            [v; 3]
        });
        let mut a = Aligner::new(Dof::SIMILARITY, true, true, 2048);
        for (i, f) in [base.clone(), other, shifted(&base, 1.0, 0.0, 1.0), shifted(&base, 2.0, 0.0, 1.0)].iter().enumerate() {
            a.push(i, f, 1.0).unwrap();
        }
        assert!(a.pairs[1].skipped, "{:?}", a.pairs[1]);
        let c = chain(&a, Reference::First, true).unwrap();
        assert_eq!(c.reference, 0);
        let p3 = c.p[3].apply(Point::new(50.0, 50.0));
        assert!((p3.x - 48.0).abs() < 0.05, "{p3:?}");
        assert!(c.warnings.iter().any(|w| w.contains("left out")));
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let mut a = Aligner::new(Dof::SIMILARITY, true, true, 2048);
        a.push(0, &scene(200, 150), 1.0).unwrap();
        assert!(matches!(a.push(1, &scene(150, 200), 1.0), Err(MergeError::Mismatch(_))));
    }
}
