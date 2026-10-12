//! Inverse-compositional Lucas–Kanade image alignment (S. Baker & I. Matthews, "Lucas-Kanade 20
//! Years On: A Unifying Framework", IJCV 2004), coarse to fine over grey pyramids.
//!
//! - The warp `W` maps template coordinates to image coordinates; its linear part is a similarity
//!   (scale, rotation, shift — each degree of freedom can be frozen) or a full affine map.
//! - Both images are normalised to zero mean and unit variance over the compared pixels, which
//!   makes the fit invariant to gain and offset (focus brackets drift in brightness).
//! - The template's steepest-descent images are computed once per level. The Gauss–Newton system
//!   is re-accumulated each iteration with Huber weights (the scale from the previous iteration)
//!   times a focus-change weight: `(min/max)⁴` of the two frames' local contrast at the pixel.
//!   Where one frame is much sharper than the other, the difference is the focus change, not
//!   motion; on synthetic brackets this weight cut the chained error from 0.57 to 0.21 px.
//! - Fine levels use the pixels with the strongest template gradients only (≤ 120 k): flat areas
//!   carry no information about the motion.
//! - Parameters live in coordinates centred on the image and scaled by half its long edge, which
//!   keeps the system well conditioned.

use lightcraft_geom::Homography;
use lightcraft_raster::Plane;
use rayon::prelude::*;

use crate::align::scale_homography;
use crate::linalg;

/// Which parameters of the motion are estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dof {
    pub shift: bool,
    pub rotation: bool,
    pub scale: bool,
    /// A full affine map (all six parameters; overrides the three flags above).
    pub affine: bool,
}

impl Dof {
    pub const SIMILARITY: Dof = Dof { shift: true, rotation: true, scale: true, affine: false };

    pub fn any(&self) -> bool {
        self.affine || self.shift || self.rotation || self.scale
    }

    /// Parameter layout: affine `[a00−1, a01, a10, a11−1, tx, ty]`; similarity family a subset of
    /// `[scale, rotation, tx, ty]` (in that order).
    fn params(&self) -> Vec<Param> {
        if self.affine {
            return vec![Param::A00, Param::A01, Param::A10, Param::A11, Param::Tx, Param::Ty];
        }
        let mut v = Vec::new();
        if self.scale {
            v.push(Param::Scale);
        }
        if self.rotation {
            v.push(Param::Rotation);
        }
        if self.shift {
            v.push(Param::Tx);
            v.push(Param::Ty);
        }
        v
    }

    /// The name reported for a fit.
    pub fn model(&self) -> &'static str {
        match (self.affine, self.scale, self.rotation, self.shift) {
            (true, ..) => "affine",
            (_, true, true, _) => "similarity",
            (_, true, false, _) => "scale",
            (_, false, true, _) => "rigid",
            (_, false, false, true) => "translation",
            _ => "identity",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    A00,
    A01,
    A10,
    A11,
    Scale,
    Rotation,
    Tx,
    Ty,
}

/// A fit: the warp at level 0 and the normalised cross-correlation it reached there.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub h: Homography,
    pub ncc: f32,
}

/// One template sample: position, normalised value and steepest-descent row.
#[derive(Clone, Copy)]
struct Sample {
    x: f32,
    y: f32,
    t: f32,
    sd: [f32; 6],
}

/// Samples per level are capped at this count (strongest gradients first).
const MAX_SAMPLES: usize = 120_000;

/// Normalised cross-correlation of the template and the image warped by `h` (both at level 0
/// of their pyramids), over the template's inner area.
pub fn ncc(t: &Plane, i: &Plane, h: &Homography) -> f32 {
    let (w, ht) = (t.width, t.height);
    if w < 4 || ht < 4 {
        return 0.0;
    }
    let (mx, my) = margins(w, ht);
    let step = ((w * ht) / 50_000).max(1);
    let pts: Vec<(f32, f32)> = (my..ht - my)
        .step_by(step.isqrt().max(1))
        .flat_map(|y| (mx..w - mx).step_by(step.isqrt().max(1)).map(move |x| (x as f32 + 0.5, y as f32 + 0.5)))
        .collect();
    let pairs: Vec<(f64, f64)> = pts
        .par_iter()
        .filter_map(|&(x, y)| {
            let (sx, sy) = h.apply_real(x as f64, y as f64);
            sample(i, sx as f32, sy as f32).map(|v| (crate::features::bilinear(t, x, y) as f64, v as f64))
        })
        .collect();
    correlation(&pairs)
}

fn correlation(pairs: &[(f64, f64)]) -> f32 {
    let n = pairs.len() as f64;
    if n < 16.0 {
        return 0.0;
    }
    let (sa, sb) = pairs.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
    let (ma, mb) = (sa / n, sb / n);
    let (mut cab, mut caa, mut cbb) = (0.0, 0.0, 0.0);
    for &(a, b) in pairs {
        cab += (a - ma) * (b - mb);
        caa += (a - ma) * (a - ma);
        cbb += (b - mb) * (b - mb);
    }
    if caa <= 1e-12 || cbb <= 1e-12 {
        return 0.0;
    }
    (cab / (caa * cbb).sqrt()) as f32
}

fn margins(w: usize, h: usize) -> (usize, usize) {
    let mx = ((w as f32 * 0.04) as usize + 2).min(w / 4);
    let my = ((h as f32 * 0.04) as usize + 2).min(h / 4);
    (mx, my)
}

/// Bilinear sample at continuous coordinates (pixel centres at +0.5); `None` outside the image.
#[inline]
fn sample(img: &Plane, x: f32, y: f32) -> Option<f32> {
    if !(x >= 0.5 && y >= 0.5 && x <= img.width as f32 - 0.5 && y <= img.height as f32 - 0.5) {
        return None;
    }
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx as usize, fy as usize);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let x1 = (x0 + 1).min(img.width - 1);
    let y1 = (y0 + 1).min(img.height - 1);
    let w = img.width;
    let d = &img.data;
    let (a, b, c, e) = (d[y0 * w + x0], d[y0 * w + x1], d[y1 * w + x0], d[y1 * w + x1]);
    let top = a + (b - a) * tx;
    let bot = c + (e - c) * tx;
    Some(top + (bot - top) * ty)
}

/// Local contrast: the modified Laplacian averaged over 5 × 5 (one map per pyramid level, made
/// once per frame: each frame is aligned twice, to its neighbours on both sides).
pub fn contrast(p: &Plane) -> Plane {
    let (w, h) = (p.width, p.height);
    let mut out = Plane::new(w, h);
    if w >= 3 && h >= 3 {
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
    }
    lightcraft_raster::blur::box_mean(&out, 2)
}

/// The template samples of one level: normalised values and steepest-descent rows.
fn template_samples(t: &Plane, params: &[Param], centre: (f64, f64), radius: f64) -> Vec<Sample> {
    let (w, h) = (t.width, t.height);
    if w < 8 || h < 8 {
        return Vec::new();
    }
    let (mx, my) = margins(w, h);
    let at = |x: usize, y: usize| t.data[y * w + x];
    let grad = |x: usize, y: usize| (0.5 * (at(x + 1, y) - at(x - 1, y)), 0.5 * (at(x, y + 1) - at(x, y - 1)));
    let (xs, ys) = (mx.max(1)..(w - mx).min(w - 1), my.max(1)..(h - my).min(h - 1));
    let area = xs.len() * ys.len();
    // large levels: candidates on a grid (at most 4 × the samples kept), and only gradients above
    // the strength a sparse subsample says keeps about MAX_SAMPLES of them
    let stride = ((area as f64 / (4 * MAX_SAMPLES) as f64).sqrt().floor() as usize).max(1);
    let candidates = area / (stride * stride);
    let thr = if candidates > MAX_SAMPLES {
        let sub = (stride * ((candidates / 20_000) as f64).sqrt().ceil() as usize).max(1);
        let mut g: Vec<f32> = ys
            .clone()
            .step_by(sub)
            .flat_map(|y| xs.clone().step_by(sub).map(move |x| (x, y)))
            .map(|(x, y)| {
                let (gx, gy) = grad(x, y);
                gx * gx + gy * gy
            })
            .collect();
        g.sort_by(|a, b| b.total_cmp(a));
        let keep = MAX_SAMPLES as f64 / candidates as f64;
        g.get(((g.len() as f64 * keep) as usize).min(g.len().saturating_sub(1))).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let mut pts: Vec<(usize, usize, f32, f32)> = ys
        .clone()
        .step_by(stride)
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map_iter(|y| {
            xs.clone().step_by(stride).filter_map(move |x| {
                let (gx, gy) = grad(x, y);
                (gx * gx + gy * gy >= thr).then_some((x, y, gx, gy))
            })
        })
        .collect();
    if pts.len() > MAX_SAMPLES {
        let s = pts.len().div_ceil(MAX_SAMPLES);
        pts = pts.into_iter().step_by(s).collect();
    }
    let n = pts.len() as f64;
    if n < 16.0 {
        return Vec::new();
    }
    let mean = pts.iter().map(|p| at(p.0, p.1) as f64).sum::<f64>() / n;
    let var = pts.iter().map(|p| (at(p.0, p.1) as f64 - mean).powi(2)).sum::<f64>() / n;
    if var <= 1e-14 {
        return Vec::new();
    }
    let inv = 1.0 / var.sqrt();
    pts.into_iter()
        .map(|(x, y, gx, gy)| {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let (xn, yn) = ((px - centre.0) / radius, (py - centre.1) / radius);
            let (gx, gy) = (gx as f64 * inv * radius, gy as f64 * inv * radius);
            let mut sd = [0f32; 6];
            for (k, p) in params.iter().enumerate() {
                sd[k] = match p {
                    Param::A00 => gx * xn,
                    Param::A01 => gx * yn,
                    Param::A10 => gy * xn,
                    Param::A11 => gy * yn,
                    Param::Scale => gx * xn + gy * yn,
                    Param::Rotation => gy * xn - gx * yn,
                    Param::Tx => gx,
                    Param::Ty => gy,
                } as f32;
            }
            Sample { x: px as f32, y: py as f32, t: ((at(x, y) as f64 - mean) * inv) as f32, sd }
        })
        .collect()
}

/// The incremental warp of parameters `dp` in absolute pixel coordinates.
fn increment(params: &[Param], dp: &[f64], centre: (f64, f64), radius: f64) -> Homography {
    let (mut a, mut t) = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]);
    for (p, v) in params.iter().zip(dp) {
        match p {
            Param::A00 => a[0] += v,
            Param::A01 => a[1] += v,
            Param::A10 => a[2] += v,
            Param::A11 => a[3] += v,
            Param::Scale => {
                a[0] += v;
                a[3] += v;
            }
            Param::Rotation => {
                a[1] -= v;
                a[2] += v;
            }
            Param::Tx => t[0] += v,
            Param::Ty => t[1] += v,
        }
    }
    let (cx, cy) = centre;
    Homography([a[0], a[1], cx - (a[0] * cx + a[1] * cy) + radius * t[0], a[2], a[3], cy - (a[2] * cx + a[3] * cy) + radius * t[1], 0.0, 0.0, 1.0])
}

/// Keep `h` inside its motion model: without scale its linear part becomes a pure rotation.
fn constrain(h: Homography, dof: Dof) -> Homography {
    if dof.affine || dof.scale {
        return h;
    }
    let m = h.0;
    let (c, s) = if dof.rotation {
        let th = (m[3] - m[1]).atan2(m[0] + m[4]);
        (th.cos(), th.sin())
    } else {
        (1.0, 0.0)
    };
    Homography([c, -s, m[2], s, c, m[5], 0.0, 0.0, 1.0])
}

/// Largest displacement of `h` at the corners of a `w × h` area (how far from the identity it is).
fn corner_motion(h: &Homography, w: f64, ht: f64) -> f64 {
    [(0.0, 0.0), (w, 0.0), (0.0, ht), (w, ht)]
        .iter()
        .map(|&(x, y)| {
            let (u, v) = h.apply_real(x, y);
            ((u - x).powi(2) + (v - y).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Refine `init` on one level (`tc`, `ic`: the images' [`contrast`] maps). Returns the warp and
/// the NCC of the last iteration.
fn refine_level(t: &Plane, i: &Plane, tc: &Plane, ic: &Plane, init: Homography, dof: Dof, iters: usize) -> (Homography, f32) {
    let params = dof.params();
    let k = params.len();
    let centre = (t.width as f64 / 2.0, t.height as f64 / 2.0);
    let radius = (t.width.max(t.height) as f64 / 2.0).max(1.0);
    let samples = template_samples(t, &params, centre, radius);
    if k == 0 || samples.len() < 32 {
        return (init, f32::NAN);
    }
    let (w, ht) = (t.width as f64, t.height as f64);
    // pixels much sharper in one frame than in the other mostly show the focus change, not motion
    let (ct, ci) = (tc, ic);
    let sw: Vec<f32> = samples
        .par_iter()
        .map(|s| {
            let (x, y) = init.apply_real(s.x as f64, s.y as f64);
            let a = crate::features::bilinear(ct, s.x, s.y);
            let b = crate::features::bilinear(ci, x as f32, y as f32);
            let r = a.min(b) / a.max(b).max(1e-9);
            r.powi(4)
        })
        .collect();
    let mut h = init;
    let mut ncc = f32::NAN;
    let mut delta = f64::INFINITY; // Huber threshold (normalised units)
    for _ in 0..iters {
        let vals: Vec<f32> = samples
            .par_iter()
            .map(|s| {
                let (x, y) = h.apply_real(s.x as f64, s.y as f64);
                sample(i, x as f32, y as f32).unwrap_or(f32::NAN)
            })
            .collect();
        let (n, sum, sum2) =
            vals.iter().filter(|v| v.is_finite()).fold((0usize, 0f64, 0f64), |(n, a, b), &v| (n + 1, a + v as f64, b + (v as f64).powi(2)));
        if n < samples.len() / 2 || n < 32 {
            break;
        }
        let mean = sum / n as f64;
        let var = (sum2 / n as f64 - mean * mean).max(0.0);
        if var <= 1e-14 {
            break;
        }
        let inv = 1.0 / var.sqrt();
        // accumulate the weighted normal equations
        let zero = || (vec![0f64; k * k], vec![0f64; k], 0f64, 0f64);
        let (hm, b, se2, sabs) = samples
            .par_iter()
            .zip(vals.par_iter())
            .zip(sw.par_iter())
            .fold(zero, |(mut hm, mut b, mut se2, mut sabs), ((s, &v), &sw)| {
                if v.is_finite() {
                    let e = (v as f64 - mean) * inv - s.t as f64;
                    let wgt = if e.abs() <= delta { 1.0 } else { delta / e.abs() } * sw as f64;
                    for r in 0..k {
                        let sr = s.sd[r] as f64 * wgt;
                        b[r] += sr * e;
                        for c in r..k {
                            hm[r * k + c] += sr * s.sd[c] as f64;
                        }
                    }
                    se2 += e * e;
                    sabs += e.abs();
                }
                (hm, b, se2, sabs)
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.0.iter_mut().zip(&b.0) {
                    *x += y;
                }
                for (x, y) in a.1.iter_mut().zip(&b.1) {
                    *x += y;
                }
                (a.0, a.1, a.2 + b.2, a.3 + b.3)
            });
        ncc = (1.0 - 0.5 * se2 / n as f64) as f32;
        delta = 1.345 * 1.25 * sabs / n as f64;
        let mut full = hm;
        for r in 0..k {
            for c in 0..r {
                full[r * k + c] = full[c * k + r];
            }
        }
        let Some(dp) = linalg::solve(full, b, k) else { break };
        let d = increment(&params, &dp, centre, radius);
        let Some(dinv) = d.inverse() else { break };
        let next = constrain(h.mul(&dinv), dof);
        if !crate::align::plausible(&next, w, ht, 0.2) {
            break;
        }
        h = next;
        if corner_motion(&d, w, ht) < 0.005 {
            break;
        }
    }
    (h, ncc)
}

/// A grey pyramid (level 0 first, each level half the size of the previous one) and its
/// [`contrast`] maps.
#[derive(Clone, Copy)]
pub struct Pyramid<'a> {
    pub grey: &'a [Plane],
    pub contrast: &'a [Plane],
}

/// Align the image pyramid `i` to the template pyramid `t`, starting from `init` (level-0
/// coordinates). Iterations per level, coarse to fine: 30, 30, 20, 15, 15…
pub fn align(t: Pyramid, i: Pyramid, init: &Homography, dof: Dof) -> Fit {
    let (tc, ic) = (t.contrast, i.contrast);
    let (t, i) = (t.grey, i.grey);
    let levels = t.len().min(i.len()).min(tc.len()).min(ic.len());
    if levels == 0 || !dof.any() {
        let ncc = match (t.first(), i.first()) {
            (Some(a), Some(b)) => ncc(a, b, init),
            _ => 0.0,
        };
        return Fit { h: *init, ncc };
    }
    let mut h = *init;
    for (step, l) in (0..levels).rev().enumerate() {
        let f = (1u64 << l) as f64;
        let hl = scale_homography(&h, f);
        let iters = [30, 30, 20][..].get(step).copied().unwrap_or(15);
        let (r, _) = refine_level(&t[l], &i[l], &tc[l], &ic[l], hl, dof, iters);
        h = scale_homography(&r, 1.0 / f);
    }
    Fit { h, ncc: ncc(&t[0], &i[0], &h) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_raster::blur::gaussian;
    use lightcraft_raster::resample::half;

    fn texture(w: usize, h: usize) -> Plane {
        let p = Plane::from_fn(w, h, |x, y| {
            let (x, y) = (x as f32, y as f32);
            0.5 + 0.2 * (x * 0.21).sin() * (y * 0.17).cos() + 0.15 * ((x + 2.0 * y) * 0.09).sin() + 0.1 * ((x * x + y * y).sqrt() * 0.3).sin()
        });
        gaussian(&p, 1.0)
    }

    fn contrasts(p: &[Plane]) -> Vec<Plane> {
        p.iter().map(contrast).collect()
    }

    fn pyr<'a>(g: &'a [Plane], c: &'a [Plane]) -> Pyramid<'a> {
        Pyramid { grey: g, contrast: c }
    }

    fn pyramid(p: Plane) -> Vec<Plane> {
        let mut v = vec![p];
        while let Some(l) = v.last() {
            if l.width.max(l.height) < 128 {
                break;
            }
            let n = half(l);
            v.push(n);
        }
        v
    }

    fn warp(p: &Plane, h: &Homography) -> Plane {
        Plane::from_fn(p.width, p.height, |x, y| {
            let (sx, sy) = h.apply_real(x as f64 + 0.5, y as f64 + 0.5);
            crate::features::bilinear(p, sx as f32, sy as f32)
        })
    }

    #[test]
    fn recovers_a_similarity() {
        let big = texture(560, 400);
        // image(truth(x)) = template(x): the fit must find `truth` (template → image coordinates)
        let (c, s) = (1.004f64 * 0.3f64.to_radians().cos(), 1.004f64 * 0.3f64.to_radians().sin());
        let (cx, cy) = (280.0, 200.0);
        let truth = Homography([c, -s, cx - (c * cx - s * cy) + 3.2, s, c, cy - (s * cx + c * cy) - 1.7, 0.0, 0.0, 1.0]);
        let img = warp(&big, &truth.inverse().unwrap());
        let (a, b) = (pyramid(big.clone()), pyramid(img));
        let (ca, cb) = (contrasts(&a), contrasts(&b));
        let fit = align(pyr(&a, &ca), pyr(&b, &cb), &Homography::IDENTITY, Dof::SIMILARITY);
        let err = (0..4)
            .map(|k| {
                let (x, y) = (100.0 + 360.0 * (k % 2) as f64, 80.0 + 240.0 * (k / 2) as f64);
                let a = fit.h.apply_real(x, y);
                let b = truth.apply_real(x, y);
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(err < 0.05, "corner error {err} px, fit {:?}", fit.h);
        assert!(fit.ncc > 0.99, "ncc {}", fit.ncc);
    }

    #[test]
    fn gain_and_offset_do_not_matter_and_frozen_dofs_stay_frozen() {
        let t = texture(300, 220);
        let shift = Homography([1.0, 0.0, 2.4, 0.0, 1.0, -1.1, 0.0, 0.0, 1.0]);
        let mut img = warp(&t, &shift);
        for v in img.data.iter_mut() {
            *v = *v * 0.7 + 0.1;
        }
        let dof = Dof { shift: true, rotation: false, scale: false, affine: false };
        let (a, b) = (pyramid(t), pyramid(img));
        let (ca, cb) = (contrasts(&a), contrasts(&b));
        let fit = align(pyr(&a, &ca), pyr(&b, &cb), &Homography::IDENTITY, dof);
        let m = fit.h.0;
        // image(x) = template(x + (2.4, −1.1)), so W(x) = x − (2.4, −1.1)
        assert!((m[2] + 2.4).abs() < 0.03 && (m[5] - 1.1).abs() < 0.03, "{m:?}");
        assert_eq!((m[0], m[1], m[3], m[4]), (1.0, 0.0, 0.0, 1.0));
        assert_eq!(dof.model(), "translation");
    }

    #[test]
    fn flat_images_keep_the_initial_warp() {
        let flat = Plane::filled(64, 48, 0.5);
        let (g, c) = (std::slice::from_ref(&flat), [contrast(&flat)]);
        let fit = align(pyr(g, &c), pyr(g, &c), &Homography::IDENTITY, Dof::SIMILARITY);
        assert_eq!(fit.h, Homography::IDENTITY);
        assert_eq!(fit.ncc, 0.0);
    }
}
