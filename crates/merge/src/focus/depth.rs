//! Method B, depth map: each pixel comes from the frame where it is sharpest, chosen on a smoothed
//! depth map. Fine detail, crossing hairs and textures stay crisp and colours come untouched from
//! one or two source frames; the price is a slight halo at depth jumps, kept narrow by the
//! confidence below.
//!
//! **Streaming.** Per pixel the state keeps the sharpest frame so far, its focus measure and its
//! neighbours' (the frame before is remembered when a new winner appears, the frame after fills
//! in when it arrives), their colours, and the sum of all measures — 46 bytes per output pixel
//! whatever the number of frames.
//!
//! **Depth.** A parabola through the log focus measures of the winner and its neighbours gives a
//! sub-frame depth (a Gaussian focus profile). Confidence is the product of
//! - contrast above noise: `smoothstep(τ, 4τ, fm)` with τ the frames' median focus measure;
//! - peakedness: `1 − mean / max` over the frames (flat profiles carry no depth);
//! - halo: `fm / (½ · max fm within 4 × Radius)` clamped to 1 (pixels much weaker than a strong
//!   edge nearby are dominated by that edge's blur), after focus-stack's halo mask
//!   (P. Aimonen, MIT, docs/Algorithms.md; idea only).
//!
//! Low-confidence pixels are filled from their surroundings (push–pull), then the depth map is
//! smoothed by a confidence-weighted guided filter with the winners' grey image as the guide
//! (radius 2 × Smoothing), which keeps depth edges on image edges. Smoothing 0 keeps the raw
//! per-pixel winners.
//!
//! **Colour.** The depth's two nearest frames are blended (or the nearest one taken). When
//! smoothing moved a pixel's depth beyond the stored neighbours of its winner, that pixel is
//! marked and its colour gathered in a second pass over just the frames it needs.

use lightcraft_raster::{Plane, Rgb32f};
use rayon::prelude::*;

use super::filters::{max_filter, smoothstep, weighted_guided};
use super::measure::{Warped, grey};
use crate::pano::blend::{Buf, push_pull};

/// Colour encoding of the stored candidates: linear 0..2 in 16 bits.
const CSCALE: f32 = 32768.0;

#[inline]
fn enc(p: [f32; 3]) -> [u16; 3] {
    p.map(|v| (v * CSCALE).round().clamp(0.0, 65535.0) as u16)
}

#[inline]
fn dec(c: [u16; 3]) -> [f32; 3] {
    c.map(|v| v as f32 / CSCALE)
}

/// No measure recorded (the neighbouring frame didn't cover the pixel, or doesn't exist).
const NONE: f32 = -1.0;

pub struct Depth {
    w: usize,
    h: usize,
    best: Vec<u16>,
    fm_best: Vec<f32>,
    fm_left: Vec<f32>,
    fm_right: Vec<f32>,
    fm_sum: Vec<f32>,
    col_best: Vec<[u16; 3]>,
    col_left: Vec<[u16; 3]>,
    col_right: Vec<[u16; 3]>,
    prev_col: Vec<[u16; 3]>,
    prev_fm: Vec<f32>,
    count: usize,
    noise: f32,
}

/// Parameters of the depth refinement (in output pixels).
#[derive(Clone, Copy, Debug)]
pub struct DepthParams {
    pub radius: usize,
    pub smoothing: f32,
    pub interpolate: bool,
}

impl Depth {
    pub fn new(w: usize, h: usize) -> Depth {
        let n = w * h;
        Depth {
            w,
            h,
            best: vec![0; n],
            fm_best: vec![NONE; n],
            fm_left: vec![NONE; n],
            fm_right: vec![NONE; n],
            fm_sum: vec![0.0; n],
            col_best: vec![[0; 3]; n],
            col_left: vec![[0; 3]; n],
            col_right: vec![[0; 3]; n],
            prev_col: vec![[0; 3]; n],
            prev_fm: vec![NONE; n],
            count: 0,
            noise: f32::INFINITY,
        }
    }

    pub fn bytes(&self) -> usize {
        self.best.len() * (2 + 4 * 6 + 6 * 4)
    }

    /// The state as another device would hand it over (tests of the device plumbing).
    pub fn into_state(self) -> super::kernels::DepthState {
        super::kernels::DepthState {
            best: self.best,
            fm_best: self.fm_best,
            fm_left: self.fm_left,
            fm_right: self.fm_right,
            fm_sum: self.fm_sum,
            prev_fm: self.prev_fm,
            col_best: self.col_best,
            col_left: self.col_left,
            col_right: self.col_right,
            prev_col: self.prev_col,
            noise: self.noise,
        }
    }

    /// The state another device built ([`super::kernels::FocusKernels`]) after `count` frames;
    /// `None` when a vector doesn't have one value per pixel.
    pub fn from_state(w: usize, h: usize, s: super::kernels::DepthState, count: usize) -> Option<Depth> {
        let n = w * h;
        let ok = [s.best.len(), s.fm_best.len(), s.fm_left.len(), s.fm_right.len(), s.fm_sum.len(), s.prev_fm.len()]
            .iter()
            .chain(&[s.col_best.len(), s.col_left.len(), s.col_right.len(), s.prev_col.len()])
            .all(|&l| l == n);
        ok.then(|| Depth {
            w,
            h,
            best: s.best,
            fm_best: s.fm_best,
            fm_left: s.fm_left,
            fm_right: s.fm_right,
            fm_sum: s.fm_sum,
            col_best: s.col_best,
            col_left: s.col_left,
            col_right: s.col_right,
            prev_col: s.prev_col,
            prev_fm: s.prev_fm,
            count,
            noise: if s.noise.is_finite() && s.noise > 0.0 { s.noise } else { f32::INFINITY },
        })
    }

    /// Add the frame at position `pos` (0, 1, 2… in order) with its focus measure and noise floor.
    pub fn add(&mut self, pos: usize, frame: &Warped, fm: &Plane, noise: f32) {
        self.count = self.count.max(pos + 1);
        if noise.is_finite() && noise > 0.0 {
            self.noise = self.noise.min(noise);
        }
        let k = pos.min(u16::MAX as usize) as u16;
        let w = self.w.max(1);
        let rgb = &frame.rgb.data;
        let flags = frame.flags;
        let st = (
            self.best.par_chunks_mut(w),
            self.fm_best.par_chunks_mut(w),
            self.fm_left.par_chunks_mut(w),
            self.fm_right.par_chunks_mut(w),
            self.fm_sum.par_chunks_mut(w),
            self.col_best.par_chunks_mut(w),
            self.col_left.par_chunks_mut(w),
            self.col_right.par_chunks_mut(w),
            self.prev_col.par_chunks_mut(w),
            self.prev_fm.par_chunks_mut(w),
        );
        st.into_par_iter().enumerate().for_each(|(y, (best, fb, fl, fr, fs, cb, cl, cr, pc, pf))| {
            for x in 0..best.len() {
                let i = y * w + x;
                if flags[i] & super::measure::VALID == 0 {
                    pf[x] = NONE;
                    continue;
                }
                let f = fm.data[i];
                let c = enc(rgb[i]);
                fs[x] += f;
                if f > fb[x] {
                    if pf[x] >= 0.0 {
                        fl[x] = pf[x];
                        cl[x] = pc[x];
                    } else {
                        fl[x] = NONE;
                    }
                    best[x] = k;
                    fb[x] = f;
                    cb[x] = c;
                    fr[x] = NONE;
                } else if k > 0 && best[x] == k - 1 {
                    fr[x] = f;
                    cr[x] = c;
                }
                pf[x] = f;
                pc[x] = c;
            }
        });
    }

    /// Build the depth map and the colours the stored candidates can provide.
    pub fn finish(self, p: DepthParams) -> Resolved {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let m = self.count.max(1);
        let last = (m - 1) as f32;
        let tau = if self.noise.is_finite() { self.noise } else { 0.0 };
        // sub-frame depth and the first two confidence terms
        let mut d = Plane::new(w, h);
        let mut conf = Plane::new(w, h);
        d.data.par_iter_mut().zip(conf.data.par_iter_mut()).enumerate().for_each(|(i, (dv, cv))| {
            let b = self.best[i] as f32;
            let fb = self.fm_best[i];
            if fb < 0.0 {
                *dv = b;
                return;
            }
            let (fl, fr) = (self.fm_left[i], self.fm_right[i]);
            let mut delta = 0.0;
            if fl >= 0.0 && fr >= 0.0 {
                let e = 1e-6;
                let (l, c, r) = ((fl + e).ln(), (fb + e).ln(), (fr + e).ln());
                let den = l - 2.0 * c + r;
                if den < -1e-6 {
                    delta = (0.5 * (l - r) / den).clamp(-0.5, 0.5);
                }
            }
            *dv = (b + delta).clamp(0.0, last);
            let mean = self.fm_sum[i] / m as f32;
            let c1 = smoothstep(tau, 3.0 * tau, fb);
            let c2 = if mean > 0.0 { smoothstep(1.2, 2.0, fb / mean) } else { 0.0 };
            *cv = c1 * c2;
        });
        // halo term
        let fmb = Plane { width: w, height: h, data: self.fm_best.iter().map(|v| v.max(0.0)).collect() };
        let dil = max_filter(&fmb, 4 * p.radius.max(1));
        conf.data.par_iter_mut().enumerate().for_each(|(i, c)| {
            let ref_ = 0.5 * dil.data[i];
            if ref_ > 0.0 {
                *c *= (fmb.data[i] / ref_).clamp(0.0, 1.0);
            }
        });
        drop(dil);
        drop(fmb);
        // fill and smooth
        if p.smoothing > 0.0 && m > 1 {
            let wgt: Vec<f32> = conf.data.iter().map(|&c| if c >= 0.05 { c } else { 0.0 }).collect();
            let mut buf = Buf { w, h, c: 1, data: std::mem::take(&mut d.data) };
            push_pull(&mut buf, &wgt);
            d.data = buf.data;
            let guide = Plane { width: w, height: h, data: self.col_best.par_iter().map(|c| grey(dec(*c))).collect() };
            let cw = conf.map(|c| c + 0.02);
            let r = (2.0 * p.smoothing).round().max(1.0) as usize;
            d = weighted_guided(&guide, &d, &cw, r, 1e-3);
            d.data.par_iter_mut().for_each(|v| *v = v.clamp(0.0, last));
        }
        // colours from the stored candidates
        let mut image = Rgb32f::new(w, h);
        let mut need = vec![false; n];
        image.data.par_iter_mut().zip(need.par_iter_mut()).enumerate().for_each(|(i, (o, nd))| {
            let b = self.best[i] as usize;
            let colour = |k: usize| -> Option<[f32; 3]> {
                if k == b {
                    Some(dec(self.col_best[i]))
                } else if k + 1 == b && self.fm_left[i] >= 0.0 {
                    Some(dec(self.col_left[i]))
                } else if k == b + 1 && self.fm_right[i] >= 0.0 {
                    Some(dec(self.col_right[i]))
                } else {
                    None
                }
            };
            let dv = d.data[i];
            let got = if p.interpolate {
                let k0 = dv.floor().max(0.0) as usize;
                let t = dv - k0 as f32;
                if t < 1e-3 {
                    colour(k0)
                } else if t > 1.0 - 1e-3 {
                    colour(k0 + 1)
                } else {
                    match (colour(k0), colour(k0 + 1)) {
                        (Some(a), Some(c)) => Some([0, 1, 2].map(|j| a[j] * (1.0 - t) + c[j] * t)),
                        _ => None,
                    }
                }
            } else {
                colour(dv.round() as usize)
            };
            match got {
                Some(c) => *o = c,
                None => {
                    // the winner's colour until the gather pass provides better
                    *o = dec(self.col_best[i]);
                    *nd = true;
                }
            }
        });
        for v in d.data.iter_mut() {
            *v /= last.max(1.0);
        }
        Resolved { image, depth: d, confidence: conf, need, wsum: Vec::new(), last: m - 1, interpolate: p.interpolate }
    }
}

/// The depth result after the fuse pass, waiting for the colours of marked pixels.
pub struct Resolved {
    pub image: Rgb32f,
    /// Depth 0..1 (0 = first frame, 1 = frame `last`).
    pub depth: Plane,
    pub confidence: Plane,
    need: Vec<bool>,
    wsum: Vec<f32>,
    last: usize,
    interpolate: bool,
}

impl Resolved {
    pub fn bytes(&self) -> usize {
        self.image.data.len() * 12 + self.depth.data.len() * 8 + self.need.len() + self.wsum.len() * 4
    }

    /// Positions whose colours marked pixels still need.
    pub fn needed(&self) -> Vec<usize> {
        let last = self.last as f32;
        let mut set = vec![false; self.last + 1];
        for (i, &nd) in self.need.iter().enumerate() {
            if !nd {
                continue;
            }
            let dv = self.depth.data[i] * last;
            if self.interpolate {
                let k0 = dv.floor().max(0.0) as usize;
                for k in [k0, k0 + 1] {
                    if k <= self.last && (dv - k as f32).abs() < 1.0 {
                        set[k] = true;
                    }
                }
            } else if let Some(s) = set.get_mut(dv.round() as usize) {
                *s = true;
            }
        }
        (0..=self.last).filter(|&k| set[k]).collect()
    }

    /// Gather the colours frame `pos` contributes to marked pixels.
    pub fn add(&mut self, pos: usize, frame: &Warped) {
        if self.wsum.is_empty() {
            self.wsum = vec![0.0; self.need.len()];
        }
        let last = self.last as f32;
        let k = pos as f32;
        let w = self.image.width.max(1);
        let interp = self.interpolate;
        let (need, depth, rgb, flags) = (&self.need, &self.depth.data, &frame.rgb.data, frame.flags);
        self.image.data.par_chunks_mut(w).zip(self.wsum.par_chunks_mut(w)).enumerate().for_each(|(y, (row, ws))| {
            for x in 0..row.len() {
                let i = y * w + x;
                if !need[i] || flags[i] & super::measure::VALID == 0 {
                    continue;
                }
                let dv = depth[i] * last;
                let wt = if interp {
                    1.0 - (dv - k).abs()
                } else if dv.round() == k {
                    1.0
                } else {
                    0.0
                };
                if wt <= 0.0 {
                    continue;
                }
                let c = rgb[i];
                if ws[x] == 0.0 {
                    row[x] = [0.0; 3];
                }
                row[x] = [row[x][0] + wt * c[0], row[x][1] + wt * c[1], row[x][2] + wt * c[2]];
                ws[x] += wt;
            }
        });
    }

    /// The final image, depth and confidence.
    pub fn done(mut self) -> (Rgb32f, Plane, Plane) {
        if !self.wsum.is_empty() {
            let ws = &self.wsum;
            self.image.data.par_iter_mut().enumerate().for_each(|(i, o)| {
                if ws[i] > 0.0 {
                    *o = o.map(|v| v / ws[i]);
                }
            });
        }
        (self.image, self.depth, self.confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::focus::measure::VALID;

    fn frame(w: usize, h: usize, v: [f32; 3]) -> (Rgb32f, Vec<u8>) {
        (Rgb32f::filled(w, h, v), vec![VALID; w * h])
    }

    fn warped(f: &(Rgb32f, Vec<u8>)) -> Warped<'_> {
        Warped { rgb: &f.0, flags: &f.1 }
    }

    #[test]
    fn picks_the_sharpest_frame_with_a_subframe_offset() {
        let (w, h) = (8, 6);
        let mut d = Depth::new(w, h);
        // focus profile peaking between frames 2 and 3 (closer to 2)
        let fms = [0.01f32, 0.05, 0.5, 0.4, 0.05, 0.01];
        for (k, f) in fms.iter().enumerate() {
            d.add(k, &warped(&frame(w, h, [k as f32 * 0.1; 3])), &Plane::filled(w, h, *f), 0.005);
        }
        let r = d.finish(DepthParams { radius: 1, smoothing: 0.0, interpolate: true });
        assert!(r.needed().is_empty());
        let depth = r.depth.get(3, 3) * 5.0;
        assert!(depth > 2.0 && depth < 2.5, "{depth}");
        let (img, _, conf) = r.done();
        let want = 0.2 + (depth - 2.0) * 0.1;
        assert!((img.get(3, 3)[0] - want).abs() < 1e-3, "{:?} vs {want}", img.get(3, 3));
        assert!(conf.get(3, 3) > 0.5);
    }

    #[test]
    fn far_depths_are_gathered_in_a_second_pass() {
        let (w, h, n) = (32, 4, 9);
        let mut d = Depth::new(w, h);
        // left part sharp in frame 0, right part in frame 8; between them no texture at all, so the
        // depth there is filled in from both sides, far from the (arbitrary) winner, frame 0
        for k in 0..n {
            let fm = Plane::from_fn(w, h, |x, _| {
                if (12..20).contains(&x) {
                    0.0
                } else if (x < 12 && k == 0) || (x >= 20 && k == n - 1) {
                    1.0
                } else {
                    0.1
                }
            });
            d.add(k, &warped(&frame(w, h, [k as f32 * 0.1; 3])), &fm, 0.05);
        }
        let mut r = d.finish(DepthParams { radius: 1, smoothing: 2.0, interpolate: false });
        let needed = r.needed();
        assert!(!needed.is_empty());
        for k in needed {
            r.add(k, &warped(&frame(w, h, [k as f32 * 0.1; 3])));
        }
        let (img, depth, _) = r.done();
        // every pixel's colour is the colour of the frame its depth names
        for x in 0..w {
            let k = (depth.get(x, 2) * (n - 1) as f32).round();
            assert!((img.get(x, 2)[0] - k * 0.1).abs() < 1e-4, "x {x}: {:?} vs frame {k}", img.get(x, 2));
        }
        assert!(img.get(2, 2)[0].abs() < 1e-4);
        assert!((img.get(30, 2)[0] - 0.8).abs() < 1e-4);
        let mid = depth.get(16, 2) * 8.0;
        assert!(mid > 1.0 && mid < 7.0, "{mid}");
    }
}
