//! Per-frame work of the fuse pass: resampling a frame into the output geometry (with its
//! brightness correction), the grey image focus is measured on, and the focus measure itself.
//!
//! The focus measure is the sum-modified Laplacian (S. K. Nayar & Y. Nakagawa, "Shape from
//! Focus", IEEE PAMI 1994): `|2Y − Y(x−1) − Y(x+1)| + |2Y − Y(y−1) − Y(y+1)|` on a
//! square-root-encoded grey image (perceptually even, cheap), averaged over a `(2r+1)²` window.
//! Saturated and invalid pixels measure 0. (Measures compared: Pertuz et al., "Analysis of focus
//! measure operators for shape-from-focus", Pattern Recognition 2013.)

use std::borrow::Cow;

use lightcraft_geom::Homography;
use lightcraft_raster::blur::box_mean;
use lightcraft_raster::{Plane, Rgb32f};
use rayon::prelude::*;

/// The pixel holds image data.
pub const VALID: u8 = 1;
/// A channel of the source pixel is at or near the clip level.
pub const SATURATED: u8 = 2;

/// A rectangle of the reference frame's pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// A frame resampled into the output geometry: brightness-corrected linear RGB and per-pixel
/// flags ([`VALID`], [`SATURATED`]).
pub struct Warped<'a> {
    pub rgb: Cow<'a, Rgb32f>,
    pub flags: Vec<u8>,
}

impl Warped<'_> {
    #[inline]
    pub fn valid(&self, i: usize) -> bool {
        self.flags.get(i).is_some_and(|f| f & VALID != 0)
    }
}

#[inline]
fn catmull_rom(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

#[inline]
fn maxc(p: [f32; 3]) -> f32 {
    p[0].max(p[1]).max(p[2])
}

/// Sample `src` at continuous coordinates (pixel centres at +0.5) with clamped edges.
#[inline]
fn sample(src: &Rgb32f, x: f32, y: f32, bicubic: bool) -> [f32; 3] {
    let (w, h) = (src.width as isize, src.height as isize);
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let d = &src.data;
    let at = |xx: isize, yy: isize| d[(yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)) as usize];
    if !bicubic {
        let (a, b, c, e) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
        return std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (e[k] - c[k]) * tx;
            top + (bot - top) * ty
        });
    }
    let (wx, wy) = (catmull_rom(tx), catmull_rom(ty));
    let mut acc = [0f32; 3];
    let interior = x0 >= 1 && y0 >= 1 && x0 + 2 < w && y0 + 2 < h;
    for (j, wyj) in wy.iter().enumerate() {
        let yy = y0 - 1 + j as isize;
        let mut row = [0f32; 3];
        if interior {
            let base = (yy * w + x0 - 1) as usize;
            for (i, wxi) in wx.iter().enumerate() {
                let p = d[base + i];
                row[0] += wxi * p[0];
                row[1] += wxi * p[1];
                row[2] += wxi * p[2];
            }
        } else {
            for (i, wxi) in wx.iter().enumerate() {
                let p = at(x0 - 1 + i as isize, yy);
                row[0] += wxi * p[0];
                row[1] += wxi * p[1];
                row[2] += wxi * p[2];
            }
        }
        for c in 0..3 {
            acc[c] += wyj * row[c];
        }
    }
    acc.map(|v| v.max(0.0))
}

/// Resample `src` into the output rectangle `out` of the reference geometry: output pixel
/// `(u, v)` shows `src` at `h(out.x + u + ½, out.y + v + ½)`, divided by the frame's brightness
/// gain. A source with the reference's geometry and no gain is borrowed, not copied.
pub fn warp<'a>(src: &'a Rgb32f, h: &Homography, out: Rect, inv_gain: [f32; 3], clip: f32, bicubic: bool) -> Warped<'a> {
    let sat = clip * 0.98;
    let identity = *h == Homography::IDENTITY && out == Rect { x: 0, y: 0, w: src.width, h: src.height };
    if identity {
        let flags: Vec<u8> = src.data.par_iter().map(|p| if maxc(*p) >= sat { VALID | SATURATED } else { VALID }).collect();
        let rgb = if inv_gain == [1.0; 3] {
            Cow::Borrowed(src)
        } else {
            let mut r = src.clone();
            r.data.par_iter_mut().for_each(|p| *p = [p[0] * inv_gain[0], p[1] * inv_gain[1], p[2] * inv_gain[2]]);
            Cow::Owned(r)
        };
        return Warped { rgb, flags };
    }
    let mut rgb = Rgb32f::new(out.w, out.h);
    let mut flags = vec![0u8; out.w * out.h];
    let (sw, sh) = (src.width as f64, src.height as f64);
    if out.w == 0 || out.h == 0 || src.width == 0 || src.height == 0 {
        return Warped { rgb: Cow::Owned(rgb), flags };
    }
    rgb.data.par_chunks_mut(out.w).zip(flags.par_chunks_mut(out.w)).enumerate().for_each(|(v, (row, frow))| {
        let y = (out.y + v) as f64 + 0.5;
        for (u, (o, f)) in row.iter_mut().zip(frow.iter_mut()).enumerate() {
            let (sx, sy) = h.apply_real((out.x + u) as f64 + 0.5, y);
            if !(sx >= 0.0 && sy >= 0.0 && sx <= sw && sy <= sh) {
                continue;
            }
            let p = sample(src, sx as f32, sy as f32, bicubic);
            let nx = (sx as usize).min(src.width - 1);
            let ny = (sy as usize).min(src.height - 1);
            let s = maxc(src.data[ny * src.width + nx]) >= sat;
            *f = if s { VALID | SATURATED } else { VALID };
            *o = [p[0] * inv_gain[0], p[1] * inv_gain[1], p[2] * inv_gain[2]];
        }
    });
    Warped { rgb: Cow::Owned(rgb), flags }
}

/// The grey value focus is measured on: √(luminance).
#[inline]
pub fn grey(p: [f32; 3]) -> f32 {
    (0.25 * p[0] + 0.6 * p[1] + 0.15 * p[2]).max(0.0).sqrt()
}

/// The modified Laplacian of the grey image, 0 at saturated or invalid pixels (neighbours outside
/// the valid area count as the centre value).
pub fn modified_laplacian(w: &Warped) -> Plane {
    let img = &*w.rgb;
    let (wd, ht) = (img.width, img.height);
    let mut y = Plane::new(wd, ht);
    y.data.par_iter_mut().zip(img.data.par_iter()).for_each(|(o, p)| *o = grey(*p));
    let mut ml = Plane::new(wd, ht);
    let fl = &w.flags;
    ml.data.par_chunks_mut(wd.max(1)).enumerate().for_each(|(yy, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = yy * wd + x;
            if fl[i] != VALID {
                continue;
            }
            let c = y.data[i];
            let at = |j: usize| if fl[j] & VALID != 0 { y.data[j] } else { c };
            let l = if x > 0 { at(i - 1) } else { c };
            let r = if x + 1 < wd { at(i + 1) } else { c };
            let u = if yy > 0 { at(i - wd) } else { c };
            let d = if yy + 1 < ht { at(i + wd) } else { c };
            *o = (2.0 * c - l - r).abs() + (2.0 * c - u - d).abs();
        }
    });
    ml
}

/// The focus measure: the modified Laplacian averaged over a `(2r+1)²` window.
pub fn focus_measure(w: &Warped, r: usize) -> Plane {
    box_mean(&modified_laplacian(w), r)
}

/// A robust noise level of a focus-measure plane: its 10th percentile over a subsample (the
/// least textured tenth of a frame — out-of-focus areas — measures little more than noise).
pub fn noise_floor(fm: &Plane) -> f32 {
    let step = (fm.data.len() / 50_000).max(1);
    let mut v: Vec<f32> = fm.data.iter().step_by(step).copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    let k = v.len() / 10;
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_borrows_and_shift_resamples() {
        let src = Rgb32f::from_fn(40, 30, |x, y| [x as f32 / 40.0, y as f32 / 30.0, 0.5]);
        let full = Rect { x: 0, y: 0, w: 40, h: 30 };
        let w = warp(&src, &Homography::IDENTITY, full, [1.0; 3], 1.0, true);
        assert!(matches!(w.rgb, Cow::Borrowed(_)));
        assert!(w.flags.iter().all(|&f| f == VALID));
        // a whole-pixel shift reproduces the source exactly (Catmull–Rom interpolates)
        let h = Homography([1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
        let out = Rect { x: 5, y: 4, w: 20, h: 15 };
        let w = warp(&src, &h, out, [0.5, 1.0, 1.0], 1.0, true);
        let p = w.rgb.get(2, 3);
        let s = src.get(5 + 2 + 3, 4 + 3 + 2);
        assert!((p[0] - s[0] * 0.5).abs() < 1e-5 && (p[1] - s[1]).abs() < 1e-5, "{p:?} {s:?}");
        // outside the source: invalid
        let far = Homography([1.0, 0.0, 100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let w = warp(&src, &far, out, [1.0; 3], 1.0, false);
        assert!(w.flags.iter().all(|&f| f == 0));
    }

    #[test]
    fn sharp_edges_measure_higher_than_blurred_ones() {
        let sharp = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.1; 3] } else { [0.6; 3] });
        let blurred = Rgb32f::from_fn(64, 32, |x, _| {
            let t = ((x as f32 - 32.0) / 6.0).clamp(-1.0, 1.0) * 0.5 + 0.5;
            [0.1 + 0.5 * t; 3]
        });
        let rect = Rect { x: 0, y: 0, w: 64, h: 32 };
        let fs = focus_measure(&warp(&sharp, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true), 2);
        let fb = focus_measure(&warp(&blurred, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true), 2);
        assert!(fs.get(32, 16) > 3.0 * fb.get(32, 16), "{} vs {}", fs.get(32, 16), fb.get(32, 16));
        assert_eq!(fs.get(5, 16), 0.0);
        // saturated pixels measure 0
        let sat = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.1; 3] } else { [1.0; 3] });
        let ml = modified_laplacian(&warp(&sat, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true));
        assert_eq!(ml.get(32, 16), 0.0);
        assert!(ml.get(31, 16) > 0.0);
    }
}
