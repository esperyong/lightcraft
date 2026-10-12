//! Per-frame work of the fuse pass: resampling a frame into the output geometry (with its
//! brightness correction), the grey image focus is measured on, and the focus measure itself.
//!
//! The focus measure is the sum-modified Laplacian (S. K. Nayar & Y. Nakagawa, "Shape from
//! Focus", IEEE PAMI 1994): `|2Y − Y(x−1) − Y(x+1)| + |2Y − Y(y−1) − Y(y+1)|` on a
//! square-root-encoded grey image (perceptually even, cheap), averaged over a `(2r+1)²` window.
//! Saturated and invalid pixels measure 0. (Measures compared: Pertuz et al., "Analysis of focus
//! measure operators for shape-from-focus", Pattern Recognition 2013.)
//!
//! Every frame-sized buffer lives in [`FrameBuf`] / [`MeasureBuf`] and is reused from frame to
//! frame: allocating fresh multi-megabyte buffers per frame costs more in page faults than the
//! arithmetic (measured: a 30 MB copy into a new buffer took 21 ms, the bilinear resample of the
//! same frame 18 ms).

use lightcraft_geom::Homography;
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
#[derive(Clone, Copy)]
pub struct Warped<'a> {
    pub rgb: &'a Rgb32f,
    pub flags: &'a [u8],
}

impl Warped<'_> {
    #[inline]
    pub fn valid(&self, i: usize) -> bool {
        self.flags.get(i).is_some_and(|f| f & VALID != 0)
    }
}

/// The resampled frame's buffers, reused from frame to frame.
pub struct FrameBuf {
    rgb: Rgb32f,
    flags: Vec<u8>,
}

impl Default for FrameBuf {
    fn default() -> Self {
        FrameBuf { rgb: Rgb32f::new(0, 0), flags: Vec::new() }
    }
}

impl FrameBuf {
    pub fn bytes(&self) -> usize {
        self.rgb.data.len() * 12 + self.flags.len()
    }
}

/// The focus measure's buffers, reused from frame to frame.
pub struct MeasureBuf {
    grey: Plane,
    ml: Plane,
    fm: Plane,
    tmp: Plane,
    /// Spare planes for callers (weights and their blur).
    pub a: Plane,
    pub b: Plane,
}

impl Default for MeasureBuf {
    fn default() -> Self {
        MeasureBuf {
            grey: Plane::new(0, 0),
            ml: Plane::new(0, 0),
            fm: Plane::new(0, 0),
            tmp: Plane::new(0, 0),
            a: Plane::new(0, 0),
            b: Plane::new(0, 0),
        }
    }
}

impl MeasureBuf {
    pub fn bytes(&self) -> usize {
        [&self.grey, &self.ml, &self.fm, &self.tmp, &self.a, &self.b].iter().map(|p| p.data.len() * 4).sum()
    }
}

/// Make `p` a `w × h` plane (its contents are overwritten by the caller).
pub fn ensure(p: &mut Plane, w: usize, h: usize) {
    if p.width != w || p.height != h || p.data.len() != w * h {
        *p = Plane::new(w, h);
    }
}

/// `x.floor()` without a libm call (x86-64's baseline has no rounding instruction), for
/// coordinates well inside `i32`.
#[inline(always)]
fn floor_i(x: f32) -> isize {
    let i = x as i32;
    (if (i as f32) > x { i - 1 } else { i }) as isize
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
#[inline(always)]
fn sample(src: &Rgb32f, x: f32, y: f32, bicubic: bool) -> [f32; 3] {
    let (w, h) = (src.width as isize, src.height as isize);
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (floor_i(fx), floor_i(fy));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let d = &src.data;
    let at = |xx: isize, yy: isize| d[(yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)) as usize];
    if !bicubic {
        let (a, b, c, e) = if x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h {
            let i = (y0 * w + x0) as usize;
            let s = w as usize;
            (d[i], d[i + 1], d[i + s], d[i + s + 1])
        } else {
            (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1))
        };
        return std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (e[k] - c[k]) * tx;
            top + (bot - top) * ty
        });
    }
    let (wx, wy) = (catmull_rom(tx), catmull_rom(ty));
    if x0 >= 1 && y0 >= 1 && x0 + 2 < w && y0 + 2 < h {
        // interior: four rows of four pixels, read as fixed-size arrays (no per-tap bounds checks)
        let stride = w as usize;
        let base = ((y0 - 1) * w + x0 - 1) as usize;
        let (mut r, mut g, mut b) = (0f32, 0f32, 0f32);
        for (j, wyj) in wy.iter().enumerate() {
            let at = base + j * stride;
            let Some(q) = d.get(at..at + 4).and_then(|s| <&[[f32; 3]; 4]>::try_from(s).ok()) else { continue };
            let rr = wx[0] * q[0][0] + wx[1] * q[1][0] + wx[2] * q[2][0] + wx[3] * q[3][0];
            let gg = wx[0] * q[0][1] + wx[1] * q[1][1] + wx[2] * q[2][1] + wx[3] * q[3][1];
            let bb = wx[0] * q[0][2] + wx[1] * q[1][2] + wx[2] * q[2][2] + wx[3] * q[3][2];
            r += wyj * rr;
            g += wyj * gg;
            b += wyj * bb;
        }
        return [r.max(0.0), g.max(0.0), b.max(0.0)];
    }
    let mut acc = [0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let yy = y0 - 1 + j as isize;
        let mut row = [0f32; 3];
        for (i, wxi) in wx.iter().enumerate() {
            let p = at(x0 - 1 + i as isize, yy);
            row[0] += wxi * p[0];
            row[1] += wxi * p[1];
            row[2] += wxi * p[2];
        }
        for c in 0..3 {
            acc[c] += wyj * row[c];
        }
    }
    acc.map(|v| v.max(0.0))
}

/// Resample `src` into the output rectangle `out` of the reference geometry: output pixel
/// `(u, v)` shows `src` at `h(out.x + u + ½, out.y + v + ½)`, divided by the frame's brightness
/// gain. A source with the reference's geometry and no gain is used as it is, not copied.
pub fn warp<'a>(src: &'a Rgb32f, h: &Homography, out: Rect, inv_gain: [f32; 3], clip: f32, bicubic: bool, buf: &'a mut FrameBuf) -> Warped<'a> {
    let sat = clip * 0.98;
    let n = out.w * out.h;
    buf.flags.resize(n, 0);
    let identity = *h == Homography::IDENTITY && out == Rect { x: 0, y: 0, w: src.width, h: src.height };
    if identity {
        buf.flags.par_iter_mut().zip(src.data.par_iter()).for_each(|(f, p)| *f = if maxc(*p) >= sat { VALID | SATURATED } else { VALID });
        if inv_gain == [1.0; 3] {
            return Warped { rgb: src, flags: &buf.flags };
        }
        if buf.rgb.width != out.w || buf.rgb.height != out.h {
            buf.rgb = Rgb32f::new(out.w, out.h);
        }
        buf.rgb.data.par_iter_mut().zip(src.data.par_iter()).for_each(|(o, p)| *o = [p[0] * inv_gain[0], p[1] * inv_gain[1], p[2] * inv_gain[2]]);
        return Warped { rgb: &buf.rgb, flags: &buf.flags };
    }
    if buf.rgb.width != out.w || buf.rgb.height != out.h {
        buf.rgb = Rgb32f::new(out.w, out.h);
    }
    let (sw, sh) = (src.width as f64, src.height as f64);
    if n == 0 || src.width == 0 || src.height == 0 {
        buf.flags.fill(0);
        return Warped { rgb: &buf.rgb, flags: &buf.flags };
    }
    let m = h.0;
    let affine = m[6] == 0.0 && m[7] == 0.0 && m[8] == 1.0;
    buf.rgb.data.par_chunks_mut(out.w).zip(buf.flags.par_chunks_mut(out.w)).enumerate().for_each(|(v, (row, frow))| {
        let y = (out.y + v) as f64 + 0.5;
        let x0 = out.x as f64 + 0.5;
        // affine maps step by a constant per pixel
        let (mut ax, mut ay) = (m[0] * x0 + m[1] * y + m[2], m[3] * x0 + m[4] * y + m[5]);
        for (u, (o, f)) in row.iter_mut().zip(frow.iter_mut()).enumerate() {
            let (sx, sy) = if affine {
                let p = (ax, ay);
                ax += m[0];
                ay += m[3];
                p
            } else {
                h.apply_real((out.x + u) as f64 + 0.5, y)
            };
            if !(sx >= 0.0 && sy >= 0.0 && sx <= sw && sy <= sh) {
                *f = 0;
                *o = [0.0; 3];
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
    Warped { rgb: &buf.rgb, flags: &buf.flags }
}

/// The grey value focus is measured on: √(luminance).
#[inline]
pub fn grey(p: [f32; 3]) -> f32 {
    (0.25 * p[0] + 0.6 * p[1] + 0.15 * p[2]).max(0.0).sqrt()
}

/// The modified Laplacian of the grey image into `ml` (using `grey` as scratch), 0 at saturated or
/// invalid pixels (neighbours outside the valid area count as the centre value).
pub fn modified_laplacian_into(w: &Warped, grey_buf: &mut Plane, ml: &mut Plane) {
    let img = w.rgb;
    let (wd, ht) = (img.width, img.height);
    ensure(grey_buf, wd, ht);
    ensure(ml, wd, ht);
    grey_buf.data.par_iter_mut().zip(img.data.par_iter()).for_each(|(o, p)| *o = grey(*p));
    let fl = w.flags;
    let y = &*grey_buf;
    ml.data.par_chunks_mut(wd.max(1)).enumerate().for_each(|(yy, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = yy * wd + x;
            if fl[i] != VALID {
                *o = 0.0;
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
}

/// The modified Laplacian as a new plane (tests, one-off use).
pub fn modified_laplacian(w: &Warped) -> Plane {
    let (mut g, mut ml) = (Plane::new(0, 0), Plane::new(0, 0));
    modified_laplacian_into(w, &mut g, &mut ml);
    ml
}

/// Mean over the `(2r+1)²` window around each pixel (clamped edges), `src → dst`, `tmp` scratch.
pub fn box_mean_into(src: &Plane, r: usize, dst: &mut Plane, tmp: &mut Plane) {
    let (w, h) = (src.width, src.height);
    ensure(dst, w, h);
    if w == 0 || h == 0 {
        return;
    }
    if r == 0 {
        dst.data.copy_from_slice(&src.data);
        return;
    }
    ensure(tmp, w, h);
    let inv = 1.0 / (2 * r + 1) as f32;
    let last = w - 1;
    tmp.data.par_chunks_mut(w).zip(src.data.par_chunks(w)).for_each(|(o, s)| {
        let mut acc = 0f32;
        for i in 0..=2 * r {
            acc += s[i.saturating_sub(r).min(last)];
        }
        for x in 0..w {
            o[x] = acc * inv;
            acc += s[(x + r + 1).min(last)] - s[x.saturating_sub(r)];
        }
    });
    const BAND: usize = 32;
    let lasty = h - 1;
    let t = &*tmp;
    let row = |y: usize| &t.data[y * w..(y + 1) * w];
    dst.data.par_chunks_mut(w * BAND).enumerate().for_each(|(band, out)| {
        let y0 = band * BAND;
        let mut acc = vec![0f32; w];
        for i in 0..=2 * r {
            for (a, v) in acc.iter_mut().zip(row((y0 + i).saturating_sub(r).min(lasty))) {
                *a += v;
            }
        }
        for (k, o) in out.chunks_mut(w).enumerate() {
            let y = y0 + k;
            for (d, a) in o.iter_mut().zip(&acc) {
                *d = a * inv;
            }
            let (add, sub) = (row((y + r + 1).min(lasty)), row(y.saturating_sub(r)));
            for ((a, p), m) in acc.iter_mut().zip(add).zip(sub) {
                *a += p - m;
            }
        }
    });
}

/// Gaussian-like blur of `p` in place (three box passes, as `lightcraft_raster::blur::gaussian`),
/// `t1`/`t2` scratch.
pub fn gaussian_in_place(p: &mut Plane, sigma: f32, t1: &mut Plane, t2: &mut Plane) {
    if sigma <= 0.3 || p.data.is_empty() {
        return;
    }
    for r in lightcraft_raster::blur::box_radii(sigma) {
        if r == 0 {
            continue;
        }
        box_mean_into(p, r, t2, t1);
        std::mem::swap(p, t2);
    }
}

/// The focus measure (the modified Laplacian averaged over a `(2r+1)²` window), in `buf`.
pub fn focus_measure<'b>(w: &Warped, r: usize, buf: &'b mut MeasureBuf) -> &'b Plane {
    modified_laplacian_into(w, &mut buf.grey, &mut buf.ml);
    box_mean_into(&buf.ml, r, &mut buf.fm, &mut buf.tmp);
    &buf.fm
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
        let mut buf = FrameBuf::default();
        let w = warp(&src, &Homography::IDENTITY, full, [1.0; 3], 1.0, true, &mut buf);
        assert!(std::ptr::eq(w.rgb, &src));
        assert!(w.flags.iter().all(|&f| f == VALID));
        // a whole-pixel shift reproduces the source exactly (Catmull–Rom interpolates)
        let h = Homography([1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
        let out = Rect { x: 5, y: 4, w: 20, h: 15 };
        let w = warp(&src, &h, out, [0.5, 1.0, 1.0], 1.0, true, &mut buf);
        let p = w.rgb.get(2, 3);
        let s = src.get(5 + 2 + 3, 4 + 3 + 2);
        assert!((p[0] - s[0] * 0.5).abs() < 1e-5 && (p[1] - s[1]).abs() < 1e-5, "{p:?} {s:?}");
        // outside the source: invalid
        let far = Homography([1.0, 0.0, 100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let w = warp(&src, &far, out, [1.0; 3], 1.0, false, &mut buf);
        assert!(w.flags.iter().all(|&f| f == 0));
        // the affine stepping agrees with the general path (a perspective map with a negligible
        // perspective term computes every pixel's position)
        let tex = Rgb32f::from_fn(60, 50, |x, y| {
            let v = 0.4 + 0.2 * ((x as f32 * 0.7).sin() * (y as f32 * 0.5).cos());
            [v, v * 0.8, 0.3]
        });
        let tilt = Homography([1.01, 0.02, 3.3, -0.015, 0.99, 2.2, 0.0, 0.0, 1.0]);
        let r = Rect { x: 6, y: 5, w: 40, h: 34 };
        let a = warp(&tex, &tilt, r, [1.0; 3], 1.0, true, &mut buf).rgb.clone();
        let persp = Homography([1.01, 0.02, 3.3, -0.015, 0.99, 2.2, 1e-12, 0.0, 1.0]);
        let b = warp(&tex, &persp, r, [1.0; 3], 1.0, true, &mut buf).rgb.clone();
        let err = a.data.iter().zip(&b.data).map(|(p, q)| (p[0] - q[0]).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "affine stepping vs direct: {err}");
    }

    #[test]
    fn sharp_edges_measure_higher_than_blurred_ones() {
        let sharp = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.1; 3] } else { [0.6; 3] });
        let blurred = Rgb32f::from_fn(64, 32, |x, _| {
            let t = ((x as f32 - 32.0) / 6.0).clamp(-1.0, 1.0) * 0.5 + 0.5;
            [0.1 + 0.5 * t; 3]
        });
        let rect = Rect { x: 0, y: 0, w: 64, h: 32 };
        let (mut fbuf, mut mbuf) = (FrameBuf::default(), MeasureBuf::default());
        let fs = focus_measure(&warp(&sharp, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true, &mut fbuf), 2, &mut mbuf).clone();
        let fb = focus_measure(&warp(&blurred, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true, &mut fbuf), 2, &mut mbuf).clone();
        assert!(fs.get(32, 16) > 3.0 * fb.get(32, 16), "{} vs {}", fs.get(32, 16), fb.get(32, 16));
        assert_eq!(fs.get(5, 16), 0.0);
        // saturated pixels measure 0
        let sat = Rgb32f::from_fn(64, 32, |x, _| if x < 32 { [0.1; 3] } else { [1.0; 3] });
        let ml = modified_laplacian(&warp(&sat, &Homography::IDENTITY, rect, [1.0; 3], 1.0, true, &mut fbuf));
        assert_eq!(ml.get(32, 16), 0.0);
        assert!(ml.get(31, 16) > 0.0);
    }

    #[test]
    fn box_mean_into_matches_the_raster_box_filter() {
        let p = Plane::from_fn(77, 45, |x, y| ((x * 31 + y * 17) % 23) as f32 / 23.0);
        let (mut d, mut t) = (Plane::new(0, 0), Plane::new(0, 0));
        for r in [1, 4, 30] {
            box_mean_into(&p, r, &mut d, &mut t);
            let want = lightcraft_raster::blur::box_mean(&p, r);
            for (a, b) in d.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-5, "r {r}: {a} vs {b}");
            }
        }
        let mut g = p.clone();
        let (mut t1, mut t2) = (Plane::new(0, 0), Plane::new(0, 0));
        gaussian_in_place(&mut g, 3.0, &mut t1, &mut t2);
        let want = lightcraft_raster::blur::gaussian(&p, 3.0);
        for (a, b) in g.data.iter().zip(&want.data) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
