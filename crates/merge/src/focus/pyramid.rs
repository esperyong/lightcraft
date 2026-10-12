//! Method C, pyramid: Laplacian pyramids (P. Burt & E. Adelson 1983) of the frames, each band
//! taken from the frame with the most local energy there (Z. Wang & Y. Chang, region-energy
//! multi-focus fusion, 2011). Best at overlapping structures (hairs, bristles, branches in front
//! of each other) where no single frame is sharp for every layer; can raise contrast and pick up
//! glare a little.
//!
//! - The pyramid is built on √RGB (perceptually even bands; squared back at the end).
//! - Energy per level: the luminance band squared, averaged over `(2r_l + 1)²` with
//!   `r_l = Radius / 2^l`, then Gaussian-smoothed (σ = Smoothing / 2^l) so the choice is spatially
//!   consistent — a streaming stand-in for the usual consistency check. One winner for all three
//!   channels keeps colours together.
//! - The coarsest level (the residual) is the frames' mean rather than a selection, which avoids
//!   the brightness and contrast lift of max-selection at low frequencies.
//!
//! State: about 24 bytes per output pixel (bands, energies, winner index) whatever the number of
//! frames.

use lightcraft_raster::{Plane, Rgb32f};
use rayon::prelude::*;

use super::measure::{VALID, Warped, box_mean_into, ensure, gaussian_in_place};
use crate::pano::blend::Buf;

const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Make `b` a `w × h × c` buffer (contents overwritten by the caller).
fn ensure_buf(b: &mut Buf, w: usize, h: usize, c: usize) {
    if b.w != w || b.h != h || b.c != c || b.data.len() != w * h * c {
        *b = Buf::new(w, h, c);
    }
}

/// Binomial 5-tap blur then decimate by 2 (as `pano::blend::down`, clamped edges), `src → dst`
/// with `tmp` scratch.
fn down_into(src: &Buf, dst: &mut Buf, tmp: &mut Buf) {
    let (w, h, c) = (src.w, src.h, src.c);
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    ensure_buf(tmp, w2, h, c);
    ensure_buf(dst, w2, h2, c);
    if w == 0 || h == 0 {
        return;
    }
    tmp.data.par_chunks_mut(w2 * c).zip(src.data.par_chunks(w * c)).for_each(|(o, row)| {
        for x in 0..w2 {
            let cx = 2 * x;
            for ch in 0..c {
                let mut s = 0.0;
                if cx >= 2 && cx + 2 < w {
                    for (k, wk) in K.iter().enumerate() {
                        s += wk * row[(cx + k - 2) * c + ch];
                    }
                } else {
                    for (k, wk) in K.iter().enumerate() {
                        let xx = (cx as isize + k as isize - 2).clamp(0, w as isize - 1) as usize;
                        s += wk * row[xx * c + ch];
                    }
                }
                o[x * c + ch] = s;
            }
        }
    });
    let t = &*tmp;
    let stride = w2 * c;
    dst.data.par_chunks_mut(stride).enumerate().for_each(|(y, o)| {
        let rows: [&[f32]; 5] = std::array::from_fn(|k| {
            let yy = (2 * y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
            &t.data[yy * stride..(yy + 1) * stride]
        });
        for (i, v) in o.iter_mut().enumerate() {
            *v = K[0] * rows[0][i] + K[1] * rows[1][i] + K[2] * rows[2][i] + K[3] * rows[3][i] + K[4] * rows[4][i];
        }
    });
}

/// Expand `src` to `w × h` (bilinear, pixel centres aligned with [`down_into`]; as
/// `pano::blend::up`) into `dst`.
fn up_into(src: &Buf, w: usize, h: usize, dst: &mut Buf) {
    let c = src.c;
    ensure_buf(dst, w, h, c);
    if src.w == 0 || src.h == 0 || w == 0 {
        return;
    }
    let (sw, sh) = (src.w as isize, src.h as isize);
    // output x = 2j → source j − 1 (¼) and j (¾); x = 2j + 1 → j (¾) and j + 1 (¼)
    let taps = |x: usize, n: isize| -> (usize, usize, f32) {
        let (a, t) = if x.is_multiple_of(2) { (x as isize / 2 - 1, 0.75) } else { (x as isize / 2, 0.25) };
        (a.clamp(0, n - 1) as usize, (a + 1).clamp(0, n - 1) as usize, t)
    };
    let xt: Vec<(usize, usize, f32)> = (0..w).map(|x| taps(x, sw)).collect();
    dst.data.par_chunks_mut(w * c).enumerate().for_each(|(y, o)| {
        let (y0, y1, ty) = taps(y, sh);
        let (r0, r1) = (&src.data[y0 * src.w * c..(y0 + 1) * src.w * c], &src.data[y1 * src.w * c..(y1 + 1) * src.w * c]);
        for (x, &(x0, x1, tx)) in xt.iter().enumerate() {
            for ch in 0..c {
                let (a, b) = (r0[x0 * c + ch], r0[x1 * c + ch]);
                let (d, e) = (r1[x0 * c + ch], r1[x1 * c + ch]);
                let top = a + (b - a) * tx;
                let bot = d + (e - d) * tx;
                o[x * c + ch] = top + (bot - top) * ty;
            }
        }
    });
}

/// One level's working buffers (reused from frame to frame).
struct LevelBuf {
    g: Buf,
    v: Buf,
}

pub struct Pyramid {
    w: usize,
    h: usize,
    levels: usize,
    acc: Vec<Buf>,
    energy: Vec<Vec<f32>>,
    winner: Vec<u16>,
    top: Buf,
    top_n: Vec<f32>,
    radius: f32,
    smoothing: f32,
    /// Per level: the frame's Gaussian level and validity (index l = level l).
    work: Vec<LevelBuf>,
    up: Buf,
    tmp: Buf,
    vtmp: Buf,
    e: Plane,
    e_tmp: Plane,
    e_tmp2: Plane,
}

/// Number of band levels for a `w × h` image: halve until the short edge is ≤ 16 (1–10 levels).
fn level_count(w: usize, h: usize) -> usize {
    let (mut m, mut l) = (w.min(h), 0);
    while m > 16 && l < 10 {
        m = m.div_ceil(2);
        l += 1;
    }
    l.max(1)
}

impl Pyramid {
    pub fn new(w: usize, h: usize, radius: f32, smoothing: f32) -> Pyramid {
        let levels = level_count(w, h);
        let (mut acc, mut energy) = (Vec::new(), Vec::new());
        let (mut lw, mut lh) = (w, h);
        for _ in 0..levels {
            acc.push(Buf::new(lw, lh, 3));
            energy.push(vec![-1.0; lw * lh]);
            lw = lw.div_ceil(2).max(1);
            lh = lh.div_ceil(2).max(1);
        }
        let empty = || Buf::new(0, 0, 3);
        Pyramid {
            w,
            h,
            levels,
            acc,
            energy,
            winner: vec![0; w * h],
            top: Buf::new(lw, lh, 3),
            top_n: vec![0.0; lw * lh],
            radius,
            smoothing,
            work: (0..=levels).map(|_| LevelBuf { g: empty(), v: Buf::new(0, 0, 1) }).collect(),
            up: empty(),
            tmp: empty(),
            vtmp: Buf::new(0, 0, 1),
            e: Plane::new(0, 0),
            e_tmp: Plane::new(0, 0),
            e_tmp2: Plane::new(0, 0),
        }
    }

    pub fn bytes(&self) -> usize {
        let bufs = self.work.iter().map(|l| l.g.data.len() + l.v.data.len()).sum::<usize>()
            + self.up.data.len()
            + self.tmp.data.len()
            + self.vtmp.data.len();
        let planes = self.e.data.len() + self.e_tmp.data.len() + self.e_tmp2.data.len();
        self.acc.iter().map(|b| b.data.len() * 4).sum::<usize>()
            + self.energy.iter().map(|e| e.len() * 4).sum::<usize>()
            + self.winner.len() * 2
            + self.top.data.len() * 4
            + self.top_n.len() * 4
            + (bufs + planes) * 4
    }

    /// Add the frame at position `pos`.
    pub fn add(&mut self, pos: usize, frame: &Warped) {
        let (w, h) = (self.w, self.h);
        let k = pos.min(u16::MAX as usize) as u16;
        {
            let l0 = &mut self.work[0];
            ensure_buf(&mut l0.g, w, h, 3);
            ensure_buf(&mut l0.v, w, h, 1);
            l0.g.data.par_chunks_mut(3).zip(frame.rgb.data.par_iter()).for_each(|(o, p)| {
                o[0] = p[0].max(0.0).sqrt();
                o[1] = p[1].max(0.0).sqrt();
                o[2] = p[2].max(0.0).sqrt();
            });
            l0.v.data.par_iter_mut().zip(frame.flags.par_iter()).for_each(|(o, f)| *o = if f & VALID != 0 { 1.0 } else { 0.0 });
        }
        for l in 0..self.levels {
            let (cur, next) = self.work.split_at_mut(l + 1);
            let (cur, next) = (&cur[l], &mut next[0]);
            down_into(&cur.g, &mut next.g, &mut self.tmp);
            down_into(&cur.v, &mut next.v, &mut self.vtmp);
            up_into(&next.g, cur.g.w, cur.g.h, &mut self.up);
            // the band (stored in `up`): g − up(next)
            self.up.data.par_iter_mut().zip(cur.g.data.par_iter()).for_each(|(u, g)| *u = g - *u);
            let lap = &self.up;
            // luminance band energy, local mean, smoothed
            ensure(&mut self.e_tmp2, lap.w, lap.h);
            self.e_tmp2.data.par_iter_mut().zip(lap.data.par_chunks(3)).for_each(|(o, c)| {
                let y = 0.25 * c[0] + 0.6 * c[1] + 0.15 * c[2];
                *o = y * y;
            });
            let f = (1u64 << l) as f32;
            let r = (self.radius / f).round().max(1.0) as usize;
            box_mean_into(&self.e_tmp2, r, &mut self.e, &mut self.e_tmp);
            gaussian_in_place(&mut self.e, self.smoothing / f, &mut self.e_tmp, &mut self.e_tmp2);
            let lw = lap.w.max(1);
            let (acc, en) = (&mut self.acc[l], &mut self.energy[l]);
            let (lapd, vd, ed) = (&lap.data, &cur.v.data, &self.e.data);
            let winner = if l == 0 { Some(&mut self.winner) } else { None };
            match winner {
                Some(win) => acc.data.par_chunks_mut(lw * 3).zip(en.par_chunks_mut(lw)).zip(win.par_chunks_mut(lw)).enumerate().for_each(
                    |(y, ((arow, erow), wrow))| {
                        for x in 0..erow.len() {
                            let i = y * lw + x;
                            if vd[i] >= 0.999 && ed[i] > erow[x] {
                                erow[x] = ed[i];
                                arow[x * 3..x * 3 + 3].copy_from_slice(&lapd[i * 3..i * 3 + 3]);
                                wrow[x] = k;
                            }
                        }
                    },
                ),
                None => acc.data.par_chunks_mut(lw * 3).zip(en.par_chunks_mut(lw)).enumerate().for_each(|(y, (arow, erow))| {
                    for x in 0..erow.len() {
                        let i = y * lw + x;
                        if vd[i] >= 0.999 && ed[i] > erow[x] {
                            erow[x] = ed[i];
                            arow[x * 3..x * 3 + 3].copy_from_slice(&lapd[i * 3..i * 3 + 3]);
                        }
                    }
                }),
            }
        }
        // the residual: mean of the frames that cover it
        let last = &self.work[self.levels];
        let (g, v) = (&last.g, &last.v);
        self.top.data.par_chunks_mut(3).zip(self.top_n.par_iter_mut()).enumerate().for_each(|(i, (t, n))| {
            if v.data.get(i).is_some_and(|&a| a >= 0.999) {
                for c in 0..3 {
                    t[c] += g.data[i * 3 + c];
                }
                *n += 1.0;
            }
        });
    }

    /// Collapse the pyramid: the result and the level-0 winners as a depth (0..1 over `last`).
    pub fn finish(mut self, last: usize) -> (Rgb32f, Plane) {
        let mut r = std::mem::replace(&mut self.top, Buf::new(0, 0, 3));
        for (i, n) in self.top_n.iter().enumerate() {
            for c in 0..3 {
                r.data[i * 3 + c] = if *n > 0.0 { r.data[i * 3 + c] / n } else { 0.0 };
            }
        }
        let mut u = Buf::new(0, 0, 3);
        for acc in self.acc.iter().rev() {
            up_into(&r, acc.w, acc.h, &mut u);
            u.data.par_iter_mut().zip(acc.data.par_iter()).for_each(|(a, b)| *a += b);
            std::mem::swap(&mut r, &mut u);
        }
        let img = Rgb32f { width: self.w, height: self.h, data: r.data.chunks_exact(3).map(|c| [0, 1, 2].map(|j| c[j].max(0.0).powi(2))).collect() };
        let inv = 1.0 / last.max(1) as f32;
        let depth = Plane { width: self.w, height: self.h, data: self.winner.iter().map(|&k| k as f32 * inv).collect() };
        (img, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_down_and_up_match_the_blender_ones() {
        let mut b = Buf::new(37, 23, 3);
        for (i, v) in b.data.iter_mut().enumerate() {
            *v = ((i * 7919) % 101) as f32 / 101.0;
        }
        let (mut d, mut t, mut u) = (Buf::new(0, 0, 3), Buf::new(0, 0, 3), Buf::new(0, 0, 3));
        down_into(&b, &mut d, &mut t);
        let want = crate::pano::blend::down(&b);
        assert_eq!((d.w, d.h), (want.w, want.h));
        assert!(d.data.iter().zip(&want.data).all(|(a, b)| (a - b).abs() < 1e-5));
        up_into(&want, 37, 23, &mut u);
        let want_up = crate::pano::blend::up(&want, 37, 23);
        assert!(u.data.iter().zip(&want_up.data).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn a_single_frame_reconstructs_exactly_and_sharp_halves_combine() {
        let (w, h) = (96, 64);
        let img = Rgb32f::from_fn(w, h, |x, y| [0.1 + 0.4 * ((x * 7 + y * 3) % 11) as f32 / 11.0, 0.3, 0.2]);
        let flags = vec![VALID; w * h];
        let f = Warped { rgb: &img, flags: &flags };
        let mut p = Pyramid::new(w, h, 2.0, 0.0);
        p.add(0, &f);
        let (out, _) = p.finish(0);
        for (a, b) in out.data.iter().zip(&img.data) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-4, "{a:?} {b:?}");
            }
        }
        // left half textured in frame 0, right half in frame 1: the result has both
        let tex = |x: usize, y: usize| 0.2 + 0.3 * ((x * 5 + y * 7) % 3) as f32 / 2.0;
        let a = Rgb32f::from_fn(w, h, |x, y| [if x < 48 { tex(x, y) } else { 0.35 }; 3]);
        let b = Rgb32f::from_fn(w, h, |x, y| [if x >= 48 { tex(x, y) } else { 0.35 }; 3]);
        let mut p = Pyramid::new(w, h, 2.0, 1.0);
        for (k, im) in [&a, &b].iter().enumerate() {
            p.add(k, &Warped { rgb: im, flags: &flags });
        }
        let (out, depth) = p.finish(1);
        let truth = Rgb32f::from_fn(w, h, |x, y| [tex(x, y); 3]);
        let err: f32 = (0..h)
            .flat_map(|y| (4..w - 4).filter(|&x| !(44..52).contains(&x)).map(move |x| (x, y)))
            .map(|(x, y)| (out.get(x, y)[0] - truth.get(x, y)[0]).abs())
            .sum::<f32>()
            / (h * (w - 16)) as f32;
        assert!(err < 0.02, "mean error {err}");
        assert_eq!(depth.get(10, 30), 0.0);
        assert_eq!(depth.get(80, 30), 1.0);
    }
}
