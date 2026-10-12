//! Adaptive Homogeneity-Directed demosaicing (K. Hirakawa, T. W. Parks, "Adaptive homogeneity-directed
//! demosaicing algorithm", IEEE Trans. Image Processing 14(3), 2005), implemented from the paper:
//!
//! 1. Interpolate green horizontally and vertically (5-tap filter [−¼ ½ ½ ½ −¼] across the missing site,
//!    clamped to the two adjacent greens), giving two candidate green planes.
//! 2. Complete each candidate with red/blue by colour-difference (C − G) interpolation.
//! 3. Convert both candidates to CIELab and measure, per pixel, how many of its 4 neighbours lie within the
//!    adaptive luminance / chrominance tolerances (ε_L, ε_C from the paper) — the homogeneity.
//! 4. Sum homogeneity over a 3×3 window and pick the more homogeneous candidate (average on ties).
//!
//! Work is done in overlapping tiles (bounded memory, rayon-parallel), each stage only over the pixels
//! the next one reads (no edge clamping inside a tile; the margin covers every stage's reach), with
//! per-thread scratch buffers. The plain implementation it must match bit for bit is kept in the
//! tests.

use super::Mosaic;
use crate::Rgb32f;
use rayon::prelude::*;

const TILE: usize = 128;
/// Context a tile needs around it: the choice reads homogeneity ±1, which reads Lab ±1, the
/// candidates ±1 (±3 for the greens' reach of ±2): 5, plus one to spare.
const MARGIN: usize = 6;

pub(crate) fn ahd(m: &Mosaic) -> Rgb32f {
    let (w, h) = (m.w, m.h);
    let mut out = Rgb32f::new(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    let tiles_x = w.div_ceil(TILE);
    out.data.par_chunks_mut(w * TILE).enumerate().for_each(|(band, rows)| {
        let (y0, th) = (band * TILE, rows.len() / w);
        let done: Vec<(usize, usize, Vec<[f32; 3]>)> = (0..tiles_x)
            .into_par_iter()
            .map_init(Scratch::default, |s, tx| {
                let (x0, tw) = (tx * TILE, TILE.min(w - tx * TILE));
                (x0, tw, tile(m, x0, y0, tw, th, s))
            })
            .collect();
        for (x0, tw, px) in done {
            for (y, src) in px.chunks_exact(tw).enumerate() {
                rows[y * w + x0..y * w + x0 + tw].copy_from_slice(src);
            }
        }
    });
    out
}

/// Linear camera RGB (treated as sRGB/D65 primaries for the homogeneity metric only) → CIELab.
#[inline]
fn lab(p: [f32; 3]) -> [f32; 3] {
    let x = (0.4124 * p[0] + 0.3576 * p[1] + 0.1805 * p[2]) / 0.95047;
    let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let z = (0.0193 * p[0] + 0.1192 * p[1] + 0.9505 * p[2]) / 1.08883;
    #[inline]
    fn f(t: f32) -> f32 {
        if t > 0.008856 { cbrt(t) } else { 7.787 * t + 16.0 / 116.0 }
    }
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Cube root of a positive normal `t`, correctly rounded (checked against f64's in the tests, so
/// the same on every platform): an estimate from the exponent (within 5 %), two Halley steps in
/// f64 (≈ 1e−15). Inlined, and cheaper than the library call.
#[inline]
fn cbrt(t: f32) -> f32 {
    let mut y = f32::from_bits(t.to_bits() / 3 + 709_921_077) as f64;
    let t = t as f64;
    for _ in 0..2 {
        let y3 = y * y * y;
        y = y * (y3 + 2.0 * t) / (2.0 * y3 + t);
    }
    y as f32
}

/// A tile's working buffers, reused from tile to tile (fresh ones page-fault their way in).
#[derive(Default)]
struct Scratch {
    mos: Vec<f32>,
    col: Vec<u8>,
    gh: Vec<f32>,
    gv: Vec<f32>,
    rh: Vec<[f32; 3]>,
    rv: Vec<[f32; 3]>,
    lh: Vec<[f32; 3]>,
    lv: Vec<[f32; 3]>,
    hh: Vec<u8>,
    hv: Vec<u8>,
}

impl Scratch {
    fn fit(&mut self, n: usize) {
        for v in [&mut self.mos, &mut self.gh, &mut self.gv] {
            v.resize(n, 0.0);
        }
        for v in [&mut self.col, &mut self.hh, &mut self.hv] {
            v.resize(n, 0);
        }
        for v in [&mut self.rh, &mut self.rv, &mut self.lh, &mut self.lv] {
            v.resize(n, [0.0; 3]);
        }
    }
}

#[inline]
fn dl(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).abs()
}

#[inline]
fn dc(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// One tile (`tw × th` at `x0, y0`), row-major.
fn tile(m: &Mosaic, x0: usize, y0: usize, tw: usize, th: usize, s: &mut Scratch) -> Vec<[f32; 3]> {
    let (bw, bh) = (tw + 2 * MARGIN, th + 2 * MARGIN);
    s.fit(bw * bh);
    let (ox, oy) = (x0 as isize - MARGIN as isize, y0 as isize - MARGIN as isize);
    // the mosaic around the tile (mirrored past the image's edges)
    let inside = ox >= 0 && oy >= 0 && x0 + tw + MARGIN <= m.w && y0 + th + MARGIN <= m.h;
    let bayer = m.cfa.width == 2 && m.cfa.height == 2;
    for y in 0..bh {
        let row = &mut s.mos[y * bw..(y + 1) * bw];
        let cols = &mut s.col[y * bw..(y + 1) * bw];
        let sy = oy + y as isize;
        if inside {
            let at = sy as usize * m.w + ox as usize;
            row.copy_from_slice(&m.data[at..at + bw]);
        } else {
            for (x, v) in row.iter_mut().enumerate() {
                *v = m.get(ox + x as isize, sy);
            }
        }
        if bayer {
            // a 2 × 2 pattern: mirroring keeps the parity, so the colour follows the coordinate's
            let pair = [m.color(ox, sy), m.color(ox + 1, sy)];
            for (x, c) in cols.iter_mut().enumerate() {
                *c = pair[x & 1];
            }
        } else {
            for (x, c) in cols.iter_mut().enumerate() {
                *c = m.color(ox + x as isize, sy);
            }
        }
    }
    let Scratch { mos, col, gh, gv, rh, rv, lh, lv, hh, hv } = s;
    let (mos, col) = (&mos[..], &col[..]);
    // 1. directional greens (rows and columns 3 … b − 3: what the candidates read)
    for y in 3..bh - 3 {
        for x in 3..bw - 3 {
            let i = y * bw + x;
            let c = mos[i];
            if col[i] == 1 {
                gh[i] = c;
                gv[i] = c;
                continue;
            }
            let (l, r) = (mos[i - 1], mos[i + 1]);
            let est = (l + r) * 0.5 + (2.0 * c - mos[i - 2] - mos[i + 2]) * 0.25;
            gh[i] = est.clamp(l.min(r), l.max(r));
            let (u, d) = (mos[i - bw], mos[i + bw]);
            let est = (u + d) * 0.5 + (2.0 * c - mos[i - 2 * bw] - mos[i + 2 * bw]) * 0.25;
            gv[i] = est.clamp(u.min(d), u.max(d));
        }
    }
    // 2. full candidates and 3. their Lab (4 … b − 4: what the homogeneity reads)
    let complete = |g: &[f32], rgb: &mut [[f32; 3]], lab_out: &mut [[f32; 3]]| {
        for y in 4..bh - 4 {
            for x in 4..bw - 4 {
                let i = y * bw + x;
                let own = col[i] as usize;
                let gvv = g[i];
                let d = |j: usize| mos[j] - g[j];
                let mut p = [0f32; 3];
                p[1] = gvv;
                if own == 1 {
                    let ch = col[i + 1] as usize;
                    let cv = col[i + bw] as usize;
                    p[ch] = gvv + (d(i - 1) + d(i + 1)) * 0.5;
                    p[cv] = gvv + (d(i - bw) + d(i + bw)) * 0.5;
                } else {
                    p[own] = mos[i];
                    p[2 - own] = gvv + (d(i - bw - 1) + d(i - bw + 1) + d(i + bw - 1) + d(i + bw + 1)) * 0.25;
                }
                rgb[i] = p;
            }
            let row = y * bw + 4..y * bw + bw - 4;
            for (o, p) in lab_out[row.clone()].iter_mut().zip(&rgb[row]) {
                *o = lab(*p);
            }
        }
    };
    complete(gh, rh, lh);
    complete(gv, rv, lv);
    // homogeneity (5 … b − 5: what the choice reads)
    for y in 5..bh - 5 {
        for x in 5..bw - 5 {
            let i = y * bw + x;
            let (l, r, u, d) = (i - 1, i + 1, i - bw, i + bw);
            let (a, b) = (&lh[i], &lv[i]);
            let eps_l = dl(a, &lh[l]).max(dl(a, &lh[r])).min(dl(b, &lv[u]).max(dl(b, &lv[d])));
            let eps_c = dc(a, &lh[l]).max(dc(a, &lh[r])).min(dc(b, &lv[u]).max(dc(b, &lv[d])));
            let mut ch = 0;
            let mut cv = 0;
            for j in [l, r, u, d] {
                if dl(a, &lh[j]) <= eps_l && dc(a, &lh[j]) <= eps_c {
                    ch += 1;
                }
                if dl(b, &lv[j]) <= eps_l && dc(b, &lv[j]) <= eps_c {
                    cv += 1;
                }
            }
            hh[i] = ch;
            hv[i] = cv;
        }
    }
    // 4. choose (3 × 3 sums from column sums)
    let mut out = Vec::with_capacity(tw * th);
    let (mut ch, mut cv) = (vec![0u32; bw], vec![0u32; bw]);
    for y in MARGIN..MARGIN + th {
        for x in MARGIN - 1..MARGIN + tw + 1 {
            let k = y * bw + x;
            ch[x] = hh[k - bw] as u32 + hh[k] as u32 + hh[k + bw] as u32;
            cv[x] = hv[k - bw] as u32 + hv[k] as u32 + hv[k + bw] as u32;
        }
        for x in MARGIN..MARGIN + tw {
            let sh = ch[x - 1] + ch[x] + ch[x + 1];
            let sv = cv[x - 1] + cv[x] + cv[x + 1];
            let i = y * bw + x;
            out.push(if sh > sv {
                rh[i]
            } else if sv > sh {
                rv[i]
            } else {
                [(rh[i][0] + rv[i][0]) * 0.5, (rh[i][1] + rv[i][1]) * 0.5, (rh[i][2] + rv[i][2]) * 0.5]
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cfa;

    /// The plain implementation (every stage over the whole tile, edges clamped, the colour of
    /// every pixel looked up): the optimised one must give the same bits.
    mod plain {
        use super::super::Mosaic;
        use crate::Rgb32f;
        use rayon::prelude::*;

        const TILE: usize = 128;
        const MARGIN: usize = 6;

        pub(crate) fn ahd(m: &Mosaic) -> Rgb32f {
            let (w, h) = (m.w, m.h);
            let tiles: Vec<(usize, usize)> =
                (0..h.div_ceil(TILE)).flat_map(|ty| (0..w.div_ceil(TILE)).map(move |tx| (tx * TILE, ty * TILE))).collect();
            let results: Vec<(usize, usize, usize, usize, Vec<[f32; 3]>)> = tiles
                .par_iter()
                .map(|&(x0, y0)| {
                    let (tw, th) = (TILE.min(w - x0), TILE.min(h - y0));
                    (x0, y0, tw, th, tile(m, x0, y0, tw, th))
                })
                .collect();
            let mut out = Rgb32f::new(w, h);
            for (x0, y0, tw, th, px) in results {
                for y in 0..th {
                    out.data[(y0 + y) * w + x0..(y0 + y) * w + x0 + tw].copy_from_slice(&px[y * tw..(y + 1) * tw]);
                }
            }
            out
        }

        /// Linear camera RGB (treated as sRGB/D65 primaries for the homogeneity metric only) → CIELab.
        #[inline]
        fn lab(p: [f32; 3]) -> [f32; 3] {
            let x = (0.4124 * p[0] + 0.3576 * p[1] + 0.1805 * p[2]) / 0.95047;
            let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            let z = (0.0193 * p[0] + 0.1192 * p[1] + 0.9505 * p[2]) / 1.08883;
            #[inline]
            fn f(t: f32) -> f32 {
                if t > 0.008856 { super::super::cbrt(t) } else { 7.787 * t + 16.0 / 116.0 }
            }
            let (fx, fy, fz) = (f(x), f(y), f(z));
            [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
        }

        fn tile(m: &Mosaic, x0: usize, y0: usize, tw: usize, th: usize) -> Vec<[f32; 3]> {
            let (bw, bh) = (tw + 2 * MARGIN, th + 2 * MARGIN);
            let n = bw * bh;
            let (ox, oy) = (x0 as isize - MARGIN as isize, y0 as isize - MARGIN as isize);
            let mut mos = vec![0f32; n];
            let mut col = vec![0u8; n];
            for y in 0..bh {
                for x in 0..bw {
                    mos[y * bw + x] = m.get(ox + x as isize, oy + y as isize);
                    col[y * bw + x] = m.color(ox + x as isize, oy + y as isize);
                }
            }
            let cl = |x: isize, y: isize| -> usize { y.clamp(0, bh as isize - 1) as usize * bw + x.clamp(0, bw as isize - 1) as usize };
            // 1. directional greens
            let mut gh = vec![0f32; n];
            let mut gv = vec![0f32; n];
            for y in 0..bh as isize {
                for x in 0..bw as isize {
                    let i = cl(x, y);
                    let c = mos[i];
                    if col[i] == 1 {
                        gh[i] = c;
                        gv[i] = c;
                        continue;
                    }
                    let (l, r) = (mos[cl(x - 1, y)], mos[cl(x + 1, y)]);
                    let est = (l + r) * 0.5 + (2.0 * c - mos[cl(x - 2, y)] - mos[cl(x + 2, y)]) * 0.25;
                    gh[i] = est.clamp(l.min(r), l.max(r));
                    let (u, d) = (mos[cl(x, y - 1)], mos[cl(x, y + 1)]);
                    let est = (u + d) * 0.5 + (2.0 * c - mos[cl(x, y - 2)] - mos[cl(x, y + 2)]) * 0.25;
                    gv[i] = est.clamp(u.min(d), u.max(d));
                }
            }
            // 2. full candidates
            let complete = |g: &[f32]| -> Vec<[f32; 3]> {
                let mut rgb = vec![[0f32; 3]; n];
                for y in 0..bh as isize {
                    for x in 0..bw as isize {
                        let i = cl(x, y);
                        let own = col[i] as usize;
                        let gvv = g[i];
                        let d = |dx: isize, dy: isize| {
                            let j = cl(x + dx, y + dy);
                            mos[j] - g[j]
                        };
                        let p = &mut rgb[i];
                        p[1] = gvv;
                        if own == 1 {
                            let ch = col[cl(x + 1, y)] as usize;
                            let cv = col[cl(x, y + 1)] as usize;
                            p[ch] = gvv + (d(-1, 0) + d(1, 0)) * 0.5;
                            p[cv] = gvv + (d(0, -1) + d(0, 1)) * 0.5;
                        } else {
                            p[own] = mos[i];
                            p[2 - own] = gvv + (d(-1, -1) + d(1, -1) + d(-1, 1) + d(1, 1)) * 0.25;
                        }
                    }
                }
                rgb
            };
            let rh = complete(&gh);
            let rv = complete(&gv);
            // 3. Lab + homogeneity
            let lh: Vec<[f32; 3]> = rh.iter().map(|&p| lab(p)).collect();
            let lv: Vec<[f32; 3]> = rv.iter().map(|&p| lab(p)).collect();
            let dl = |a: &[f32; 3], b: &[f32; 3]| (a[0] - b[0]).abs();
            let dc = |a: &[f32; 3], b: &[f32; 3]| (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            let mut hh = vec![0u8; n];
            let mut hv = vec![0u8; n];
            for y in 1..bh as isize - 1 {
                for x in 1..bw as isize - 1 {
                    let i = cl(x, y);
                    let (l, r, u, d) = (cl(x - 1, y), cl(x + 1, y), cl(x, y - 1), cl(x, y + 1));
                    let eps_l = dl(&lh[i], &lh[l]).max(dl(&lh[i], &lh[r])).min(dl(&lv[i], &lv[u]).max(dl(&lv[i], &lv[d])));
                    let eps_c = dc(&lh[i], &lh[l]).max(dc(&lh[i], &lh[r])).min(dc(&lv[i], &lv[u]).max(dc(&lv[i], &lv[d])));
                    let mut ch = 0;
                    let mut cv = 0;
                    for j in [l, r, u, d] {
                        if dl(&lh[i], &lh[j]) <= eps_l && dc(&lh[i], &lh[j]) <= eps_c {
                            ch += 1;
                        }
                        if dl(&lv[i], &lv[j]) <= eps_l && dc(&lv[i], &lv[j]) <= eps_c {
                            cv += 1;
                        }
                    }
                    hh[i] = ch;
                    hv[i] = cv;
                }
            }
            // 4. choose
            let mut out = Vec::with_capacity(tw * th);
            for y in MARGIN..MARGIN + th {
                for x in MARGIN..MARGIN + tw {
                    let (mut sh, mut sv) = (0u32, 0u32);
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            let j = cl(x as isize + dx, y as isize + dy);
                            sh += hh[j] as u32;
                            sv += hv[j] as u32;
                        }
                    }
                    let i = y * bw + x;
                    out.push(if sh > sv {
                        rh[i]
                    } else if sv > sh {
                        rv[i]
                    } else {
                        [(rh[i][0] + rv[i][0]) * 0.5, (rh[i][1] + rv[i][1]) * 0.5, (rh[i][2] + rv[i][2]) * 0.5]
                    });
                }
            }
            out
        }
    }

    #[test]
    fn cube_root_is_correctly_rounded() {
        // every 5th float over the range Lab takes it in (0.008856 … 64)
        let mut t = 0.008857f32;
        let mut off = 0;
        while t < 64.0 {
            if cbrt(t).to_bits() != ((t as f64).cbrt() as f32).to_bits() {
                off += 1;
            }
            t = f32::from_bits(t.to_bits() + 5);
        }
        assert_eq!(off, 0);
        for e in -6..6 {
            let t = 2f32.powi(e);
            assert!((cbrt(t) - t.cbrt()).abs() <= f32::EPSILON * t.cbrt(), "{t}");
        }
    }

    #[test]
    fn matches_the_plain_implementation_bit_for_bit() {
        // noise, edges and flat areas; sizes off the tile grid; every Bayer phase; tiny images
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for (w, h) in [(300, 190), (128, 128), (129, 7), (5, 260), (12, 12), (4, 4)] {
            let data: Vec<f32> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let edge = if (x / 9 + y / 13) % 2 == 0 { 0.7 } else { 0.1 };
                    if x > w / 2 { edge } else { rnd() }
                })
                .collect();
            for pat in ["RGGB", "BGGR", "GRBG", "GBRG"] {
                let cfa = Cfa::bayer(pat).unwrap();
                let m = Mosaic { w, h, data: &data, cfa: &cfa };
                let (a, b) = (ahd(&m), plain::ahd(&m));
                assert_eq!((a.width, a.height), (b.width, b.height));
                let same = a.data.iter().zip(&b.data).all(|(p, q)| p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits()));
                assert!(same, "{w}×{h} {pat}");
            }
        }
    }
}
