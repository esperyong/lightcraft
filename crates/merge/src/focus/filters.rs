//! Plane filters for the depth map: a separable max filter (M. van Herk 1992; J. Gil & M. Werman
//! 1993: three comparisons per pixel whatever the radius), a confidence-weighted guided filter
//! (K. He, J. Sun & X. Tang, "Guided Image Filtering", IEEE PAMI 2013, with every local mean
//! weighted — a normalised convolution — so unreliable depths don't steer the local models), and
//! an edge-aware fill that spreads values along the image and stops at its edges (a normalised
//! convolution with the domain transform's recursive filter: E. S. L. Gastal & M. M. Oliveira,
//! "Domain Transform for Edge-Aware Image and Video Processing", ACM TOG 2011).

use lightcraft_raster::Plane;
use lightcraft_raster::blur::box_mean;
use rayon::prelude::*;

use crate::pano::blend::{Buf, push_pull};

/// Max over a `(2r+1)` window of one row (missing samples outside the row).
fn max_row(row: &[f32], r: usize, out: &mut [f32], g: &mut Vec<f32>, h: &mut Vec<f32>) {
    let n = row.len();
    let k = 2 * r + 1;
    let len = n + 2 * r;
    let at = |i: usize| if i >= r && i - r < n { row[i - r] } else { f32::NEG_INFINITY };
    g.clear();
    h.clear();
    g.resize(len, f32::NEG_INFINITY);
    h.resize(len, f32::NEG_INFINITY);
    for i in 0..len {
        g[i] = if i % k == 0 { at(i) } else { g[i - 1].max(at(i)) };
    }
    for i in (0..len).rev() {
        h[i] = if i % k == k - 1 || i + 1 == len { at(i) } else { h[i + 1].max(at(i)) };
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = h[i].max(g[(i + k - 1).min(len - 1)]);
    }
}

fn max_rows(p: &Plane, r: usize) -> Plane {
    let mut out = Plane::new(p.width, p.height);
    let w = p.width.max(1);
    out.data.par_chunks_mut(w).zip(p.data.par_chunks(w)).for_each_init(|| (Vec::new(), Vec::new()), |(g, h), (o, row)| max_row(row, r, o, g, h));
    out
}

/// Transpose in 64 × 64 blocks.
fn transpose(p: &Plane) -> Plane {
    let (w, h) = (p.width, p.height);
    let mut out = Plane::new(h, w);
    const B: usize = 64;
    out.data.par_chunks_mut(h.max(1) * B).enumerate().for_each(|(bx, chunk)| {
        // output rows bx*B .. = input columns
        for by in (0..h).step_by(B) {
            for x in bx * B..((bx + 1) * B).min(w) {
                let orow = &mut chunk[(x - bx * B) * h..(x - bx * B + 1) * h];
                for y in by..(by + B).min(h) {
                    orow[y] = p.data[y * w + x];
                }
            }
        }
    });
    out
}

/// Max over the `(2r+1)²` window around each pixel (the window clipped to the plane).
pub fn max_filter(p: &Plane, r: usize) -> Plane {
    if r == 0 || p.data.is_empty() {
        return p.clone();
    }
    let h = max_rows(p, r);
    transpose(&max_rows(&transpose(&h), r))
}

/// Guided filter of `p` with guide `guide`, every local mean weighted by `c` (≥ 0), over
/// `(2r+1)²` windows; `eps` regularises flat guide areas (larger = smoother).
pub fn weighted_guided(guide: &Plane, p: &Plane, c: &Plane, r: usize, eps: f32) -> Plane {
    let (w, h) = (p.width, p.height);
    let prod = |f: &(dyn Fn(usize) -> f32 + Sync)| {
        let mut o = Plane::new(w, h);
        o.data.par_iter_mut().enumerate().for_each(|(i, v)| *v = f(i));
        box_mean(&o, r)
    };
    let (gi, pi, ci) = (&guide.data, &p.data, &c.data);
    let cw = box_mean(c, r);
    let ci_ = prod(&|i| ci[i] * gi[i]);
    let cp = prod(&|i| ci[i] * pi[i]);
    let cip = prod(&|i| ci[i] * gi[i] * pi[i]);
    let cii = prod(&|i| ci[i] * gi[i] * gi[i]);
    let mut a = Plane::new(w, h);
    let mut b = Plane::new(w, h);
    a.data.par_iter_mut().zip(b.data.par_iter_mut()).enumerate().for_each(|(i, (av, bv))| {
        let n = cw.data[i];
        if n <= 1e-12 {
            *bv = pi[i];
            return;
        }
        let (mi, mp) = (ci_.data[i] / n, cp.data[i] / n);
        let cov = cip.data[i] / n - mi * mp;
        let var = (cii.data[i] / n - mi * mi).max(0.0);
        *av = cov / (var + eps);
        *bv = mp - *av * mi;
    });
    let (ma, mb) = (box_mean(&a, r), box_mean(&b, r));
    let mut q = Plane::new(w, h);
    q.data.par_iter_mut().enumerate().for_each(|(i, v)| *v = ma.data[i] * gi[i] + mb.data[i]);
    q
}

/// The fill works on a grid this many times coarser than the image (it spreads over hundreds of
/// pixels; edges stay within a cell, and the guided filter after it puts them back on the pixel).
const FILL_SCALE: usize = 4;
/// Spatial reach (σs) of the fill, as a share of the image's longer side.
const FILL_REACH: f32 = 0.1;
/// Range σr of the fill, in grey units (√luminance): crossing an edge this strong costs as much
/// as σs of distance.
const FILL_EDGE: f32 = 0.05;
/// Grey steps between neighbouring cells up to this size (noise, shading) cost no distance.
const FILL_FLAT: f32 = 0.005;

/// Domain distances between horizontal neighbours: `1 + σs/σr · max(|ΔI| − flat, 0)`.
fn domain_dist(guide: &Plane, ratio: f32) -> Plane {
    let w = guide.width.max(1);
    let mut d = Plane::new(guide.width, guide.height);
    d.data.par_chunks_mut(w).zip(guide.data.par_chunks(w)).for_each(|(o, g)| {
        for x in 0..o.len() {
            o[x] = 1.0 + if x > 0 { ratio * ((g[x] - g[x - 1]).abs() - FILL_FLAT).max(0.0) } else { 0.0 };
        }
    });
    d
}

/// One recursive-filter pass along each row of `planes`, left to right then back; the feedback
/// from pixel `i − 1` to pixel `i` is `exp(ln_a · dist[i])`.
fn rf_rows(planes: [&mut Plane; 2], dist: &Plane, ln_a: f32) {
    let w = dist.width.max(1);
    let fb: Vec<f32> = dist.data.par_iter().map(|d| (ln_a * d).exp()).collect();
    for p in planes {
        p.data.par_chunks_mut(w).zip(fb.par_chunks(w)).for_each(|(row, f)| {
            for x in 1..row.len() {
                row[x] += f[x] * (row[x - 1] - row[x]);
            }
            for x in (0..row.len().saturating_sub(1)).rev() {
                row[x] += f[x + 1] * (row[x + 1] - row[x]);
            }
        });
    }
}

/// Every pixel's value spread from the weighted pixels (`wgt` 0..1) along the guide: a weighted
/// average over the pixels reachable without crossing strong guide edges, nearer ones counting
/// more. Pixels no weighted pixel reaches take the nearest ones' values (push–pull).
pub fn spread_along_edges(v: &Plane, wgt: &[f32], guide: &Plane) -> Plane {
    let (w, h) = (v.width, v.height);
    let s = FILL_SCALE;
    let (cw, ch) = (w.div_ceil(s).max(1), h.div_ceil(s).max(1));
    // the coarse grid: weight sum, weighted value sum, guide sum, pixel count per cell
    let mut acc = vec![[0.0f32; 4]; cw * ch];
    acc.par_chunks_mut(cw).enumerate().for_each(|(j, row)| {
        for y in j * s..((j + 1) * s).min(h) {
            for x in 0..w {
                let i = y * w + x;
                let (Some(&wt), Some(&val), Some(&g), Some(c)) = (wgt.get(i), v.data.get(i), guide.data.get(i), row.get_mut(x / s)) else { continue };
                *c = [c[0] + wt, c[1] + wt * val, c[2] + g, c[3] + 1.0];
            }
        }
    });
    let plane = |f: &(dyn Fn(&[f32; 4]) -> f32 + Sync)| Plane { width: cw, height: ch, data: acc.par_iter().map(f).collect() };
    let g = plane(&|c| c[2] / c[3].max(1.0));
    let mut den = plane(&|c| c[0] / c[3].max(1.0));
    let mut num = plane(&|c| c[1] / c[3].max(1.0));
    drop(acc);
    // normalised convolution with the recursive filter, three iterations of shrinking σ
    let sigma_s = FILL_REACH * cw.max(ch) as f32;
    let ratio = sigma_s / FILL_EDGE;
    let dh = domain_dist(&g, ratio);
    let dv = domain_dist(&transpose(&g), ratio);
    let n = 3;
    for i in 0..n {
        let sh = sigma_s * 3f32.sqrt() * 2f32.powi(n - i - 1) / (4f32.powi(n) - 1.0).sqrt();
        let ln_a = -(2f32.sqrt()) / sh.max(1e-3);
        rf_rows([&mut num, &mut den], &dh, ln_a);
        let (mut nt, mut dt) = (transpose(&num), transpose(&den));
        rf_rows([&mut nt, &mut dt], &dv, ln_a);
        num = transpose(&nt);
        den = transpose(&dt);
    }
    let reached: Vec<f32> = den.data.iter().map(|&d| if d > 1e-9 { 1.0 } else { 0.0 }).collect();
    let mut fill = Buf { w: cw, h: ch, c: 1, data: num.data.iter().zip(&den.data).map(|(n, d)| if *d > 1e-9 { n / d } else { 0.0 }).collect() };
    push_pull(&mut fill, &reached);
    // back to pixels: bilinear between cell centres, each cell weighted by how close its grey is
    // to the pixel's (joint bilateral upsampling: J. Kopf, M. F. Cohen, D. Lischinski &
    // M. Uyttendaele, ACM TOG 2007), so a cell across an edge doesn't bleed into the other side
    let mut out = Plane::new(w, h);
    let at = |p: &[f32], x: usize, y: usize| p.get(y * cw + x).copied().unwrap_or(0.0);
    out.data.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) / s as f32 - 0.5).clamp(0.0, (ch - 1) as f32);
        let (y0, ty) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(ch - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) / s as f32 - 0.5).clamp(0.0, (cw - 1) as f32);
            let (x0, tx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(cw - 1);
            let gp = guide.data.get(y * w + x).copied().unwrap_or(0.0);
            let (mut sum, mut wsum, mut plain) = (0.0f32, 0.0f32, 0.0f32);
            for (cx, cy, b) in [(x0, y0, (1.0 - tx) * (1.0 - ty)), (x1, y0, tx * (1.0 - ty)), (x0, y1, (1.0 - tx) * ty), (x1, y1, tx * ty)] {
                let v = at(&fill.data, cx, cy);
                let r = (at(&g.data, cx, cy) - gp) / FILL_EDGE;
                let wt = b * (-r * r).exp();
                sum += wt * v;
                wsum += wt;
                plain += b * v;
            }
            *o = if wsum > 1e-6 { sum / wsum } else { plain };
        }
    });
    out
}

#[inline]
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return if x >= b { 1.0 } else { 0.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_filter_matches_brute_force() {
        let p = Plane::from_fn(37, 23, |x, y| (((x * 7919 + y * 104_729) % 97) as f32) / 97.0);
        for r in [1, 3, 8] {
            let m = max_filter(&p, r);
            for y in 0..23usize {
                for x in 0..37usize {
                    let mut want = f32::NEG_INFINITY;
                    for yy in y.saturating_sub(r)..=(y + r).min(22) {
                        for xx in x.saturating_sub(r)..=(x + r).min(36) {
                            want = want.max(p.get(xx, yy));
                        }
                    }
                    assert_eq!(m.get(x, y), want, "r {r} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn the_fill_stays_on_its_side_of_an_edge() {
        // left half dark, right half bright; one known value on each side, far from the edge
        let (w, h) = (120, 60);
        let guide = Plane::from_fn(w, h, |x, _| if x < 60 { 0.3 } else { 0.7 });
        let v = Plane::from_fn(w, h, |x, _| if x < 60 { 2.0 } else { 9.0 });
        let wgt: Vec<f32> = (0..w * h).map(|i| if [(10, 30), (110, 30)].contains(&(i % w, i / w)) { 1.0 } else { 0.0 }).collect();
        let f = spread_along_edges(&v, &wgt, &guide);
        // just beside the edge each side keeps its own value, though the other is nearer
        for y in [5, 30, 55] {
            assert!((f.get(52, y) - 2.0).abs() < 0.2, "{}", f.get(52, y));
            assert!((f.get(67, y) - 9.0).abs() < 0.2, "{}", f.get(67, y));
        }
        // nothing weighted: no values to spread
        let f = spread_along_edges(&v, &vec![0.0; w * h], &guide);
        assert!(f.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn guided_filter_keeps_edges_of_the_guide_and_ignores_unweighted_values() {
        // guide: a step; input: the step plus an outlier with no weight
        let guide = Plane::from_fn(40, 20, |x, _| if x < 20 { 0.2 } else { 0.8 });
        let mut p = Plane::from_fn(40, 20, |x, _| if x < 20 { 3.0 } else { 7.0 });
        p.set(10, 10, 100.0);
        let mut c = Plane::filled(40, 20, 1.0);
        c.set(10, 10, 0.0);
        let q = weighted_guided(&guide, &p, &c, 3, 1e-4);
        assert!((q.get(10, 10) - 3.0).abs() < 0.05, "{}", q.get(10, 10));
        assert!((q.get(18, 5) - 3.0).abs() < 0.05 && (q.get(21, 5) - 7.0).abs() < 0.05, "{} {}", q.get(18, 5), q.get(21, 5));
    }
}
