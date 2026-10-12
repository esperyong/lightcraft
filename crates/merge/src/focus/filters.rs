//! Plane filters for the depth map: a separable max filter (M. van Herk 1992; J. Gil & M. Werman
//! 1993: three comparisons per pixel whatever the radius) and a confidence-weighted guided filter
//! (K. He, J. Sun & X. Tang, "Guided Image Filtering", IEEE PAMI 2013, with every local mean
//! weighted — a normalised convolution — so unreliable depths don't steer the local models).

use lightcraft_raster::Plane;
use lightcraft_raster::blur::box_mean;
use rayon::prelude::*;

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
