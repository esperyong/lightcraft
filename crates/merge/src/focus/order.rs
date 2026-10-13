//! Focus order from the frames themselves. Each frame's sharpness grid (focus measure per cell)
//! is turned into a focus profile — per cell, the log ratio to the cell's mean over all frames —
//! and frames are compared by the correlation of their profiles: frames focused at neighbouring
//! distances are sharp in overlapping places.
//!
//! - [`violations`] tells how many frames' most similar frame isn't a near neighbour in the given
//!   order (a hint that the order is wrong).
//! - [`spectral`] orders frames by the Fiedler vector of the similarity graph's Laplacian
//!   (spectral seriation: J. E. Atkins, E. G. Boman & B. Hendrickson, "A spectral algorithm for
//!   seriation and the consecutive ones problem", SIAM J. Computing 1998), on a graph of each
//!   frame's few most similar frames, then polishes the path with 2-opt segment reversals (the
//!   Fiedler vector flattens at the ends of a sequence).

use crate::linalg::sym_eigen;

/// Pairwise profile correlations (`n × n`, row-major).
pub fn similarity(grids: &[&[f32]]) -> Vec<f64> {
    let n = grids.len();
    let cells = grids.iter().map(|g| g.len()).min().unwrap_or(0);
    if n == 0 || cells == 0 {
        return vec![0.0; n * n];
    }
    let eps = 1e-6f64;
    let mean: Vec<f64> = (0..cells).map(|c| grids.iter().map(|g| g[c] as f64).sum::<f64>() / n as f64 + eps).collect();
    let prof: Vec<Vec<f64>> = grids
        .iter()
        .map(|g| {
            let v: Vec<f64> = (0..cells).map(|c| ((g[c] as f64 + eps) / mean[c]).ln()).collect();
            let m = v.iter().sum::<f64>() / cells as f64;
            let v: Vec<f64> = v.iter().map(|x| x - m).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 { v.iter().map(|x| x / norm).collect() } else { vec![0.0; cells] }
        })
        .collect();
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v: f64 = prof[i].iter().zip(&prof[j]).map(|(a, b)| a * b).sum();
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    s
}

/// Fraction of frames whose most similar other frame is more than two places away in this order.
pub fn violations(grids: &[&[f32]]) -> f64 {
    let n = grids.len();
    if n < 4 {
        return 0.0;
    }
    let s = similarity(grids);
    let bad = (0..n)
        .filter(|&i| {
            let best = (0..n).filter(|&j| j != i).max_by(|&a, &b| s[i * n + a].total_cmp(&s[i * n + b]));
            best.is_some_and(|j| j.abs_diff(i) > 2)
        })
        .count();
    bad as f64 / n as f64
}

/// A focus order of the frames (indices into `grids`), oriented to agree with the given order as
/// much as possible. `None` when the similarity graph falls apart (no consistent order).
pub fn spectral(grids: &[&[f32]]) -> Option<Vec<usize>> {
    let n = grids.len();
    if n < 3 {
        return Some((0..n).collect());
    }
    let s = similarity(grids);
    // symmetric k-nearest-neighbour graph, weights = positive correlations
    let k = 4.min(n - 1);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        let mut js: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        js.sort_by(|&x, &y| s[i * n + y].total_cmp(&s[i * n + x]));
        for &j in js.iter().take(k) {
            let wgt = s[i * n + j].max(0.0) + 1e-3;
            a[i * n + j] = wgt;
            a[j * n + i] = wgt;
        }
    }
    let mut lap = vec![0.0; n * n];
    for i in 0..n {
        let d: f64 = (0..n).map(|j| a[i * n + j]).sum();
        for j in 0..n {
            lap[i * n + j] = if i == j { d } else { -a[i * n + j] };
        }
    }
    let (vals, vecs) = sym_eigen(&lap, n);
    let (l2, f) = (vals.get(1).copied()?, vecs.get(1)?);
    let scale = vals.last().copied().unwrap_or(1.0).abs().max(1e-12);
    if l2 < 1e-9 * scale {
        return None;
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&x, &y| f[x].total_cmp(&f[y]).then(x.cmp(&y)));
    two_opt(&mut ord, &s, n);
    // orient like the given order (Spearman sign)
    let agree: f64 = ord.iter().enumerate().map(|(p, &i)| (p as f64 - (n - 1) as f64 / 2.0) * (i as f64 - (n - 1) as f64 / 2.0)).sum();
    if agree < 0.0 {
        ord.reverse();
    }
    Some(ord)
}

/// Improve an open path through the frames by segment reversals (2-opt) that raise the summed
/// similarity of consecutive frames; fixes the ends, where the Fiedler vector flattens.
fn two_opt(ord: &mut [usize], s: &[f64], n: usize) {
    let m = ord.len();
    let sim = |a: usize, b: usize| s[a * n + b];
    for _ in 0..50 {
        let mut improved = false;
        for i in 0..m {
            for j in i + 1..m {
                // reversing ord[i..=j] swaps edges (i−1, i) and (j, j+1) for (i−1, j) and (i, j+1)
                let mut delta = 0.0;
                if i > 0 {
                    delta += sim(ord[i - 1], ord[j]) - sim(ord[i - 1], ord[i]);
                }
                if j + 1 < m {
                    delta += sim(ord[i], ord[j + 1]) - sim(ord[j], ord[j + 1]);
                }
                if delta > 1e-12 {
                    ord[i..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grids of a focus sweep: frame `k` is sharp where the cell's depth is near `k`.
    fn sweep(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|k| {
                (0..64 * 40)
                    .map(|c| {
                        let (x, y) = ((c % 64) as f32, (c / 64) as f32);
                        let z = (x / 63.0 * 0.7 + y / 39.0 * 0.3) * (n - 1) as f32;
                        let texture = 1.0 + 0.5 * ((c * 7919) % 13) as f32 / 13.0;
                        texture * (0.02 + (-(z - k as f32).powi(2) / 4.0).exp())
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn recovers_a_shuffled_order_and_flags_it() {
        let g = sweep(12);
        let refs: Vec<&[f32]> = g.iter().map(|v| v.as_slice()).collect();
        assert_eq!(violations(&refs), 0.0);
        assert_eq!(spectral(&refs).unwrap(), (0..12).collect::<Vec<_>>());
        let perm = [5, 0, 11, 3, 8, 1, 10, 6, 2, 9, 4, 7];
        let shuffled: Vec<&[f32]> = perm.iter().map(|&i| g[i].as_slice()).collect();
        assert!(violations(&shuffled) > 0.5);
        let ord = spectral(&shuffled).unwrap();
        let recovered: Vec<usize> = ord.iter().map(|&p| perm[p]).collect();
        let fwd: Vec<usize> = (0..12).collect();
        let back: Vec<usize> = (0..12).rev().collect();
        assert!(recovered == fwd || recovered == back, "{recovered:?}");
    }
}
