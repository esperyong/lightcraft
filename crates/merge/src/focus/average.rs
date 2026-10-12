//! Method A, weighted average: every frame contributes to every pixel with weight `fm²` (its focus
//! measure squared), the weights Gaussian-smoothed (σ = 2 × Smoothing) so neighbouring pixels
//! agree. Smooth transitions, no hard seams, colours stay true; in-focus detail is slightly
//! softened by the out-of-focus frames' small weights. Flat areas (all weights alike) become the
//! mean of the frames, which also averages their noise.
//!
//! State: 20 bytes per output pixel (weighted RGB sum, weight sum, weighted position sum for the
//! depth estimate), independent of the number of frames.

use lightcraft_raster::blur::gaussian;
use lightcraft_raster::{Plane, Rgb32f};
use rayon::prelude::*;

use super::measure::Warped;

pub struct Average {
    w: usize,
    h: usize,
    sum: Vec<[f32; 3]>,
    wsum: Vec<f32>,
    dsum: Vec<f32>,
    sigma: f32,
}

impl Average {
    pub fn new(w: usize, h: usize, sigma: f32) -> Average {
        Average { w, h, sum: vec![[0.0; 3]; w * h], wsum: vec![0.0; w * h], dsum: vec![0.0; w * h], sigma }
    }

    pub fn bytes(&self) -> usize {
        self.sum.len() * 12 + self.wsum.len() * 4 + self.dsum.len() * 4
    }

    /// Add the frame at sequence position `pos` with its focus measure.
    pub fn add(&mut self, pos: usize, frame: &Warped, fm: &Plane) {
        let mut wt = fm.map(|f| f * f + 1e-10);
        if self.sigma > 0.3 {
            wt = gaussian(&wt, self.sigma);
        }
        let w = self.w.max(1);
        let rgb = &frame.rgb.data;
        let flags = &frame.flags;
        let p = pos as f32;
        self.sum.par_chunks_mut(w).zip(self.wsum.par_chunks_mut(w)).zip(self.dsum.par_chunks_mut(w)).enumerate().for_each(|(y, ((s, ws), ds))| {
            for x in 0..s.len() {
                let i = y * w + x;
                if flags[i] & super::measure::VALID == 0 {
                    continue;
                }
                let k = wt.data[i];
                let c = rgb[i];
                s[x] = [s[x][0] + k * c[0], s[x][1] + k * c[1], s[x][2] + k * c[2]];
                ws[x] += k;
                ds[x] += k * p;
            }
        });
    }

    /// The result and its depth (0 = first frame, 1 = frame `last`).
    pub fn finish(self, last: usize) -> (Rgb32f, Plane) {
        let mut img = Rgb32f::new(self.w, self.h);
        let mut depth = Plane::new(self.w, self.h);
        let inv_last = 1.0 / last.max(1) as f32;
        img.data.par_iter_mut().zip(depth.data.par_iter_mut()).enumerate().for_each(|(i, (o, d))| {
            let ws = self.wsum[i];
            if ws > 0.0 {
                let s = self.sum[i];
                *o = [s[0] / ws, s[1] / ws, s[2] / ws];
                *d = (self.dsum[i] / ws * inv_last).clamp(0.0, 1.0);
            }
        });
        (img, depth)
    }
}
