//! The fuse pass on another device. A caller with a GPU (the engine, layer L4) hands the stacker a
//! [`FocusKernels`] implementation; the stacker then uploads each frame to it instead of measuring
//! and folding it on the CPU, keeps the method's per-pixel state there, and reads the state back
//! once at the end of the pass — everything after (the depth refinement, the gather pass, method
//! C) stays on the CPU, which remains the reference.
//!
//! A device that fails mid-pass costs nothing but time: the stacker repeats the pass on the CPU.

use lightcraft_geom::Homography;
use lightcraft_raster::Rgb32f;

use super::FocusMethod;
use super::measure::Rect;

/// Method B's per-pixel state as the CPU keeps it (see `depth.rs`), each vector one value per
/// output pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DepthState {
    pub best: Vec<u16>,
    pub fm_best: Vec<f32>,
    pub fm_left: Vec<f32>,
    pub fm_right: Vec<f32>,
    pub fm_sum: Vec<f32>,
    pub prev_fm: Vec<f32>,
    pub col_best: Vec<[u16; 3]>,
    pub col_left: Vec<[u16; 3]>,
    pub col_right: Vec<[u16; 3]>,
    pub prev_col: Vec<[u16; 3]>,
    /// The frames' noise floor (lowest of their focus measures' 10th percentiles).
    pub noise: f32,
}

/// A method's state after the fuse pass.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelState {
    /// Method A: weighted RGB sums, weight sums, weighted position sums.
    Average {
        sum: Vec<[f32; 3]>,
        wsum: Vec<f32>,
        dsum: Vec<f32>,
    },
    Depth(DepthState),
}

/// The fuse pass of methods A and B on another device.
pub trait FocusKernels: Send {
    /// The device, for reports ("Apple M5 (Metal)").
    fn name(&self) -> String;
    /// Start a pass over `w × h` output pixels: focus measured over `(2·radius+1)²`, method A's
    /// weights blurred by `sigma`. `Err` = not on this device (the CPU stacks instead).
    fn begin(&mut self, w: usize, h: usize, method: FocusMethod, radius: usize, sigma: f32) -> Result<(), String>;
    /// Fold in the frame at sequence position `pos`: `src` resampled through `h` (output →
    /// source pixels) into `out`, divided by `1 / inv_gain`, saturated at 0.98 × `clip`.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, pos: usize, src: &Rgb32f, h: &Homography, out: Rect, inv_gain: [f32; 3], clip: f32, bicubic: bool) -> Result<(), String>;
    /// The state after `frames` frames.
    fn finish(&mut self, frames: usize) -> Result<KernelState, String>;
}
