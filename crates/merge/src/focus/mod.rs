//! Focus stacking: a focus bracket (2–300 photos of the same scene focused at different
//! distances) → one photo sharp throughout, plus a depth map.
//!
//! Three methods, matching what photographers know from dedicated stacking software:
//! - **A — weighted average** ([`average`]): every frame weighted by its local sharpness. Smooth,
//!   true colours, slightly soft.
//! - **B — depth map** ([`depth`]): each pixel from the frame where it is sharpest, on a smoothed
//!   depth map. Crisp detail and texture; the default.
//! - **C — pyramid** ([`pyramid`]): Laplacian pyramid bands from the frames with the most local
//!   energy. Best at overlapping fine structures.
//!
//! *Radius* (pixels at full resolution) is the window sharpness is measured over; *Smoothing*
//! how much the choice is smoothed (A: the weights, B: the depth map, C: the band energies).
//!
//! **Streaming, three passes.** Memory doesn't grow with the number of frames: the caller hands
//! the frames to a [`FocusStacker`] one at a time, pass by pass ([`FocusStacker::next_pass`]
//! says which frames, at what size), and each frame is dropped once pushed.
//! 1. *Order* (only for [`FocusOrder::Auto`]): ≤ 512 px frames, focus order from sharpness grids
//!    ([`order`]).
//! 2. *Align* (≤ 2048 px): each frame registered to its neighbour and chained to a reference;
//!    brightness gains; the area every frame covers ([`align`], [`lk`]).
//! 3. *Fuse* (full resolution): each frame resampled into the reference geometry, its focus
//!    measured ([`measure`]) and folded into the method's state.
//! 4. *Gather* (method B, only when needed): colours for pixels whose smoothed depth moved away
//!    from the frames the fuse pass kept. At half size: those are the low-confidence pixels, with
//!    little detail of their own, and a raw is then binned instead of demosaiced (the slowest part
//!    of decoding it).
//!
//! Colours always come from the source frames (no sharpening, no tone changes); the result is
//! scene-linear like its sources, written as a linear DNG by the caller. Our own implementation
//! from the papers cited in each module; MIT-licensed stackers (focus-stack, LAPSTACK, z-stackr,
//! stackaroni) were read for ideas only, no GPL code was consulted.

pub mod align;
pub mod average;
pub mod depth;
pub mod filters;
pub mod kernels;
pub mod lk;
pub mod measure;
pub mod order;
mod pyramid;

use std::borrow::Cow;

use lightcraft_geom::{Homography, Orientation};
use lightcraft_meta::Metadata;
use lightcraft_raster::resample::{Filter, fit};
use lightcraft_raster::{Plane, Rgb32f};
use serde::{Deserialize, Serialize};

use crate::frame::{Frame, FrameColor};
use crate::{MergeError, Progress, Result, check};
pub use measure::Rect;

/// `SemanticName` of a focus stack's depth map stored in its DNG (a DNG semantic mask, 0 = the
/// first frame of the focus order … 65535 = the last).
pub const DEPTH_MASK_NAME: &str = "urn:app:lightcraft:focus:depth";

/// The depth map as a DNG semantic mask: 16-bit, fitted within 4096 px (a matte doesn't need the
/// full resolution; the reader caps masks at 32 M samples).
pub fn depth_mask(depth: &Plane) -> Option<lightcraft_raw::SemanticMask> {
    if depth.width == 0 || depth.height == 0 {
        return None;
    }
    let d = if depth.width.max(depth.height) > 4096 { fit(depth, 4096, 4096, Filter::Box) } else { depth.clone() };
    Some(lightcraft_raw::SemanticMask {
        name: DEPTH_MASK_NAME.into(),
        instance_id: None,
        width: d.width,
        height: d.height,
        sub_area: None,
        data: d.data.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect(),
    })
}

/// How the frames are combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FocusMethod {
    /// A: weighted average.
    #[serde(alias = "a", alias = "A")]
    Average,
    /// B: depth map.
    #[default]
    #[serde(alias = "b", alias = "B")]
    Depth,
    /// C: pyramid.
    #[serde(alias = "c", alias = "C")]
    Pyramid,
}

impl FocusMethod {
    pub fn letter(self) -> &'static str {
        match self {
            FocusMethod::Average => "A",
            FocusMethod::Depth => "B",
            FocusMethod::Pyramid => "C",
        }
    }
}

/// The focus order of the frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FocusOrder {
    /// By capture time (the caller sorts; cameras shoot brackets near → far or far → near).
    #[default]
    Capture,
    /// As selected (the caller's order).
    Selection,
    /// The given order, reversed.
    Reverse,
    /// Worked out from the frames.
    Auto,
}

/// The frame the others are aligned to (its geometry and colour tags are the result's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Reference {
    #[default]
    Middle,
    First,
    Sharpest,
}

/// The output area.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CropMode {
    /// The area every aligned frame covers.
    #[default]
    Common,
    /// The whole reference frame (edges some frames don't reach use the others).
    Reference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AlignOptions {
    /// Align the frames at all.
    pub enabled: bool,
    pub shift: bool,
    pub rotation: bool,
    /// Magnification (focus breathing).
    pub scale: bool,
    /// A full affine model instead of a similarity.
    pub affine: bool,
    /// Even out brightness differences between frames.
    pub brightness: bool,
    pub reference: Reference,
    pub crop: CropMode,
    /// Long edge of the alignment images.
    pub work_edge: usize,
}

impl Default for AlignOptions {
    fn default() -> Self {
        AlignOptions {
            enabled: true,
            shift: true,
            rotation: true,
            scale: true,
            affine: false,
            brightness: true,
            reference: Reference::Middle,
            crop: CropMode::Common,
            work_edge: 2048,
        }
    }
}

impl AlignOptions {
    fn dof(&self) -> lk::Dof {
        lk::Dof { shift: self.shift, rotation: self.rotation, scale: self.scale, affine: self.affine }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct FocusOptions {
    pub method: FocusMethod,
    /// Sharpness window radius, in pixels of the frames stacked (1–30 at full resolution).
    pub radius: f32,
    /// Smoothing (0–10, pixels-ish at full resolution).
    pub smoothing: f32,
    pub align: AlignOptions,
    pub order: FocusOrder,
    /// Stack every n-th frame of the focus order (the first and last always); 1 = all.
    pub every: usize,
    /// Method B: blend the two frames nearest the depth (else take the nearest one).
    pub interpolate: bool,
    /// Bicubic (Catmull–Rom) resampling of aligned frames, else bilinear.
    pub bicubic: bool,
    /// Long edge of the frames to stack (`None` = full size; previews use ~1024).
    pub max_edge: Option<usize>,
}

impl Default for FocusOptions {
    fn default() -> Self {
        FocusOptions {
            method: FocusMethod::Depth,
            radius: 8.0,
            smoothing: 4.0,
            align: AlignOptions::default(),
            order: FocusOrder::Capture,
            every: 1,
            interpolate: true,
            bicubic: true,
            max_edge: None,
        }
    }
}

impl FocusOptions {
    /// The options for frames `s` times the full size (previews): pixel radii scale with it.
    pub fn scaled(&self, s: f32) -> FocusOptions {
        let s = if s.is_finite() && s > 0.0 { s.min(1.0) } else { 1.0 };
        FocusOptions { radius: self.radius * s, smoothing: self.smoothing * s, ..self.clone() }
    }

    fn radius_px(&self) -> usize {
        if self.radius.is_finite() { self.radius.round().clamp(1.0, 200.0) as usize } else { 8 }
    }

    fn smoothing_px(&self) -> f32 {
        if self.smoothing.is_finite() { self.smoothing.clamp(0.0, 100.0) } else { 0.0 }
    }
}

/// A pass of the stacker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pass {
    Order,
    Align,
    Fuse,
    Gather,
}

impl Pass {
    fn stage(self) -> &'static str {
        match self {
            Pass::Order => "Ordering photos",
            Pass::Align => "Aligning",
            Pass::Fuse => "Stacking",
            Pass::Gather => "Collecting colours",
        }
    }
}

/// What the stacker needs next: these frames (indices into the inputs), in this order, no
/// larger than `max_edge` (`None` = full size; the stacker scales larger frames down itself).
#[derive(Clone, Debug, PartialEq)]
pub struct PassRequest {
    pub pass: Pass,
    pub frames: Vec<usize>,
    pub max_edge: Option<usize>,
}

/// How one stacked frame maps onto the result.
#[derive(Clone, Debug, PartialEq)]
pub struct FocusAlignment {
    /// Index into the inputs.
    pub index: usize,
    /// Maps reference pixels (full stacked size) to this frame's pixels.
    pub h: Homography,
    /// Brightness relative to the reference, per channel (the frame was divided by it).
    pub gain: [f32; 3],
    /// Correlation with its neighbour after alignment (1 = identical).
    pub ncc: f32,
    /// The alignment needed a feature-based starting point.
    pub seeded: bool,
}

/// The stacked image and how it was made.
#[derive(Clone, Debug)]
pub struct FocusResult {
    /// Scene-linear pixels in the reference frame's units, the crop of the reference geometry.
    pub image: Rgb32f,
    /// Depth 0..1 (0 = the first frame of the focus order, 1 = the last), same size as `image`.
    pub depth: Plane,
    /// Method B: confidence of the depth (0..1).
    pub confidence: Option<Plane>,
    pub method: FocusMethod,
    /// The focus order (indices into the inputs) after ordering and `every`.
    pub order: Vec<usize>,
    /// The frames stacked, in focus order.
    pub used: Vec<usize>,
    /// Frames left out (they didn't match their neighbours).
    pub skipped: Vec<usize>,
    /// The reference frame (index into the inputs).
    pub reference: usize,
    pub alignments: Vec<FocusAlignment>,
    /// The output area within the reference frame (`frame_size` pixels).
    pub crop: Rect,
    pub frame_size: (usize, usize),
    pub warnings: Vec<String>,
    /// Where the fuse pass ran: "CPU", or the device's name (with why it fell back, if it did).
    pub device: String,
    /// Method B: share of the pixels whose colours came from the gather pass.
    pub gathered: f32,
    pub color: FrameColor,
    pub raw: bool,
    pub orientation: Orientation,
    pub metadata: Metadata,
    pub baseline_exposure: f64,
}

/// A stack's geometry fixed in advance — re-stacking a saved focus project: no ordering or
/// alignment pass, the inputs stacked in the given order through these maps.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGeometry {
    /// The full-size frame the maps and the crop are in.
    pub frame_size: (usize, usize),
    /// Per input: reference → frame pixels at `frame_size`.
    pub maps: Vec<Homography>,
    /// Per input: brightness relative to the reference (the frame is divided by it).
    pub gains: Vec<[f32; 3]>,
    pub crop: Rect,
    /// The reference input.
    pub reference: usize,
}

/// Frames on demand (decoded files, test images…).
pub trait FrameSource: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Frame `index`, no larger than `max_edge` when given.
    fn frame(&self, index: usize, max_edge: Option<usize>) -> Result<Frame>;
}

impl FrameSource for Vec<Frame> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn frame(&self, index: usize, max_edge: Option<usize>) -> Result<Frame> {
        let f = self.get(index).ok_or_else(|| MergeError::Mismatch(format!("no photo {index}")))?;
        let mut f = f.clone();
        if let Some(m) = max_edge
            && f.image.width.max(f.image.height) > m
        {
            f.image = fit(&f.image, m, m, Filter::Box);
        }
        Ok(f)
    }
}

/// Stack the frames of `src` (in their focus order unless `opts.order` says otherwise).
pub fn stack_focus(src: &dyn FrameSource, opts: &FocusOptions, progress: &Progress) -> Result<FocusResult> {
    let mut st = FocusStacker::new(src.len(), opts.clone())?;
    while let Some(req) = st.next_pass()? {
        for &i in &req.frames {
            let f = src.frame(i, req.max_edge)?;
            st.push(i, &f, progress)?;
        }
        st.end_pass(progress)?;
    }
    st.finish(progress)
}

/// Below this long edge the gather pass takes its frames at full size.
const MIN_GATHER_EDGE: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Order,
    Align,
    Fuse,
    Gather,
    Done,
}

struct Active {
    pass: Pass,
    frames: Vec<usize>,
    pushed: usize,
}

/// The output geometry, fixed by the first full-size frame.
struct Geometry {
    size: (usize, usize),
    rect: Rect,
    /// Per fuse position: reference (full size) → frame pixels.
    p: Vec<Homography>,
    inv_gain: Vec<[f32; 3]>,
}

struct RefMeta {
    color: FrameColor,
    raw: bool,
    metadata: Metadata,
    baseline_exposure: f64,
}

enum Fuser {
    Average(average::Average),
    Depth(depth::Depth),
    Pyramid(pyramid::Pyramid),
}

/// Progress fractions of the passes.
#[derive(Clone, Copy)]
struct Spans {
    order: (f32, f32),
    align: (f32, f32),
    fuse: (f32, f32),
    gather: (f32, f32),
}

/// The streaming focus stacker (see the module docs for the protocol).
pub struct FocusStacker {
    n: usize,
    opts: FocusOptions,
    /// Inputs in focus order (after `every`).
    seq: Vec<usize>,
    stage: Stage,
    active: Option<Active>,
    order_grids: Vec<(usize, Vec<f32>)>,
    aligner: Option<align::Aligner>,
    chain: Option<align::Chain>,
    /// Sequence positions stacked (the skipped ones left out).
    fuse_seq: Vec<usize>,
    geometry: Option<Geometry>,
    fuser: Option<Fuser>,
    resolved: Option<depth::Resolved>,
    gather: Vec<usize>,
    /// Share of the pixels the gather pass colours.
    gathered: f32,
    outcome: Option<(Rgb32f, Plane, Option<Plane>)>,
    reference_meta: Option<RefMeta>,
    warnings: Vec<String>,
    spans: Spans,
    /// Frame-sized buffers reused from frame to frame.
    frame_buf: measure::FrameBuf,
    measure_buf: measure::MeasureBuf,
    /// Another device for the fuse pass, and its state in the current pass.
    kernels: Option<Box<dyn kernels::FocusKernels>>,
    fixed: Option<FixedGeometry>,
    device: Device,
    device_note: String,
}

/// Where the current fuse pass runs.
#[derive(Clone, Debug, PartialEq)]
enum Device {
    Cpu,
    Kernels,
    /// The device failed in this pass (it is repeated on the CPU).
    Failed(String),
}

/// `seq` thinned to every n-th entry, keeping the first and the last.
fn thin(seq: Vec<usize>, every: usize) -> Vec<usize> {
    if every <= 1 || seq.len() <= 2 {
        return seq;
    }
    let last = seq.len() - 1;
    seq.iter().enumerate().filter(|(p, _)| p % every == 0 || *p == last).map(|(_, &i)| i).collect()
}

impl FocusStacker {
    /// A stacker for `n` input frames.
    pub fn new(n: usize, opts: FocusOptions) -> Result<FocusStacker> {
        if n < 2 {
            return Err(MergeError::TooFew(n, 2));
        }
        let auto = opts.order == FocusOrder::Auto;
        let aligning = opts.align.enabled || opts.align.brightness;
        let mut seq: Vec<usize> = (0..n).collect();
        if opts.order == FocusOrder::Reverse {
            seq.reverse();
        }
        if !auto {
            seq = thin(seq, opts.every);
        }
        let stage = if auto {
            Stage::Order
        } else if aligning {
            Stage::Align
        } else {
            Stage::Fuse
        };
        let o = if auto { 0.05 } else { 0.0 };
        let a = if aligning { o + 0.2 } else { o };
        let spans = Spans { order: (0.0, o), align: (o, a), fuse: (a, 0.9), gather: (0.9, 0.97) };
        let fuse_seq = if aligning { Vec::new() } else { (0..seq.len()).collect() };
        Ok(FocusStacker {
            n,
            opts,
            seq,
            stage,
            active: None,
            order_grids: Vec::new(),
            aligner: None,
            chain: None,
            fuse_seq,
            geometry: None,
            fuser: None,
            resolved: None,
            gather: Vec::new(),
            gathered: 0.0,
            outcome: None,
            reference_meta: None,
            warnings: Vec::new(),
            spans,
            frame_buf: measure::FrameBuf::default(),
            measure_buf: measure::MeasureBuf::default(),
            kernels: None,
            fixed: None,
            device: Device::Cpu,
            device_note: "CPU".into(),
        })
    }

    /// A stacker for `fixed.maps.len()` inputs with the geometry decided before (a saved focus
    /// project): one fuse pass (plus gather), inputs in the given order, frames at any size
    /// (the maps and the crop scale with them).
    pub fn with_geometry(opts: FocusOptions, fixed: FixedGeometry) -> Result<FocusStacker> {
        let n = fixed.maps.len();
        if fixed.gains.len() != n || fixed.reference >= n {
            return Err(MergeError::Mismatch("focus stack: the saved geometry doesn't match its inputs".into()));
        }
        let opts = FocusOptions { order: FocusOrder::Selection, every: 1, ..opts };
        let mut st = FocusStacker::new(n, FocusOptions { align: AlignOptions { enabled: false, brightness: false, ..opts.align.clone() }, ..opts })?;
        st.seq = (0..n).collect();
        st.fuse_seq = (0..n).collect();
        st.stage = Stage::Fuse;
        st.spans = Spans { order: (0.0, 0.0), align: (0.0, 0.0), fuse: (0.0, 0.9), gather: (0.9, 0.97) };
        st.fixed = Some(fixed);
        Ok(st)
    }

    /// Run the fuse pass of methods A and B on another device (a GPU). Method C, the depth
    /// refinement and the gather pass stay on the CPU; a device that fails repeats the pass there.
    pub fn set_kernels(&mut self, k: Box<dyn kernels::FocusKernels>) {
        self.kernels = Some(k);
    }

    /// Bytes of image-sized state held now (alignment images, the method's per-pixel state, the
    /// reused per-frame buffers); the per-frame bookkeeping (≈ 12 KB a frame) is not counted.
    pub fn state_bytes(&self) -> usize {
        let fuser = match &self.fuser {
            Some(Fuser::Average(a)) => a.bytes(),
            Some(Fuser::Depth(d)) => d.bytes(),
            Some(Fuser::Pyramid(p)) => p.bytes(),
            None => 0,
        };
        fuser
            + self.aligner.as_ref().map_or(0, align::Aligner::bytes)
            + self.resolved.as_ref().map_or(0, depth::Resolved::bytes)
            + self.frame_buf.bytes()
            + self.measure_buf.bytes()
    }

    fn cap(&self, edge: usize) -> Option<usize> {
        Some(self.opts.max_edge.map_or(edge, |m| m.min(edge)))
    }

    /// The next pass, or `None` when everything is stacked ([`FocusStacker::finish`]).
    pub fn next_pass(&mut self) -> Result<Option<PassRequest>> {
        if let Some(a) = &self.active {
            return Err(MergeError::Mismatch(format!("focus stack: the {:?} pass isn't finished", a.pass)));
        }
        let (pass, frames, max_edge) = match self.stage {
            Stage::Order => (Pass::Order, (0..self.n).collect(), self.cap(512)),
            Stage::Align => {
                self.aligner = Some(align::Aligner::new(
                    self.opts.align.dof(),
                    self.opts.align.enabled,
                    self.opts.align.brightness,
                    self.cap(self.opts.align.work_edge).unwrap_or(2048),
                ));
                (Pass::Align, self.seq.clone(), self.cap(self.opts.align.work_edge))
            }
            Stage::Fuse => (Pass::Fuse, self.fuse_seq.iter().map(|&p| self.seq[p]).collect(), self.opts.max_edge),
            Stage::Gather => {
                // TEMPORARY (#18, M5 round 3, to be removed before merging): `LIGHTCRAFT_FOCUS_GATHER=full`
                // gathers from full-size frames, to compare with the half-size default on real brackets
                let full = std::env::var("LIGHTCRAFT_FOCUS_GATHER").is_ok_and(|v| v == "full");
                let half = self.geometry.as_ref().map(|g| g.size.0.max(g.size.1).div_ceil(2)).filter(|&e| e >= MIN_GATHER_EDGE && !full);
                (Pass::Gather, self.gather.iter().map(|&p| self.seq[self.fuse_seq[p]]).collect(), half.or(self.opts.max_edge))
            }
            Stage::Done => return Ok(None),
        };
        self.active = Some(Active { pass, frames: Vec::clone(&frames), pushed: 0 });
        Ok(Some(PassRequest { pass, frames, max_edge }))
    }

    fn span(&self, pass: Pass) -> (f32, f32) {
        match pass {
            Pass::Order => self.spans.order,
            Pass::Align => self.spans.align,
            Pass::Fuse => self.spans.fuse,
            Pass::Gather => self.spans.gather,
        }
    }

    /// Hand over the next frame of the current pass (`index` = the input it is).
    pub fn push(&mut self, index: usize, frame: &Frame, progress: &Progress) -> Result<()> {
        let Some(a) = &self.active else { return Err(MergeError::Mismatch("focus stack: no pass is running".into())) };
        let (pass, pos, total) = (a.pass, a.pushed, a.frames.len());
        let expected = a.frames.get(pos).copied();
        if expected != Some(index) {
            return Err(MergeError::Mismatch(format!(
                "focus stack: got photo {} while the {:?} pass expected {}",
                index + 1,
                pass,
                expected.map_or("none".to_string(), |e| (e + 1).to_string())
            )));
        }
        let span = self.span(pass);
        check(progress, span.0 + (span.1 - span.0) * pos as f32 / total.max(1) as f32, pass.stage())?;
        let img: Cow<Rgb32f> =
            if frame.orientation == Orientation::Normal { Cow::Borrowed(&frame.image) } else { Cow::Owned(frame.image.oriented(frame.orientation)) };
        let clip = if frame.clip.is_finite() && frame.clip > 0.0 { frame.clip } else { f32::INFINITY };
        match pass {
            Pass::Order => {
                let wf = align::work_frame(&img, if clip.is_finite() { clip } else { 1.0 }, 512);
                self.order_grids.push((index, wf.grid));
            }
            Pass::Align => {
                let Some(al) = self.aligner.as_mut() else { return Err(MergeError::Mismatch("focus stack: alignment not started".into())) };
                al.push(index, &img, if clip.is_finite() { clip } else { 1.0 })?;
            }
            Pass::Fuse => self.fuse(pos, index, &img, frame, clip)?,
            Pass::Gather => self.gather_one(pos, &img, clip)?,
        }
        if let Some(a) = self.active.as_mut() {
            a.pushed += 1;
        }
        check(progress, span.0 + (span.1 - span.0) * (pos + 1) as f32 / total.max(1) as f32, pass.stage())
    }

    fn geometry_for(&self, size: (usize, usize)) -> Geometry {
        let m = self.fuse_seq.len();
        if let Some(f) = &self.fixed {
            let sx = size.0 as f64 / f.frame_size.0.max(1) as f64;
            let sy = size.1 as f64 / f.frame_size.1.max(1) as f64;
            // S · M · S⁻¹ with S = diag(sx, sy): the maps in this frame size's pixels
            let rescale = |h: &Homography| {
                let m = h.0;
                Homography([m[0], m[1] * sx / sy, m[2] * sx, m[3] * sy / sx, m[4], m[5] * sy, m[6] / sx, m[7] / sy, m[8]])
            };
            let c = f.crop;
            let x0 = ((c.x as f64 * sx).round() as usize).min(size.0);
            let y0 = ((c.y as f64 * sy).round() as usize).min(size.1);
            let w = ((c.w as f64 * sx).round() as usize).clamp(1, size.0.saturating_sub(x0).max(1));
            let h = ((c.h as f64 * sy).round() as usize).clamp(1, size.1.saturating_sub(y0).max(1));
            return Geometry {
                size,
                rect: Rect { x: x0, y: y0, w, h },
                p: f.maps.iter().map(rescale).collect(),
                inv_gain: f.gains.iter().map(|g| g.map(|v| if v > 0.0 && v.is_finite() { 1.0 / v } else { 1.0 })).collect(),
            };
        }
        let Some(ch) = &self.chain else {
            return Geometry { size, rect: Rect { x: 0, y: 0, w: size.0, h: size.1 }, p: vec![Homography::IDENTITY; m], inv_gain: vec![[1.0; 3]; m] };
        };
        let (ww, wh) = ch.work_size;
        let sx = ww as f64 / size.0.max(1) as f64;
        let sy = wh as f64 / size.1.max(1) as f64;
        let rescale = |h: &Homography| {
            let m = h.0;
            Homography([m[0], m[1] * sy / sx, m[2] / sx, m[3] * sx / sy, m[4], m[5] / sy, m[6] * sx, m[7] * sy, m[8]])
        };
        let p: Vec<Homography> = self.fuse_seq.iter().map(|&q| rescale(&ch.p[q])).collect();
        let inv_gain = self.fuse_seq.iter().map(|&q| ch.gain[q].map(|g| if g > 0.0 && g.is_finite() { 1.0 / g } else { 1.0 })).collect();
        let full = Rect { x: 0, y: 0, w: size.0, h: size.1 };
        let rect = match (self.opts.align.crop, ch.common) {
            (CropMode::Common, Some(c)) => {
                let x0 = (c[0] / sx).ceil().max(0.0) as usize;
                let y0 = (c[1] / sy).ceil().max(0.0) as usize;
                let x1 = (((c[0] + c[2]) / sx).floor() as usize).min(size.0);
                let y1 = (((c[1] + c[3]) / sy).floor() as usize).min(size.1);
                if x1 > x0 + 8 && y1 > y0 + 8 { Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } } else { full }
            }
            _ => full,
        };
        Geometry { size, rect, p, inv_gain }
    }

    fn reference_index(&self) -> usize {
        if let Some(f) = &self.fixed {
            return f.reference;
        }
        match &self.chain {
            Some(c) => self.seq.get(c.reference).copied().unwrap_or(0),
            None => self.seq.get(self.seq.len() / 2).copied().unwrap_or(0),
        }
    }

    fn fuse(&mut self, pos: usize, index: usize, img: &Rgb32f, frame: &Frame, clip: f32) -> Result<()> {
        let size = (img.width, img.height);
        if self.geometry.is_none() {
            self.geometry = Some(self.geometry_for(size));
        }
        if self.fuser.is_none() && self.device == Device::Cpu {
            let Some(g) = &self.geometry else { return Ok(()) };
            let (w, h) = (g.rect.w, g.rect.h);
            let s = self.opts.smoothing_px();
            let on_device = match (self.kernels.as_mut(), self.opts.method) {
                (Some(k), FocusMethod::Average | FocusMethod::Depth) => match k.begin(w, h, self.opts.method, self.opts.radius_px(), 2.0 * s) {
                    Ok(()) => {
                        self.device_note = k.name();
                        true
                    }
                    Err(e) => {
                        self.device_note = format!("CPU ({}: {e})", k.name());
                        self.kernels = None;
                        false
                    }
                },
                _ => false,
            };
            if on_device {
                self.device = Device::Kernels;
            } else {
                self.fuser = Some(match self.opts.method {
                    FocusMethod::Average => Fuser::Average(average::Average::new(w, h, 2.0 * s)),
                    FocusMethod::Depth => Fuser::Depth(depth::Depth::new(w, h)),
                    FocusMethod::Pyramid => Fuser::Pyramid(pyramid::Pyramid::new(w, h, self.opts.radius_px() as f32, s)),
                });
            }
        }
        let Some(g) = &self.geometry else { return Ok(()) };
        if g.size != size {
            return Err(MergeError::Mismatch(format!(
                "photo {} is {}×{}, the others {}×{}; focus stacking needs photos of the same size",
                index + 1,
                size.0,
                size.1,
                g.size.0,
                g.size.1
            )));
        }
        if index == self.reference_index() {
            self.reference_meta = Some(RefMeta {
                color: frame.color.clone(),
                raw: frame.raw,
                metadata: frame.metadata.clone(),
                baseline_exposure: frame.baseline_exposure,
            });
        }
        let h = g.p.get(pos).copied().unwrap_or(Homography::IDENTITY);
        let ig = g.inv_gain.get(pos).copied().unwrap_or([1.0; 3]);
        match (&self.device, self.kernels.as_mut()) {
            (Device::Kernels, Some(k)) => {
                if let Err(e) = k.add(pos, img, &h, g.rect, ig, clip, self.opts.bicubic) {
                    self.device = Device::Failed(format!("{}: {e}", k.name()));
                }
                return Ok(());
            }
            // the pass is repeated on the CPU
            (Device::Failed(_), _) => return Ok(()),
            _ => {}
        }
        let warped = measure::warp(img, &h, g.rect, ig, clip, self.opts.bicubic, &mut self.frame_buf);
        let r = self.opts.radius_px();
        match self.fuser.as_mut() {
            Some(Fuser::Average(a)) => {
                let fm = measure::focus_measure(&warped, r, &mut self.measure_buf);
                a.add(pos, &warped, fm);
            }
            Some(Fuser::Depth(d)) => {
                let fm = measure::focus_measure(&warped, r, &mut self.measure_buf);
                let noise = measure::noise_floor(fm);
                d.add(pos, &warped, fm, noise);
            }
            Some(Fuser::Pyramid(p)) => p.add(pos, &warped),
            None => {}
        }
        Ok(())
    }

    fn gather_one(&mut self, k: usize, img: &Rgb32f, clip: f32) -> Result<()> {
        let pos = self.gather.get(k).copied().ok_or_else(|| MergeError::Mismatch("focus stack: unexpected photo".into()))?;
        let (Some(g), Some(r)) = (&self.geometry, self.resolved.as_mut()) else { return Ok(()) };
        // the frame at the size asked for (half), or any size up to the stack's of the same shape
        let (sx, sy) = (img.width as f64 / g.size.0.max(1) as f64, img.height as f64 / g.size.1.max(1) as f64);
        if img.width == 0 || img.height == 0 || sx > 1.0 || sy > 1.0 || (sx / sy - 1.0).abs() > 0.02 {
            return Err(MergeError::Mismatch("focus stacking needs photos of the same size".into()));
        }
        let scale = Homography([sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0]);
        let h = scale.mul(&g.p.get(pos).copied().unwrap_or(Homography::IDENTITY));
        let ig = g.inv_gain.get(pos).copied().unwrap_or([1.0; 3]);
        let warped = measure::warp(img, &h, g.rect, ig, clip, self.opts.bicubic, &mut self.frame_buf);
        r.add(pos, &warped);
        Ok(())
    }

    /// Finish the current pass (after all its frames were pushed).
    pub fn end_pass(&mut self, progress: &Progress) -> Result<()> {
        let Some(a) = self.active.take() else { return Err(MergeError::Mismatch("focus stack: no pass is running".into())) };
        if a.pushed != a.frames.len() {
            return Err(MergeError::Mismatch(format!("focus stack: {} of {} photos were handed to the {:?} pass", a.pushed, a.frames.len(), a.pass)));
        }
        match a.pass {
            Pass::Order => {
                let grids: Vec<&[f32]> = self.order_grids.iter().map(|(_, g)| g.as_slice()).collect();
                let seq: Vec<usize> = match order::spectral(&grids) {
                    Some(o) => o.into_iter().filter_map(|p| self.order_grids.get(p).map(|g| g.0)).collect(),
                    None => {
                        self.warnings.push("the focus order couldn't be worked out from the photos; using the capture order".into());
                        (0..self.n).collect()
                    }
                };
                self.order_grids.clear();
                self.seq = thin(seq, self.opts.every);
                if self.opts.align.enabled || self.opts.align.brightness {
                    self.stage = Stage::Align;
                } else {
                    self.fuse_seq = (0..self.seq.len()).collect();
                    self.stage = Stage::Fuse;
                }
            }
            Pass::Align => {
                let al = self.aligner.take().ok_or_else(|| MergeError::Mismatch("focus stack: alignment not started".into()))?;
                let ch = align::chain(&al, self.opts.align.reference, self.opts.align.crop == CropMode::Common)?;
                self.warnings.extend(ch.warnings.iter().cloned());
                self.fuse_seq = (0..self.seq.len()).filter(|&p| ch.used(p)).collect();
                self.chain = Some(ch);
                if self.fuse_seq.len() < 2 {
                    return Err(MergeError::TooFew(self.fuse_seq.len(), 2));
                }
                self.stage = Stage::Fuse;
            }
            Pass::Fuse => {
                let last = self.fuse_seq.len().saturating_sub(1);
                // the state another device kept, or the pass again on the CPU when it failed
                let device = std::mem::replace(&mut self.device, Device::Cpu);
                let failed = match (device, self.kernels.as_mut()) {
                    (Device::Kernels, Some(k)) => match k.finish(last + 1) {
                        Ok(state) => {
                            let Some(g) = &self.geometry else { return Err(MergeError::TooFew(0, 2)) };
                            let (w, h) = (g.rect.w, g.rect.h);
                            let fuser = match state {
                                kernels::KernelState::Average { sum, wsum, dsum } => {
                                    average::Average::from_state(w, h, 2.0 * self.opts.smoothing_px(), sum, wsum, dsum).map(Fuser::Average)
                                }
                                kernels::KernelState::Depth(s) => depth::Depth::from_state(w, h, s, last + 1).map(Fuser::Depth),
                            };
                            match fuser {
                                Some(f) => {
                                    self.fuser = Some(f);
                                    None
                                }
                                None => Some(format!("{}: the state read back has the wrong size", k.name())),
                            }
                        }
                        Err(e) => Some(format!("{}: {e}", k.name())),
                    },
                    (Device::Failed(why), _) => Some(why),
                    _ => None,
                };
                if let Some(why) = failed {
                    self.kernels = None;
                    self.device_note = format!("CPU (after {why})");
                    self.stage = Stage::Fuse;
                    return Ok(());
                }
                check(progress, self.spans.fuse.1, "Building depth map")?;
                match self.fuser.take() {
                    Some(Fuser::Average(av)) => {
                        let (img, d) = av.finish(last);
                        self.outcome = Some((img, d, None));
                        self.stage = Stage::Done;
                    }
                    Some(Fuser::Pyramid(p)) => {
                        let (img, d) = p.finish(last);
                        self.outcome = Some((img, d, None));
                        self.stage = Stage::Done;
                    }
                    Some(Fuser::Depth(d)) => {
                        let r = d.finish(depth::DepthParams {
                            radius: self.opts.radius_px(),
                            smoothing: self.opts.smoothing_px(),
                            interpolate: self.opts.interpolate,
                        });
                        let needed = r.needed();
                        self.gathered = r.gathered();
                        if needed.is_empty() {
                            let (img, d, c) = r.done();
                            self.outcome = Some((img, d, Some(c)));
                            self.stage = Stage::Done;
                        } else {
                            self.gather = needed;
                            self.resolved = Some(r);
                            self.stage = Stage::Gather;
                        }
                    }
                    None => return Err(MergeError::TooFew(0, 2)),
                }
            }
            Pass::Gather => {
                let r = self.resolved.take().ok_or_else(|| MergeError::Mismatch("focus stack: nothing to gather".into()))?;
                let (img, d, c) = r.done();
                self.outcome = Some((img, d, Some(c)));
                self.stage = Stage::Done;
            }
        }
        Ok(())
    }

    /// The result (after the last pass).
    pub fn finish(self, progress: &Progress) -> Result<FocusResult> {
        if self.stage != Stage::Done || self.active.is_some() {
            return Err(MergeError::Mismatch("focus stack: not all passes have run".into()));
        }
        let (image, depth, confidence) = self.outcome.ok_or_else(|| MergeError::Mismatch("focus stack: nothing stacked".into()))?;
        let geometry = self.geometry.ok_or_else(|| MergeError::Mismatch("focus stack: nothing stacked".into()))?;
        let meta = self.reference_meta.ok_or_else(|| MergeError::Mismatch("focus stack: the reference photo wasn't stacked".into()))?;
        check(progress, 0.98, "Stacking")?;
        let used: Vec<usize> = self.fuse_seq.iter().filter_map(|&p| self.seq.get(p).copied()).collect();
        let skipped: Vec<usize> = (0..self.seq.len()).filter(|p| !self.fuse_seq.contains(p)).filter_map(|p| self.seq.get(p).copied()).collect();
        let alignments = self
            .fuse_seq
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                let pair = self.chain.as_ref().and_then(|c| c.pairs.get(p));
                FocusAlignment {
                    index: self.seq.get(p).copied().unwrap_or(0),
                    h: geometry.p.get(k).copied().unwrap_or(Homography::IDENTITY),
                    gain: geometry.inv_gain.get(k).map_or([1.0; 3], |g| g.map(|v| 1.0 / v)),
                    ncc: pair.map_or(1.0, |q| q.ncc),
                    seeded: pair.is_some_and(|q| q.seeded),
                }
            })
            .collect();
        let reference = match (&self.fixed, &self.chain) {
            (Some(f), _) => f.reference,
            (None, Some(c)) => self.seq.get(c.reference).copied().unwrap_or(0),
            (None, None) => self.seq.get(self.seq.len() / 2).copied().unwrap_or(0),
        };
        Ok(FocusResult {
            image,
            depth,
            confidence,
            method: self.opts.method,
            order: self.seq,
            used,
            skipped,
            reference,
            alignments,
            crop: geometry.rect,
            frame_size: geometry.size,
            warnings: self.warnings,
            device: self.device_note,
            gathered: self.gathered,
            color: meta.color,
            raw: meta.raw,
            orientation: Orientation::Normal,
            metadata: meta.metadata,
            baseline_exposure: meta.baseline_exposure,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in device running the CPU code behind [`kernels::FocusKernels`], optionally
    /// failing at one frame.
    struct Mock {
        fail_at: Option<usize>,
        avg: Option<average::Average>,
        dep: Option<depth::Depth>,
        radius: usize,
        fb: measure::FrameBuf,
        mb: measure::MeasureBuf,
    }

    impl kernels::FocusKernels for Mock {
        fn name(&self) -> String {
            "mock".into()
        }
        fn begin(&mut self, w: usize, h: usize, method: FocusMethod, radius: usize, sigma: f32) -> std::result::Result<(), String> {
            self.radius = radius;
            match method {
                FocusMethod::Average => self.avg = Some(average::Average::new(w, h, sigma)),
                FocusMethod::Depth => self.dep = Some(depth::Depth::new(w, h)),
                FocusMethod::Pyramid => return Err("not here".into()),
            }
            Ok(())
        }
        fn add(
            &mut self,
            pos: usize,
            src: &Rgb32f,
            h: &Homography,
            out: Rect,
            g: [f32; 3],
            clip: f32,
            bicubic: bool,
        ) -> std::result::Result<(), String> {
            if self.fail_at == Some(pos) {
                return Err("device lost".into());
            }
            let w = measure::warp(src, h, out, g, clip, bicubic, &mut self.fb);
            let fm = measure::focus_measure(&w, self.radius, &mut self.mb);
            if let Some(a) = self.avg.as_mut() {
                a.add(pos, &w, fm);
            }
            if let Some(d) = self.dep.as_mut() {
                d.add(pos, &w, fm, measure::noise_floor(fm));
            }
            Ok(())
        }
        fn finish(&mut self, _frames: usize) -> std::result::Result<kernels::KernelState, String> {
            match (self.avg.take(), self.dep.take()) {
                (Some(a), _) => Ok(a.into_state()),
                (_, Some(d)) => Ok(kernels::KernelState::Depth(d.into_state())),
                _ => Err("nothing".into()),
            }
        }
    }

    #[test]
    fn another_device_gives_the_cpu_result_and_a_failing_one_falls_back() {
        let b = crate::synth::focus_bracket(160, 110, &crate::synth::FocusSynth { n: 5, ..Default::default() }).unwrap();
        let frames: Vec<Frame> = b.frames.iter().map(|f| Frame::from_linear_rec2020(f.clone(), None)).collect();
        for method in [FocusMethod::Average, FocusMethod::Depth] {
            let opts = FocusOptions { method, ..Default::default() };
            let cpu = stack_focus(&frames, &opts, &crate::no_progress).unwrap();
            assert_eq!(cpu.device, "CPU");
            for fail_at in [None, Some(2)] {
                let mut st = FocusStacker::new(frames.len(), opts.clone()).unwrap();
                st.set_kernels(Box::new(Mock { fail_at, avg: None, dep: None, radius: 0, fb: Default::default(), mb: Default::default() }));
                let mut fuse_passes = 0;
                while let Some(req) = st.next_pass().unwrap() {
                    fuse_passes += usize::from(req.pass == Pass::Fuse);
                    for &i in &req.frames {
                        st.push(i, &frames[i], &crate::no_progress).unwrap();
                    }
                    st.end_pass(&crate::no_progress).unwrap();
                }
                let r = st.finish(&crate::no_progress).unwrap();
                assert_eq!(r.image.data, cpu.image.data, "{method:?} {fail_at:?}");
                assert_eq!(fuse_passes, if fail_at.is_some() { 2 } else { 1 });
                if fail_at.is_some() {
                    assert!(r.device.starts_with("CPU (after mock: device lost"), "{}", r.device);
                } else {
                    assert_eq!(r.device, "mock");
                }
            }
        }
    }

    #[test]
    fn a_fixed_geometry_restacks_like_the_original() {
        let b = crate::synth::focus_bracket(200, 140, &crate::synth::FocusSynth { n: 5, ..Default::default() }).unwrap();
        let frames: Vec<Frame> = b.frames.iter().map(|f| Frame::from_linear_rec2020(f.clone(), None)).collect();
        let opts = FocusOptions { method: FocusMethod::Average, ..Default::default() };
        let r = stack_focus(&frames, &opts, &crate::no_progress).unwrap();
        let fixed = FixedGeometry {
            frame_size: r.frame_size,
            maps: r.alignments.iter().map(|a| a.h).collect(),
            gains: r.alignments.iter().map(|a| a.gain).collect(),
            crop: r.crop,
            reference: r.used.iter().position(|&i| i == r.reference).unwrap(),
        };
        let mut st = FocusStacker::with_geometry(opts.clone(), fixed.clone()).unwrap();
        let req = st.next_pass().unwrap().unwrap();
        assert_eq!(req.pass, Pass::Fuse);
        for &i in &req.frames {
            st.push(i, &frames[i], &crate::no_progress).unwrap();
        }
        st.end_pass(&crate::no_progress).unwrap();
        assert!(st.next_pass().unwrap().is_none());
        let again = st.finish(&crate::no_progress).unwrap();
        assert_eq!(again.crop, r.crop);
        let err = again.image.data.iter().zip(&r.image.data).map(|(a, b)| (a[1] - b[1]).abs()).fold(0.0, f32::max);
        assert!(err < 1e-5, "{err}");
        // at half size: the maps and crop scale
        let half: Vec<Frame> = frames
            .iter()
            .map(|f| Frame { image: lightcraft_raster::resample::fit(&f.image, 100, 100, lightcraft_raster::resample::Filter::Box), ..f.clone() })
            .collect();
        let small = stack_focus_fixed(&half, &opts, fixed);
        assert_eq!((small.crop.w as f64 / r.crop.w as f64 * 10.0).round(), 5.0);
    }

    fn stack_focus_fixed(frames: &[Frame], opts: &FocusOptions, fixed: FixedGeometry) -> FocusResult {
        let mut st = FocusStacker::with_geometry(opts.clone(), fixed).unwrap();
        while let Some(req) = st.next_pass().unwrap() {
            for &i in &req.frames {
                st.push(i, &frames[i], &crate::no_progress).unwrap();
            }
            st.end_pass(&crate::no_progress).unwrap();
        }
        st.finish(&crate::no_progress).unwrap()
    }

    #[test]
    fn thinning_keeps_both_ends() {
        assert_eq!(thin((0..10).collect(), 3), vec![0, 3, 6, 9]);
        assert_eq!(thin((0..11).collect(), 3), vec![0, 3, 6, 9, 10]);
        assert_eq!(thin((0..5).collect(), 1), vec![0, 1, 2, 3, 4]);
        assert_eq!(thin(vec![4, 7], 5), vec![4, 7]);
    }

    #[test]
    fn options_parse_letters_and_scale() {
        let o: FocusOptions = serde_json::from_value(serde_json::json!({"method": "c", "radius": 10, "order": "auto"})).unwrap();
        assert_eq!(o.method, FocusMethod::Pyramid);
        assert_eq!(o.order, FocusOrder::Auto);
        assert!(o.align.enabled);
        let s = o.scaled(0.25);
        assert_eq!((s.radius, s.smoothing), (2.5, 1.0));
        assert!(FocusStacker::new(1, FocusOptions::default()).is_err());
    }

    #[test]
    fn the_protocol_is_enforced() {
        let mut st = FocusStacker::new(3, FocusOptions::default()).unwrap();
        let req = st.next_pass().unwrap().unwrap();
        assert_eq!(req.pass, Pass::Align);
        assert_eq!(req.frames, vec![0, 1, 2]);
        assert_eq!(req.max_edge, Some(2048));
        assert!(st.next_pass().is_err());
        let f = Frame::from_linear_rec2020(Rgb32f::filled(64, 48, [0.2; 3]), None);
        assert!(st.push(2, &f, &crate::no_progress).is_err());
        st.push(0, &f, &crate::no_progress).unwrap();
        assert!(st.end_pass(&crate::no_progress).is_err());
    }
}
