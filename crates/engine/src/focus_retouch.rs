//! Focus-stack projects and retouching.
//!
//! A focus stack keeps what it was made from next to its DNG, `<dng>.lcfocus.json`
//! ([`FocusProject`]): the frames stacked (paths, in focus order), the options, and the geometry
//! the stacker found — per frame its map onto the result and its brightness gain, the crop. With
//! it the stack can be redone without aligning again (another method, another frame) and
//! *retouched*: brush strokes that take pixels from another source — a single frame, or method
//! A, B or C's result — where the stack went wrong (a halo round a stem, a bristle the depth map
//! lost, a background better from one frame), as Zerene Stacker's and Helicon Focus's retouching
//! do.
//!
//! - [`Retouch`]: a preview session (≤ 1600 px): the sources at preview size, made on first use,
//!   and the strokes composited and rendered with the photo's settings. `Send`: the app runs it
//!   on a worker thread.
//! - [`RetouchJob`]: the same at full size; [`crate::Session::finish_retouch`] writes the DNG
//!   over the stack's own file (its un-retouched version kept as `<dng>.lcfocus-base`, so strokes
//!   can be changed or removed later), saves the strokes in the project and reloads the photo.
//!
//! Strokes are in normalised coordinates of the result (x, y in 0..1), sizes a fraction of its
//! long edge, so previews and the full-size render agree.

use std::collections::HashMap;

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_develop::DevelopSettings;
use lightcraft_geom::Homography;
use lightcraft_merge::focus::{FixedGeometry, FocusMethod, FocusOptions, FocusResult, FocusStacker, Rect, measure};
use lightcraft_merge::{DngSamples, Frame};
use lightcraft_pipeline::{Quality, RenderRequest};
use lightcraft_raster::resample::{Filter, resize};
use lightcraft_raster::{Rgb32f, Rgba8};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::merge::ByteReader;
use crate::{EngineError, Result, Session};

/// Version of the project file this build writes.
pub const PROJECT_VERSION: u32 = 1;

/// Limits on strokes (they come from users and agents): strokes per retouch, points per stroke,
/// and the brush radius (a fraction of the long edge).
pub const MAX_STROKES: usize = 10_000;
pub const MAX_POINTS: usize = 100_000;
pub const MAX_SIZE: f32 = 0.5;

/// The project file of a focus stack's DNG.
pub fn project_path(dng: &str) -> String {
    format!("{dng}.lcfocus.json")
}

/// The un-retouched stack kept beside a retouched one (not an image extension: imports skip it).
pub fn base_path(dng: &str) -> String {
    format!("{dng}.lcfocus-base")
}

/// Where a retouch stroke takes its pixels from.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RetouchSource {
    /// Frame `k` of the project's inputs (0-based, focus order).
    Frame(usize),
    /// `"base"` (the stack as merged: erases earlier strokes), `"a"`, `"b"` or `"c"` (a method's
    /// result).
    Named(String),
}

impl RetouchSource {
    fn method(&self) -> Option<Option<FocusMethod>> {
        match self {
            RetouchSource::Named(n) => match n.to_ascii_lowercase().as_str() {
                "base" => Some(None),
                "a" | "average" => Some(Some(FocusMethod::Average)),
                "b" | "depth" => Some(Some(FocusMethod::Depth)),
                "c" | "pyramid" => Some(Some(FocusMethod::Pyramid)),
                _ => None,
            },
            RetouchSource::Frame(_) => None,
        }
    }
}

/// One brush stroke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetouchStroke {
    pub source: RetouchSource,
    /// Normalised (x, y) along the stroke.
    pub points: Vec<[f32; 2]>,
    /// Radius as a fraction of the long edge.
    pub size: f32,
    /// 0 (soft) … 1 (hard edge).
    #[serde(default = "half")]
    pub hardness: f32,
    #[serde(default = "one")]
    pub opacity: f32,
}

fn half() -> f32 {
    0.5
}

fn one() -> f32 {
    1.0
}

/// What a focus stack was made from (see the module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusProject {
    pub version: u32,
    /// The frames stacked, in focus order: their files and (when stacked) catalog ids.
    pub inputs: Vec<String>,
    #[serde(default)]
    pub ids: Vec<u64>,
    /// The options of the stack (`max_edge`: the size the frames were stacked at).
    pub options: FocusOptions,
    /// The reference input.
    pub reference: usize,
    /// Per input: reference → frame pixels (row-major 3 × 3) at `frame_size`.
    pub maps: Vec<[f64; 9]>,
    pub gains: Vec<[f32; 3]>,
    /// The result's area within the reference frame: x, y, w, h.
    pub crop: [usize; 4],
    pub frame_size: [usize; 2],
    #[serde(default)]
    pub strokes: Vec<RetouchStroke>,
}

impl FocusProject {
    /// The project of a stack just made from `sources` (all the inputs the stack was asked for).
    pub fn of(r: &FocusResult, sources: &[(PhotoId, String)], options: &FocusOptions) -> FocusProject {
        let inputs = r.alignments.iter().filter_map(|a| sources.get(a.index)).collect::<Vec<_>>();
        FocusProject {
            version: PROJECT_VERSION,
            inputs: inputs.iter().map(|s| s.1.clone()).collect(),
            ids: inputs.iter().map(|s| s.0.0).collect(),
            options: FocusOptions { every: 1, ..options.clone() },
            reference: r.alignments.iter().position(|a| a.index == r.reference).unwrap_or(0),
            maps: r.alignments.iter().map(|a| a.h.0).collect(),
            gains: r.alignments.iter().map(|a| a.gain).collect(),
            crop: [r.crop.x, r.crop.y, r.crop.w, r.crop.h],
            frame_size: [r.frame_size.0, r.frame_size.1],
            strokes: Vec::new(),
        }
    }

    fn fixed(&self) -> FixedGeometry {
        FixedGeometry {
            frame_size: (self.frame_size[0], self.frame_size[1]),
            maps: self.maps.iter().map(|m| Homography(*m)).collect(),
            gains: self.gains.clone(),
            crop: Rect { x: self.crop[0], y: self.crop[1], w: self.crop[2], h: self.crop[3] },
            reference: self.reference,
        }
    }

    /// Check the strokes against the project (sources that exist, sane numbers).
    pub fn check(&self, strokes: &[RetouchStroke]) -> std::result::Result<(), String> {
        if strokes.len() > MAX_STROKES {
            return Err(format!("{} strokes (at most {MAX_STROKES})", strokes.len()));
        }
        for (i, s) in strokes.iter().enumerate() {
            if s.points.len() > MAX_POINTS {
                return Err(format!("stroke {}: {} points (at most {MAX_POINTS})", i + 1, s.points.len()));
            }
            match &s.source {
                RetouchSource::Frame(k) if *k >= self.inputs.len() => {
                    return Err(format!("stroke {}: frame {k} doesn't exist (the stack has {} frames)", i + 1, self.inputs.len()));
                }
                RetouchSource::Named(n) if s.source.method().is_none() => {
                    return Err(format!("stroke {}: unknown source `{n}` (base, a, b, c or a frame number)", i + 1));
                }
                _ => {}
            }
            if !(s.size.is_finite() && s.size > 0.0 && s.size <= MAX_SIZE) || s.points.iter().flatten().any(|v| !v.is_finite() || v.abs() > 4.0) {
                return Err(format!("stroke {}: size or points out of range", i + 1));
            }
        }
        Ok(())
    }
}

/// Read a DNG's project, `None` when it has none (not a focus stack, or made before projects).
pub fn read_project(dng: &str) -> Option<FocusProject> {
    let text = std::fs::read_to_string(project_path(dng)).ok()?;
    serde_json::from_str::<FocusProject>(&text)
        .ok()
        .filter(|p| p.version <= PROJECT_VERSION && p.maps.len() == p.inputs.len() && p.gains.len() == p.inputs.len())
}

/// The sources' scale against the stored DNG: sources × this = DNG units.
fn dng_scale(src_baseline: f64, dng_baseline: f64) -> f32 {
    let k = 2f64.powf(src_baseline - dng_baseline) as f32;
    if k.is_finite() && k > 0.0 { k } else { 1.0 }
}

/// Paint `strokes` from their sources over `img` (`source` returns a source at `img`'s size).
fn composite(
    img: &mut Rgb32f,
    strokes: &[RetouchStroke],
    mut source: impl FnMut(&RetouchSource) -> std::result::Result<std::sync::Arc<Rgb32f>, String>,
) -> std::result::Result<(), String> {
    let (w, h) = (img.width, img.height);
    let long = w.max(h) as f32;
    for s in strokes {
        let src = source(&s.source)?;
        if (src.width, src.height) != (w, h) {
            return Err("a retouch source has the wrong size".into());
        }
        let r = (s.size * long).max(0.5);
        let inner = r * s.hardness.clamp(0.0, 1.0);
        let opacity = s.opacity.clamp(0.0, 1.0);
        // stamps along the stroke, a quarter radius apart
        let mut stamps: Vec<(f32, f32)> = Vec::new();
        let pts: Vec<(f32, f32)> = s.points.iter().map(|p| (p[0] * w as f32, p[1] * h as f32)).collect();
        for (i, &(x, y)) in pts.iter().enumerate() {
            if let Some(&(px, py)) = i.checked_sub(1).and_then(|j| pts.get(j)) {
                let d = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
                let n = (d / (r / 4.0).max(0.5)).ceil().min(10_000.0) as usize;
                for k in 1..n {
                    let t = k as f32 / n as f32;
                    stamps.push((px + (x - px) * t, py + (y - py) * t));
                }
            }
            stamps.push((x, y));
        }
        let Some(bb) = stamps.iter().fold(None::<(f32, f32, f32, f32)>, |b, &(x, y)| {
            Some(match b {
                None => (x - r, y - r, x + r, y + r),
                Some((a, c, e, g)) => (a.min(x - r), c.min(y - r), e.max(x + r), g.max(y + r)),
            })
        }) else {
            continue;
        };
        let (x0, y0) = (bb.0.floor().max(0.0) as usize, bb.1.floor().max(0.0) as usize);
        let (x1, y1) = ((bb.2.ceil().max(0.0) as usize).min(w), (bb.3.ceil().max(0.0) as usize).min(h));
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let bw = x1 - x0;
        let mut alpha = vec![0f32; bw * (y1 - y0)];
        for &(sx, sy) in &stamps {
            let (ax, ay) = (((sx - r).floor().max(0.0) as usize).max(x0), ((sy - r).floor().max(0.0) as usize).max(y0));
            let (bx, by) = (((sx + r).ceil().max(0.0) as usize).min(x1), ((sy + r).ceil().max(0.0) as usize).min(y1));
            for y in ay..by {
                for x in ax..bx {
                    let d = ((x as f32 + 0.5 - sx).powi(2) + (y as f32 + 0.5 - sy).powi(2)).sqrt();
                    if d >= r {
                        continue;
                    }
                    let a = if d <= inner || r <= inner { 1.0 } else { 1.0 - smoothstep(inner, r, d) } * opacity;
                    if let Some(v) = alpha.get_mut((y - y0) * bw + (x - x0)) {
                        *v = v.max(a);
                    }
                }
            }
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let a = alpha.get((y - y0) * bw + (x - x0)).copied().unwrap_or(0.0);
                if a <= 0.0 {
                    continue;
                }
                let i = y * w + x;
                let (Some(o), Some(q)) = (img.data.get(i).copied(), src.data.get(i)) else { continue };
                img.data[i] = [o[0] + (q[0] - o[0]) * a, o[1] + (q[1] - o[1]) * a, o[2] + (q[2] - o[2]) * a];
            }
        }
    }
    Ok(())
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The sources of a project at one size: frames and method results made on first use.
struct Sources {
    project: FocusProject,
    read: ByteReader,
    /// The size frames are decoded at (`None` = as stacked).
    frame_edge: Option<usize>,
    /// The result's size.
    size: (usize, usize),
    /// Sources × this = DNG units.
    dng_baseline: f64,
    base: std::sync::Arc<Rgb32f>,
    made: HashMap<RetouchSource, std::sync::Arc<Rgb32f>>,
}

impl Sources {
    fn frame(&self, k: usize) -> std::result::Result<Frame, String> {
        let path = self.project.inputs.get(k).ok_or("no such frame")?;
        let bytes = (self.read)(path)?;
        lightcraft_merge::load_frame(&bytes, self.frame_edge, true).map_err(|e| format!("{path}: {e}"))
    }

    fn get(&mut self, s: &RetouchSource) -> std::result::Result<std::sync::Arc<Rgb32f>, String> {
        if let Some(img) = self.made.get(s) {
            return Ok(img.clone());
        }
        let (img, baseline) = match (s, s.method()) {
            (_, Some(None)) => return Ok(self.base.clone()),
            (RetouchSource::Frame(k), _) => {
                let f = self.frame(*k)?;
                let fixed = self.project.fixed();
                let (sx, sy) = (f.image.width as f64 / fixed.frame_size.0.max(1) as f64, f.image.height as f64 / fixed.frame_size.1.max(1) as f64);
                let m = fixed.maps.get(*k).ok_or("no such frame")?.0;
                let map = Homography([m[0], m[1] * sx / sy, m[2] * sx, m[3] * sy / sx, m[4], m[5] * sy, m[6] / sx, m[7] / sy, m[8]]);
                let c = fixed.crop;
                let rect = Rect {
                    x: (c.x as f64 * sx).round() as usize,
                    y: (c.y as f64 * sy).round() as usize,
                    w: ((c.w as f64 * sx).round() as usize).max(1),
                    h: ((c.h as f64 * sy).round() as usize).max(1),
                };
                let g = fixed.gains.get(*k).copied().unwrap_or([1.0; 3]).map(|v| if v > 0.0 { 1.0 / v } else { 1.0 });
                let mut buf = measure::FrameBuf::default();
                let warped = measure::warp(&f.image, &map, rect, g, f.clip, self.project.options.bicubic, &mut buf).rgb.clone();
                (warped, f.baseline_exposure)
            }
            (_, Some(Some(method))) => {
                let opts = FocusOptions { method, max_edge: self.frame_edge, ..self.project.options.clone() };
                let mut st = FocusStacker::with_geometry(opts, self.project.fixed()).map_err(|e| e.to_string())?;
                while let Some(req) = st.next_pass().map_err(|e| e.to_string())? {
                    for &i in &req.frames {
                        let f = self.frame(i)?;
                        st.push(i, &f, &lightcraft_merge::no_progress).map_err(|e| e.to_string())?;
                    }
                    st.end_pass(&lightcraft_merge::no_progress).map_err(|e| e.to_string())?;
                }
                let r = st.finish(&lightcraft_merge::no_progress).map_err(|e| e.to_string())?;
                (r.image, r.baseline_exposure)
            }
            _ => return Err("unknown retouch source".into()),
        };
        let k = dng_scale(baseline, self.dng_baseline);
        let mut img = if (img.width, img.height) == self.size { img } else { resize(&img, self.size.0, self.size.1, Filter::Bilinear) };
        img.data.iter_mut().for_each(|p| *p = p.map(|v| v * k));
        let img = std::sync::Arc::new(img);
        self.made.insert(s.clone(), img.clone());
        Ok(img)
    }
}

/// A retouch preview session (see the module docs). `Send`.
pub struct Retouch {
    src: Sources,
    base_frame: Frame,
    /// The stack DNG's private data (the source's camera look), for the previews' DNGs.
    private_data: Option<Vec<u8>>,
    settings: DevelopSettings,
    edge: usize,
}

impl Retouch {
    pub fn project(&self) -> &FocusProject {
        &self.src.project
    }

    /// The preview's size.
    pub fn size(&self) -> (usize, usize) {
        self.src.size
    }

    /// Make a source now (the app does this off the UI thread when one is picked).
    pub fn prepare(&mut self, s: &RetouchSource) -> std::result::Result<(), String> {
        self.src.get(s).map(|_| ())
    }

    /// The strokes painted over the stack, rendered with the photo's settings (uncropped).
    pub fn render(&mut self, strokes: &[RetouchStroke]) -> std::result::Result<Rgba8, String> {
        self.src.project.check(strokes)?;
        let mut img = (*self.src.base).clone();
        composite(&mut img, strokes, |s| self.src.get(s))?;
        let f = &self.base_frame;
        let extra = lightcraft_merge::DngExtra { masks: Vec::new(), private_data: self.private_data.clone(), unique_id: None };
        let dng = lightcraft_merge::write_linear_dng_with(
            &img,
            &f.color,
            lightcraft_geom::Orientation::Normal,
            &f.metadata,
            f.baseline_exposure,
            DngSamples::U16,
            extra,
        )
        .map_err(|e| e.to_string())?;
        let (src, info) = crate::files::load_bytes(&dng, self.edge)?;
        let req = RenderRequest { quality: Quality::Full, apply_crop: false, ..RenderRequest::fit(self.edge, self.edge) };
        Ok(lightcraft_pipeline::render(&src, &info, &self.settings, &req).image)
    }
}

/// A full-size retouch, detached from the session.
pub struct RetouchJob {
    pub id: PhotoId,
    pub path: String,
    pub strokes: Vec<RetouchStroke>,
    project: FocusProject,
    base_bytes: Vec<u8>,
    read: ByteReader,
}

impl RetouchJob {
    /// The retouched DNG. `progress(fraction, stage)` returns false to cancel.
    pub fn run(&self, progress: &(dyn Fn(f32, &str) -> bool + Sync)) -> std::result::Result<Vec<u8>, String> {
        self.project.check(&self.strokes)?;
        let base = lightcraft_merge::load_frame(&self.base_bytes, None, true).map_err(|e| e.to_string())?;
        let mut src = Sources {
            project: self.project.clone(),
            read: self.read.clone(),
            frame_edge: self.project.options.max_edge,
            size: (base.image.width, base.image.height),
            dng_baseline: base.baseline_exposure,
            base: std::sync::Arc::new(base.image.clone()),
            made: HashMap::new(),
        };
        // the sources first (the slow part), then the strokes
        let mut needed: Vec<RetouchSource> = Vec::new();
        for s in &self.strokes {
            if !needed.contains(&s.source) {
                needed.push(s.source.clone());
            }
        }
        for (i, s) in needed.iter().enumerate() {
            if !progress(0.05 + 0.8 * i as f32 / needed.len().max(1) as f32, "Preparing sources") {
                return Err("cancelled".into());
            }
            src.get(s)?;
        }
        if !progress(0.9, "Retouching") {
            return Err("cancelled".into());
        }
        let mut img = base.image.clone();
        composite(&mut img, &self.strokes, |s| src.get(s))?;
        // what the stack's DNG carries besides its image: the depth map, the source's camera look
        let extra = lightcraft_merge::DngExtra {
            masks: lightcraft_raw::semantic_masks(&self.base_bytes),
            private_data: crate::camera_preview::private_data_of(&self.base_bytes),
            unique_id: None,
        };
        let dng = lightcraft_merge::write_linear_dng_with(
            &img,
            &base.color,
            lightcraft_geom::Orientation::Normal,
            &base.metadata,
            base.baseline_exposure,
            DngSamples::U16,
            extra,
        )
        .map_err(|e| e.to_string())?;
        progress(1.0, "Done");
        Ok(dng)
    }
}

impl Session {
    fn project_of(&self, id: PhotoId) -> Result<(String, FocusProject)> {
        let p = self.catalog.photo(id).ok_or(lightcraft_catalog::CatalogError::NoPhoto(id))?;
        let Source::File { path } = &p.source else { return Err(EngineError::Other(format!("{} isn't a file", p.file_name))) };
        let project = read_project(path).ok_or_else(|| {
            EngineError::Other(format!("{} has no focus-stack project (only stacks made by LightCraft can be retouched)", p.file_name))
        })?;
        Ok((path.clone(), project))
    }

    fn reader(&self) -> ByteReader {
        match &self.media.file_bytes {
            Some(r) => r.clone(),
            None => std::sync::Arc::new(|path: &str| std::fs::read(path).map_err(|e| format!("{path}: {e}"))),
        }
    }

    /// The un-retouched stack of `path`.
    fn base_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let read = self.reader();
        read(&base_path(path)).or_else(|_| read(path)).map_err(EngineError::Other)
    }

    /// Does photo `id` have a focus-stack project (can it be retouched)?
    pub fn has_focus_project(&self, id: PhotoId) -> bool {
        self.catalog.photo(id).is_some_and(|p| matches!(&p.source, Source::File { path } if std::path::Path::new(&project_path(path)).exists()))
    }

    /// A preview session for retouching `id` at ≤ `edge` px, with the photo's develop settings.
    pub fn open_retouch(&self, id: PhotoId, edge: usize) -> Result<Retouch> {
        self.plan_open_retouch(id, edge)?.open().map_err(EngineError::Other)
    }

    /// [`Session::open_retouch`] in two steps: this one cheap (on the UI thread), then
    /// [`RetouchOpen::open`] (decoding, on a worker).
    pub fn plan_open_retouch(&self, id: PhotoId, edge: usize) -> Result<RetouchOpen> {
        let (path, project) = self.project_of(id)?;
        let settings = self.develop_of(id).map(|d| (*d).clone()).unwrap_or_default();
        Ok(RetouchOpen { path, project, settings, edge, read: self.reader() })
    }
}

/// A retouch session to open (see [`Session::plan_open_retouch`]). `Send`.
pub struct RetouchOpen {
    path: String,
    project: FocusProject,
    settings: DevelopSettings,
    edge: usize,
    read: ByteReader,
}

impl RetouchOpen {
    pub fn open(self) -> std::result::Result<Retouch, String> {
        let RetouchOpen { path, project, settings, edge, read } = self;
        let bytes = read(&base_path(&path)).or_else(|_| read(&path))?;
        let base = lightcraft_merge::load_frame(&bytes, Some(edge), true).map_err(|e| e.to_string())?;
        let (rw, rh) = (project.crop[2].max(1), project.crop[3].max(1));
        let s = base.image.width.max(base.image.height) as f64 / rw.max(rh) as f64;
        let full_edge = project.options.max_edge.unwrap_or(project.frame_size[0].max(project.frame_size[1]));
        let frame_edge = ((full_edge as f64 * s).round() as usize).max(16);
        Ok(Retouch {
            src: Sources {
                project,
                read,
                frame_edge: Some(frame_edge),
                size: (base.image.width, base.image.height),
                dng_baseline: base.baseline_exposure,
                base: std::sync::Arc::new(base.image.clone()),
                made: HashMap::new(),
            },
            base_frame: base,
            private_data: crate::camera_preview::private_data_of(&bytes),
            settings,
            edge,
        })
    }
}

impl Session {
    /// Plan the full-size retouch of `id` with `strokes` (they replace the project's).
    pub fn plan_retouch(&self, id: PhotoId, strokes: Vec<RetouchStroke>) -> Result<RetouchJob> {
        let (path, project) = self.project_of(id)?;
        project.check(&strokes).map_err(EngineError::Other)?;
        let base_bytes = self.base_bytes(&path)?;
        Ok(RetouchJob { id, path, strokes, project, base_bytes, read: self.reader() })
    }

    /// Write a finished retouch over the stack's DNG (keeping the un-retouched one beside it),
    /// save the strokes in its project and reload the photo.
    pub fn finish_retouch(&mut self, job: &RetouchJob, dng: Vec<u8>) -> Result<Value> {
        let base = base_path(&job.path);
        if !std::path::Path::new(&base).exists() {
            crate::export::write_file_durable(&base, &job.base_bytes).map_err(EngineError::Other)?;
        }
        // the stack's own file (its project proves it): replaced in one step
        crate::export::write_file_durable(&job.path, &dng).map_err(EngineError::Other)?;
        let project = FocusProject { strokes: job.strokes.clone(), ..job.project.clone() };
        let text = serde_json::to_string_pretty(&project).map_err(|e| EngineError::Other(e.to_string()))?;
        crate::export::write_file_durable(&project_path(&job.path), text.as_bytes()).map_err(EngineError::Other)?;
        self.execute("photo.reload", &json!({"ids": [job.id.0]}))?;
        Ok(json!({"id": job.id.0, "path": job.path, "strokes": job.strokes.len(), "bytes": dng.len()}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> FocusProject {
        FocusProject {
            version: PROJECT_VERSION,
            inputs: vec!["a.dng".into(), "b.dng".into()],
            ids: vec![],
            options: FocusOptions::default(),
            reference: 0,
            maps: vec![Homography::IDENTITY.0; 2],
            gains: vec![[1.0; 3]; 2],
            crop: [0, 0, 10, 10],
            frame_size: [10, 10],
            strokes: vec![],
        }
    }

    fn stroke(points: usize, size: f32) -> RetouchStroke {
        RetouchStroke { source: RetouchSource::Frame(1), points: vec![[0.5, 0.5]; points], size, hardness: 0.5, opacity: 1.0 }
    }

    /// Hostile strokes (too many, too many points, a huge brush, NaN) are refused before any work.
    #[test]
    fn hostile_strokes_are_refused() {
        let p = project();
        assert!(p.check(&[stroke(10, 0.1)]).is_ok());
        assert!(p.check(&[stroke(MAX_POINTS + 1, 0.1)]).is_err());
        assert!(p.check(&vec![stroke(1, 0.1); MAX_STROKES + 1]).is_err());
        assert!(p.check(&[stroke(1, 0.9)]).is_err());
        assert!(p.check(&[stroke(1, f32::NAN)]).is_err());
        let mut far = stroke(2, 0.1);
        far.points[1] = [1e30, 0.5];
        assert!(p.check(&[far]).is_err());
        assert!(p.check(&[RetouchStroke { source: RetouchSource::Frame(2), ..stroke(1, 0.1) }]).is_err());
    }

    /// A stroke paints its source over the image within its radius, and nothing outside.
    #[test]
    fn a_stroke_composites_within_its_radius() {
        let mut img = Rgb32f::filled(40, 20, [0.1; 3]);
        let src = std::sync::Arc::new(Rgb32f::filled(40, 20, [0.9; 3]));
        let s = RetouchStroke { source: RetouchSource::Frame(0), points: vec![[0.25, 0.5], [0.5, 0.5]], size: 0.1, hardness: 1.0, opacity: 1.0 };
        composite(&mut img, &[s], |_| Ok(src.clone())).unwrap();
        assert_eq!(img.get(15, 10), [0.9; 3]);
        assert_eq!(img.get(35, 10), [0.1; 3]);
        assert_eq!(img.get(15, 1), [0.1; 3]);
    }
}
