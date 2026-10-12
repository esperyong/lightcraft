//! Photo Merge (HDR, Panorama, HDR Panorama, Focus Stack) in the library.
//!
//! A merge is planned from the session ([`crate::Session::plan_merge`] → [`MergeJob`], which is
//! `Send` and self-contained), run anywhere (a worker thread in the app, inline for the CLI/MCP:
//! [`MergeJob::run`], with progress and cancellation), and finished on the session
//! ([`crate::Session::finish_merge`]): the DNG is written next to the first source photo
//! (`<name>-HDR.dng`, `-Pano.dng`, `-HDR-Pano.dng`, `-Focus.dng`), imported, selected, and given
//! Auto Settings / the auto-crop rectangle when asked. A *preview* run works on ≤ 1024 px frames
//! and returns a rendered image (with the deghost overlay or the depth map when asked) instead of
//! a file.
//!
//! A focus stack streams its frames through [`lightcraft_merge::FocusStacker`] pass by pass
//! (decoding the next few while the stacker works on the current ones), so memory stays flat
//! however many photos the bracket has.

use std::sync::Arc;

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_develop::DevelopSettings;
use lightcraft_merge::focus::{FocusOptions, FocusOrder, FocusResult, FocusStacker, Pass};
use lightcraft_merge::{Deghost, Frame, HdrOptions, PanoOptions, Projection};
use lightcraft_pipeline::{Quality, RenderRequest, SourceInfo};
use lightcraft_raster::{Rgb32f, Rgba8};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

/// Long edge of the frames of a preview merge.
pub const PREVIEW_EDGE: usize = 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MergeKind {
    Hdr {
        hdr: HdrOptions,
    },
    Panorama {
        pano: PanoOptions,
    },
    /// Merge each bracket (`bracket` consecutive photos; 0 = detect from EXIF), then stitch.
    HdrPanorama {
        hdr: HdrOptions,
        pano: PanoOptions,
        bracket: usize,
    },
    /// Focus stack. `focus.every` 0 = automatic (every photo for the merge, about 12 photos for a
    /// preview); `max_megapixels` > 0 limits the result's size.
    FocusStack {
        focus: FocusOptions,
        max_megapixels: f64,
    },
}

impl MergeKind {
    fn suffix(&self) -> &'static str {
        match self {
            MergeKind::Hdr { .. } => "HDR",
            MergeKind::Panorama { .. } => "Pano",
            MergeKind::HdrPanorama { .. } => "HDR-Pano",
            MergeKind::FocusStack { .. } => "Focus",
        }
    }
    fn min_photos(&self) -> usize {
        match self {
            MergeKind::HdrPanorama { .. } => 4,
            _ => 2,
        }
    }
}

/// Options that apply after the merge.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MergeFinish {
    /// Run Auto Settings on the result.
    pub auto_settings: bool,
    /// Stack the result on top of its sources (collapsed); the result's caption also lists them.
    pub stack: bool,
    /// Preview only: tint deghosted areas red.
    pub show_overlay: bool,
    /// Preview only (focus stack): show the depth map instead of the result.
    pub show_depth: bool,
    /// Focus stack: also write the depth map here (16-bit grey PNG; 0 = first photo of the focus
    /// order, white = last).
    pub depth_path: Option<String>,
}

/// Reads a source photo's bytes (by catalog path).
pub type ByteReader = Arc<dyn Fn(&str) -> std::result::Result<Vec<u8>, String> + Send + Sync>;

/// A planned merge, detached from the session.
#[derive(Clone)]
pub struct MergeJob {
    pub kind: MergeKind,
    pub finish: MergeFinish,
    pub sources: Vec<(PhotoId, String)>,
    /// `Some(edge)`: preview at that frame size; `None`: the full-resolution merge.
    pub preview: Option<usize>,
    pub read: ByteReader,
    /// The largest source size (catalog pixels, upright): previews scale pixel radii by it, focus
    /// stacks size their decode window by it.
    pub size: (usize, usize),
}

/// What a merge produced.
pub struct MergeOutput {
    /// The DNG (also for previews, at preview size).
    pub dng: Vec<u8>,
    /// Rendered preview (previews only).
    pub preview: Option<Rgba8>,
    /// Auto-crop rectangle (normalised x, y, w, h) for panoramas.
    pub crop: Option<[f64; 4]>,
    /// Statistics for the caller (exposures, alignment, projection, …).
    pub info: Value,
    /// Focus stacks: the depth map (0..1, same size as the result).
    pub depth: Option<lightcraft_raster::Plane>,
}

pub type ProgressFn<'a> = dyn Fn(f32, &str) -> bool + Sync + 'a;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

impl MergeJob {
    fn load(&self, orient: bool, progress: &ProgressFn, span: (f32, f32)) -> std::result::Result<Vec<Frame>, String> {
        use rayon::prelude::*;
        let done = std::sync::atomic::AtomicUsize::new(0);
        let n = self.sources.len();
        let frames: std::result::Result<Vec<Frame>, String> = self
            .sources
            .par_iter()
            .map(|(_, path)| {
                let bytes = (self.read)(path)?;
                let f = lightcraft_merge::load_frame(&bytes, self.preview, orient).map_err(|e| format!("{path}: {e}"));
                let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(span.0 + (span.1 - span.0) * k as f32 / n as f32, "Reading photos");
                f
            })
            .collect();
        frames
    }

    /// Run the merge. `progress(fraction, stage)` returns false to cancel.
    pub fn run(&self, progress: &ProgressFn) -> std::result::Result<MergeOutput, String> {
        if self.sources.len() < self.kind.min_photos() {
            return Err(format!("select at least {} photos", self.kind.min_photos()));
        }
        if !progress(0.0, "Reading photos") {
            return Err("cancelled".into());
        }
        let sub = |a: f32, b: f32| move |f: f32, s: &str| progress(a + (b - a) * f, s);
        let mut depth = None;
        let (image, color, orientation, metadata, baseline, samples, crop, mut info, ghost) = match &self.kind {
            MergeKind::FocusStack { focus, .. } => {
                let p = sub(0.0, 0.9);
                let (r, seconds) = self.run_focus(focus, &p)?;
                let mut info = focus_info(&r, &self.sources, focus);
                info["seconds"] = seconds;
                depth = Some(r.depth);
                (r.image, r.color, r.orientation, r.metadata, r.baseline_exposure, lightcraft_merge::DngSamples::U16, None, info, None)
            }
            MergeKind::Hdr { hdr } => {
                let frames = self.load(false, progress, (0.0, 0.3))?;
                let p = sub(0.3, 0.9);
                let r = lightcraft_merge::merge_hdr(frames, hdr, &p).map_err(err)?;
                let info = json!({
                    "reference": self.sources[r.reference].0.0,
                    "ev": r.ev, "evExif": r.ev_exif,
                    "alignment": r.alignments.iter().map(|a| json!({"model": a.model, "inliers": a.inliers, "rmsPx": a.rms})).collect::<Vec<_>>(),
                });
                let ghost = (!r.ghost.data.is_empty()).then(|| r.ghost.oriented(r.orientation));
                (r.radiance, r.color, r.orientation, r.metadata, r.baseline_exposure, lightcraft_merge::DngSamples::Half, None, info, ghost)
            }
            MergeKind::Panorama { pano } => {
                let frames = self.load(true, progress, (0.0, 0.25))?;
                let p = sub(0.25, 0.9);
                let r = lightcraft_merge::stitch(frames, pano, &p).map_err(err)?;
                let info = pano_info(&r, &self.sources);
                (
                    r.image,
                    r.color,
                    lightcraft_geom::Orientation::Normal,
                    r.metadata,
                    r.baseline_exposure,
                    lightcraft_merge::DngSamples::U16,
                    r.crop,
                    info,
                    None,
                )
            }
            MergeKind::HdrPanorama { hdr, pano, bracket } => {
                let frames = self.load(false, progress, (0.0, 0.2))?;
                let b =
                    if *bracket >= 2 { *bracket } else { detect_bracket(&frames).ok_or("could not detect the bracket size from EXIF; set it")? };
                if !frames.len().is_multiple_of(b) || frames.len() / b < 2 {
                    return Err(format!("{} photos don't form brackets of {b} for at least two positions", frames.len()));
                }
                let groups = frames.len() / b;
                let mut merged = Vec::new();
                let mut it = frames.into_iter();
                for g in 0..groups {
                    let set: Vec<Frame> = it.by_ref().take(b).collect();
                    let p = sub(0.2 + 0.4 * g as f32 / groups as f32, 0.2 + 0.4 * (g + 1) as f32 / groups as f32);
                    let r = lightcraft_merge::merge_hdr(set, hdr, &p).map_err(err)?;
                    let mut f = Frame {
                        image: r.radiance,
                        clip: f32::INFINITY,
                        exposure: None,
                        raw: r.raw,
                        color: r.color,
                        orientation: r.orientation,
                        metadata: r.metadata,
                        baseline_exposure: r.baseline_exposure,
                    };
                    f.orient();
                    merged.push(f);
                }
                let p = sub(0.6, 0.9);
                let r = lightcraft_merge::stitch(merged, pano, &p).map_err(err)?;
                let mut info = pano_info(&r, &self.sources);
                info["bracket"] = json!(b);
                (
                    r.image,
                    r.color,
                    lightcraft_geom::Orientation::Normal,
                    r.metadata,
                    r.baseline_exposure,
                    lightcraft_merge::DngSamples::Half,
                    r.crop,
                    info,
                    None,
                )
            }
        };
        if !progress(0.9, "Writing DNG") {
            return Err("cancelled".into());
        }
        let mut metadata = metadata;
        if self.finish.stack {
            let names: Vec<&str> =
                self.sources.iter().map(|(_, p)| std::path::Path::new(p).file_name().and_then(|n| n.to_str()).unwrap_or(p)).collect();
            metadata.caption = Some(merged_caption(&names));
        }
        metadata.software = Some("LightCraft Photo Merge".into());
        let dng = lightcraft_merge::write_linear_dng(&image, &color, orientation, &metadata, baseline, samples).map_err(err)?;
        info["width"] = json!(image.width);
        info["height"] = json!(image.height);
        let preview = match (self.preview, depth.as_ref().filter(|_| self.finish.show_depth)) {
            (Some(_), Some(d)) => Some(depth_preview(d)),
            (Some(edge), None) => {
                Some(render_preview(&dng, edge, self.finish.auto_settings, crop, ghost.as_ref().filter(|_| self.finish.show_overlay))?)
            }
            (None, _) => None,
        };
        progress(1.0, "Done");
        Ok(MergeOutput { dng, preview, crop, info, depth })
    }
}

impl MergeJob {
    /// Frames decoded at once (and held while the stacker works on the previous ones): as many as
    /// the memory room for user-started work allows, 1–4.
    fn focus_window(&self, max_edge: Option<usize>) -> usize {
        let (w, h) = (self.size.0.max(1), self.size.1.max(1));
        let full = w * h;
        // the mosaic and file of a raw, plus the decoded frame and the decoder's work (demosaicing
        // at full size, binning below it)
        let px = match max_edge {
            Some(m) => {
                let s = (m as f64 / w.max(h) as f64).min(1.0);
                (full as f64 * s * s) as usize
            }
            None => full,
        };
        let cost = full * 4 + px * if max_edge.is_none() { 48 } else { 24 };
        (crate::memory::export_room() / (2 * cost.max(1))).clamp(1, 4)
    }

    fn decode_focus(&self, frames: &[usize], max_edge: Option<usize>) -> Vec<std::result::Result<Frame, String>> {
        use rayon::prelude::*;
        frames
            .par_iter()
            .map(|&i| {
                let (_, path) = self.sources.get(i).ok_or_else(|| format!("no photo {i}"))?;
                let bytes = (self.read)(path)?;
                lightcraft_merge::load_frame(&bytes, max_edge, true).map_err(|e| format!("{path}: {e}"))
            })
            .collect()
    }

    /// Stream the sources through a focus stacker. Returns the result and seconds per pass.
    fn run_focus(&self, opts: &FocusOptions, progress: &ProgressFn) -> std::result::Result<(FocusResult, Value), String> {
        let mut st = FocusStacker::new(self.sources.len(), opts.clone()).map_err(err)?;
        let profile = lightcraft_pipeline::profiling();
        let mut seconds = serde_json::Map::new();
        while let Some(req) = st.next_pass().map_err(err)? {
            let t0 = web_time::Instant::now();
            let k = self.focus_window(req.max_edge);
            let chunks: Vec<&[usize]> = req.frames.chunks(k).collect();
            let mut pending = chunks.first().map(|c| self.decode_focus(c, req.max_edge));
            for (ci, chunk) in chunks.iter().enumerate() {
                let cur = pending.take().unwrap_or_default();
                let next = chunks.get(ci + 1);
                let stacker = &mut st;
                let (pushed, decoded) = rayon::join(
                    move || -> std::result::Result<(), String> {
                        for (&i, f) in chunk.iter().zip(cur) {
                            stacker.push(i, &f?, progress).map_err(err)?;
                        }
                        Ok(())
                    },
                    || next.map(|c| self.decode_focus(c, req.max_edge)),
                );
                pushed?;
                pending = decoded;
            }
            st.end_pass(progress).map_err(err)?;
            let name = match req.pass {
                Pass::Order => "order",
                Pass::Align => "align",
                Pass::Fuse => "stack",
                Pass::Gather => "gather",
            };
            let secs = t0.elapsed().as_secs_f64();
            if profile {
                eprintln!("focus stack: {name} pass, {} photos at {:?} px, {k} decoded at a time: {secs:.2} s", req.frames.len(), req.max_edge);
            }
            seconds.insert(name.into(), json!(secs));
        }
        let r = st.finish(progress).map_err(err)?;
        Ok((r, Value::Object(seconds)))
    }
}

/// "Merged from a, b, c" — abbreviated past eight names.
fn merged_caption(names: &[&str]) -> String {
    if names.len() <= 8 {
        return format!("Merged from {}", names.join(", "));
    }
    let (first, last) = (&names[..3], names.last().copied().unwrap_or(""));
    format!("Merged from {}, … {last} ({} photos)", first.join(", "), names.len())
}

fn focus_info(r: &FocusResult, sources: &[(PhotoId, String)], opts: &FocusOptions) -> Value {
    let id = |i: usize| sources.get(i).map(|s| s.0.0);
    let (mut shift, mut rot, mut smin, mut smax, mut ncc) = (0f64, 0f64, f64::MAX, f64::MIN, 1f32);
    for a in &r.alignments {
        let m = a.h.0;
        let (cx, cy) = (r.frame_size.0 as f64 / 2.0, r.frame_size.1 as f64 / 2.0);
        let (x, y) = a.h.apply_real(cx, cy);
        shift = shift.max(((x - cx).powi(2) + (y - cy).powi(2)).sqrt());
        rot = rot.max((m[3] - m[1]).atan2(m[0] + m[4]).to_degrees().abs());
        let sc = (m[0] * m[4] - m[1] * m[3]).abs().sqrt();
        smin = smin.min(sc);
        smax = smax.max(sc);
        ncc = ncc.min(a.ncc);
    }
    json!({
        "method": r.method,
        "radius": opts.radius,
        "smoothing": opts.smoothing,
        "order": r.order.iter().filter_map(|&i| id(i)).collect::<Vec<_>>(),
        "used": r.used.len(),
        "skipped": r.skipped.iter().filter_map(|&i| id(i)).collect::<Vec<_>>(),
        "reference": id(r.reference),
        "crop": [r.crop.x, r.crop.y, r.crop.w, r.crop.h],
        "frameSize": [r.frame_size.0, r.frame_size.1],
        "alignment": {
            "maxShiftPx": shift,
            "maxRotationDeg": rot,
            "scale": [if smin.is_finite() && smin < f64::MAX { smin } else { 1.0 }, if smax > f64::MIN { smax } else { 1.0 }],
            "minNcc": ncc,
            "seeded": r.alignments.iter().filter(|a| a.seeded).count(),
        },
        "warnings": r.warnings,
    })
}

/// The depth map as a grey preview (near = black, far = white).
fn depth_preview(d: &lightcraft_raster::Plane) -> Rgba8 {
    let mut img = Rgba8::new(d.width, d.height);
    for (o, v) in img.data.iter_mut().zip(&d.data) {
        let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        *o = [g, g, g, 255];
    }
    img
}

/// The depth map as a 16-bit grey PNG.
pub fn depth_png(d: &lightcraft_raster::Plane) -> std::result::Result<Vec<u8>, String> {
    let s: Vec<u16> = d.data.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
    let img = lightcraft_codecs::EncodeImage::new(d.width as u32, d.height as u32, 1, lightcraft_codecs::Samples::U16(&s));
    lightcraft_codecs::encode_png(&img, &lightcraft_codecs::EncodeMeta::default()).map_err(|e| e.to_string())
}

fn pano_info(r: &lightcraft_merge::PanoResult, sources: &[(PhotoId, String)]) -> Value {
    json!({
        "projection": r.projection,
        "used": r.used.iter().filter_map(|&i| sources.get(i).map(|s| s.0.0)).collect::<Vec<_>>(),
        "fov": [r.fov.0, r.fov.1],
        "rmsPx": r.rms,
        "gains": r.gains,
        "crop": r.crop,
    })
}

/// Bracket size from EXIF: the smallest period of the exposure sequence (≥ 2).
pub fn detect_bracket(frames: &[Frame]) -> Option<usize> {
    let ev: Vec<f64> = frames.iter().map(|f| f.exposure).collect::<Option<_>>()?;
    let n = ev.len();
    (2..=n / 2).find(|&b| n.is_multiple_of(b) && (0..n).all(|k| (ev[k] - ev[k % b]).abs() < 0.2) && (1..b).any(|k| (ev[k] - ev[0]).abs() > 0.5))
}

/// Render a merged DNG like the library will show it (default raw settings, optionally Auto
/// Settings and the auto crop), with the deghost overlay tinted red.
fn render_preview(
    dng: &[u8],
    edge: usize,
    auto: bool,
    crop: Option<[f64; 4]>,
    ghost: Option<&lightcraft_raster::Plane>,
) -> std::result::Result<Rgba8, String> {
    let (src, info) = crate::files::load_bytes(dng, edge)?;
    let mut s = DevelopSettings::for_raw(info.as_shot_temp, info.as_shot_tint);
    if auto {
        apply_auto(&mut s, &src, &info);
    }
    if let Some(c) = crop {
        s.crop.geometry.rect = lightcraft_geom::Rect::from_xywh(c[0], c[1], c[2], c[3]);
    }
    let req = RenderRequest { quality: Quality::Full, apply_crop: false, ..RenderRequest::fit(edge, edge) };
    let mut img = lightcraft_pipeline::render(&src, &info, &s, &req).image;
    if let Some(g) = ghost {
        let (w, h) = (img.width, img.height);
        for y in 0..h {
            for x in 0..w {
                let gx = ((x as f32 + 0.5) / w as f32 * g.width as f32) as usize;
                let gy = ((y as f32 + 0.5) / h as f32 * g.height as f32) as usize;
                let a = g.get(gx.min(g.width - 1), gy.min(g.height - 1)).clamp(0.0, 1.0) * 0.6;
                if a > 0.01 {
                    let p = &mut img.data[y * w + x];
                    p[0] = (p[0] as f32 * (1.0 - a) + 255.0 * a) as u8;
                    p[1] = (p[1] as f32 * (1.0 - a)) as u8;
                    p[2] = (p[2] as f32 * (1.0 - a)) as u8;
                }
            }
        }
    }
    if let Some(c) = crop {
        // draw the auto-crop frame
        let (w, h) = (img.width as f64, img.height as f64);
        let (x0, y0, x1, y1) = ((c[0] * w) as usize, (c[1] * h) as usize, ((c[0] + c[2]) * w) as usize, ((c[1] + c[3]) * h) as usize);
        for y in y0..y1.min(img.height) {
            for x in x0..x1.min(img.width) {
                if y == y0 || y + 1 == y1 || x == x0 || x + 1 == x1 {
                    img.data[y * img.width + x] = [255, 255, 255, 255];
                }
            }
        }
    }
    Ok(img)
}

fn apply_auto(s: &mut DevelopSettings, src: &Rgb32f, info: &SourceInfo) {
    let a = lightcraft_pipeline::auto::auto_tone(src, info, s);
    s.light.exposure = a.exposure;
    s.light.contrast = a.contrast;
    s.light.highlights = a.highlights;
    s.light.shadows = a.shadows;
    s.light.whites = a.whites;
    s.light.blacks = a.blacks;
    s.color.vibrance = a.vibrance;
    s.color.saturation = a.saturation;
}

/// `dir/<stem>-<suffix>.dng`, then `-2`, `-3`… (the candidates for a merge's output).
fn output_names(first: &str, suffix: &str) -> impl Iterator<Item = std::path::PathBuf> {
    let p = std::path::Path::new(first);
    let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
    let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Merge".into());
    let suffix = suffix.to_string();
    (1u64..).map(move |k| dir.join(if k == 1 { format!("{stem}-{suffix}.dng") } else { format!("{stem}-{suffix}-{k}.dng") }))
}

/// Parse `merge.*` command parameters into a kind and finishing options.
pub fn parse(id: &str, p: &Value) -> Result<(MergeKind, MergeFinish)> {
    let bad = |m: String| EngineError::BadParams { cmd: id.into(), msg: m };
    let deghost: Deghost = serde_json::from_value(p.get("deghost").cloned().unwrap_or(json!("none"))).map_err(|e| bad(format!("deghost: {e}")))?;
    let hdr = HdrOptions { align: p.get("align").and_then(Value::as_bool).unwrap_or(true), deghost };
    let projection: Projection =
        serde_json::from_value(p.get("projection").cloned().unwrap_or(json!("auto"))).map_err(|e| bad(format!("projection: {e}")))?;
    let pano = PanoOptions {
        projection,
        boundary_warp: p.get("boundaryWarp").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0),
        auto_crop: p.get("autoCrop").and_then(Value::as_bool).unwrap_or(false),
        fill_edges: p.get("fillEdges").and_then(Value::as_bool).unwrap_or(false),
        max_megapixels: p.get("maxMegapixels").and_then(Value::as_f64).unwrap_or(40.0),
    };
    let kind = match id {
        "merge.hdr" => MergeKind::Hdr { hdr },
        "merge.panorama" => MergeKind::Panorama { pano },
        "merge.focusStack" => MergeKind::FocusStack {
            focus: parse_focus(p).map_err(bad)?,
            max_megapixels: p.get("maxMegapixels").and_then(Value::as_f64).unwrap_or(0.0).max(0.0),
        },
        _ => MergeKind::HdrPanorama { hdr, pano, bracket: p.get("bracket").and_then(Value::as_u64).unwrap_or(0) as usize },
    };
    let finish = MergeFinish {
        auto_settings: p.get("autoSettings").and_then(Value::as_bool).unwrap_or(true),
        stack: p.get("stack").and_then(Value::as_bool).unwrap_or(false),
        show_overlay: p.get("showOverlay").and_then(Value::as_bool).unwrap_or(false),
        show_depth: p.get("showDepth").and_then(Value::as_bool).unwrap_or(false),
        depth_path: p.get("depthPath").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
    };
    Ok((kind, finish))
}

/// `merge.focusStack` parameters → focus options (`every` 0 = automatic).
fn parse_focus(p: &Value) -> std::result::Result<FocusOptions, String> {
    fn field<T: serde::de::DeserializeOwned>(p: &Value, key: &str) -> std::result::Result<Option<T>, String> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => serde_json::from_value(v.clone()).map(Some).map_err(|e| format!("{key}: {e}")),
        }
    }
    let flag = |key: &str| p.get(key).and_then(Value::as_bool);
    let mut f = FocusOptions::default();
    if let Some(m) = field(p, "method")? {
        f.method = m;
    }
    if let Some(r) = p.get("radius").and_then(Value::as_f64) {
        f.radius = r.clamp(1.0, 30.0) as f32;
    }
    if let Some(s) = p.get("smoothing").and_then(Value::as_f64) {
        f.smoothing = s.clamp(0.0, 10.0) as f32;
    }
    if let Some(o) = field(p, "order")? {
        f.order = o;
    }
    f.every = p.get("every").and_then(Value::as_u64).map_or(0, |e| e.min(1000) as usize);
    let align = flag("align").unwrap_or(true);
    f.align.enabled = align;
    f.align.shift = flag("alignShift").unwrap_or(align);
    f.align.rotation = flag("alignRotation").unwrap_or(align);
    f.align.scale = flag("alignScale").unwrap_or(align);
    f.align.brightness = flag("alignBrightness").unwrap_or(align);
    f.align.affine = flag("alignAffine").unwrap_or(false);
    if let Some(r) = field(p, "reference")? {
        f.align.reference = r;
    }
    if let Some(c) = field(p, "crop")? {
        f.align.crop = c;
    }
    if let Some(i) = flag("interpolate") {
        f.interpolate = i;
    }
    Ok(f)
}

impl Session {
    /// Plan a merge of `ids` (catalog order of the given ids is kept).
    pub fn plan_merge(&self, kind: MergeKind, finish: MergeFinish, ids: &[PhotoId], preview: bool) -> Result<MergeJob> {
        let mut sources: Vec<(PhotoId, String)> = Vec::new();
        for id in ids {
            let p = self.catalog.photo(*id).ok_or(lightcraft_catalog::CatalogError::NoPhoto(*id))?;
            match &p.source {
                Source::File { path } => sources.push((*id, path.clone())),
                Source::Demo { .. } => return Err(EngineError::Other(format!("{} is a demo photo; Photo Merge needs photo files", p.file_name))),
            }
        }
        if sources.len() < kind.min_photos() {
            return Err(EngineError::Other(format!("select at least {} photos to merge", kind.min_photos())));
        }
        let read: ByteReader = match &self.media.file_bytes {
            Some(r) => r.clone(),
            None => Arc::new(|path: &str| std::fs::read(path).map_err(|e| format!("{path}: {e}"))),
        };
        let size = ids
            .iter()
            .filter_map(|id| self.catalog.photo(*id))
            .map(|p| (p.width as usize, p.height as usize))
            .max_by_key(|(w, h)| w * h)
            .unwrap_or((0, 0));
        let edge = preview.then_some(PREVIEW_EDGE);
        let mut kind = kind;
        if let MergeKind::FocusStack { focus, max_megapixels } = &mut kind {
            // capture order: by capture time, then file name (brackets are numbered in order)
            if focus.order != FocusOrder::Selection {
                let key = |id: PhotoId| self.catalog.photo(id).map(|p| (p.captured.clone().unwrap_or_default(), p.file_name.clone()));
                sources.sort_by_key(|(id, _)| key(*id));
            }
            let n = sources.len();
            let long = size.0.max(size.1);
            match edge {
                Some(e) => {
                    if focus.every == 0 {
                        focus.every = n.div_ceil(12).max(1);
                    }
                    *focus = focus.scaled(if long > e { e as f32 / long as f32 } else { 1.0 });
                    focus.max_edge = Some(e);
                }
                None => {
                    focus.every = focus.every.max(1);
                    let px = (size.0 * size.1) as f64;
                    if *max_megapixels > 0.0 && px > *max_megapixels * 1e6 && long > 0 {
                        let s = (*max_megapixels * 1e6 / px).sqrt();
                        focus.max_edge = Some(((long as f64 * s) as usize).max(64));
                        *focus = focus.scaled(s as f32);
                    }
                }
            }
        }
        Ok(MergeJob { kind, finish, sources, preview: edge, read, size })
    }

    /// Write a finished (non-preview) merge next to its first source, import it, select it and
    /// apply Auto Settings / the auto crop. Returns `{id, path, …info}`.
    pub fn finish_merge(&mut self, job: &MergeJob, out: MergeOutput) -> Result<Value> {
        let first = &job.sources.first().ok_or_else(|| EngineError::Other("nothing merged".into()))?.1;
        // a new file under a free name (never replacing one), complete and synced before it appears
        let path = lightcraft_catalog::safe_file::write_new_unique(&mut output_names(first, job.kind.suffix()), &out.dng)
            .map_err(|e| EngineError::Other(format!("could not write the merged DNG next to {first}: {e}")))?
            .to_string_lossy()
            .to_string();
        let report = crate::import::import(self, std::slice::from_ref(&path), crate::import::ImportMode::Add)?;
        let id = report
            .imported
            .first()
            .copied()
            .map(PhotoId)
            .ok_or_else(|| EngineError::Other(format!("the merged file {path} could not be imported: {:?}", report.failed)))?;
        if job.finish.stack {
            let sources: Vec<PhotoId> = job.sources.iter().map(|(p, _)| *p).collect();
            if let Some(op) = self.catalog.stack_with_ops(id, &sources) {
                self.commit("Stack with Sources", op)?;
            }
        }
        self.selection = crate::Selection::single(id);
        if job.finish.auto_settings {
            self.execute("develop.auto", &json!({}))?;
        }
        if let Some(c) = out.crop
            && let Some(d) = self.develop_of(id)
        {
            let mut d = (*d).clone();
            d.crop.geometry.rect = lightcraft_geom::Rect::from_xywh(c[0], c[1], c[2], c[3]);
            self.set_develop(id, d, "Auto Crop")?;
        }
        let mut info = out.info;
        if let (Some(dp), Some(d)) = (&job.finish.depth_path, &out.depth) {
            self.check_write_target(dp).map_err(EngineError::Other)?;
            crate::export::write_file(dp, &depth_png(d).map_err(EngineError::Other)?).map_err(EngineError::Other)?;
            info["depthPath"] = json!(dp);
        }
        info["id"] = json!(id.0);
        info["path"] = json!(path);
        info["bytes"] = json!(out.dng.len());
        Ok(info)
    }

    /// Plan, run and finish a merge synchronously (CLI, MCP, control channel).
    pub fn merge_now(&mut self, kind: MergeKind, finish: MergeFinish, ids: &[PhotoId]) -> Result<Value> {
        let job = self.plan_merge(kind, finish, ids, false)?;
        let out = job.run(&|_, _| true).map_err(EngineError::Other)?;
        self.finish_merge(&job, out)
    }
}
