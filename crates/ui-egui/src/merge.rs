//! Photo Merge in the UI: the HDR / Panorama / HDR Panorama / Focus Stack dialog (options + live
//! preview) and the background merge jobs.
//!
//! Previews and the final merge run [`lightcraft_engine::merge::MergeJob`]s on worker threads with
//! progress and cancellation; the window stays responsive. Changing an option cancels the running
//! preview and starts a new one (slider changes after a 250 ms pause, so a drag doesn't start a
//! preview per frame). Finished previews are kept per option set, so switching back (say, between
//! focus-stack methods A, B and C) is instant; a focus stack's depth map comes with its preview, so
//! showing it doesn't run anything. The final merge keeps running after the dialog closes
//! (progress in a toast); when it finishes, the result is written, imported and selected
//! ([`lightcraft_engine::Session::finish_merge`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use lightcraft_catalog::PhotoId;
use lightcraft_engine::merge::{MergeJob, MergeOutput};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LightcraftApp;

/// The dialog's options (also the parameters of the `merge.*` commands).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MergeDialog {
    /// "merge.hdr" | "merge.panorama" | "merge.hdrPanorama".
    pub command: String,
    pub align: bool,
    /// none | low | medium | high
    pub deghost: String,
    pub show_overlay: bool,
    pub auto_settings: bool,
    pub stack: bool,
    /// auto | spherical | cylindrical | perspective
    pub projection: String,
    pub boundary_warp: f64,
    pub auto_crop: bool,
    pub fill_edges: bool,
    /// HDR panorama: photos per bracket (0 = from EXIF).
    pub bracket: u32,
    /// Focus stack: depth | average | pyramid (B, A, C).
    pub method: String,
    pub radius: f64,
    pub smoothing: f64,
    /// capture | selection | reverse | auto
    pub order: String,
    pub align_shift: bool,
    pub align_rotation: bool,
    pub align_scale: bool,
    pub align_brightness: bool,
    /// Preview every n-th photo (0 = automatic, about 12 photos).
    pub every: u32,
    /// Merge with the preview's `every` too (else every photo).
    pub every_final: bool,
    /// Show the depth map instead of the stacked preview (no new run).
    pub show_depth: bool,
    /// Store the depth map in the DNG (Depth Range masks select on it).
    pub depth_map: bool,
}

impl Default for MergeDialog {
    fn default() -> Self {
        MergeDialog {
            command: "merge.hdr".into(),
            align: true,
            deghost: "none".into(),
            show_overlay: false,
            auto_settings: true,
            stack: false,
            projection: "auto".into(),
            boundary_warp: 0.0,
            auto_crop: true,
            fill_edges: false,
            bracket: 0,
            method: "depth".into(),
            radius: 8.0,
            smoothing: 4.0,
            order: "capture".into(),
            align_shift: true,
            align_rotation: true,
            align_scale: true,
            align_brightness: true,
            every: 0,
            every_final: false,
            show_depth: false,
            depth_map: true,
        }
    }
}

impl MergeDialog {
    pub fn for_command(command: &str) -> MergeDialog {
        MergeDialog { command: command.into(), ..Default::default() }
    }
    pub fn title(&self) -> &'static str {
        match self.command.as_str() {
            "merge.panorama" => "Panorama Merge Preview",
            "merge.hdrPanorama" => "HDR Panorama Merge Preview",
            "merge.focusStack" => "Focus Stack Preview",
            _ => "HDR Merge Preview",
        }
    }
    pub fn is_hdr(&self) -> bool {
        matches!(self.command.as_str(), "merge.hdr" | "merge.hdrPanorama")
    }
    pub fn is_pano(&self) -> bool {
        matches!(self.command.as_str(), "merge.panorama" | "merge.hdrPanorama")
    }
    pub fn is_focus(&self) -> bool {
        self.command == "merge.focusStack"
    }
    /// The command parameters for a preview or the merge itself.
    pub fn params(&self, preview: bool) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        if self.is_focus() {
            // the depth map comes with every preview; the dialog picks what to show
            v["showDepth"] = json!(false);
            v["every"] = json!(if preview {
                self.every
            } else if self.every_final {
                self.every.max(1)
            } else {
                1
            });
        }
        v
    }
    /// The options a preview depends on (what is shown of it aside).
    pub fn preview_key(&self) -> MergeDialog {
        MergeDialog { show_depth: false, every_final: false, depth_map: true, ..self.clone() }
    }
    /// Only slider values differ from `other` (the preview waits for the drag to pause).
    fn slider_change(&self, other: &MergeDialog) -> bool {
        let strip = |d: &MergeDialog| MergeDialog { radius: 0.0, smoothing: 0.0, every: 0, boundary_warp: 0.0, bracket: 0, ..d.preview_key() };
        strip(self) == strip(other)
    }
}

/// A finished preview: the rendered result, the depth map (focus stacks), their size, the stats.
#[derive(Clone)]
pub struct CachedPreview {
    pub key: MergeDialog,
    pub tex: egui::TextureHandle,
    pub pixels: Arc<egui::ColorImage>,
    pub depth: Option<(egui::TextureHandle, Arc<egui::ColorImage>)>,
    pub size: [usize; 2],
    pub info: Value,
}

/// Previews kept for instant switching back.
const CACHED_PREVIEWS: usize = 8;

/// A merge running on a worker thread.
pub struct MergeTask {
    pub preview: bool,
    pub job: MergeJob,
    /// The dialog options it was started with.
    pub options: MergeDialog,
    pub progress: Arc<Mutex<(f32, String)>>,
    pub cancel: Arc<AtomicBool>,
    rx: Receiver<Result<MergeOutput, String>>,
}

/// Merge state of the app.
#[derive(Default)]
pub struct MergeState {
    pub preview_task: Option<MergeTask>,
    pub final_task: Option<MergeTask>,
    /// The latest preview (texture, size, its options, stats) and the latest error.
    pub preview: Option<(egui::TextureHandle, [usize; 2], MergeDialog, Value)>,
    pub preview_pixels: Option<Arc<egui::ColorImage>>,
    pub error: Option<String>,
    /// Options whose preview failed (not retried until they change).
    pub failed_options: Option<MergeDialog>,
    /// The last finished merge (`{id, path, …}`).
    pub last_result: Option<Value>,
    /// Photos the dialog merges (the selection when it opened).
    pub ids: Vec<PhotoId>,
    /// The options of the last merge started, per kind ("merge.hdr" …): Merge with Last Settings.
    pub last: std::collections::HashMap<String, MergeDialog>,
    /// Finished previews, most recent last.
    pub cache: Vec<CachedPreview>,
    /// Options changed by a slider and when (the preview starts once they rest).
    pub pending: Option<(MergeDialog, f64)>,
}

impl MergeState {
    /// A preview or a merge is running.
    pub fn busy(&self) -> bool {
        self.preview_task.is_some() || self.final_task.is_some()
    }
}

fn spawn(app: &LightcraftApp, options: &MergeDialog, ids: &[PhotoId], preview: bool) -> Result<MergeTask, String> {
    let (kind, finish) = lightcraft_engine::merge::parse(&options.command, &options.params(preview)).map_err(|e| e.to_string())?;
    let job = app.session.plan_merge(kind, finish, ids, preview).map_err(|e| e.to_string())?;
    let progress = Arc::new(Mutex::new((0.0, "Starting".to_string())));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = channel();
    let (j, p, c) = (job.clone(), progress.clone(), cancel.clone());
    let work = move || {
        let run = || {
            j.run(&|f, stage| {
                if let Ok(mut g) = p.lock() {
                    *g = (f, stage.to_string());
                }
                !c.load(Ordering::Relaxed)
            })
        };
        // a panicking merge reports an error instead of leaving "Merging…" up forever
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|_| Err("the merge failed unexpectedly".into()));
        let _ = tx.send(r);
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name("photo-merge".into()).spawn(work).map_err(|e| e.to_string())?;
    #[cfg(target_arch = "wasm32")]
    work();
    Ok(MergeTask { preview, job, options: options.clone(), progress, cancel, rx })
}

/// Open the merge dialog for the selection.
pub fn open(app: &mut LightcraftApp, command: &str) -> Result<Value, String> {
    let ids = app.session.targets(&json!({}));
    if ids.len() < 2 {
        return Err("select at least 2 photos to merge".into());
    }
    app.merge.ids = ids;
    app.merge.error = None;
    app.merge.preview = None;
    app.merge.preview_pixels = None;
    app.merge.cache.clear();
    app.merge.pending = None;
    app.merge.failed_options = None;
    if let Some(t) = app.merge.preview_task.take() {
        t.cancel.store(true, Ordering::Relaxed);
    }
    // the dialog opens with the options of the last merge of its kind
    let opts = app.merge.last.get(command).cloned().unwrap_or_else(|| MergeDialog::for_command(command));
    app.ui.dialog = Some(crate::state::Dialog::Merge { opts });
    Ok(json!({"photos": app.merge.ids.len()}))
}

/// Start the full-resolution merge (the dialog's Merge button / `ui.dialog.confirm`).
pub fn start_final(app: &mut LightcraftApp, opts: &MergeDialog) -> Result<Value, String> {
    if app.merge.final_task.is_some() {
        return Err("a merge is already running".into());
    }
    if let Some(t) = app.merge.preview_task.take() {
        t.cancel.store(true, Ordering::Relaxed);
    }
    let ids = app.merge.ids.clone();
    let task = spawn(app, opts, &ids, false)?;
    app.merge.final_task = Some(task);
    app.merge.last.insert(opts.command.clone(), opts.clone());
    Ok(json!({"started": true, "photos": ids.len()}))
}

/// Merge the selection without the dialog, with the options last used for `command` (the
/// defaults the first time).
pub fn start_last(app: &mut LightcraftApp, command: &str) -> Result<Value, String> {
    let ids = app.session.targets(&json!({}));
    if ids.len() < 2 {
        return Err("select at least 2 photos to merge".into());
    }
    app.merge.ids = ids;
    let opts = app.merge.last.get(command).cloned().unwrap_or_else(|| MergeDialog::for_command(command));
    start_final(app, &opts)
}

/// Show `c` as the dialog's preview (the depth map when asked).
fn show_cached(app: &mut LightcraftApp, c: &CachedPreview, depth: bool) {
    let (tex, px) = match (&c.depth, depth) {
        (Some((t, p)), true) => (t.clone(), p.clone()),
        _ => (c.tex.clone(), c.pixels.clone()),
    };
    app.merge.preview_pixels = Some(px);
    app.merge.preview = Some((tex, c.size, c.key.clone(), c.info.clone()));
}

/// Per frame: keep the preview in sync with the dialog's options, collect finished jobs.
pub fn poll(app: &mut LightcraftApp, ctx: &egui::Context) {
    // preview for the open dialog
    let dialog_opts = match &app.ui.dialog {
        Some(crate::state::Dialog::Merge { opts }) => Some(opts.clone()),
        _ => None,
    };
    match &dialog_opts {
        Some(opts) => {
            let key = opts.preview_key();
            let now = ctx.input(|i| i.time);
            if let Some(c) = app.merge.cache.iter().find(|c| c.key == key).cloned() {
                // a preview for these options exists: show it, stop working on others
                if app.merge.preview.as_ref().is_none_or(|p| p.2 != key)
                    || app.merge.preview_pixels.is_none()
                    || opts.show_depth != showing_depth(app)
                {
                    show_cached(app, &c, opts.show_depth);
                }
                if let Some(t) = app.merge.preview_task.take_if(|t| t.options != key) {
                    t.cancel.store(true, Ordering::Relaxed);
                }
                app.merge.pending = None;
                app.merge.error = None;
            } else {
                let running = app.merge.preview_task.as_ref().map(|t| &t.options);
                let stale = running != Some(&key) && !last_failed_matches(app, &key);
                if stale {
                    // slider drags wait for a 250 ms pause; other changes start at once
                    let base = running.or(app.merge.preview.as_ref().map(|p| &p.2));
                    let wait = base.is_some_and(|b| key.slider_change(b));
                    let ready = match &app.merge.pending {
                        Some((k, t)) if *k == key => now - t >= 0.25,
                        _ => {
                            app.merge.pending = Some((key.clone(), now));
                            !wait
                        }
                    };
                    if ready {
                        app.merge.pending = None;
                        if let Some(t) = app.merge.preview_task.take() {
                            t.cancel.store(true, Ordering::Relaxed);
                        }
                        let ids = app.merge.ids.clone();
                        match spawn(app, &key, &ids, true) {
                            Ok(t) => {
                                app.merge.error = None;
                                app.merge.preview_task = Some(t);
                            }
                            Err(e) => {
                                app.merge.error = Some(e);
                                app.merge.failed_options = Some(key.clone());
                            }
                        }
                    } else {
                        ctx.request_repaint_after(std::time::Duration::from_millis(60));
                    }
                }
            }
        }
        None => {
            if let Some(t) = app.merge.preview_task.take() {
                t.cancel.store(true, Ordering::Relaxed);
            }
        }
    }
    if let Some(t) = &app.merge.preview_task {
        match t.rx.try_recv() {
            Ok(Ok(out)) => {
                let Some(t) = app.merge.preview_task.take() else { return };
                if let Some(img) = out.preview {
                    let color = Arc::new(egui::ColorImage::from_rgba_unmultiplied([img.width, img.height], &img.as_bytes()));
                    let tex = ctx.load_texture("merge-preview", color.clone(), egui::TextureOptions::LINEAR);
                    let depth = out.depth.as_ref().map(|d| {
                        let rgba: Vec<u8> = d
                            .data
                            .iter()
                            .flat_map(|v| {
                                let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                                [g, g, g, 255]
                            })
                            .collect();
                        let px = Arc::new(egui::ColorImage::from_rgba_unmultiplied([d.width, d.height], &rgba));
                        (ctx.load_texture("merge-depth", px.clone(), egui::TextureOptions::LINEAR), px)
                    });
                    let c = CachedPreview { key: t.options.clone(), tex, pixels: color, depth, size: [img.width, img.height], info: out.info };
                    let show = dialog_opts.as_ref().is_some_and(|o| o.show_depth);
                    show_cached(app, &c, show);
                    app.merge.cache.retain(|x| x.key != c.key);
                    app.merge.cache.push(c);
                    if app.merge.cache.len() > CACHED_PREVIEWS {
                        app.merge.cache.remove(0);
                    }
                }
            }
            Ok(Err(e)) => {
                let Some(t) = app.merge.preview_task.take() else { return };
                if e != "cancelled" {
                    app.merge.error = Some(e);
                    app.merge.failed_options = Some(t.options);
                }
            }
            Err(_) => ctx.request_repaint_after(std::time::Duration::from_millis(50)),
        }
    }
    // the final merge
    if let Some(t) = &app.merge.final_task {
        match t.rx.try_recv() {
            Ok(r) => {
                let Some(t) = app.merge.final_task.take() else { return };
                let what = match t.options.command.as_str() {
                    "merge.panorama" => "Panorama",
                    "merge.hdrPanorama" => "HDR panorama",
                    "merge.focusStack" => "Focus stack",
                    _ => "HDR",
                };
                match r.map_err(lightcraft_engine::EngineError::Other).and_then(|out| app.session.finish_merge(&t.job, out)) {
                    Ok(v) => {
                        app.merge.last_result = Some(v);
                        app.toast(ctx, crate::i18n::tr_format!("{what} merge added", what = what));
                    }
                    Err(e) => {
                        app.ui.status = e.to_string();
                        app.toast(ctx, crate::i18n::tr_format!("{what} merge failed: {e}", e = e, what = what));
                    }
                }
            }
            Err(_) => {
                let (f, stage) = t.progress.lock().map(|g| g.clone()).unwrap_or_default();
                let now = ctx.input(|i| i.time);
                app.ui.toast = Some((crate::i18n::tr_format!("Merging… {stage} {:.0}%", f * 100.0, stage = stage), now + 0.5, None));
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
    }
}

fn last_failed_matches(app: &LightcraftApp, opts: &MergeDialog) -> bool {
    app.merge.failed_options.as_ref() == Some(opts)
}

/// The dialog shows the depth map now.
fn showing_depth(app: &LightcraftApp) -> bool {
    match (&app.merge.preview, app.merge.cache.iter().find(|c| app.merge.preview.as_ref().is_some_and(|p| p.2 == c.key))) {
        (Some((tex, ..)), Some(c)) => c.depth.as_ref().is_some_and(|(d, _)| d.id() == tex.id()),
        _ => false,
    }
}

/// The dialog body (options on the left, preview on the right).
pub fn body(app: &mut LightcraftApp, ui: &mut egui::Ui, opts: &mut MergeDialog) {
    let t = crate::theme::Tokens::get(ui.ctx());
    ui.horizontal_top(|ui| {
        // preview
        ui.vertical(|ui| {
            let box_size = egui::vec2(520.0, 360.0);
            let (rect, _) = ui.allocate_exact_size(box_size, egui::Sense::hover());
            ui.painter().rect_filled(rect, 4.0, t.canvas);
            crate::widgets::register(ui.ctx(), "merge.preview", rect);
            if let Some((tex, size, _, _)) = &app.merge.preview {
                let s = (box_size.x / size[0] as f32).min(box_size.y / size[1] as f32);
                let r = egui::Rect::from_center_size(rect.center(), egui::vec2(size[0] as f32 * s, size[1] as f32 * s));
                ui.painter().image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            if let Some(task) = &app.merge.preview_task {
                let (f, stage) = task.progress.lock().map(|g| g.clone()).unwrap_or_default();
                ui.painter().text(
                    rect.center_bottom() - egui::vec2(0.0, 14.0),
                    egui::Align2::CENTER_CENTER,
                    crate::i18n::tr_format!("{stage}… {:.0}%", f * 100.0, stage = stage),
                    t.font(12.0),
                    t.text,
                );
            }
            if let Some(e) = &app.merge.error {
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, e, t.font(13.0), t.text);
            }
            if opts.is_focus() {
                // compare the methods (each preview is kept once computed)
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(crate::i18n::tr("Compare")).color(t.text_label));
                    for (key, label) in [("a", "A"), ("b", "B"), ("c", "C")] {
                        let m = method_of(key);
                        let cached = app.merge.cache.iter().any(|c| c.key == MergeDialog { method: m.into(), ..opts.preview_key() });
                        let text = if cached && opts.method != m { format!("{label}·") } else { label.to_string() };
                        if crate::widgets::text_button(ui, &format!("mergeCompare-{key}"), &text, opts.method == m).clicked() {
                            opts.method = m.into();
                        }
                    }
                    ui.add_space(8.0);
                    let r = ui.checkbox(&mut opts.show_depth, crate::i18n::tr("Depth Map"));
                    crate::widgets::register(ui.ctx(), "check:mergeShowDepth", r.rect);
                });
            }
            let info = app.merge.preview.as_ref().map(|p| p.3.clone());
            if let Some(info) = info {
                let txt = if opts.is_focus() {
                    focus_summary(app, &info)
                } else if opts.is_pano() {
                    format!(
                        "{} photos · {} · {:.0}° × {:.0}°",
                        info["used"].as_array().map_or(0, Vec::len),
                        info["projection"].as_str().unwrap_or(""),
                        info["fov"][0].as_f64().unwrap_or(0.0),
                        info["fov"][1].as_f64().unwrap_or(0.0)
                    )
                } else {
                    let ev: Vec<String> =
                        info["ev"].as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|v| format!("{v:+.1}")).collect()).unwrap_or_default();
                    crate::i18n::tr_format!("{} photos · exposures {} EV", ev.len(), ev.join(" / "))
                };
                ui.label(egui::RichText::new(txt).color(t.text_dim));
            }
        });
        ui.add_space(12.0);
        // options
        ui.vertical(|ui| {
            ui.set_width(if opts.is_focus() { 250.0 } else { 220.0 });
            if opts.is_focus() {
                focus_options(ui, opts);
            }
            if opts.is_pano() {
                ui.label(egui::RichText::new(crate::i18n::tr("Projection")).color(t.text_label));
                for (v, l) in [("auto", "Auto Select"), ("spherical", "Spherical"), ("cylindrical", "Cylindrical"), ("perspective", "Perspective")] {
                    if ui.radio(opts.projection == v, l).clicked() {
                        opts.projection = v.to_string();
                    }
                }
                ui.add_space(4.0);
                let mut bw = opts.boundary_warp;
                if num(ui, &BOUNDARY_WARP, &mut bw) {
                    opts.boundary_warp = bw.round();
                }
                ui.checkbox(&mut opts.fill_edges, crate::i18n::tr("Fill Edges"));
                ui.checkbox(&mut opts.auto_crop, crate::i18n::tr("Auto Crop"));
                ui.add_space(4.0);
            }
            if opts.is_hdr() {
                ui.checkbox(&mut opts.align, crate::i18n::tr("Auto Align"));
                ui.label(egui::RichText::new(crate::i18n::tr("Deghost Amount")).color(t.text_label));
                ui.horizontal(|ui| {
                    for (i, (v, l)) in [("none", "None"), ("low", "Low"), ("medium", "Med"), ("high", "High")].iter().enumerate() {
                        if crate::widgets::text_button(ui, &format!("mergeDeghost-{i}"), l, opts.deghost == *v).clicked() {
                            opts.deghost = v.to_string();
                        }
                    }
                });
                if opts.command == "merge.hdr" {
                    ui.add_enabled_ui(opts.deghost != "none", |ui| ui.checkbox(&mut opts.show_overlay, crate::i18n::tr("Show Deghost Overlay")));
                }
                if opts.command == "merge.hdrPanorama" {
                    let mut b = opts.bracket as f64;
                    if num(ui, &BRACKET, &mut b) {
                        opts.bracket = b.round() as u32;
                    }
                }
                ui.add_space(4.0);
            }
            ui.checkbox(&mut opts.auto_settings, crate::i18n::tr("Auto Settings"));
            ui.checkbox(&mut opts.stack, crate::i18n::tr("Create Stack"));
            ui.add_space(8.0);
            ui.label(egui::RichText::new(crate::i18n::tr_format!("{} photos", app.merge.ids.len())).color(t.text_dim));
        });
    });
}

fn method_of(key: &str) -> &'static str {
    match key {
        "a" => "average",
        "c" => "pyramid",
        _ => "depth",
    }
}

/// A checkbox addressable as `check:{id}`.
fn check(ui: &mut egui::Ui, id: &str, value: &mut bool, label: &str) {
    let r = ui.checkbox(value, crate::i18n::tr(label));
    crate::widgets::register(ui.ctx(), format!("check:{id}"), r.rect);
}

/// The focus-stack options column.
fn focus_options(ui: &mut egui::Ui, opts: &mut MergeDialog) {
    let t = crate::theme::Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(crate::i18n::tr("Method")).color(t.text_label));
    let methods = [("A Average", "a"), ("B Depth Map", "b"), ("C Pyramid", "c")];
    let active = methods.iter().position(|(_, k)| method_of(k) == opts.method);
    if let Some(i) = crate::widgets::segmented(ui, "mergeMethod", &methods, active, 3) {
        opts.method = method_of(methods[i].1).into();
    }
    let hint = match opts.method.as_str() {
        "average" => "Weights every photo by its sharpness: smooth, true colours, a little soft.",
        "pyramid" => "Detail bands from the sharpest photos: best for crossing hairs and bristles.",
        _ => "Each pixel from its sharpest photo: crisp detail and texture.",
    };
    ui.label(egui::RichText::new(crate::i18n::tr(hint)).color(t.text_dim).size(11.0));
    ui.add_space(2.0);
    let mut r = opts.radius;
    if num(ui, &RADIUS, &mut r) {
        opts.radius = r.round();
    }
    let mut sm = opts.smoothing;
    if num(ui, &SMOOTHING, &mut sm) {
        opts.smoothing = sm.round();
    }
    ui.label(egui::RichText::new(crate::i18n::tr("Auto-Adjust")).color(t.text_label));
    ui.horizontal_wrapped(|ui| {
        check(ui, "mergeAlignShift", &mut opts.align_shift, "Shift");
        check(ui, "mergeAlignRotation", &mut opts.align_rotation, "Rotation");
        check(ui, "mergeAlignScale", &mut opts.align_scale, "Magnification");
        check(ui, "mergeAlignBrightness", &mut opts.align_brightness, "Brightness");
    });
    ui.label(egui::RichText::new(crate::i18n::tr("Focus Order")).color(t.text_label));
    let orders = [("Capture Time", "capture"), ("Selection", "selection"), ("Reverse", "reverse"), ("Auto", "auto")];
    let active = orders.iter().position(|(_, k)| *k == opts.order);
    if let Some(i) = crate::widgets::segmented(ui, "mergeOrder", &orders, active, 2) {
        opts.order = orders[i].1.into();
    }
    let mut e = opts.every as f64;
    if num(ui, &EVERY, &mut e) {
        opts.every = e.round().max(0.0) as u32;
    }
    if opts.every == 0 {
        ui.label(egui::RichText::new(crate::i18n::tr("0 = about 12 photos")).color(t.text_dim).size(11.0));
    }
    ui.add_enabled_ui(opts.every > 1, |ui| check(ui, "mergeEveryFinal", &mut opts.every_final, "Merge Every n-th Photo Too"));
    check(ui, "mergeDepthMap", &mut opts.depth_map, "Embed Depth Map");
    ui.add_space(4.0);
}

/// The stats line under a focus-stack preview.
fn focus_summary(app: &LightcraftApp, info: &Value) -> String {
    let used = info["used"].as_u64().unwrap_or(0);
    let order: Vec<u64> = info["order"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    let every = if used > 0 && order.len() as u64 == used && (app.merge.ids.len() as u64) > used {
        format!(" · {}/{}", used, app.merge.ids.len())
    } else {
        String::new()
    };
    let reference = info["reference"].as_u64().and_then(|id| app.session.catalog.photo(PhotoId(id))).map(|p| p.file_name.clone()).unwrap_or_default();
    let a = &info["alignment"];
    let skipped = info["skipped"].as_array().map_or(0, Vec::len);
    let mut txt = crate::i18n::tr_format!("{} photos", used);
    txt.push_str(&every);
    if !reference.is_empty() {
        txt.push_str(&format!(" · {reference}"));
    }
    txt.push_str(&format!(
        " · ≤ {:.1} px, {:.2}°, ×{:.3}–{:.3}",
        a["maxShiftPx"].as_f64().unwrap_or(0.0),
        a["maxRotationDeg"].as_f64().unwrap_or(0.0),
        a["scale"][0].as_f64().unwrap_or(1.0),
        a["scale"][1].as_f64().unwrap_or(1.0)
    ));
    if skipped > 0 {
        txt.push_str(&format!(" · {skipped} ✕"));
    }
    if let Some(w) = info["warnings"].as_array().and_then(|w| w.first()).and_then(Value::as_str) {
        txt.push('\n');
        txt.push_str(w);
    }
    txt
}

const RADIUS: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "merge.radius",
    label: "Radius",
    section: lightcraft_develop::Section::Light,
    min: 1.0,
    max: 30.0,
    default: 8.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
const SMOOTHING: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "merge.smoothing",
    label: "Smoothing",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 10.0,
    default: 4.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
const EVERY: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "merge.every",
    label: "Preview Every n-th Photo",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 20.0,
    default: 0.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};

const BOUNDARY_WARP: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "merge.boundaryWarp",
    label: "Boundary Warp",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 100.0,
    default: 0.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
const BRACKET: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "merge.bracket",
    label: "Photos per bracket (0 = auto)",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 9.0,
    default: 0.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};

fn num(ui: &mut egui::Ui, spec: &lightcraft_develop::ControlSpec, v: &mut f64) -> bool {
    match crate::widgets::slider(ui, spec, *v, true, None).value {
        Some(n) => {
            *v = n;
            true
        }
        None => false,
    }
}
