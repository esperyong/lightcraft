//! Retouching a focus stack in the app: a dialog with the stack on a canvas, brush strokes that
//! take pixels from a frame or another method's result, and its background work (the engine side
//! is `lightcraft_engine::focus_retouch`).
//!
//! Opening decodes the stack at ≤ 1600 px on a worker; every stroke re-renders the preview there
//! (a source is made the first time a stroke uses it — a frame decodes and aligns in a moment, a
//! method restacks every frame at preview size). Apply redoes the DNG at full size in the
//! background, like a merge, and reloads the photo.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use lightcraft_catalog::PhotoId;
use lightcraft_engine::focus_retouch::{Retouch, RetouchJob, RetouchSource, RetouchStroke};
use lightcraft_raster::Rgba8;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LightcraftApp;

/// Long edge of the preview.
pub const PREVIEW_EDGE: usize = 1600;

/// The dialog's state (also settable through the control channel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RetouchDialog {
    pub id: u64,
    /// base | a | b | c | frame
    pub source: String,
    /// The frame painted from when `source` is `frame` (0-based).
    pub frame: usize,
    /// Brush radius, % of the long edge.
    pub size: f64,
    /// 0..100
    pub hardness: f64,
    /// 0..100
    pub opacity: f64,
    pub strokes: Vec<RetouchStroke>,
}

impl Default for RetouchDialog {
    fn default() -> Self {
        RetouchDialog { id: 0, source: "frame".into(), frame: 0, size: 3.0, hardness: 50.0, opacity: 100.0, strokes: Vec::new() }
    }
}

impl RetouchDialog {
    fn source(&self) -> RetouchSource {
        match self.source.as_str() {
            "frame" => RetouchSource::Frame(self.frame),
            s => RetouchSource::Named(s.to_string()),
        }
    }
}

/// The app's retouch state.
#[derive(Default)]
pub struct RetouchState {
    session: Option<Arc<Mutex<Retouch>>>,
    opening: Option<Receiver<Result<Retouch, String>>>,
    /// Frames' file names (from the project).
    pub frames: Vec<String>,
    render: Option<(Receiver<Result<Rgba8, String>>, Vec<RetouchStroke>)>,
    /// The strokes the preview shows.
    shown: Option<Vec<RetouchStroke>>,
    pub preview: Option<(egui::TextureHandle, [usize; 2])>,
    pub pixels: Option<Arc<egui::ColorImage>>,
    pub error: Option<String>,
    /// The stroke being painted (normalised points).
    painting: Vec<[f32; 2]>,
    apply: Option<ApplyTask>,
}

struct ApplyTask {
    job: Arc<RetouchJob>,
    rx: Receiver<Result<Vec<u8>, String>>,
    progress: Arc<Mutex<(f32, String)>>,
    #[allow(dead_code)]
    cancel: Arc<AtomicBool>,
}

impl RetouchState {
    /// A preview, the opening or an apply is running.
    pub fn busy(&self) -> bool {
        self.opening.is_some() || self.render.is_some() || self.apply.is_some()
    }
}

fn spawn<T: Send + 'static>(name: &str, f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<Receiver<Result<T, String>>, String> {
    let (tx, rx) = channel();
    let work = move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| Err("the retouch failed unexpectedly".into()));
        let _ = tx.send(r);
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name(name.into()).spawn(work).map_err(|e| e.to_string())?;
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        work();
    }
    Ok(rx)
}

/// Open the dialog for the active photo (a focus stack made by LightCraft).
pub fn open(app: &mut LightcraftApp) -> Result<Value, String> {
    let id = app.session.active().ok_or("no photo")?;
    let plan = app.session.plan_open_retouch(id, PREVIEW_EDGE).map_err(|e| e.to_string())?;
    let project = app.session.execute("merge.focusProject", &json!({"id": id.0})).map_err(|e| e.to_string())?;
    let strokes: Vec<RetouchStroke> = serde_json::from_value(project["strokes"].clone()).unwrap_or_default();
    app.retouch = RetouchState {
        frames: project["frames"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default(),
        opening: Some(spawn("focus-retouch", move || plan.open())?),
        ..Default::default()
    };
    let opts = RetouchDialog { id: id.0, strokes, ..Default::default() };
    let n = app.retouch.frames.len();
    app.ui.dialog = Some(crate::state::Dialog::FocusRetouch { opts });
    Ok(json!({"frames": n}))
}

/// Apply the dialog's strokes at full size (in the background).
pub fn apply(app: &mut LightcraftApp, opts: &RetouchDialog) -> Result<Value, String> {
    if app.retouch.apply.is_some() {
        return Err("a retouch is already being applied".into());
    }
    let job = Arc::new(app.session.plan_retouch(PhotoId(opts.id), opts.strokes.clone()).map_err(|e| e.to_string())?);
    let progress = Arc::new(Mutex::new((0.0, "Starting".to_string())));
    let cancel = Arc::new(AtomicBool::new(false));
    let (j, p, c) = (job.clone(), progress.clone(), cancel.clone());
    let rx = spawn("focus-retouch-apply", move || {
        j.run(&|f, stage| {
            if let Ok(mut g) = p.lock() {
                *g = (f, stage.to_string());
            }
            !c.load(Ordering::Relaxed)
        })
    })?;
    app.retouch.apply = Some(ApplyTask { job, rx, progress, cancel });
    app.retouch.session = None;
    Ok(json!({"started": true, "strokes": opts.strokes.len()}))
}

/// Per frame: collect the opening, keep the preview in step with the strokes, finish applies.
pub fn poll(app: &mut LightcraftApp, ctx: &egui::Context) {
    let st = &mut app.retouch;
    if let Some(rx) = &st.opening {
        match rx.try_recv() {
            Ok(Ok(r)) => {
                st.session = Some(Arc::new(Mutex::new(r)));
                st.opening = None;
            }
            Ok(Err(e)) => {
                st.error = Some(e);
                st.opening = None;
            }
            Err(_) => ctx.request_repaint_after(std::time::Duration::from_millis(50)),
        }
    }
    let wanted = match &app.ui.dialog {
        Some(crate::state::Dialog::FocusRetouch { opts }) => Some(opts.strokes.clone()),
        _ => None,
    };
    let st = &mut app.retouch;
    if let Some((rx, strokes)) = &st.render {
        match rx.try_recv() {
            Ok(r) => {
                let strokes = strokes.clone();
                st.render = None;
                match r {
                    Ok(img) => {
                        let color = Arc::new(egui::ColorImage::from_rgba_unmultiplied([img.width, img.height], &img.as_bytes()));
                        st.preview = Some((ctx.load_texture("focus-retouch", color.clone(), egui::TextureOptions::LINEAR), [img.width, img.height]));
                        st.pixels = Some(color);
                        st.shown = Some(strokes);
                        st.error = None;
                    }
                    Err(e) => {
                        st.error = Some(e);
                        st.shown = Some(strokes);
                    }
                }
            }
            Err(_) => ctx.request_repaint_after(std::time::Duration::from_millis(50)),
        }
    }
    if let (Some(strokes), Some(session), None) = (wanted, st.session.clone(), &st.render)
        && st.shown.as_ref() != Some(&strokes)
    {
        let s = strokes.clone();
        match spawn("focus-retouch-preview", move || session.lock().map_err(|_| "retouch session poisoned".to_string())?.render(&s)) {
            Ok(rx) => st.render = Some((rx, strokes)),
            Err(e) => st.error = Some(e),
        }
    }
    // the full-size apply
    if let Some(t) = &app.retouch.apply {
        match t.rx.try_recv() {
            Ok(r) => {
                let Some(t) = app.retouch.apply.take() else { return };
                match r.map_err(lightcraft_engine::EngineError::Other).and_then(|dng| app.session.finish_retouch(&t.job, dng)) {
                    Ok(_) => app.toast(ctx, crate::i18n::tr("Focus stack retouched")),
                    Err(e) => {
                        app.ui.status = e.to_string();
                        app.toast(ctx, e.to_string());
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

/// The dialog body: the stack on the left (paint on it), sources and brush on the right.
pub fn body(app: &mut LightcraftApp, ui: &mut egui::Ui, opts: &mut RetouchDialog) {
    let t = crate::theme::Tokens::get(ui.ctx());
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            let box_size = egui::vec2(760.0, 520.0);
            let (rect, resp) = ui.allocate_exact_size(box_size, egui::Sense::click_and_drag());
            crate::widgets::register(ui.ctx(), "focusRetouch.canvas", rect);
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 4.0, t.canvas);
            let Some((tex, size)) = &app.retouch.preview else {
                let msg = app.retouch.error.clone().unwrap_or_else(|| crate::i18n::tr("Opening the focus stack…").to_string());
                p.text(rect.center(), egui::Align2::CENTER_CENTER, msg, t.font(13.0), t.text);
                return;
            };
            let s = (box_size.x / size[0] as f32).min(box_size.y / size[1] as f32);
            let img = egui::Rect::from_center_size(rect.center(), egui::vec2(size[0] as f32 * s, size[1] as f32 * s));
            p.image(tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            let radius = (opts.size / 100.0) as f32 * img.width().max(img.height());
            let norm = |q: egui::Pos2| [((q.x - img.left()) / img.width()).clamp(0.0, 1.0), ((q.y - img.top()) / img.height()).clamp(0.0, 1.0)];
            if let Some(h) = resp.hover_pos() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::None);
                p.circle_stroke(h, radius, egui::Stroke::new(1.0, egui::Color32::WHITE));
                p.circle_stroke(h, radius * (opts.hardness / 100.0) as f32, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(110)));
            }
            if let Some(q) = resp.interact_pointer_pos()
                && (resp.drag_started() || resp.dragged() || resp.clicked())
                && img.contains(q)
            {
                app.retouch.painting.push(norm(q));
            }
            for q in &app.retouch.painting {
                let at = egui::pos2(img.left() + q[0] * img.width(), img.top() + q[1] * img.height());
                p.circle_filled(at, radius, egui::Color32::from_rgba_unmultiplied(80, 160, 255, 50));
            }
            if (resp.drag_stopped() || resp.clicked()) && !app.retouch.painting.is_empty() {
                let points = std::mem::take(&mut app.retouch.painting);
                opts.strokes.push(RetouchStroke {
                    source: opts.source(),
                    points,
                    size: (opts.size / 100.0) as f32,
                    hardness: (opts.hardness / 100.0) as f32,
                    opacity: (opts.opacity / 100.0) as f32,
                });
            }
            if app.retouch.render.is_some() || app.retouch.opening.is_some() {
                p.text(rect.center_bottom() - egui::vec2(0.0, 14.0), egui::Align2::CENTER_CENTER, crate::i18n::tr("Updating…"), t.font(12.0), t.text);
            }
            if let Some(e) = &app.retouch.error {
                p.text(rect.center_top() + egui::vec2(0.0, 14.0), egui::Align2::CENTER_CENTER, e, t.font(12.0), t.text);
            }
        });
        ui.add_space(12.0);
        ui.vertical(|ui| {
            ui.set_width(230.0);
            ui.label(egui::RichText::new(crate::i18n::tr("Paint From")).color(t.text_label));
            let sources = [("Frame", "frame"), ("As Merged", "base"), ("A", "a"), ("B", "b"), ("C", "c")];
            let active = sources.iter().position(|(_, k)| *k == opts.source);
            if let Some(i) = crate::widgets::segmented(ui, "retouchSource", &sources[..2], active.filter(|i| *i < 2), 2) {
                opts.source = sources[i].1.into();
            }
            if let Some(i) = crate::widgets::segmented(ui, "retouchMethod", &sources[2..], active.and_then(|i| i.checked_sub(2)), 3) {
                opts.source = sources[i + 2].1.into();
            }
            let n = app.retouch.frames.len();
            if opts.source == "frame" && n > 0 {
                let mut f = opts.frame as f64 + 1.0;
                let spec = lightcraft_develop::ControlSpec { max: n as f64, ..FRAME };
                if let Some(v) = crate::widgets::slider(ui, &spec, f, true, None).value {
                    f = v.round();
                    opts.frame = (f as usize).clamp(1, n) - 1;
                }
                if let Some(name) = app.retouch.frames.get(opts.frame) {
                    ui.label(egui::RichText::new(name).color(t.text_dim).size(11.0));
                }
            }
            let hint = match opts.source.as_str() {
                "base" => "Erases strokes underneath: the stack as merged.",
                "a" | "b" | "c" => "The stack redone with that method (the first stroke takes a moment).",
                _ => "One photo of the bracket, aligned to the stack.",
            };
            ui.label(egui::RichText::new(crate::i18n::tr(hint)).color(t.text_dim).size(11.0));
            ui.add_space(4.0);
            for (spec, v) in [(&SIZE, &mut opts.size), (&HARDNESS, &mut opts.hardness), (&OPACITY, &mut opts.opacity)] {
                if let Some(n) = crate::widgets::slider(ui, spec, *v, true, None).value {
                    *v = n;
                }
            }
            ui.horizontal(|ui| {
                if crate::widgets::text_button(ui, "retouchUndo", "Undo Stroke", false).clicked() {
                    opts.strokes.pop();
                }
                if crate::widgets::text_button(ui, "retouchClear", "Clear", false).clicked() {
                    opts.strokes.clear();
                }
            });
            ui.label(egui::RichText::new(format!("{}: {}", crate::i18n::tr("Strokes"), opts.strokes.len())).color(t.text_dim));
        });
    });
}

const FRAME: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "retouch.frame",
    label: "Frame",
    section: lightcraft_develop::Section::Light,
    min: 1.0,
    max: 2.0,
    default: 1.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
const SIZE: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "retouch.size",
    label: "Size",
    section: lightcraft_develop::Section::Light,
    min: 0.3,
    max: 20.0,
    default: 3.0,
    step: 0.1,
    decimals: 1,
    track: lightcraft_develop::Track::Plain,
};
const HARDNESS: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "retouch.hardness",
    label: "Hardness",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 100.0,
    default: 50.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
const OPACITY: lightcraft_develop::ControlSpec = lightcraft_develop::ControlSpec {
    id: "retouch.opacity",
    label: "Opacity",
    section: lightcraft_develop::Section::Light,
    min: 0.0,
    max: 100.0,
    default: 100.0,
    step: 1.0,
    decimals: 0,
    track: lightcraft_develop::Track::Plain,
};
