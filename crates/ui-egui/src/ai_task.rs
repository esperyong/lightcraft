//! AI analysis in the background (Photo → Analyze with Apple Vision, `ai.analyze`): the engine's workers analyze the
//! photos while the window stays responsive; every few frames `ai.status` stores what they finished, and a panel shows
//! the count with a Cancel button. When the run ends, a toast says how it went.

use egui::Align2;
use serde_json::json;

use crate::LightcraftApp;
use crate::theme::Tokens;

/// How often the engine is asked for progress while photos are analyzed (seconds).
const POLL_SECS: f64 = 0.2;

#[derive(Default)]
pub struct AiTask {
    /// A run is in progress (seen busy since it last ended).
    running: bool,
    next_poll: f64,
    done: u64,
    total: u64,
    failed: usize,
    stopping: bool,
}

/// Called every frame: cheap while nothing is analyzed.
pub fn poll(app: &mut LightcraftApp, ctx: &egui::Context) {
    if !app.session.ai.busy() && !app.ai_task.running {
        return;
    }
    let now = ctx.input(|i| i.time);
    if now >= app.ai_task.next_poll {
        app.ai_task.next_poll = now + POLL_SECS;
        if let Ok(v) = app.session.execute("ai.status", &json!({})) {
            let t = &mut app.ai_task;
            t.running = true;
            t.done = v["done"].as_u64().unwrap_or(0);
            t.total = v["total"].as_u64().unwrap_or(0);
            t.failed = v["failed"].as_array().map_or(0, Vec::len);
            if !v["photos"].as_array().is_none_or(Vec::is_empty) {
                // new labels and scores: a search, smart album or aesthetic sort in view changes
                ctx.request_repaint();
            }
            if v["pending"] == json!(0) {
                let msg = if t.failed > 0 {
                    crate::i18n::tr_format!("Analyzed {} photos ({} could not be analyzed)", t.done, t.failed)
                } else {
                    crate::i18n::tr_format!("Analyzed {} photos", t.done)
                };
                *t = AiTask::default();
                app.toast(ctx, msg);
                return;
            }
        }
    }
    let (done, total, stopping) = (app.ai_task.done, app.ai_task.total, app.ai_task.stopping);
    let t = Tokens::get(ctx);
    let mut cancel = false;
    egui::Window::new(crate::i18n::tr("Analyzing"))
        .title_bar(false)
        .resizable(false)
        .anchor(Align2::LEFT_BOTTOM, [16.0, -136.0])
        .fixed_size([300.0, 48.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(crate::i18n::tr_format!("Analyzing photos: {done} of {total}", done = done, total = total)).color(t.text),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::widgets::text_button(ui, "aiCancel", if stopping { "Stopping…" } else { "Cancel" }, false).clicked() {
                        cancel = true;
                    }
                });
            });
            ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).desired_width(280.0));
        });
    if cancel && !stopping {
        app.ai_task.stopping = true;
        let _ = app.session.execute("ai.cancel", &json!({}));
    }
    ctx.request_repaint_after(std::time::Duration::from_millis((POLL_SECS * 1000.0) as u64));
}
