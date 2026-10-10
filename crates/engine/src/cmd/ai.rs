//! AI analysis (`ai.*`): labels and an aesthetic score per photo from Apple's Vision framework (see [`crate::ai`]).

use std::time::{Duration, Instant};

use lightcraft_catalog::{Photo, PhotoId};
use serde_json::{Value, json};

use super::{CommandSpec, bool_or, cmd};
use crate::{Result, Session};

/// In the Photo menu where the model exists (a Mac); elsewhere the commands stay, disabled, for scripts to ask about.
const MENU: &[&str] = if cfg!(target_os = "macos") { &["Photo"] } else { &[] };
/// Labels a result lists (the catalog keeps more).
const SHOWN_LABELS: usize = 8;
/// Longest a waiting `ai.analyze` waits without a `budgetMs`.
const MAX_WAIT: Duration = Duration::from_secs(6 * 3600);

fn available(s: &Session) -> std::result::Result<(), String> {
    if s.ai.analyzer.is_some() { Ok(()) } else { Err(crate::ai::UNAVAILABLE.into()) }
}

/// A photo's analysis as commands report it.
pub(crate) fn ai_summary(p: &Photo) -> Value {
    match &p.ai {
        Some(a) => json!({
            "id": p.id.0,
            "labels": a.labels.iter().take(SHOWN_LABELS).map(|l| json!({"name": l.name, "confidence": (l.confidence * 1000.0).round() / 1000.0})).collect::<Vec<_>>(),
            "aesthetic": a.aesthetic.map(|v| (v * 10.0).round() / 10.0),
            "utility": a.utility,
        }),
        None => json!({"id": p.id.0}),
    }
}

fn finished(s: &mut Session) -> Vec<Value> {
    let ids = s.ai_take_finished();
    ids.iter().filter_map(|id| s.catalog.photo(*id)).map(|p| ai_summary(p)).collect()
}

fn progress(s: &Session) -> Value {
    let p = s.ai.progress();
    json!({
        "done": p.done,
        "total": p.total,
        "running": p.running,
        "queued": p.queued,
        "pending": p.running + p.queued,
        "failed": p.failed.iter().map(|(id, why)| json!({"id": id, "error": why})).collect::<Vec<_>>(),
    })
}

fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let ids: Vec<PhotoId> =
        if p.get("ids").is_some() || p.get("id").is_some() || s.selection.ids.len() > 1 { s.targets(p) } else { s.visible_cloned() };
    let queued = s.ai_enqueue(&ids, bool_or(p, "force", false));
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(!s.ai.background);
    if wait {
        let budget = p.get("budgetMs").and_then(Value::as_u64).map_or(MAX_WAIT, Duration::from_millis).min(MAX_WAIT);
        s.ai_wait(Instant::now() + budget);
    } else {
        s.ai_pump();
    }
    let mut out = progress(s);
    out["queued"] = json!(queued);
    out["skipped"] = json!(ids.len() - queued);
    if wait {
        out["photos"] = json!(finished(s));
    }
    Ok(out)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.analyze",
            label: "Analyze with Apple Vision",
            menu: MENU,
            shortcut: None,
            params: "{ids?, force?: bool (also photos analyzed before), wait?: bool, budgetMs?} — what each photo shows (labels, e.g. dog, beach, sports_car) and how good it looks (aesthetic 0..100, macOS 15+), from Apple's Vision framework on the Mac; the selected photos (else all in view; photos analyzed before are skipped unless `force`). Stored in the catalog: search `ai:dog`, `aesthetic:>70`, `utility:yes`, `analyzed:no`, smart-album fields aesthetic / aiLabels / utility, `library.sort {key: aesthetic}`. Waits (CLI, MCP; at most `budgetMs`, the rest keeps going and the next ai.* command collects it) or, in the app, returns at once → {queued, skipped, done, total, pending, failed: [{id, error}], photos?: [{id, labels: [{name, confidence}], aesthetic, utility}]}",
            enabled: available,
            run: analyze,
            journal: true,
        },
        cmd!(
            query "ai.status",
            "AI Analysis Status",
            [],
            None,
            "{} — stores what has finished and reports progress → {available, features: {labels, aesthetics, subject}, done, total, pending, failed, photos: [what finished since the last report]}",
            super::always,
            |s, _| {
                s.ai_pump();
                let f = lightcraft_vision::features();
                let mut out = progress(s);
                out["available"] = json!(s.ai.analyzer.is_some());
                out["features"] = json!({"labels": f.labels, "aesthetics": f.aesthetics, "subject": f.subject});
                out["photos"] = json!(finished(s));
                Ok(out)
            }
        ),
        cmd!(
            "ai.cancel",
            "Stop AI Analysis",
            [],
            None,
            "{} — drops the photos still waiting (the ones being analyzed finish) → {cancelled}",
            super::always,
            |s, _| {
                let n = s.ai_cancel();
                s.ai_pump();
                Ok(json!({"cancelled": n}))
            }
        ),
    ]
}
