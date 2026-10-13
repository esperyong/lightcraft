//! Photo Merge commands: `merge.hdr`, `merge.panorama`, `merge.hdrPanorama`, `merge.focusStack`,
//! and for many focus brackets at once `merge.detectFocusBrackets` / `merge.focusStackBatch`.
//!
//! Run synchronously here (CLI, MCP, control channel); the desktop UI plans the same job with
//! [`Session::plan_merge`] and runs it on a worker thread. `preview: true` merges ≤ 1024 px frames
//! and (with `previewPath`) writes the rendered preview as PNG instead of adding a photo.

use lightcraft_catalog::stacks::BracketRule;
use lightcraft_catalog::{PhotoId, StackKind};
use serde_json::{Value, json};

use super::{CommandSpec, bad, bool_or, cmd, str_param};
use crate::merge::MergeKind;
use crate::stack_scope::Members;
use crate::{Result, Session};

fn can_merge(s: &Session) -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") && s.media.file_bytes.is_none() {
        return Err("Photo Merge needs access to the photo files".into());
    }
    // the photo count is checked when run (an explicit `ids` parameter may name them)
    if s.catalog.photos().nth(1).is_none() { Err("the library has fewer than 2 photos".into()) } else { Ok(()) }
}

/// Bracket detection reads no files (only the catalog), so it also works where merges can't.
fn has_photos(s: &Session) -> std::result::Result<(), String> {
    if s.catalog.photos().nth(1).is_none() { Err("the library has fewer than 2 photos".into()) } else { Ok(()) }
}

fn run(s: &mut Session, id: &str, p: &Value) -> Result<Value> {
    let (kind, finish) = crate::merge::parse(id, p)?;
    let scope = s.stack_targets(p, Members::Sources);
    // two or more collapsed stacks (focus or exposure brackets): one merge each; panoramas join them
    let per_stack = scope.per_stack() && matches!(kind, MergeKind::FocusStack { .. } | MergeKind::Hdr { .. });
    if !bool_or(p, "preview", false) {
        if per_stack {
            let mut r = s.merge_groups(&kind, &finish, &scope.groups());
            r["perStack"] = json!(true);
            return Ok(r);
        }
        return s.merge_now(kind, finish, &scope.ids);
    }
    // a preview of several stacks shows the first
    let ids = if per_stack { scope.groups().into_iter().next().unwrap_or_default() } else { scope.ids.clone() };
    let job = s.plan_merge(kind, finish, &ids, true)?;
    let out = job.run(&|_, _| true).map_err(|e| bad(id, e))?;
    let mut info = out.info;
    if per_stack {
        info["stacks"] = json!(scope.stacks.len());
    }
    if let (Some(path), Some(img)) = (str_param(p, "previewPath"), &out.preview) {
        let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default())
            .map_err(|e| bad(id, e.to_string()))?;
        s.check_write_target(path).map_err(|e| bad(id, e))?;
        crate::export::write_file(path, &png).map_err(|e| bad(id, e))?;
        info["previewPath"] = json!(path);
    }
    if let Some(img) = &out.preview {
        info["previewSize"] = json!([img.width, img.height]);
    }
    Ok(info)
}

/// The bracket rule from `{gapSecs=2, minFrames=3, sameExposure=true}`.
pub fn bracket_rule(p: &Value) -> BracketRule {
    let d = BracketRule::default();
    BracketRule {
        gap_secs: p.get("gapSecs").and_then(Value::as_f64).filter(|g| g.is_finite() && *g >= 0.0).unwrap_or(d.gap_secs),
        min_frames: p.get("minFrames").and_then(Value::as_u64).map_or(d.min_frames, |m| (m as usize).clamp(2, 10_000)),
        same_exposure: bool_or(p, "sameExposure", d.same_exposure),
        ..d
    }
}

/// Focus brackets among the photos (see [`BracketRule`]) — the selection or the photos shown,
/// members of collapsed stacks included; `stack: true` makes each a focus-bracket stack of its own
/// (one undo step).
fn detect(s: &mut Session, p: &Value) -> Result<Value> {
    let (scope, _) = s.bracket_scope(p, Members::All);
    let groups = s.catalog.focus_bracket_groups(&scope.ids, &bracket_rule(p));
    let (mut stacked, mut unchanged) = (0, 0);
    if bool_or(p, "stack", false) {
        for g in &groups {
            match s.catalog.restack_ops(g, StackKind::FocusBracket) {
                Some(op) => {
                    s.commit("Stack Focus Bracket", op)?;
                    stacked += 1;
                }
                None => unchanged += 1,
            }
        }
        s.merge_undo(stacked, "Stack Focus Brackets");
        if stacked > 0 {
            s.keep_active_visible();
        }
    }
    Ok(json!({
        "groups": groups.iter().map(|g| g.iter().map(|id| id.0).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "count": groups.len(),
        "photos": groups.iter().map(Vec::len).sum::<usize>(),
        "stacked": stacked,
        "unchanged": unchanged,
    }))
}

/// Focus-stack each bracket (`groups`, else [`Session::focus_batch_groups`]) with the same options,
/// one after the other; a bracket that fails is reported and the others still run.
fn batch(s: &mut Session, p: &Value) -> Result<Value> {
    let groups: Vec<Vec<PhotoId>> = match p.get("groups").and_then(Value::as_array) {
        Some(a) => a.iter().map(|g| g.as_array().map(|v| v.iter().filter_map(Value::as_u64).map(PhotoId).collect()).unwrap_or_default()).collect(),
        None => s.focus_batch_groups(p, &bracket_rule(p)),
    };
    // one depth PNG path would be overwritten by every bracket
    let mut params = p.clone();
    if let Some(o) = params.as_object_mut() {
        o.remove("depthPath");
        o.remove("preview");
    }
    let (kind, finish) = crate::merge::parse("merge.focusStack", &params)?;
    Ok(s.merge_groups(&kind, &finish, &groups))
}

/// The strokes parameter of `merge.focusRetouch`.
fn strokes(p: &Value) -> Result<Vec<crate::focus_retouch::RetouchStroke>> {
    serde_json::from_value(p.get("strokes").cloned().unwrap_or(json!([]))).map_err(|e| bad("merge.focusRetouch", format!("strokes: {e}")))
}

/// Retouch a focus stack: preview (a PNG) or the full-size DNG over the stack's own file.
fn retouch(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.targets(p).first().copied().ok_or_else(|| bad("merge.focusRetouch", "no photo"))?;
    let strokes = strokes(p)?;
    if bool_or(p, "preview", false) {
        let edge = p.get("edge").and_then(Value::as_u64).map_or(1600, |e| (e as usize).clamp(64, 4096));
        let mut r = s.open_retouch(id, edge)?;
        let img = r.render(&strokes).map_err(|e| bad("merge.focusRetouch", e))?;
        let mut out = json!({"previewSize": [img.width, img.height], "strokes": strokes.len()});
        if let Some(path) = str_param(p, "previewPath") {
            let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default())
                .map_err(|e| bad("merge.focusRetouch", e.to_string()))?;
            s.check_write_target(path).map_err(|e| bad("merge.focusRetouch", e))?;
            crate::export::write_file(path, &png).map_err(|e| bad("merge.focusRetouch", e))?;
            out["previewPath"] = json!(path);
        }
        return Ok(out);
    }
    let job = s.plan_retouch(id, strokes)?;
    let dng = job.run(&|_, _| true).map_err(|e| bad("merge.focusRetouch", e))?;
    s.finish_retouch(&job, dng)
}

/// A focus stack's project: its frames, options and strokes.
fn project(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.targets(p).first().copied().ok_or_else(|| bad("merge.focusProject", "no photo"))?;
    let Some(lightcraft_catalog::Source::File { path }) = s.catalog.photo(id).map(|ph| ph.source.clone()) else {
        return Err(bad("merge.focusProject", "not a file"));
    };
    let pr = crate::focus_retouch::read_project(&path).ok_or_else(|| bad("merge.focusProject", "this photo has no focus-stack project"))?;
    Ok(json!({
        "id": id.0,
        "frames": pr.inputs.iter().map(|f| std::path::Path::new(f).file_name().and_then(|n| n.to_str()).unwrap_or(f)).collect::<Vec<_>>(),
        "ids": pr.ids,
        "method": pr.options.method,
        "radius": pr.options.radius,
        "smoothing": pr.options.smoothing,
        "reference": pr.reference,
        "strokes": pr.strokes,
    }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "merge.hdr",
            "HDR Merge",
            [],
            None,
            "{ids? (default: the selection, a collapsed stack standing for its photos; two or more collapsed stacks are merged one by one → {results, failed, count, perStack}), expandStacks=false (explicit ids: take a collapsed stack's photos too), align=true, deghost=none|low|medium|high, autoSettings (default: on unless the result takes edited sources' settings), sourceSettings=true (when the photos were edited, the result starts from the reference photo's settings: white balance, tone, colour, detail, lens corrections, profile — not crop, geometry, masks, spots, lens blur or AI enhance), stack (default: when the photos are one stack, the result goes on its top), preview=false, showOverlay=false, previewPath?}",
            can_merge,
            |s, p| run(s, "merge.hdr", p)
        ),
        cmd!(
            "merge.panorama",
            "Panorama Merge",
            [],
            None,
            "{ids? (default: the selection, a collapsed stack standing for its photos), expandStacks=false, projection=auto|spherical|cylindrical|perspective, boundaryWarp=0..100, autoCrop=false, fillEdges=false, autoSettings (default: on unless the result takes edited sources' settings), sourceSettings=true (lens geometry and vignetting stay the panorama's own), stack (default: when the photos are one stack), maxMegapixels=40, preview=false, previewPath?}",
            can_merge,
            |s, p| run(s, "merge.panorama", p)
        ),
        cmd!(
            "merge.hdrPanorama",
            "HDR Panorama Merge",
            [],
            None,
            "{ids? (default: the selection, collapsed stacks standing for their photos), expandStacks=false, bracket=0 (auto from EXIF), align, deghost, projection, boundaryWarp, autoCrop, fillEdges, autoSettings (default: on unless the result takes edited sources' settings), sourceSettings=true, stack (default: when the photos are one stack), preview, previewPath?}",
            can_merge,
            |s, p| run(s, "merge.hdrPanorama", p)
        ),
        cmd!(
            "merge.focusStack",
            "Focus Stack",
            [],
            None,
            "{ids? (default: the selection, a collapsed stack standing for its photos — merge results, virtual copies and videos left out; two or more collapsed stacks are stacked one by one → {results, failed, count, perStack}), expandStacks=false (explicit ids: take a collapsed stack's photos too), method=depth|average|pyramid (a|b|c; depth), radius=1..30 (8), smoothing=0..10 (4), align=true, alignShift, alignRotation, alignScale, alignBrightness (default: align), order=capture|selection|reverse|auto, every=1 (every n-th photo, first and last always; previews ~12 photos), reference=middle|first|sharpest, crop=common|reference, maxMegapixels=0 (no limit), depthMap=true (embed the depth map in the DNG for Depth Range masks), depthPath? (also as a 16-bit PNG), autoSettings (default: on unless the result takes edited sources' settings), sourceSettings=true (when the photos were edited, the result starts from the reference photo's settings: white balance, tone, colour, detail, lens corrections, profile — not crop, geometry, masks, spots, lens blur or AI enhance), stack (default: when the photos are one stack, e.g. a focus bracket, the result goes on its top), preview=false, showDepth=false, previewPath?}",
            can_merge,
            |s, p| run(s, "merge.focusStack", p)
        ),
        cmd!(
            "merge.focusProject",
            "Focus Stack Project",
            [],
            None,
            "{id?} → {frames, ids, method, radius, smoothing, reference, strokes}: what a focus stack made by LightCraft was made from (saved as <dng>.lcfocus.json)",
            can_merge,
            project
        ),
        cmd!(
            "merge.focusRetouch",
            "Retouch Focus Stack",
            [],
            None,
            "{id?, strokes: [{source: \"base\"|\"a\"|\"b\"|\"c\"|frame index (0-based, focus order), points: [[x, y]…] (0..1 of the result), size (radius, fraction of the long edge), hardness=0.5, opacity=1}], preview=false, previewPath?, edge=1600} — paint pixels from a frame or another method over the stack. Preview: a PNG with the photo's settings; else the DNG is redone at full size over the stack's own file (the un-retouched one kept as <dng>.lcfocus-base; the strokes replace earlier ones, [] removes them) → {id, path, strokes}",
            can_merge,
            retouch
        ),
        cmd!(
            "merge.detectFocusBrackets",
            "Detect Focus Brackets",
            [],
            None,
            "{ids? (default: the selection of two or more photos, else the photos shown; members of collapsed stacks included), gapSecs=2 (most seconds between frames), minFrames=3, sameExposure=true, stack=false (make each bracket a collapsed focus-bracket stack of its own, taking its photos out of other stacks; one undo step)} → {groups: [[ids]], count, photos, stacked (stacks made), unchanged (already stacked)}. Frames of a bracket share camera, lens, focal length and (by default) exposure; merge results and virtual copies are left out",
            has_photos,
            detect
        ),
        cmd!(
            "merge.focusStackBatch",
            "Focus Stack Each Bracket",
            [],
            None,
            "{groups? [[ids]] (default: each selected collapsed stack — with no selection, each focus-bracket stack shown — and the brackets merge.detectFocusBrackets finds among the other photos, with gapSecs/minFrames/sameExposure), …merge.focusStack options} → {results: [{group, id, path, …}], failed: [{group, error}], count}. Runs the brackets one after the other",
            can_merge,
            batch
        ),
    ]
}
