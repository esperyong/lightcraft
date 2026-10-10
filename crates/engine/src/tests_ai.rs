//! AI analysis (`ai.*`) with a stand-in model: its labels and score come from the picture's brightness, which is
//! enough to test the queue, the workers, the catalog and the commands anywhere (Apple's Vision itself is tested on a
//! Mac, in `lightcraft-vision`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lightcraft_catalog::{AiAnalysis, AiLabel, Filter, PhotoId, Sort, SortKey};
use serde_json::{Value, json};

use crate::Session;

/// A demo session whose model calls a picture `bright` or `dark` and scores it by its mean brightness (0..100);
/// counts its calls.
fn session() -> (Session, Arc<AtomicUsize>) {
    let mut s = Session::with_demo();
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    s.ai.analyzer = Some(Arc::new(move |img: &lightcraft_raster::Rgba8| {
        c.fetch_add(1, Ordering::SeqCst);
        if img.data.is_empty() {
            return Err("an empty picture".into());
        }
        let mean = img.data.iter().map(|p| (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) as f32 / 765.0).sum::<f32>() / img.data.len() as f32;
        let name = if mean > 0.4 { "bright" } else { "dark" };
        Ok(AiAnalysis {
            labels: vec![AiLabel { name: name.into(), confidence: 0.9 }, AiLabel { name: "photo".into(), confidence: 0.5 }],
            aesthetic: Some(mean * 100.0),
            utility: false,
            model: "test".into(),
        })
    }));
    (s, calls)
}

fn all(s: &Session) -> Vec<PhotoId> {
    s.catalog.query(&Filter::default(), &Sort::default())
}

#[test]
fn analyze_waits_stores_and_makes_photos_searchable() {
    let (mut s, calls) = session();
    let ids = all(&s);
    assert!(ids.len() >= 4, "the demo library has photos");
    let r = s.execute("ai.analyze", &json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>()})).unwrap();
    assert_eq!(r["queued"], json!(ids.len()), "{r}");
    assert_eq!(r["done"], json!(ids.len()), "{r}");
    assert_eq!(r["pending"], json!(0), "{r}");
    assert_eq!(r["photos"].as_array().map(Vec::len), Some(ids.len()), "{r}");
    assert_eq!(calls.load(Ordering::SeqCst), ids.len());
    for id in &ids {
        let p = s.catalog.photo(*id).unwrap();
        let a = p.ai.as_ref().unwrap();
        assert_eq!(a.model, "test");
        assert!(a.aesthetic.is_some_and(|v| (0.0..=100.0).contains(&v)));
    }
    // not an undo step: it is the model's bookkeeping, not an edit
    assert!(s.undo.last().is_none_or(|u| !u.label.contains("AI")));
    // searchable, sortable, and in the photo summaries
    let q = |s: &Session, text: &str| s.catalog.query(&Filter { text: text.into(), ..Default::default() }, &Sort::default()).len();
    assert_eq!(q(&s, "ai:photo"), ids.len());
    assert_eq!(q(&s, "ai:bright") + q(&s, "ai:dark"), ids.len());
    assert_eq!(q(&s, "aesthetic:>-1"), ids.len());
    assert_eq!(q(&s, "analyzed:no"), 0);
    let best = s.catalog.query(&Filter::default(), &Sort { key: SortKey::Aesthetic, ascending: false, ..Default::default() });
    let score = |id: &PhotoId| s.catalog.photo(*id).and_then(|p| p.ai.as_ref()?.aesthetic).unwrap();
    assert!(best.windows(2).all(|w| score(&w[0]) >= score(&w[1])), "best first");
    let listed = s.execute("catalog.query", &json!({"filter": {"text": "ai:photo"}, "sort": {"key": "aesthetic"}, "limit": 2})).unwrap();
    let first = &listed["photos"][0];
    assert!(first["aesthetic"].is_number() && first["aiLabels"].as_array().is_some_and(|l| !l.is_empty()), "{listed}");
    // photos analyzed before are skipped unless forced
    let again = s.execute("ai.analyze", &json!({"ids": [ids[0].0]})).unwrap();
    assert_eq!((again["queued"].clone(), again["skipped"].clone()), (json!(0), json!(1)), "{again}");
    assert_eq!(calls.load(Ordering::SeqCst), ids.len());
    let forced = s.execute("ai.analyze", &json!({"ids": [ids[0].0], "force": true})).unwrap();
    assert_eq!(forced["queued"], json!(1), "{forced}");
    assert_eq!(calls.load(Ordering::SeqCst), ids.len() + 1);
}

#[test]
fn without_ids_the_view_is_analyzed_and_failures_are_reported() {
    let (mut s, _) = session();
    let n = s.visible_cloned().len();
    // a model that fails on every second call
    let k = Arc::new(AtomicUsize::new(0));
    let ok = s.ai.analyzer.clone().unwrap();
    s.ai.analyzer = Some(Arc::new(
        move |img: &lightcraft_raster::Rgba8| {
            if k.fetch_add(1, Ordering::SeqCst) % 2 == 1 { Err("the model said no".into()) } else { ok(img) }
        },
    ));
    let r = s.execute("ai.analyze", &json!({})).unwrap();
    let failed = r["failed"].as_array().map_or(0, Vec::len);
    assert_eq!(r["queued"], json!(n), "{r}");
    assert_eq!(r["done"].as_u64().unwrap() as usize + failed, n, "{r}");
    assert!(failed > 0 && r["failed"][0]["error"].as_str().unwrap().contains("said no"), "{r}");
    // the failed ones are still unanalyzed: a second run takes only them
    let again = s.execute("ai.analyze", &json!({})).unwrap();
    assert_eq!(again["queued"], json!(failed), "{again}");
}

#[test]
fn background_mode_returns_at_once_and_status_collects() {
    let (mut s, _) = session();
    s.ai.background = true;
    let ids = all(&s);
    let r = s.execute("ai.analyze", &json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>()})).unwrap();
    assert_eq!(r["queued"], json!(ids.len()), "{r}");
    assert!(r.get("photos").is_none(), "the app is not made to wait: {r}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut seen = 0;
    loop {
        let st = s.execute("ai.status", &json!({})).unwrap();
        seen += st["photos"].as_array().map_or(0, Vec::len);
        if st["pending"] == json!(0) {
            assert_eq!(st["done"], json!(ids.len()), "{st}");
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the analysis never finished: {st}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(seen, ids.len(), "every finished photo is reported once");
    assert!(!s.ai.busy());
}

#[test]
fn a_budget_returns_early_and_cancel_drops_the_rest() {
    let (mut s, _) = session();
    // a slow model: the budget runs out first
    let ok = s.ai.analyzer.clone().unwrap();
    s.ai.analyzer = Some(Arc::new(move |img: &lightcraft_raster::Rgba8| {
        std::thread::sleep(std::time::Duration::from_millis(150));
        ok(img)
    }));
    let ids = all(&s);
    let r = s.execute("ai.analyze", &json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>(), "budgetMs": 1})).unwrap();
    assert!(r["pending"].as_u64().unwrap() > 0, "{r}");
    let c = s.execute("ai.cancel", &json!({})).unwrap();
    let dropped = c["cancelled"].as_u64().unwrap() as usize;
    // what was running finishes and is stored; nothing else starts
    let r = s.execute("ai.analyze", &json!({"ids": [], "budgetMs": 30_000})).unwrap();
    assert_eq!(r["pending"], json!(0), "{r}");
    let analyzed = ids.iter().filter(|id| s.catalog.photo(**id).is_some_and(|p| p.ai.is_some())).count();
    assert_eq!(analyzed + dropped, ids.len(), "{r}");
}

#[test]
fn without_a_model_the_commands_say_why() {
    let mut s = Session::with_demo();
    s.ai.analyzer = None;
    let e = s.execute("ai.analyze", &json!({})).unwrap_err().to_string();
    assert!(e.contains("Vision"), "{e}");
    let st: Value = s.execute("ai.status", &json!({})).unwrap();
    assert_eq!(st["available"], json!(false));
    #[cfg(not(target_os = "macos"))]
    {
        // off macOS there is no model by default, and the command is not in the menus
        assert!(Session::with_demo().ai.analyzer.is_none());
        let spec = crate::find_command("ai.analyze").unwrap();
        assert!(spec.menu.is_empty());
    }
}

/// A stand-in for Vision's subject: a disc in the middle of whatever picture it is given (or nothing).
fn disc_finder(found: bool) -> crate::subject::SubjectFinder {
    Arc::new(move |img: &lightcraft_raster::Rgba8| {
        if !found {
            return Ok(None);
        }
        let (w, h) = (img.width, img.height);
        let r = w.min(h) as f32 * 0.25;
        let alpha = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - w as f32 / 2.0, (i / w) as f32 - h as f32 / 2.0);
                if (x * x + y * y).sqrt() < r { 1.0 } else { 0.0 }
            })
            .collect();
        Ok(Some(lightcraft_vision::Matte { width: w, height: h, alpha }))
    })
}

fn component_shape(s: &mut Session, mask: u64, comp: usize) -> lightcraft_develop::MaskShape {
    let id = s.active().unwrap();
    let d = s.develop_of(id).unwrap();
    d.masks.iter().find(|m| u64::from(m.id) == mask).unwrap().components[comp].shape.clone()
}

/// The stored subject's coverage at a normalized point.
fn coverage(shape: &lightcraft_develop::MaskShape, x: f64, y: f64) -> f32 {
    use lightcraft_develop::MaskShape;
    let (MaskShape::Subject { seg: Some(seg) } | MaskShape::Background { seg: Some(seg) }) = shape else { panic!("no segmentation: {shape:?}") };
    let side = seg.side as usize;
    let l = seg.logits().unwrap();
    let (i, j) = (((x * side as f64) as usize).min(side - 1), ((y * side as f64) as usize).min(side - 1));
    1.0 / (1.0 + (-l[j * side + i]).exp())
}

#[test]
fn select_subject_stores_visions_subject_and_background_its_inverse() {
    let mut s = Session::with_demo();
    s.ai.subject.finder = Some(disc_finder(true));
    let first = all(&s)[0];
    s.execute("library.select", &json!({"ids": [first.0]})).unwrap();
    let r = s.execute("mask.add", &json!({"kind": "subject"})).unwrap();
    let mask = r["activeMask"].as_u64().unwrap();
    let subject = component_shape(&mut s, mask, 0);
    assert!(coverage(&subject, 0.5, 0.5) > 0.99 && coverage(&subject, 0.02, 0.02) < 0.01, "{subject:?}");
    s.execute("mask.addComponent", &json!({"id": mask, "op": "subtract", "kind": "background"})).unwrap();
    let background = component_shape(&mut s, mask, 1);
    assert!(coverage(&background, 0.5, 0.5) < 0.01 && coverage(&background, 0.02, 0.02) > 0.99);
    // it renders (the segmentation lives in normalized coordinates: any size)
    for edge in [48, 160] {
        let job = s.render_job(first, edge, edge, false, true).unwrap();
        assert!(job.run().rendered.is_ok());
    }
    // no subject: the estimate, and a note saying so
    s.ai.subject.finder = Some(disc_finder(false));
    let r = s.execute("mask.add", &json!({"kind": "subject"})).unwrap();
    assert!(r["note"].as_str().is_some_and(|n| n.contains("no subject")), "{r}");
    let shape = component_shape(&mut s, r["activeMask"].as_u64().unwrap(), 0);
    assert_eq!(shape, lightcraft_develop::MaskShape::Subject { seg: None });
    // no finder (other systems): the estimate, silently
    s.ai.subject.finder = None;
    let r = s.execute("mask.add", &json!({"kind": "background"})).unwrap();
    assert!(r.get("note").is_none() && r.get("pending").is_none(), "{r}");
}

#[test]
fn in_the_app_the_subject_arrives_after_the_mask() {
    let mut s = Session::with_demo();
    s.segmenter.background = true;
    s.ai.subject.finder = Some(disc_finder(true));
    let first = all(&s)[0];
    s.execute("library.select", &json!({"ids": [first.0]})).unwrap();
    let r = s.execute("mask.add", &json!({"kind": "subject"})).unwrap();
    assert_eq!(r["pending"], json!(true), "{r}");
    let mask = r["activeMask"].as_u64().unwrap();
    assert_eq!(component_shape(&mut s, mask, 0), lightcraft_develop::MaskShape::Subject { seg: None }, "the estimate first");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let polled = s.segment_poll();
        assert!(polled.messages.is_empty(), "{:?}", polled.messages);
        if polled.changed {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the subject never arrived");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(coverage(&component_shape(&mut s, mask, 0), 0.5, 0.5) > 0.99);
    assert!(!s.ai.subject.busy());
    // one undo step brings the estimate back, another removes the mask
    s.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(component_shape(&mut s, mask, 0), lightcraft_develop::MaskShape::Subject { seg: None });
}

#[test]
fn labels_become_keywords_on_request_in_one_undo_step() {
    let (mut s, _) = session();
    let ids = all(&s);
    let list = json!(ids.iter().map(|i| i.0).collect::<Vec<_>>());
    s.execute("ai.analyze", &json!({"ids": list})).unwrap();
    // the analysis itself touches no keywords
    assert!(ids.iter().all(|id| s.catalog.photo(*id).unwrap().meta.keywords.iter().all(|k| !k.starts_with("AI"))));
    let r = s.execute("ai.keywords", &json!({"ids": list})).unwrap();
    assert_eq!(r["changed"], json!(ids.len()), "{r}");
    let kw = |s: &Session, id: PhotoId| s.catalog.photo(id).unwrap().meta.keywords.clone();
    let first = kw(&s, ids[0]);
    assert!(first.iter().any(|k| k == "AI|photo") && first.iter().any(|k| k == "AI|bright" || k == "AI|dark"), "{first:?}");
    let q = |s: &Session, f: Filter| s.catalog.query(&f, &Sort::default()).len();
    assert_eq!(q(&s, Filter { keyword: Some("AI".into()), ..Default::default() }), ids.len());
    // again: nothing new; a higher bar with replace keeps only the confident label
    assert_eq!(s.execute("ai.keywords", &json!({"ids": list})).unwrap()["changed"], json!(0));
    s.execute("ai.keywords", &json!({"ids": list, "minConfidence": 0.8, "replace": true})).unwrap();
    assert!(!kw(&s, ids[0]).iter().any(|k| k == "AI|photo"));
    // one undo step per command
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(kw(&s, ids[0]).iter().any(|k| k == "AI|photo"));
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(kw(&s, ids[0]).iter().all(|k| !k.starts_with("AI")));
}

/// The model sees each photo at `ai::EDGE` px (the grid's 512 px thumbnails gave Vision unstable aesthetics scores), and a
/// pass leaves the caches the views use as they were.
#[test]
fn photos_are_analyzed_at_1024_px_without_filling_the_view_caches() {
    let mut s = Session::with_demo();
    let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = sizes.clone();
    s.ai.analyzer = Some(Arc::new(move |img: &lightcraft_raster::Rgba8| {
        seen.lock().unwrap().push((img.width, img.height));
        Ok(AiAnalysis { labels: vec![], aesthetic: Some(50.0), utility: false, model: "test".into() })
    }));
    let ids = all(&s);
    let before = s.memory_report();
    s.execute("ai.analyze", &json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>()})).unwrap();
    let sizes = sizes.lock().unwrap().clone();
    assert_eq!(sizes.len(), ids.len());
    assert!(sizes.iter().any(|(w, h)| (*w).max(*h) > 512), "larger than a grid thumbnail: {sizes:?}");
    for (id, (w, h)) in ids.iter().zip(&sizes) {
        let p = s.catalog.photo(*id).unwrap();
        // fitted within 1,024 px, as large as the photo allows
        let expected = (p.width.max(p.height) as usize).min(crate::ai::EDGE);
        assert!((*w).max(*h).abs_diff(expected) <= 1, "{id:?}: {w} × {h}, expected a long edge of {expected}");
    }
    let after = s.memory_report();
    assert_eq!(after.rendered.bytes, before.rendered.bytes, "no render kept");
    assert_eq!(after.preview_sources.bytes, before.preview_sources.bytes, "no source kept");
    assert_eq!(after.thumb_sources.bytes, before.thumb_sources.bytes);
}
