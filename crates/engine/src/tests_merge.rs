//! Photo Merge through the command layer: synthetic brackets / panorama views on disk → merged
//! DNG next to the sources → imported, selected, Auto Settings / auto crop applied.

use serde_json::json;

use crate::Session;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lc-merge-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn import(s: &mut Session, paths: Vec<String>) -> Vec<u64> {
    let r = s.execute("library.import", &json!({"paths": paths})).unwrap();
    r["imported"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()
}

#[test]
fn hdr_merge_command_creates_and_imports_a_dng() {
    let dir = temp_dir("hdr");
    let mut paths = Vec::new();
    for (i, b) in lightcraft_merge::synth::bracket_dngs(480, 320, &[-2.0, 0.0, 2.0]).unwrap().into_iter().enumerate() {
        let p = dir.join(format!("IMG_{i}.dng"));
        std::fs::write(&p, b).unwrap();
        paths.push(p.to_string_lossy().to_string());
    }
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, paths);
    assert_eq!(ids.len(), 3);
    // preview: rendered PNG, nothing added
    let png = dir.join("preview.png");
    let pv = s.execute("merge.hdr", &json!({"ids": ids, "deghost": "medium", "preview": true, "showOverlay": true, "previewPath": png})).unwrap();
    assert!(png.exists() && pv["previewSize"][0].as_u64().unwrap() > 100, "{pv}");
    assert_eq!(s.catalog.photos().count(), 3);
    let ev: Vec<f64> = pv["ev"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((ev[0] + 2.0).abs() < 0.1 && (ev[2] - 2.0).abs() < 0.1, "{ev:?}");
    // full merge
    let r = s.execute("merge.hdr", &json!({"ids": ids, "autoSettings": true, "stack": true})).unwrap();
    let path = r["path"].as_str().unwrap();
    assert!(path.ends_with("IMG_0-HDR.dng"), "{path}");
    let id = lightcraft_catalog::PhotoId(r["id"].as_u64().unwrap());
    let p = s.catalog.photo(id).unwrap().clone();
    assert_eq!((p.width, p.height), (480, 320));
    assert_eq!(p.kind, lightcraft_catalog::MediaKind::Raw);
    assert_eq!(s.selection.active, Some(id));
    let d = s.develop_of(id).unwrap();
    assert!(d.light.exposure != 0.0 || d.light.highlights != 0.0, "auto settings applied");
    let img = s.render_now(id, 256, 256).unwrap();
    assert_eq!(img.image.width, 256);
    // Create Stack: the result tops a collapsed stack of its sources, so the grid shows the result
    let st = s.catalog.stack_of(id).expect("merge result stacked").clone();
    assert_eq!(st.top(), id);
    assert_eq!(st.photos[1..].iter().map(|p| p.0).collect::<Vec<_>>(), ids);
    assert!(st.collapsed);
    assert_eq!(s.visible_cloned(), vec![id]);
    // the second merge doesn't overwrite the first (its sources are one stack now: asked not to stack)
    let r2 = s.execute("merge.hdr", &json!({"ids": ids, "autoSettings": false, "stack": false})).unwrap();
    assert!(r2["path"].as_str().unwrap().ends_with("IMG_0-HDR-2.dng"));
    let id2 = lightcraft_catalog::PhotoId(r2["id"].as_u64().unwrap());
    assert!(s.catalog.stack_of(id2).is_none(), "no stack without the option");
    // the type filter finds merge results (and only them)
    s.execute("library.filter", &json!({"merged": "hdr"})).unwrap();
    let found: std::collections::HashSet<_> = s.catalog.query(&s.filter, &Default::default()).into_iter().collect();
    assert_eq!(found, [id, id2].into_iter().collect());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn panorama_command_with_auto_crop() {
    let dir = temp_dir("pano");
    let mut paths = Vec::new();
    for (i, v) in lightcraft_merge::synth::pano_views(480, 360, 420.0, &[-25.0, 0.0, 25.0]).unwrap().into_iter().enumerate() {
        let img = v.to_srgb8();
        let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
        let p = dir.join(format!("P_{i}.png"));
        std::fs::write(&p, png).unwrap();
        paths.push(p.to_string_lossy().to_string());
    }
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, paths);
    let r = s.execute("merge.panorama", &json!({"ids": ids, "projection": "cylindrical", "autoCrop": true, "autoSettings": false})).unwrap();
    assert_eq!(r["projection"], "cylindrical");
    assert_eq!(r["used"].as_array().unwrap().len(), 3);
    let id = lightcraft_catalog::PhotoId(r["id"].as_u64().unwrap());
    let p = s.catalog.photo(id).unwrap().clone();
    assert!(p.width > 800 && p.height > 300, "{}×{}", p.width, p.height);
    let crop = s.develop_of(id).unwrap().crop.geometry.rect;
    assert!(crop.width() < 1.0 && crop.width() * crop.height() > 0.5, "{crop:?}");
    // too few photos
    assert!(s.execute("merge.panorama", &json!({"ids": [ids[0]]})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn focus_stack_command_streams_a_bracket_into_a_dng() {
    let dir = temp_dir("focus");
    let n = 6;
    let mut paths = Vec::new();
    // file names run against the capture order: the merge must follow the capture times
    for (i, b) in lightcraft_merge::synth::focus_bracket_dngs(480, 320, n, 0).unwrap().into_iter().enumerate() {
        let p = dir.join(format!("F_{}.dng", n - 1 - i));
        std::fs::write(&p, b).unwrap();
        paths.push(p.to_string_lossy().to_string());
    }
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, paths);
    assert_eq!(ids.len(), n);
    // preview of the depth map: a PNG, nothing added
    let png = dir.join("depth-preview.png");
    let pv = s.execute("merge.focusStack", &json!({"ids": ids, "preview": true, "showDepth": true, "previewPath": png})).unwrap();
    assert!(png.exists(), "{pv}");
    assert_eq!(s.catalog.photos().count(), n);
    // the full stack, method B, with the depth map written next to it
    let depth = dir.join("depth.png");
    let r = s.execute("merge.focusStack", &json!({"ids": ids, "method": "b", "stack": true, "autoSettings": false, "depthPath": depth})).unwrap();
    assert_eq!(r["method"], "depth");
    assert_eq!(r["used"], n);
    assert_eq!(r["order"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>(), ids, "capture order");
    assert!(r["warnings"].as_array().unwrap().is_empty(), "{r}");
    assert!(r["alignment"]["minNcc"].as_f64().unwrap() > 0.9, "{r}");
    let path = r["path"].as_str().unwrap();
    assert!(path.ends_with("F_5-Focus.dng"), "{path}");
    let id = lightcraft_catalog::PhotoId(r["id"].as_u64().unwrap());
    let p = s.catalog.photo(id).unwrap().clone();
    assert!(p.width > 440 && p.height > 290 && p.width <= 480, "{}×{}", p.width, p.height);
    assert_eq!(s.catalog.stack_of(id).map(|st| st.top()), Some(id));
    let d = lightcraft_codecs::decode(&std::fs::read(&depth).unwrap(), Default::default()).unwrap();
    assert_eq!((d.image.width, d.image.height), (p.width as usize, p.height as usize));
    // the depth map travels in the DNG: the source has a depth matte, and a Depth Range mask
    // selects on it
    let masks = lightcraft_raw::semantic_masks(&std::fs::read(path).unwrap());
    assert_eq!(masks.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), [lightcraft_merge::focus::DEPTH_MASK_NAME]);
    s.render_now(id, 256, 256).unwrap();
    assert!(s.source_info(id).mattes.is_some_and(|m| m.has(lightcraft_pipeline::masks::MatteKind::Depth)));
    s.selection = crate::Selection::single(id);
    s.execute("mask.add", &json!({"kind": "depthRange", "lo": 0.0, "hi": 0.5})).unwrap();
    // every other photo, the pyramid method, size limit
    let r2 =
        s.execute("merge.focusStack", &json!({"ids": ids, "method": "pyramid", "every": 2, "maxMegapixels": 0.05, "autoSettings": false})).unwrap();
    assert_eq!(r2["used"], 4);
    assert!(r2["path"].as_str().unwrap().ends_with("F_5-Focus-2.dng"));
    let id2 = lightcraft_catalog::PhotoId(r2["id"].as_u64().unwrap());
    let p2 = s.catalog.photo(id2).unwrap();
    assert!((p2.width * p2.height) as f64 <= 0.05e6, "{}×{}", p2.width, p2.height);
    // the type filter
    s.execute("library.filter", &json!({"merged": "focus"})).unwrap();
    let found: std::collections::HashSet<_> = s.catalog.query(&s.filter, &Default::default()).into_iter().collect();
    assert_eq!(found, [id, id2].into_iter().collect());
    // bad parameters are errors, not panics
    assert!(s.execute("merge.focusStack", &json!({"ids": ids, "method": "x"})).is_err());
    assert!(s.execute("merge.focusStack", &json!({"ids": [ids[0]]})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn focus_brackets_are_detected_and_stacked_in_a_batch() {
    let dir = temp_dir("focus-batch");
    let mut paths = Vec::new();
    // two brackets of 4, half a minute apart
    for (b, start) in [(0, 0), (1, 30)] {
        for (i, d) in lightcraft_merge::synth::focus_bracket_dngs(240, 160, 4, start).unwrap().into_iter().enumerate() {
            let p = dir.join(format!("B{b}_{i}.dng"));
            std::fs::write(&p, d).unwrap();
            paths.push(p.to_string_lossy().to_string());
        }
    }
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, paths);
    let r = s.execute("merge.detectFocusBrackets", &json!({"ids": ids, "stack": true})).unwrap();
    assert_eq!(r["count"], 2, "{r}");
    assert_eq!(r["groups"][0].as_array().unwrap().len(), 4);
    assert_eq!(r["stacked"], 2);
    let r = s.execute("merge.focusStackBatch", &json!({"ids": ids, "method": "a", "autoSettings": false})).unwrap();
    assert_eq!(r["count"], 2, "{r}");
    assert!(r["failed"].as_array().unwrap().is_empty(), "{r}");
    let paths: Vec<&str> = r["results"].as_array().unwrap().iter().map(|v| v["path"].as_str().unwrap()).collect();
    assert!(paths[0].ends_with("B0_0-Focus.dng") && paths[1].ends_with("B1_0-Focus.dng"), "{paths:?}");
    // the results are not brackets themselves
    let all: Vec<u64> = s.catalog.photos().map(|p| p.id.0).collect();
    assert_eq!(s.execute("merge.detectFocusBrackets", &json!({"ids": all})).unwrap()["count"], 2);
    // a bad group fails alone
    let r = s.execute("merge.focusStackBatch", &json!({"groups": [[ids[0]], [ids[4], ids[5]]], "autoSettings": false})).unwrap();
    assert_eq!((r["results"].as_array().unwrap().len(), r["failed"].as_array().unwrap().len()), (1, 1), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Issue #24: brackets become collapsed focus-bracket stacks; one selected stack is merged whole and
/// takes the result on top; several selected stacks are merged one by one.
#[test]
fn a_selected_collapsed_bracket_is_merged_whole_with_the_result_on_top() {
    use lightcraft_catalog::{PhotoId, StackKind};
    let dir = temp_dir("focus-stacks");
    let mut paths = Vec::new();
    for (b, start) in [(0, 0), (1, 30)] {
        for (i, d) in lightcraft_merge::synth::focus_bracket_dngs(240, 160, 4, start).unwrap().into_iter().enumerate() {
            let p = dir.join(format!("S{b}_{i}.dng"));
            std::fs::write(&p, d).unwrap();
            paths.push(p.to_string_lossy().to_string());
        }
    }
    let mut s = Session::new().with_fs();
    let ids: Vec<PhotoId> = import(&mut s, paths).into_iter().map(PhotoId).collect();
    // a stack by capture time holding both brackets, collapsed
    s.execute("library.selectNone", &json!({})).unwrap();
    s.execute("stack.auto", &json!({"gap": 100})).unwrap();
    assert_eq!(s.visible_cloned().len(), 1);
    // nothing selected: the brackets are found inside it and become stacks of their own
    let r = s.execute("merge.detectFocusBrackets", &json!({"stack": true})).unwrap();
    assert_eq!((r["count"].as_u64(), r["stacked"].as_u64(), r["photos"].as_u64()), (Some(2), Some(2), Some(8)), "{r}");
    let (b0, b1) = (s.catalog.stack_of(ids[0]).unwrap().clone(), s.catalog.stack_of(ids[4]).unwrap().clone());
    assert_eq!((b0.photos.clone(), b0.kind, b0.collapsed), (ids[0..4].to_vec(), StackKind::FocusBracket, true));
    assert_eq!(b1.photos, ids[4..8].to_vec());
    let shown = |s: &mut Session| {
        let mut v = s.visible_cloned();
        v.sort();
        v
    };
    assert_eq!(shown(&mut s), vec![ids[0], ids[4]]);
    // again: nothing new; one undo brings the time stack back
    let r = s.execute("merge.detectFocusBrackets", &json!({"stack": true})).unwrap();
    assert_eq!((r["stacked"].as_u64(), r["unchanged"].as_u64()), (Some(0), Some(2)), "{r}");
    s.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(s.catalog.stack_of(ids[0]).map(|st| st.photos.len()), Some(8));
    s.execute("edit.redo", &json!({})).unwrap();
    assert_eq!(s.catalog.stack_of(ids[0]).map(|st| st.photos.len()), Some(4));
    // one collapsed bracket selected (one cell): all its frames are merged, the result goes on top
    s.execute("library.select", &json!({"ids": [ids[0].0]})).unwrap();
    let r = s.execute("merge.focusStack", &json!({"method": "a", "autoSettings": false})).unwrap();
    assert_eq!(r["used"], 4, "{r}");
    let res = PhotoId(r["id"].as_u64().unwrap());
    let st = s.catalog.stack_of(res).unwrap().clone();
    assert_eq!((st.id, st.kind, st.collapsed), (b0.id, StackKind::FocusBracket, true));
    assert_eq!(st.photos, [vec![res], ids[0..4].to_vec()].concat(), "on top, the frames in order");
    assert_eq!(shown(&mut s), vec![ids[4], res]);
    // the result's cell again: merged from the frames (not from the result), the new one on top
    let r = s.execute("merge.focusStack", &json!({"method": "c", "autoSettings": false})).unwrap();
    assert_eq!(r["used"], 4, "{r}");
    let res2 = PhotoId(r["id"].as_u64().unwrap());
    assert_eq!(s.catalog.stack_of(res2).map(|st| st.photos[..2].to_vec()), Some(vec![res2, res]));
    // two collapsed brackets selected: one merge each, each result on its own stack
    s.execute("library.select", &json!({"ids": [res2.0, ids[4].0]})).unwrap();
    let r = s.execute("merge.focusStack", &json!({"method": "a", "autoSettings": false})).unwrap();
    assert_eq!((r["count"].as_u64(), r["perStack"].as_bool()), (Some(2), Some(true)), "{r}");
    assert_eq!(r["failed"].as_array().map(Vec::len), Some(0), "{r}");
    let made: Vec<PhotoId> = r["results"].as_array().unwrap().iter().map(|v| PhotoId(v["id"].as_u64().unwrap())).collect();
    assert_eq!(
        made.iter().map(|id| s.catalog.stack_of(*id).map(|st| (st.id, st.top()))).collect::<Vec<_>>(),
        [Some((b0.id, made[0])), Some((b1.id, made[1]))]
    );
    // a preview of both shows the first
    let pv = s.execute("merge.focusStack", &json!({"preview": true})).unwrap();
    assert_eq!(pv["stacks"], 2, "{pv}");
    // nothing selected: Focus Stack Each Bracket finds the collapsed brackets
    s.execute("library.selectNone", &json!({})).unwrap();
    let (photos, files) = (s.catalog.photos().count(), std::fs::read_dir(&dir).unwrap().count());
    let r = s.execute("merge.focusStackBatch", &json!({"method": "a", "autoSettings": false})).unwrap();
    assert_eq!((r["count"].as_u64(), r["failed"].as_array().map(Vec::len)), (Some(2), Some(0)), "{r}");
    assert!(r["results"].as_array().unwrap().iter().all(|v| v["used"] == 4), "{r}");
    // the same merges as before (same frames, same method): the earlier results stand for them
    assert!(r["results"].as_array().unwrap().iter().all(|v| v["sameAs"].is_u64()), "{r}");
    assert_eq!((s.catalog.photos().count(), std::fs::read_dir(&dir).unwrap().count()), (photos, files), "no copies left behind");
    let info = s.execute("photo.inspect", &json!({"id": ids[4].0})).unwrap();
    assert_eq!(info["stack"]["kind"], "focusBracket");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_focus_stack_is_retouched_from_a_frame_and_back() {
    let dir = temp_dir("focus-retouch");
    let mut paths = Vec::new();
    for (i, d) in lightcraft_merge::synth::focus_bracket_dngs(240, 160, 5, 0).unwrap().into_iter().enumerate() {
        let p = dir.join(format!("R_{i}.dng"));
        std::fs::write(&p, d).unwrap();
        paths.push(p.to_string_lossy().to_string());
    }
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, paths);
    let r = s.execute("merge.focusStack", &json!({"ids": ids, "autoSettings": false})).unwrap();
    let id = lightcraft_catalog::PhotoId(r["id"].as_u64().unwrap());
    let path = r["path"].as_str().unwrap().to_string();
    assert!(std::path::Path::new(&crate::focus_retouch::project_path(&path)).exists(), "{r}");
    assert!(s.has_focus_project(id));
    let pr = s.execute("merge.focusProject", &json!({"id": id.0})).unwrap();
    assert_eq!(pr["frames"].as_array().unwrap().len(), 5, "{pr}");
    let centre = |s: &Session| {
        let b = std::fs::read(&path).unwrap();
        let f = lightcraft_merge::load_frame(&b, None, true).unwrap();
        let _ = s;
        (f.image.get(f.image.width / 2, f.image.height / 2), f.image.get(4, 4))
    };
    let (base_c, base_corner) = centre(&s);
    // a preview: nothing written
    let png = dir.join("retouch.png");
    let stroke = json!([{"source": 0, "points": [[0.5, 0.5]], "size": 0.1, "hardness": 1.0}]);
    let pv = s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": stroke, "preview": true, "previewPath": png})).unwrap();
    assert!(png.exists() && pv["previewSize"][0].as_u64().unwrap() > 100, "{pv}");
    assert_eq!(centre(&s).0, base_c, "a preview leaves the DNG alone");
    // the stroke at full size: the centre now comes from frame 0, the corner is untouched
    let done = s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": stroke})).unwrap();
    assert_eq!(done["strokes"], 1);
    assert!(std::path::Path::new(&crate::focus_retouch::base_path(&path)).exists());
    let (c, corner) = centre(&s);
    assert!((0..3).any(|k| (c[k] - base_c[k]).abs() > 1e-3), "{c:?} vs {base_c:?}");
    assert!((0..3).all(|k| (corner[k] - base_corner[k]).abs() < 1e-4), "{corner:?} vs {base_corner:?}");
    assert_eq!(s.execute("merge.focusProject", &json!({"id": id.0})).unwrap()["strokes"].as_array().unwrap().len(), 1);
    // no strokes: back to the stack as merged
    s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": []})).unwrap();
    let (c, _) = centre(&s);
    assert!((0..3).all(|k| (c[k] - base_c[k]).abs() < 1e-4), "{c:?} vs {base_c:?}");
    // a stroke from method C, and bad strokes are errors
    s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": [{"source": "c", "points": [[0.2, 0.2], [0.3, 0.3]], "size": 0.05}]})).unwrap();
    assert!(s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": [{"source": 9, "points": [[0.5, 0.5]], "size": 0.1}]})).is_err());
    assert!(s.execute("merge.focusRetouch", &json!({"id": id.0, "strokes": [{"source": "x", "points": [[0.5, 0.5]], "size": 0.1}]})).is_err());
    assert!(s.execute("merge.focusRetouch", &json!({"id": ids[0], "strokes": []})).is_err(), "not a stack");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A merge of vendor raws (no colour matrix: white balance as multipliers, look fitted to the
/// camera JPEG) opens with the sources' colours. Before, it lost the white balance (green) and the
/// camera look (darker). Uses the raw corpus (`cargo xtask corpus --download`); skips without it.
#[test]
fn a_merge_of_vendor_raws_keeps_their_colours() {
    let corpus = std::env::var_os("LIGHTCRAFT_CORPUS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus"));
    let Ok(arw) = std::fs::read(corpus.join("raw/arw-sony-a7m3-compressed.arw")) else {
        eprintln!("skip: no raw corpus");
        return;
    };
    let dir = temp_dir("vendor");
    let (a, b) = (dir.join("A.arw"), dir.join("B.arw"));
    std::fs::write(&a, &arw).unwrap();
    // the same picture twice (a byte more, so the library keeps both)
    let mut arw2 = arw;
    arw2.push(0);
    std::fs::write(&b, arw2).unwrap();
    let mut s = Session::new().with_fs();
    let ids = import(&mut s, vec![a.to_string_lossy().into(), b.to_string_lossy().into()]);
    assert_eq!(ids.len(), 2);
    let r = s.execute("merge.focusStack", &json!({"ids": ids, "method": "a", "maxMegapixels": 1.5, "autoSettings": false})).unwrap();
    let id = lightcraft_catalog::PhotoId(r["id"].as_u64().unwrap());
    let mean = |s: &mut Session, id: lightcraft_catalog::PhotoId| {
        let img = s.render_now(id, 320, 320).unwrap().image;
        let n = (img.width * img.height) as f64;
        let mut m = [0f64; 3];
        for p in &img.data {
            for c in 0..3 {
                m[c] += p[c] as f64 / n;
            }
        }
        m
    };
    let src = mean(&mut s, lightcraft_catalog::PhotoId(ids[0]));
    let merged = mean(&mut s, id);
    for c in 0..3 {
        assert!((src[c] - merged[c]).abs() < 3.0, "source {src:?}, merged {merged:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
