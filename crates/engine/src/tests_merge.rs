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
    // the second merge doesn't overwrite the first
    let r2 = s.execute("merge.hdr", &json!({"ids": ids, "autoSettings": false})).unwrap();
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
    for (i, b) in lightcraft_merge::synth::focus_bracket_dngs(480, 320, n).unwrap().into_iter().enumerate() {
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
