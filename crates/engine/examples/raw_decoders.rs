//! The built-in and Apple's raw decoders on the same files (issue #2, docs/apple-raw.md): size, speed, tone response
//! (is Apple's output as linear as the built-in decoder's?), highlights above white, colour (a fitted 3×3 matrix and
//! Oklab ΔE at matched exposure) and side-by-side renders with the default raw settings.
//!
//! ```sh
//! cargo run --release -p lightcraft-engine --example raw_decoders -- [--edge 1536] [--out DIR] [--timing] [--sweep] FILE...
//! ```
//!
//! One JSON line per file on stdout, a summary on stderr. `--timing` adds both decoders' times at 512 px, 2,560 px and
//! full size; `--sweep` repeats the comparison with Core Image's tuning changed one setting at a time (what the
//! defaults in `lightcraft_coreimage::Options` were chosen from); `--out` writes `<name>-builtin-vs-apple.jpg`
//! (built-in left, Apple right). Needs macOS 12 or later; elsewhere it says so. `--self-check` compares the built-in
//! decoder with itself made twice as bright (anywhere): +1 EV, a flat tone response, the identity matrix and ΔE 0.

use std::time::Instant;

use lightcraft_codecs::{ChromaSubsampling, EncodeImage, EncodeMeta, Samples, encode_jpeg};
use lightcraft_color::perceptual::oklab_from_2020;
use lightcraft_coreimage as ci;
use lightcraft_engine::apple_raw;
use lightcraft_pipeline::{RenderRequest, SourceInfo};
use lightcraft_raster::Rgb32f;
use lightcraft_raster::resample::{Filter, resize};
use serde_json::{Value, json};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut edge, mut out, mut timing, mut sweep, mut check, mut files) = (1536usize, None, false, false, false, Vec::new());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--edge" => edge = args.next().and_then(|v| v.parse().ok()).unwrap_or(edge),
            "--out" => out = args.next(),
            "--timing" => timing = true,
            "--sweep" => sweep = true,
            "--self-check" => check = true,
            _ => files.push(a),
        }
    }
    if files.is_empty() {
        eprintln!("usage: raw_decoders [--edge N] [--out DIR] [--timing] [--sweep] FILE...");
        std::process::exit(2);
    }
    if check {
        for f in &files {
            let r = std::fs::read(f).map_err(|e| e.to_string()).and_then(|bytes| lightcraft_engine::files::load_bytes(&bytes, edge));
            let v = match r {
                Ok((b, _)) => {
                    let doubled = Rgb32f { width: b.width, height: b.height, data: b.data.iter().map(|p| p.map(|c| c * 2.0)).collect() };
                    json!({"file": f, "selfCheck": stats(&b, &doubled)})
                }
                Err(e) => json!({"file": f, "error": e}),
            };
            println!("{v}");
        }
        return;
    }
    if !ci::available() {
        eprintln!("Apple's raw decoder is not available here ({}): run this on a Mac with macOS 12 or later", ci::identity());
        return;
    }
    eprintln!("Apple's raw decoder: {}", ci::identity());
    for f in &files {
        let v = compare(f, edge, out.as_deref(), timing, sweep).unwrap_or_else(|e| json!({"file": f, "error": e}));
        summary(&v);
        println!("{v}");
    }
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

fn facts(img: &Rgb32f, info: &SourceInfo, ms: f64) -> Value {
    json!({
        "size": [img.width, img.height],
        "ms": ms,
        "raw": info.raw,
        "asShot": [info.as_shot_temp, info.as_shot_tint],
        "relativeWb": info.relative_wb,
        "lens": info.lens.is_some(),
    })
}

fn compare(path: &str, edge: usize, out: Option<&str>, timing: bool, sweep: bool) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let t = Instant::now();
    let (b, b_info) = lightcraft_engine::files::load_bytes(&bytes, edge)?;
    let builtin = facts(&b, &b_info, ms(t));
    let t = Instant::now();
    let apple = apple_raw::develop_now(path, &bytes, edge, |_| {});
    let mut v = json!({"file": path, "edge": edge, "builtin": builtin});
    let (a, a_info) = match apple {
        Ok(r) => r,
        Err(e) => {
            v["apple"] = json!({"error": e});
            return Ok(v);
        }
    };
    v["apple"] = facts(&a, &a_info, ms(t));
    // (a built-in "raw: false" is the embedded preview: only Apple develops this raw, there is nothing to compare)
    if b_info.raw {
        v["compare"] = stats(&b, &a);
        if sweep {
            let variants: [(&str, fn(&mut ci::Options)); 6] = [
                ("edr0", |o| o.edr_amount = 0.0),
                ("edr1", |o| o.edr_amount = 1.0),
                ("boostShadow1", |o| o.boost_shadow_amount = 1.0),
                ("appleShadowBias", |o| o.shadow_bias = None),
                ("appleBaseline", |o| o.baseline_in_rust = false),
                ("noHighlightRecovery", |o| o.highlight_recovery = false),
            ];
            let mut s = serde_json::Map::new();
            for (name, tune) in variants {
                let r = apple_raw::develop_now(path, &bytes, edge, tune).map(|(img, _)| stats(&b, &img));
                s.insert(name.into(), r.unwrap_or_else(|e| json!({"error": e})));
            }
            v["sweep"] = Value::Object(s);
        }
    }
    if timing {
        let mut tm = serde_json::Map::new();
        for (name, e) in [("512", 512), ("2560", 2560), ("full", usize::MAX)] {
            let t = Instant::now();
            let built = lightcraft_engine::files::load_bytes(&bytes, e).map(|(img, _)| (img.width, img.height));
            let built_ms = ms(t);
            let t = Instant::now();
            let app = apple_raw::develop_now(path, &bytes, e, |_| {}).map(|(img, _)| (img.width, img.height));
            tm.insert(name.into(), json!({"builtinMs": built_ms, "builtinSize": built.ok(), "appleMs": ms(t), "appleSize": app.ok()}));
        }
        v["timing"] = Value::Object(tm);
    }
    if let Some(dir) = out {
        v["sideBySide"] = json!(side_by_side(path, dir, (&b, &b_info), (&a, &a_info))?);
    }
    Ok(v)
}

fn luminance(p: [f32; 3]) -> f32 {
    0.2627 * p[0] + 0.6780 * p[1] + 0.0593 * p[2]
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Apple's picture against the built-in one, both at the built-in picture's shape, at most 768 px long.
fn stats(b: &Rgb32f, a: &Rgb32f) -> Value {
    let scale = (768.0 / b.width.max(b.height) as f64).min(1.0);
    let (w, h) = (((b.width as f64 * scale).round() as usize).max(1), ((b.height as f64 * scale).round() as usize).max(1));
    let (b, a) = (resize(b, w, h, Filter::Box), resize(a, w, h, Filter::Box));
    let n = (w * h) as f64;
    // tone response: Apple / built-in luminance, by the built-in luminance's stop below white
    let mut bins: Vec<Vec<f32>> = vec![Vec::new(); 13];
    let mut mid = Vec::new();
    for (pb, pa) in b.data.iter().zip(&a.data) {
        let (yb, ya) = (luminance(*pb), luminance(*pa));
        if yb > 1e-6 && ya > 0.0 {
            let i = yb.log2().floor() as i32 + 10;
            if let Some(bin) = usize::try_from(i).ok().and_then(|i| bins.get_mut(i)) {
                bin.push(ya / yb);
            }
            if (1.0 / 32.0..0.5).contains(&yb) {
                mid.push(ya / yb);
            }
        }
    }
    let gain = median(&mut mid);
    let tone: Vec<Value> = bins
        .iter_mut()
        .enumerate()
        .filter(|(_, v)| v.len() >= 50)
        .map(|(i, v)| json!({"stop": i as i32 - 10, "n": v.len(), "ratio": round4(median(v) / gain)}))
        .collect();
    let over = |img: &Rgb32f| round4((img.data.iter().filter(|p| p.iter().any(|&c| c > 1.0)).count() as f64 / n) as f32);
    let max = |img: &Rgb32f| round4(img.data.iter().flatten().fold(0f32, |m, &c| m.max(c)));
    // colour on unclipped mid-tones, Apple's exposure matched to the built-in one
    let pairs: Vec<([f32; 3], [f32; 3])> = b
        .data
        .iter()
        .zip(&a.data)
        .filter(|(pb, pa)| (1.0 / 64.0..0.8).contains(&luminance(**pb)) && pb.iter().chain(pa.iter()).all(|&c| c < 0.95 * gain.max(1.0)))
        .map(|(pb, pa)| (*pb, pa.map(|c| c / gain)))
        .collect();
    let matrix = fit_matrix(&pairs);
    let rms = matrix.map(|m| {
        let e: f64 = pairs
            .iter()
            .map(|(pb, pa)| (0..3).map(|c| (f64::from(pa[c]) - (0..3).map(|k| m[c][k] * f64::from(pb[k])).sum::<f64>()).powi(2)).sum::<f64>())
            .sum();
        round4((e / (3.0 * pairs.len().max(1) as f64)).sqrt() as f32)
    });
    let mut de: Vec<f32> = pairs
        .iter()
        .map(|(pb, pa)| {
            let (lb, la) = (oklab_from_2020(*pb), oklab_from_2020(*pa));
            100.0 * ((lb[0] - la[0]).powi(2) + (lb[1] - la[1]).powi(2) + (lb[2] - la[2]).powi(2)).sqrt()
        })
        .collect();
    de.sort_by(f32::total_cmp);
    let mean = de.iter().sum::<f32>() / de.len().max(1) as f32;
    let p95 = de.get(de.len() * 95 / 100).copied().unwrap_or(f32::NAN);
    json!({
        "size": [w, h],
        "gain": round4(gain),
        "gainEv": round4(gain.log2()),
        "tone": tone,
        "builtinMax": max(&b),
        "appleMax": max(&a),
        "builtinOver1": over(&b),
        "appleOver1": over(&a),
        "matrix": matrix.map(|m| m.map(|r| r.map(|v| (v * 1e4).round() / 1e4))),
        "matrixRms": rms,
        "deltaE": {"pixels": de.len(), "mean": round4(mean), "p95": round4(p95)},
    })
}

fn round4(v: f32) -> f32 {
    (v * 1e4).round() / 1e4
}

/// The 3×3 matrix `m` with Apple ≈ m · built-in (least squares), `None` for too few or degenerate pixels.
fn fit_matrix(pairs: &[([f32; 3], [f32; 3])]) -> Option<[[f64; 3]; 3]> {
    if pairs.len() < 100 {
        return None;
    }
    let mut btb = [[0f64; 3]; 3];
    let mut bta = [[0f64; 3]; 3];
    for (pb, pa) in pairs {
        for i in 0..3 {
            for j in 0..3 {
                btb[i][j] += f64::from(pb[i]) * f64::from(pb[j]);
                bta[i][j] += f64::from(pb[i]) * f64::from(pa[j]);
            }
        }
    }
    let inv = invert(btb)?;
    let mut m = [[0f64; 3]; 3];
    for (c, row) in m.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|j| inv[k][j] * bta[j][c]).sum();
        }
    }
    Some(m)
}

fn invert(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ])
}

/// Both pictures rendered with the default raw settings (as imported), side by side: `<dir>/<name>-builtin-vs-apple.jpg`.
fn side_by_side(path: &str, dir: &str, built: (&Rgb32f, &SourceInfo), apple: (&Rgb32f, &SourceInfo)) -> Result<String, String> {
    let render = |(img, info): (&Rgb32f, &SourceInfo)| {
        let settings = lightcraft_develop::DevelopSettings::for_raw(info.as_shot_temp, info.as_shot_tint);
        lightcraft_pipeline::render(img, info, &settings, &RenderRequest::fit(1200, 1200)).image
    };
    let (l, r) = (render(built), render(apple));
    let (w, h) = (l.width + 8 + r.width, l.height.max(r.height));
    let mut rgb = vec![0u8; w * h * 3];
    for (img, x0) in [(&l, 0), (&r, l.width + 8)] {
        for y in 0..img.height {
            for x in 0..img.width {
                let p = img.data[y * img.width + x];
                let o = (y * w + x0 + x) * 3;
                rgb[o..o + 3].copy_from_slice(&p[..3]);
            }
        }
    }
    let jpeg = encode_jpeg(&EncodeImage::new(w as u32, h as u32, 3, Samples::U8(&rgb)), 90, ChromaSubsampling::S444, &EncodeMeta::default())
        .map_err(|e| e.to_string())?;
    let name = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "raw".into());
    let out = std::path::Path::new(dir).join(format!("{name}-builtin-vs-apple.jpg"));
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(&out, jpeg).map_err(|e| e.to_string())?;
    Ok(out.to_string_lossy().to_string())
}

fn summary(v: &Value) {
    let f = v["file"].as_str().unwrap_or("?");
    if let Some(e) = v["error"].as_str() {
        eprintln!("{f}: {e}");
        return;
    }
    let side = |k: &str| {
        let s = &v[k];
        match s["error"].as_str() {
            Some(e) => format!("{k}: {e}"),
            None => format!("{k} {}×{} {} ms{}", s["size"][0], s["size"][1], s["ms"], if s["raw"] == false { " (embedded preview)" } else { "" }),
        }
    };
    eprintln!("{f}: {}; {}", side("builtin"), side("apple"));
    let c = &v["compare"];
    if c.is_object() {
        eprintln!(
            "  Apple/built-in {} EV; over white {} vs {}; max {} vs {}; ΔE mean {} p95 {}; matrix rms {}",
            c["gainEv"], c["builtinOver1"], c["appleOver1"], c["builtinMax"], c["appleMax"], c["deltaE"]["mean"], c["deltaE"]["p95"], c["matrixRms"]
        );
        let tone: Vec<String> = c["tone"].as_array().into_iter().flatten().map(|t| format!("{}:{}", t["stop"], t["ratio"])).collect();
        eprintln!("  tone (stop: Apple/built-in, 1 = same response): {}", tone.join(" "));
    }
}
