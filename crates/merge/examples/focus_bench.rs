//! Focus-stacking benchmark on Helicon Focus's terms: 100 frames of 5616 × 3744 already in memory
//! (no decoding), one method at a time, reported in megapixels per second (Helicon's own table:
//! Apple M5 753 MPixel/s for method B).
//!
//! `cargo run --release -p lightcraft-merge --example focus_bench` (`FRAMES=100`, `SIZE=5616x3744`,
//! `METHODS=b,a,c`, `BASE=8` distinct synthetic frames cycled forth and back so that neighbours
//! stay neighbours, `ALIGN=0` to skip alignment). Needs about `BASE × 0.6 GB` of memory at the
//! default size (the synthetic frames plus the stacker); `SIZE=1920x1280` for a small machine.
//!
//! Prints lines `cargo xtask bench --focus` records (`… ms wall, … ms cpu`, minimum process CPU
//! time across threads) and a MPixel/s summary per method.

use std::time::Instant;

use lightcraft_merge::focus::{FocusMethod, FocusOptions, FocusStacker, Pass};
use lightcraft_merge::synth::{FocusSynth, focus_bracket};
use lightcraft_merge::{Frame, no_progress};
use lightcraft_raster::resample::{Filter, fit};

/// Process CPU time in ms (all threads). The only `unsafe` is this libc clock read, in a dev-only example.
#[allow(unsafe_code)]
fn cpu_ms() -> f64 {
    #[cfg(unix)]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `clock_gettime` only writes into the timespec we pass.
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
        ts.tv_sec as f64 * 1e3 + ts.tv_nsec as f64 / 1e6
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn main() {
    let frames: usize = env("FRAMES", 100);
    let base_n: usize = env::<usize>("BASE", 8).clamp(2, 64);
    let (w, h) = std::env::var("SIZE")
        .ok()
        .and_then(|s| s.split_once(['x', '×']).and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?))))
        .unwrap_or((5616usize, 3744usize));
    let align = env("ALIGN", 1) != 0;
    let methods: Vec<FocusMethod> = std::env::var("METHODS")
        .unwrap_or_else(|_| "b,a,c".into())
        .split(',')
        .filter_map(|m| serde_json::from_value(serde_json::json!(m.trim())).ok())
        .collect();
    let t0 = Instant::now();
    let bracket = match focus_bracket(w, h, &FocusSynth { n: base_n, ..Default::default() }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("focus_bench: {e}");
            std::process::exit(1);
        }
    };
    let full: Vec<Frame> = bracket.frames.into_iter().map(|f| Frame::from_linear_rec2020(f, None)).collect();
    let small: Vec<Frame> = full.iter().map(|f| Frame { image: fit(&f.image, 2048, 2048, Filter::Box), ..f.clone() }).collect();
    eprintln!("focus_bench: {base_n} synthetic {w}×{h} frames in {:.1} s; {frames} frames per stack", t0.elapsed().as_secs_f64());
    // frame k of the stack is base frame `tri(k)`: 0, 1, … n−1, n−2, … 1, 0, 1, … (neighbours stay neighbours)
    let period = 2 * (base_n - 1);
    let tri = |k: usize| {
        let p = k % period;
        if p < base_n { p } else { period - p }
    };
    let mpx = frames as f64 * (w * h) as f64 / 1e6;
    for m in methods {
        let opts = FocusOptions {
            method: m,
            align: lightcraft_merge::focus::AlignOptions { enabled: align, brightness: align, ..Default::default() },
            ..Default::default()
        };
        let mut st = match FocusStacker::new(frames, opts) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("focus_bench: {e}");
                std::process::exit(1);
            }
        };
        let (tw, tc) = (Instant::now(), cpu_ms());
        let mut lines = Vec::new();
        let mut peak = 0usize;
        loop {
            let req = match st.next_pass() {
                Ok(Some(r)) => r,
                Ok(None) => break,
                Err(e) => {
                    eprintln!("focus_bench: {e}");
                    std::process::exit(1);
                }
            };
            let (pw, pc) = (Instant::now(), cpu_ms());
            for &i in &req.frames {
                let src = if req.max_edge.is_some_and(|e| e < w.max(h)) { &small[tri(i)] } else { &full[tri(i)] };
                if let Err(e) = st.push(i, src, &no_progress) {
                    eprintln!("focus_bench: {e}");
                    std::process::exit(1);
                }
                peak = peak.max(st.state_bytes());
            }
            if let Err(e) = st.end_pass(&no_progress) {
                eprintln!("focus_bench: {e}");
                std::process::exit(1);
            }
            let name = match req.pass {
                Pass::Order => "order",
                Pass::Align => "align",
                Pass::Fuse => "stack",
                Pass::Gather => "gather",
            };
            lines.push(format!(
                "focus {} {name} {frames}×{w}×{h}:  {:.1} ms wall, {:.1} ms cpu",
                m.letter(),
                pw.elapsed().as_secs_f64() * 1e3,
                cpu_ms() - pc
            ));
        }
        let r = match st.finish(&no_progress) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("focus_bench: {e}");
                std::process::exit(1);
            }
        };
        let (wall, cpu) = (tw.elapsed().as_secs_f64() * 1e3, cpu_ms() - tc);
        for l in &lines {
            println!("{l}");
        }
        println!("focus {} total {frames}×{w}×{h}:  {wall:.1} ms wall, {cpu:.1} ms cpu", m.letter());
        let stack_ms = lines
            .iter()
            .find(|l| l.contains(" stack "))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|s| s.split_whitespace().next())
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(wall);
        println!(
            "focus {}: {:.0} MPixel/s end to end, {:.0} MPixel/s stacking only; state {:.0} MB ({:.1} B/px); output {}×{}, {} skipped",
            m.letter(),
            mpx / (wall / 1e3),
            mpx / (stack_ms / 1e3),
            peak as f64 / 1e6,
            peak as f64 / (r.image.width * r.image.height).max(1) as f64,
            r.image.width,
            r.image.height,
            r.skipped.len()
        );
    }
    println!("reference: Helicon Focus benchmark (100 × 5616×3744 16-bit in memory, method B): Apple M5 753, M4 580, M5 Max 1499 MPixel/s");
}
