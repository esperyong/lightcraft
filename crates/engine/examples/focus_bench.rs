//! Focus-stacking benchmark on Helicon Focus's terms: 100 frames of 5616 × 3744 already in memory
//! (no decoding), one method at a time, reported in megapixels per second (Helicon's own table:
//! Apple M5 753 MPixel/s for method B).
//!
//! `cargo run --release -p lightcraft-engine --example focus_bench` (`FRAMES=100`, `SIZE=5616x3744`,
//! `METHODS=b,a,c`, `BASE=8` distinct synthetic frames cycled forth and back so that neighbours
//! stay neighbours, `ALIGN=0` to skip alignment, `LIGHTCRAFT_GPU=0` for the CPU column only).
//! Needs about `BASE × 0.6 GB` of memory at the default size (the synthetic frames plus the
//! stacker); `SIZE=1920x1280` for a small machine.
//!
//! Methods A and B run on the CPU and, when there is a GPU, with their fuse pass on it (second
//! column; method C is CPU only). Prints lines `cargo xtask bench --focus` records
//! (`cpu: … ms wall, … ms cpu | gpu: …`, process CPU time across threads) and a MPixel/s summary.

use std::time::Instant;

use lightcraft_engine::merge::GpuFocusKernels;
use lightcraft_merge::focus::{FocusMethod, FocusOptions, FocusResult, FocusStacker, Pass};
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

/// One stack: the result, per pass (name, wall ms, cpu ms), the total, peak state bytes.
struct Run {
    r: FocusResult,
    passes: Vec<(&'static str, f64, f64)>,
    total: (f64, f64),
    peak: usize,
}

fn stack(
    frames: usize,
    opts: &FocusOptions,
    gpu: bool,
    small: &[Frame],
    full: &[Frame],
    edge: usize,
    tri: &dyn Fn(usize) -> usize,
) -> Result<Run, String> {
    let mut st = FocusStacker::new(frames, opts.clone()).map_err(|e| e.to_string())?;
    if gpu {
        st.set_kernels(Box::new(GpuFocusKernels::available().ok_or("no GPU")?));
    }
    let (tw, tc) = (Instant::now(), cpu_ms());
    let (mut passes, mut peak) = (Vec::new(), 0usize);
    while let Some(req) = st.next_pass().map_err(|e| e.to_string())? {
        let (pw, pc) = (Instant::now(), cpu_ms());
        for &i in &req.frames {
            // in memory the gather pass samples the full frames (it asks for half size to spare
            // a raw's demosaicing, which frames in memory don't need)
            let src = if req.pass != Pass::Gather && req.max_edge.is_some_and(|e| e < edge) { &small[tri(i)] } else { &full[tri(i)] };
            st.push(i, src, &no_progress).map_err(|e| e.to_string())?;
            peak = peak.max(st.state_bytes());
        }
        st.end_pass(&no_progress).map_err(|e| e.to_string())?;
        let name = match req.pass {
            Pass::Order => "order",
            Pass::Align => "align",
            Pass::Fuse => "stack",
            Pass::Gather => "gather",
        };
        passes.push((name, pw.elapsed().as_secs_f64() * 1e3, cpu_ms() - pc));
    }
    let r = st.finish(&no_progress).map_err(|e| e.to_string())?;
    Ok(Run { r, passes, total: (tw.elapsed().as_secs_f64() * 1e3, cpu_ms() - tc), peak })
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
    let gpu = GpuFocusKernels::available().map(|k| lightcraft_merge::focus::kernels::FocusKernels::name(&k));
    eprintln!(
        "focus_bench: {base_n} synthetic {w}×{h} frames in {:.1} s; {frames} frames per stack; GPU: {}",
        t0.elapsed().as_secs_f64(),
        gpu.as_deref().unwrap_or("none")
    );
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
        let run = |on_gpu: bool| match stack(frames, &opts, on_gpu, &small, &full, w.max(h), &tri) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("focus_bench: {e}");
                std::process::exit(1);
            }
        };
        let cpu = run(false);
        let gpu_run = (gpu.is_some() && m != FocusMethod::Pyramid).then(|| run(true));
        let col = |r: &Run, name: &str| r.passes.iter().find(|p| p.0 == name).map(|p| format!("{:.1} ms wall, {:.1} ms cpu", p.1, p.2));
        for (name, ..) in &cpu.passes {
            let c = col(&cpu, name).unwrap_or_default();
            match gpu_run.as_ref().and_then(|g| col(g, name)) {
                Some(g) => println!("focus {} {name} {frames}×{w}×{h}:  cpu: {c}  |  gpu: {g}", m.letter()),
                None => println!("focus {} {name} {frames}×{w}×{h}:  cpu: {c}", m.letter()),
            }
        }
        let tot = |r: &Run| format!("{:.1} ms wall, {:.1} ms cpu", r.total.0, r.total.1);
        match &gpu_run {
            Some(g) => println!("focus {} total {frames}×{w}×{h}:  cpu: {}  |  gpu: {}", m.letter(), tot(&cpu), tot(g)),
            None => println!("focus {} total {frames}×{w}×{h}:  cpu: {}", m.letter(), tot(&cpu)),
        }
        for (label, r) in std::iter::once(("CPU", &cpu)).chain(gpu_run.as_ref().map(|g| ("GPU", g))) {
            let stack_ms = r.passes.iter().find(|p| p.0 == "stack").map_or(r.total.0, |p| p.1);
            println!(
                "focus {} {label}: {:.0} MPixel/s end to end, {:.0} MPixel/s stacking only; state {:.0} MB ({:.1} B/px); output {}×{}, {} skipped; ran on {}",
                m.letter(),
                mpx / (r.total.0 / 1e3),
                mpx / (stack_ms / 1e3),
                r.peak as f64 / 1e6,
                r.peak as f64 / (r.r.image.width * r.r.image.height).max(1) as f64,
                r.r.image.width,
                r.r.image.height,
                r.r.skipped.len(),
                r.r.device
            );
        }
    }
    println!("reference: Helicon Focus benchmark (100 × 5616×3744 16-bit in memory, method B): Apple M5 753, M4 580, M5 Max 1499 MPixel/s");
}
