//! The focus stacker's fuse pass on the GPU (`lightcraft_gpu::focus`, plugged in by the engine)
//! against the CPU reference: methods A and B on a synthetic bracket. Skipped without a GPU
//! adapter (a software one counts with `LIGHTCRAFT_GPU_SOFTWARE=1`).

use lightcraft_engine::merge::GpuFocusKernels;
use lightcraft_merge::focus::{FocusMethod, FocusOptions, FocusResult, FocusStacker, stack_focus};
use lightcraft_merge::synth::{FocusSynth, focus_bracket};
use lightcraft_merge::{Frame, no_progress};

fn on_gpu(frames: &[Frame], opts: &FocusOptions, k: GpuFocusKernels) -> FocusResult {
    let mut st = FocusStacker::new(frames.len(), opts.clone()).unwrap();
    st.set_kernels(Box::new(k));
    while let Some(req) = st.next_pass().unwrap() {
        for &i in &req.frames {
            st.push(i, &frames[i], &no_progress).unwrap();
        }
        st.end_pass(&no_progress).unwrap();
    }
    st.finish(&no_progress).unwrap()
}

/// Differences in 8-bit steps of the √-encoded values: (mean, 99.9th percentile, max).
fn lsb(a: &lightcraft_raster::Rgb32f, b: &lightcraft_raster::Rgb32f) -> (f64, f64, f64) {
    let mut d: Vec<f64> = a
        .data
        .iter()
        .zip(&b.data)
        .flat_map(|(p, q)| (0..3).map(move |c| ((p[c].max(0.0).sqrt() - q[c].max(0.0).sqrt()) as f64).abs() * 255.0))
        .collect();
    let mean = d.iter().sum::<f64>() / d.len().max(1) as f64;
    d.sort_by(|x, y| x.total_cmp(y));
    let p999 = d.get(d.len() * 999 / 1000).copied().unwrap_or(0.0);
    (mean, p999, d.last().copied().unwrap_or(0.0))
}

#[test]
fn gpu_fuse_pass_matches_the_cpu() {
    if GpuFocusKernels::available().is_none() {
        eprintln!("no GPU adapter: skipped");
        return;
    }
    let b = focus_bracket(320, 220, &FocusSynth { n: 8, ..Default::default() }).unwrap();
    let frames: Vec<Frame> = b.frames.iter().map(|f| Frame::from_linear_rec2020(f.clone(), None)).collect();
    for method in [FocusMethod::Average, FocusMethod::Depth] {
        let opts = FocusOptions { method, ..Default::default() };
        let cpu = stack_focus(&frames, &opts, &no_progress).unwrap();
        let Some(k) = GpuFocusKernels::available() else { return };
        let gpu = on_gpu(&frames, &opts, k);
        assert!(!gpu.device.starts_with("CPU"), "{method:?} fell back: {}", gpu.device);
        assert_eq!((gpu.image.width, gpu.image.height), (cpu.image.width, cpu.image.height));
        let (mean, p999, max) = lsb(&gpu.image, &cpu.image);
        eprintln!("{method:?} on {}: mean {mean:.4} LSB, 99.9 % {p999:.3}, max {max:.2}", gpu.device);
        assert!(mean < 0.5 && p999 <= 3.0, "{method:?}: mean {mean}, 99.9 % {p999}, max {max}");
        // the depth maps agree (B: the same winners nearly everywhere)
        let same = gpu.depth.data.iter().zip(&cpu.depth.data).filter(|(a, b)| (*a - *b).abs() < 0.02).count();
        let frac = same as f64 / cpu.depth.data.len() as f64;
        assert!(frac > 0.995, "{method:?}: depth agrees at {:.3} %", frac * 100.0);
    }
}
