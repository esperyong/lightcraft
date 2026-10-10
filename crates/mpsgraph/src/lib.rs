//! Denoise networks on the Mac's GPU through Apple's MetalPerformanceShadersGraph (MPSGraph).
//!
//! The same [`Net`] the pure-Rust CPU and wgpu runners take (no model conversion: the graph is built from it when the
//! runner is made) becomes one MPSGraph: Apple's own convolution kernels, in f32 or f16, on the system's Metal device.
//! It is another [`TileRunner`]; the engine checks its answer on a tile against the CPU's before using it, as it does
//! for the wgpu runner, and falls back to that when this one is not available or not right.
//!
//! The Objective-C calls are `unsafe` and live in one module (`macos`), every block with its `SAFETY:` reasoning; the
//! API here is safe. Elsewhere [`runner`] says it is macOS only. Graph building cannot be wrapped against Objective-C
//! exceptions without a C helper, so everything MPSGraph is given is checked here first: shapes come from a [`Net`]
//! that [`Net::new`] validated, and the tile must fit it.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::{Mutex, PoisonError};

use lightcraft_denoise::net::Net;
use lightcraft_denoise::run::{Error, TileRunner};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as imp;

/// The arithmetic the graph computes in. Inputs and outputs are f32 either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    F32,
    /// Half precision: weights and activations in f16 (Apple GPUs run it at least as fast, with half the memory).
    F16,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
        }
    }

    /// `f32` or `f16`.
    pub fn parse(name: &str) -> Option<Precision> {
        match name.trim() {
            "f32" => Some(Precision::F32),
            "f16" => Some(Precision::F16),
            _ => None,
        }
    }
}

/// Largest tile side this runner builds a graph for (the model's tile is 512).
pub const MAX_TILE: usize = 4096;

/// A network built as an MPSGraph for one tile size. Tiles from several threads take turns: the GPU runs one graph at
/// a time anyway.
pub struct MpsRunner {
    graph: Mutex<imp::Graph>,
    adapter: String,
    in_floats: usize,
    out_floats: usize,
}

/// Is there a Metal device MPSGraph can run on? (Always `false` off macOS.)
pub fn available() -> bool {
    #[cfg(target_os = "macos")]
    {
        imp::device_name().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Build `net` for tiles of `tile × tile` cells, computing in `precision`.
pub fn runner(net: &Net, tile: usize, precision: Precision) -> Result<MpsRunner, String> {
    if tile == 0 || tile > MAX_TILE || !net.fits_tile(tile) {
        return Err(format!("the network does not take tiles of {tile} cells"));
    }
    let in_floats = net.in_channels.checked_mul(tile * tile).ok_or("the tile is too large")?;
    let out_floats = net.out_channels().checked_mul(4 * tile * tile).ok_or("the tile is too large")?;
    #[cfg(target_os = "macos")]
    {
        let (graph, device) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| imp::build(net, tile, precision)))
            .map_err(|_| "building the MPSGraph failed unexpectedly".to_string())??;
        Ok(MpsRunner { graph: Mutex::new(graph), adapter: format!("{device} (MPSGraph, {})", precision.name()), in_floats, out_floats })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (in_floats, out_floats, precision);
        Err("MetalPerformanceShadersGraph is only on macOS".into())
    }
}

impl MpsRunner {
    /// The device and how it computes, e.g. `Apple M5 (MPSGraph, f32)`.
    pub fn adapter(&self) -> String {
        self.adapter.clone()
    }
}

impl TileRunner for MpsRunner {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        if input.len() != self.in_floats {
            return Err(Error::Input("the tile is not the size the model wants".into()));
        }
        let graph = self.graph.lock().unwrap_or_else(PoisonError::into_inner);
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.run(input, self.out_floats)))
            .map_err(|_| Error::Runtime("the MPSGraph runner gave up".into()))?
            .map_err(Error::Runtime)?;
        if out.len() != self.out_floats {
            return Err(Error::Model(format!("{} numbers instead of {}", out.len(), self.out_floats)));
        }
        Ok(out)
    }
}

/// Off macOS there is never a graph: [`runner`] says so, and no [`MpsRunner`] exists.
#[cfg(not(target_os = "macos"))]
mod imp {
    pub enum Graph {}

    impl Graph {
        pub fn run(&self, _input: &[f32], _out_floats: usize) -> Result<Vec<f32>, String> {
            match *self {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_denoise::{onnx, reference, synthetic};

    fn net(tile: u64, ch: u64, depth: usize, seed: u64) -> Net {
        let path = std::env::temp_dir().join(format!("lc-mps-{}-{tile}-{ch}-{depth}-{seed}.onnx", std::process::id()));
        std::fs::write(&path, synthetic::unet_onnx(tile, ch, depth, seed)).unwrap();
        let n = onnx::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        n
    }

    fn picture(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                (s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    /// Largest difference, as a share of the reference's largest value.
    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
    }

    /// The runner, or `None` (with a note) where MPSGraph can't run: off macOS, or a Mac without a Metal device.
    fn mps(net: &Net, tile: usize, precision: Precision) -> Option<MpsRunner> {
        match runner(net, tile, precision) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("skipped: no MPSGraph runner here ({e})");
                None
            }
        }
    }

    #[test]
    fn tiles_the_network_does_not_take_are_refused_everywhere() {
        let n = net(32, 8, 2, 1);
        for tile in [0, 3, 30, MAX_TILE * 2] {
            assert!(runner(&n, tile, Precision::F32).is_err(), "tile {tile}");
        }
        assert_eq!(Precision::parse("f16"), Some(Precision::F16));
        assert_eq!(Precision::parse(" f32 "), Some(Precision::F32));
        assert_eq!(Precision::parse("f64"), None);
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!available());
            assert!(runner(&n, 32, Precision::F32).err().unwrap_or_default().contains("only on macOS"));
        }
    }

    /// Every layer kind (3×3 and 1×1 convolutions, skip joins, max pooling, transposed convolutions, the last
    /// depth-to-space) against the scalar reference, at sizes that are not multiples of anything convenient.
    #[test]
    fn mpsgraph_gives_what_the_reference_gives() {
        for (tile, ch, depth, seed) in [(1u64, 4u64, 0usize, 1u64), (6, 4, 1, 9), (20, 12, 2, 2), (32, 8, 2, 3), (64, 16, 3, 5)] {
            let n = net(tile, ch, depth, seed);
            let x = picture(4 * (tile * tile) as usize, seed);
            let want = reference::run(&n, tile as usize, &x).unwrap();
            for (precision, bound) in [(Precision::F32, 2e-4), (Precision::F16, 5e-2)] {
                let Some(r) = mps(&n, tile as usize, precision) else { return };
                let got = r.run(&x).unwrap();
                let e = worst(&got, &want);
                eprintln!("{}: tile {tile}, {ch} channels, depth {depth}: off by {e:e} of the output's size", r.adapter());
                assert!(e < bound, "{}: tile {tile}, {ch} channels, depth {depth}: off by {e:e}", r.adapter());
            }
        }
    }

    #[test]
    fn a_runner_serves_tiles_from_many_threads_and_refuses_wrong_sizes() {
        let n = net(32, 8, 2, 7);
        let Some(r) = mps(&n, 32, Precision::F32) else { return };
        assert!(matches!(r.run(&[0.0; 10]), Err(Error::Input(_))));
        let inputs: Vec<Vec<f32>> = (0..8).map(|i| picture(4 * 32 * 32, 100 + i)).collect();
        let wants: Vec<Vec<f32>> = inputs.iter().map(|x| reference::run(&n, 32, x).unwrap()).collect();
        std::thread::scope(|s| {
            for chunk in inputs.chunks(2).zip(wants.chunks(2)) {
                let r = &r;
                s.spawn(move || {
                    for (x, want) in chunk.0.iter().zip(chunk.1) {
                        let e = worst(&r.run(x).unwrap(), want);
                        assert!(e < 2e-4, "off by {e:e}");
                    }
                });
            }
        });
    }

    /// The real model on MPSGraph (f32 and f16) against the pure-Rust CPU runner, with timings:
    /// `LC_DENOISE_MODEL=<model_bayer.onnx> cargo test --release -p lightcraft-mpsgraph --lib real_model -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real model: see the doc comment"]
    fn real_model_on_mpsgraph_matches_cpu() {
        use lightcraft_denoise::cpu::NetRunner;
        use std::time::Instant;
        let Some(model) = std::env::var_os("LC_DENOISE_MODEL") else { return };
        let net = onnx::read(std::path::Path::new(&model)).unwrap();
        let tile = 512usize;
        let noise = picture(4 * tile * tile, 42);
        let input: Vec<f32> =
            (0..4 * tile * tile).map(|i| (0.2 + 0.3 * ((i % tile) as f32 / tile as f32) + 0.1 * (noise[i] - 0.5)).clamp(0.0, 1.0)).collect();
        let t = Instant::now();
        let want = NetRunner::new(&net, tile).unwrap().run(&input).unwrap();
        println!("CPU: {:.0} ms for one tile (one call runs on one core)", t.elapsed().as_secs_f64() * 1000.0);
        for precision in [Precision::F32, Precision::F16] {
            let t = Instant::now();
            let Some(r) = mps(&net, tile, precision) else { return };
            println!("{}: built in {:.2} s", r.adapter(), t.elapsed().as_secs_f64());
            let t = Instant::now();
            let got = r.run(&input).unwrap();
            println!("{}: first tile (compiles the graph) {:.0} ms", r.adapter(), t.elapsed().as_secs_f64() * 1000.0);
            let e = worst(&got, &want);
            println!("{}: worst difference {e:e} of the output's size", r.adapter());
            let mut times: Vec<f64> = (0..8)
                .map(|_| {
                    let t = Instant::now();
                    r.run(&input).unwrap();
                    t.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            times.sort_by(|a, b| a.total_cmp(b));
            println!(
                "{}: one tile at a time: median {:.1} ms (best {:.1}, worst {:.1})",
                r.adapter(),
                times[times.len() / 2],
                times[0],
                times[times.len() - 1]
            );
            if precision == Precision::F32 {
                assert!(e < 1e-3, "MPSGraph f32 does not give the CPU answer ({e:e})");
            }
        }
    }
}
