//! Focus stacking's fuse pass on the render device (`wgsl/focus.wgsl`): each frame is uploaded,
//! resampled into the output geometry, its focus measured, and folded into method A's or B's
//! per-pixel state, which stays on the device until [`FocusPass::finish_average`] /
//! [`FocusPass::finish_depth`] read it back once.
//!
//! Uploads are 16-bit (scaled to the frame's peak: finer than a raw's 14 bits everywhere), packed
//! by all cores straight into one of two mapped staging buffers while the device works on the
//! previous frame from the other; a frame waits only for its staging buffer's last copy. The CPU stacker (`lightcraft_merge::focus`)
//! is the reference: the engine plugs this in behind `lightcraft_merge::focus::kernels`, and any
//! failure is an `Err` the stacker answers by repeating the pass on the CPU.

use lightcraft_raster::Rgb32f;

#[cfg(not(target_arch = "wasm32"))]
use crate::ctx::{self, Buf, ErrorScopes, Gpu, groups1, groups2};
#[cfg(not(target_arch = "wasm32"))]
use crate::render::{Cx, gaussian};

/// Method B's state read back: per pixel the sharpest frame, the measures (`fm_best`,
/// `fm_left`, `fm_right`, `fm_sum`, `prev_fm`), the colours (`col_best`, `col_left`,
/// `col_right`, `prev_col`, 16-bit linear 0..2), and the frames' noise floor.
#[derive(Clone, Debug, Default)]
pub struct DepthOut {
    pub best: Vec<u16>,
    pub measures: [Vec<f32>; 5],
    pub colours: [Vec<[u16; 3]>; 4],
    pub noise: f32,
}

/// One fuse pass on the device.
pub struct FocusPass {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Inner,
}

#[cfg(not(target_arch = "wasm32"))]
struct Inner {
    gpu: &'static Gpu,
    w: usize,
    h: usize,
    depth: bool,
    radius: usize,
    sigma: f32,
    fs: Buf,
    cs: Option<Buf>,
    /// The frame on the device: 16-bit samples, two a word.
    frame: Option<Buf>,
    /// Staging buffers, used in turn.
    staging: [Option<Staging>; 2],
    /// Frames added so far.
    added: usize,
    rgb: Buf,
    flags: Buf,
    grey: Buf,
    ml: Buf,
    fm: Buf,
    tmp: Buf,
    /// Method B: every `step`-th focus measure of each frame (noise floors).
    subs: Vec<(Buf, usize)>,
    /// Tiny buffers for the bindings a kernel doesn't use (one buffer can't be bound both
    /// read-only and writable in one dispatch).
    spare: Vec<Buf>,
}

/// A staging buffer the CPU fills while the device copies from the other.
#[cfg(not(target_arch = "wasm32"))]
struct Staging {
    buf: wgpu::Buffer,
    /// The mapping asked for after its last copy: its result arrives here.
    mapping: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// The submission whose copy last read it.
    used_by: Option<wgpu::SubmissionIndex>,
}

/// Bytes packed per task.
#[cfg(not(target_arch = "wasm32"))]
const PACK_CHUNK: usize = 1 << 17;

/// `src` as 16-bit samples (`× scale`, rounded, clamped; NaN → 0), written in parallel into `dst`
/// (zeros past the end of `src`).
#[cfg(not(target_arch = "wasm32"))]
fn pack(dst: wgpu::WriteOnly<'_, [u8]>, src: &[f32], scale: f32) {
    use rayon::prelude::*;
    let fill = |at: usize, mut part: wgpu::WriteOnly<'_, [u8]>| {
        let mut tmp = vec![0u8; part.len()];
        for (j, o) in tmp.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = src.get(at / 2 + j).copied().unwrap_or(0.0);
            *o = ((v * scale).round().clamp(0.0, 65535.0) as u16).to_le_bytes();
        }
        part.copy_from_slice(&tmp);
    };
    let (chunks, rest) = dst.into_chunks::<PACK_CHUNK>();
    let parts: Vec<wgpu::WriteOnly<'_, [u8; PACK_CHUNK]>> = chunks.into_iter().collect();
    let tail = parts.len() * PACK_CHUNK;
    parts.into_par_iter().enumerate().for_each(|(k, part)| fill(k * PACK_CHUNK, part.into()));
    fill(tail, rest);
}

/// The module's seven bindings, unused ones on distinct spare buffers.
#[cfg(not(target_arch = "wasm32"))]
fn slots<'a>(b: [Option<&'a Buf>; 7], spare: &'a [Buf]) -> Vec<Option<&'a Buf>> {
    let mut k = 0;
    b.iter()
        .map(|s| {
            s.or_else(|| {
                k += 1;
                spare.get(k - 1)
            })
        })
        .collect()
}

/// Values per readback (and per staging buffer): 64 MiB.
#[cfg(not(target_arch = "wasm32"))]
const READ_CHUNK: usize = 16 << 20;

impl FocusPass {
    /// The device's name, when there is one to use.
    pub fn device_name() -> Option<String> {
        crate::enabled().then(crate::adapter_name).flatten()
    }

    /// Start a pass over `w × h` output pixels: method B (`depth`) or A, focus measured over
    /// `(2·radius+1)²`, A's weights blurred by `sigma`.
    pub fn begin(w: usize, h: usize, depth: bool, radius: usize, sigma: f32) -> Result<FocusPass, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if !crate::enabled() {
                return Err("GPU rendering is off".into());
            }
            let gpu = crate::device().ok_or_else(|| crate::unavailable_reason().unwrap_or_else(|| "no GPU".into()))?;
            let n = w * h;
            if n == 0 {
                return Err("empty output".into());
            }
            let state = if depth { 9 * n } else { 5 * n };
            if !gpu.fits(state) || !gpu.fits(3 * n) {
                return Err(format!("{w}×{h} needs buffers over the device's {} MiB limit", gpu.max_buffer >> 20));
            }
            let inner = guarded(gpu, || {
                let mut cx = Cx::new(gpu);
                let fs = gpu.buffer(5 * n);
                let cs = depth.then(|| gpu.buffer(9 * n));
                let spare: Vec<Buf> = (0..7).map(|_| gpu.buffer(1)).collect();
                let p = [n as u32];
                if depth {
                    cx.run("fdep_init", &p, &slots([None, None, None, None, Some(&fs), cs.as_ref(), None], &spare), groups1(n));
                } else {
                    cx.run("favg_init", &p, &slots([None, None, None, None, Some(&fs), None, None], &spare), groups1(n));
                }
                cx.flush();
                Inner {
                    gpu,
                    w,
                    h,
                    depth,
                    radius,
                    sigma,
                    fs,
                    cs,
                    frame: None,
                    staging: [None, None],
                    added: 0,
                    rgb: gpu.buffer(3 * n),
                    flags: gpu.buffer(n),
                    grey: gpu.buffer(n),
                    ml: gpu.buffer(n),
                    fm: gpu.buffer(n),
                    tmp: gpu.buffer(n),
                    subs: Vec::new(),
                    spare,
                }
            })?;
            Ok(FocusPass { inner })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (w, h, depth, radius, sigma);
            Err("no GPU in the browser build".into())
        }
    }

    /// Fold in the frame at sequence position `pos`: `src` resampled through `hm` (output →
    /// source pixels, row-major 3 × 3) into the output rectangle `out` = (x, y, w, h), multiplied
    /// by `inv_gain`, saturated at `sat`.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        pos: usize,
        src: &Rgb32f,
        hm: &[f64; 9],
        out: [usize; 4],
        inv_gain: [f32; 3],
        sat: f32,
        bicubic: bool,
    ) -> Result<(), String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let s = &mut self.inner;
            let (gpu, n, w, h) = (s.gpu, s.w * s.h, s.w, s.h);
            if out[2] != w || out[3] != h {
                return Err("the output rectangle changed".into());
            }
            let samples = 3 * src.width * src.height;
            let words = samples.div_ceil(2) + 1;
            if samples == 0 || !gpu.fits(words) {
                return Err(format!("a {}×{} frame is over the device's buffer limit", src.width, src.height));
            }
            if s.frame.as_ref().is_none_or(|b| b.len != words) {
                s.frame = Some(gpu.buffer(words));
            }
            // the frame's peak sets the 16-bit scale
            let flat: &[f32] = bytemuck::cast_slice(&src.data);
            let peak = {
                use rayon::prelude::*;
                flat.par_iter().copied().filter(|v| v.is_finite()).reduce(|| 0.0, f32::max)
            };
            let scale = if peak > 0.0 { 65535.0 / peak } else { 1.0 };
            let bytes = (words * 4) as u64;
            let slot = s.added % 2;
            let staging = stage(gpu, &mut s.staging[slot], bytes)?;
            let Some(fb) = s.frame.as_ref() else { return Err("no frame buffer".into()) };
            // pack, copy to the device, and map again once the copy is done
            let (copied, rx) = guarded(gpu, || {
                if let Ok(mut view) = staging.buf.get_mapped_range_mut(..bytes) {
                    pack(view.slice(..), flat, scale);
                }
                staging.buf.unmap();
                let mut enc = gpu.encoder();
                enc.copy_buffer_to_buffer(&staging.buf, 0, fb.raw(), 0, bytes);
                let copied = gpu.submit_indexed([enc.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                staging.buf.map_async(wgpu::MapMode::Write, .., move |r| {
                    let _ = tx.send(r);
                });
                (copied, rx)
            })?;
            staging.mapping = Some(rx);
            staging.used_by = Some(copied);
            s.added += 1;
            guarded(gpu, || {
                let Some(sb) = s.frame.as_ref() else { return };
                let mut cx = Cx::new(gpu);
                let mut p: Vec<u32> = vec![n as u32, w as u32, out[0] as u32, out[1] as u32, src.width as u32, src.height as u32, bicubic as u32];
                p.extend(hm.iter().map(|v| (*v as f32).to_bits()));
                p.extend(inv_gain.iter().map(|v| v.to_bits()));
                p.push(sat.to_bits());
                p.push((1.0 / scale).to_bits());
                let sp = &s.spare;
                cx.run("fwarp", &p, &slots([None, Some(&s.rgb), None, Some(&s.flags), None, None, Some(sb)], sp), groups1(n));
                cx.run("fgrey", &[n as u32], &slots([Some(&s.rgb), Some(&s.grey), None, None, None, None, None], sp), groups1(n));
                cx.run(
                    "fml",
                    &[n as u32, w as u32, h as u32],
                    &slots([Some(&s.grey), Some(&s.ml), None, Some(&s.flags), None, None, None], sp),
                    groups1(n),
                );
                box_mean(&mut cx, &s.ml, &s.tmp, &s.fm, w, h, s.radius);
                if s.depth {
                    let step = (n / 50_000).max(1);
                    let count = n.div_ceil(step);
                    let sub = gpu.buffer(count);
                    cx.run("fsub", &[count as u32, step as u32], &slots([Some(&s.fm), Some(&sub), None, None, None, None, None], sp), groups1(count));
                    cx.run(
                        "fdep_add",
                        &[n as u32, pos as u32],
                        &slots([Some(&s.rgb), None, Some(&s.fm), Some(&s.flags), Some(&s.fs), s.cs.as_ref(), None], sp),
                        groups1(n),
                    );
                    s.subs.push((sub, count));
                } else {
                    cx.run("favg_w", &[n as u32], &slots([Some(&s.fm), Some(&s.grey), None, None, None, None, None], sp), groups1(n));
                    let wt = gaussian(&mut cx, &s.grey, w, h, 1, s.sigma);
                    cx.run(
                        "favg_add",
                        &[n as u32, (pos as f32).to_bits()],
                        &slots([Some(&s.rgb), None, Some(&wt), Some(&s.flags), Some(&s.fs), None, None], sp),
                        groups1(n),
                    );
                }
                cx.flush();
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (pos, src, hm, out, inv_gain, sat, bicubic);
            Err("no GPU in the browser build".into())
        }
    }

    /// Method A's state: weighted RGB sums, weight sums, weighted position sums.
    #[allow(clippy::type_complexity)]
    pub fn finish_average(self) -> Result<(Vec<[f32; 3]>, Vec<f32>, Vec<f32>), String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let s = self.inner;
            let n = s.w * s.h;
            let f: Vec<f32> = read(s.gpu, &s.fs, 5 * n)?;
            let sum = f.get(..3 * n).unwrap_or_default().as_chunks::<3>().0.to_vec();
            Ok((sum, f.get(3 * n..4 * n).unwrap_or_default().to_vec(), f.get(4 * n..5 * n).unwrap_or_default().to_vec()))
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err("no GPU in the browser build".into())
        }
    }

    /// Method B's state.
    pub fn finish_depth(self) -> Result<DepthOut, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let s = self.inner;
            let n = s.w * s.h;
            let cs = s.cs.as_ref().ok_or("not a depth-map pass")?;
            let f: Vec<f32> = read(s.gpu, &s.fs, 5 * n)?;
            let c: Vec<u32> = read(s.gpu, cs, 9 * n)?;
            let plane = |k: usize| f.get(k * n..(k + 1) * n).unwrap_or_default().to_vec();
            let colours = |k: usize| -> Vec<[u16; 3]> {
                c.get(n + 2 * k * n..n + 2 * (k + 1) * n)
                    .unwrap_or_default()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|q| [(q[0] & 0xffff) as u16, (q[0] >> 16) as u16, (q[1] & 0xffff) as u16])
                    .collect()
            };
            let mut noise = f32::INFINITY;
            for (b, count) in &s.subs {
                let mut v: Vec<f32> = read::<f32>(s.gpu, b, *count)?.into_iter().filter(|v| v.is_finite() && *v > 0.0).collect();
                if v.is_empty() {
                    continue;
                }
                let k = v.len() / 10;
                v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                if let Some(x) = v.get(k).copied().filter(|x| *x > 0.0) {
                    noise = noise.min(x);
                }
            }
            Ok(DepthOut {
                best: c.get(..n).unwrap_or_default().iter().map(|&b| b.min(u16::MAX as u32) as u16).collect(),
                measures: [plane(0), plane(1), plane(2), plane(3), plane(4)],
                colours: [colours(0), colours(1), colours(2), colours(3)],
                noise,
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err("no GPU in the browser build".into())
        }
    }
}

/// The staging buffer of `slot`, mapped and at least `bytes` long: a new one (mapped at creation),
/// or the old one once the device has copied from it and mapped it again.
#[cfg(not(target_arch = "wasm32"))]
fn stage<'a>(gpu: &Gpu, slot: &'a mut Option<Staging>, bytes: u64) -> Result<&'a mut Staging, String> {
    if slot.as_ref().is_some_and(|st| st.buf.size() < bytes) {
        *slot = None;
    }
    let st = match slot {
        Some(st) => st,
        None => {
            let buf = guarded(gpu, || {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("focus frame staging"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                })
            })?;
            slot.insert(Staging { buf, mapping: None, used_by: None })
        }
    };
    if let Some(rx) = st.mapping.take() {
        // the copy that last read it: once done (and the frames before it), it maps again
        gpu.device
            .poll(wgpu::PollType::Wait { submission_index: st.used_by.take(), timeout: Some(std::time::Duration::from_secs(60)) })
            .map_err(|e| format!("waiting for an upload: {e}"))?;
        // another thread's poll may be the one that runs the callback: allow it a moment
        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("staging buffer: {e}")),
            Err(e) => return Err(format!("staging buffer: {e}")),
        }
    }
    Ok(st)
}

/// Mean over `(2r+1)²` (clamped edges): `src → tmp → dst` (`measure::box_mean_into`).
#[cfg(not(target_arch = "wasm32"))]
fn box_mean(cx: &mut Cx<'_>, src: &Buf, tmp: &Buf, dst: &Buf, w: usize, h: usize, r: usize) {
    let chunk = (2 * r + 1).clamp(16, 128);
    let p = [w as u32, h as u32, 1, r as u32, chunk as u32];
    cx.run("box_h", &p, &[Some(src), Some(tmp)], groups2(w.div_ceil(chunk), h, [64, 4]));
    cx.run("box_v", &p, &[Some(tmp), Some(dst)], groups2(w, h.div_ceil(chunk), [64, 4]));
}

/// Run `f` and report what went wrong on the device (an error raised, a buffer over the limit,
/// a panic) as an `Err`; a fatal error stops GPU use for the process, like a failed render.
#[cfg(not(target_arch = "wasm32"))]
fn guarded<R>(gpu: &Gpu, f: impl FnOnce() -> R) -> Result<R, String> {
    let _ = ctx::take_failure();
    let errors = ErrorScopes::push(gpu);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let scoped = errors.pop();
    let failure = ctx::take_failure().or(scoped);
    match (r, failure) {
        (Err(_), _) => {
            crate::mark_broken("a focus-stack kernel panicked");
            Err("a GPU focus-stack step panicked".into())
        }
        (Ok(_), Some(f)) => {
            if f.kind == ctx::FailKind::Fatal {
                crate::mark_broken(&f.reason);
            }
            Err(f.reason)
        }
        (Ok(r), None) => Ok(r),
    }
}

/// Read `len` values of `b` in chunks.
#[cfg(not(target_arch = "wasm32"))]
fn read<T: bytemuck::Pod>(gpu: &Gpu, b: &Buf, len: usize) -> Result<Vec<T>, String> {
    guarded(gpu, || {
        let mut out = Vec::with_capacity(len);
        let mut at = 0;
        while at < len {
            let k = READ_CHUNK.min(len - at);
            out.extend(gpu.read_range::<T>(b, at, k));
            at += k;
        }
        out
    })
}
