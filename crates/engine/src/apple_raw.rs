//! Apple's raw decoder (Core Image's `CIRAWFilter`, [`lightcraft_coreimage`]) as a second raw decoder on a Mac.
//!
//! Which decoder develops raws is a [`Mode`]: LightCraft's own only, Apple's for the raws LightCraft's can't develop
//! (`auto`, the default), or Apple's first for every raw but DNG (`apple`). `LIGHTCRAFT_RAW_DECODER` forces one.
//! Elsewhere (and before macOS 12) there is no Apple decoder and nothing changes.
//!
//! Apple's pictures are made to look like the built-in decoder's to the rest of LightCraft: scene-linear Rec.2020,
//! upright, white-balanced for the as-shot white, fitted to the frame the file's own headers give (when the built-in
//! decoder can read them), with the white-balance scale the catalog uses for the format (relative for the formats whose
//! look LightCraft fits from the camera's JPEG). See `docs/apple-raw.md`.
//!
//! Core Image runs in this process, so a crash in it would take LightCraft down: every decode runs on its own thread
//! with a time limit (and at most [`Limits::slots`] at once), Apple is turned off for the session after
//! [`Limits::timeouts_before_off`] decodes that ran over, and the desktop app leaves a marker file while a decode runs
//! ([`set_crash_markers`]): a marker still there at the next start means Apple crashed on that file, which is then left
//! to the built-in decoder.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering::AcqRel, Ordering::Acquire, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use lightcraft_color::Xy;
use lightcraft_color::cct::xy_to_temp_tint;
use lightcraft_coreimage as ci;
use lightcraft_pipeline::SourceInfo;
use lightcraft_raster::Rgb32f;
use lightcraft_raster::resample::{Filter, fit};
use lightcraft_raw::{RawFormat, RawInfo};
use serde_json::{Value, json};

use crate::files::{AltFacts, AltRaw, AltWhen};
use crate::media::{ProbeInfo, SourceLevel};

/// Which raw decoder develops raws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// LightCraft's own decoders only.
    Builtin,
    /// Apple's for the raws LightCraft's own can't develop (those it would show from their embedded preview).
    #[default]
    Auto,
    /// Apple's first for every raw but DNG (LightCraft's DNG support is complete: opcodes, profiles, masks), LightCraft's
    /// when Apple's fails.
    Apple,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "builtin" | "built-in" => Some(Mode::Builtin),
            "auto" => Some(Mode::Auto),
            "apple" => Some(Mode::Apple),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Builtin => "builtin",
            Mode::Auto => "auto",
            Mode::Apple => "apple",
        }
    }

    fn from_u8(v: u8) -> Mode {
        match v {
            0 => Mode::Builtin,
            2 => Mode::Apple,
            _ => Mode::Auto,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Mode::Builtin => 0,
            Mode::Auto => 1,
            Mode::Apple => 2,
        }
    }
}

/// The environment variable that forces a [`Mode`] over the preference: `builtin`, `auto` or `apple`.
pub const ENV: &str = "LIGHTCRAFT_RAW_DECODER";

/// The mode [`ENV`] forces (read once).
pub fn env_mode() -> Option<Mode> {
    static FORCED: OnceLock<Option<Mode>> = OnceLock::new();
    *FORCED.get_or_init(|| {
        let v = std::env::var(ENV).ok()?;
        let mode = Mode::parse(&v);
        if mode.is_none() {
            log::warn!("{ENV}={v} is not builtin, auto or apple: ignored");
        }
        mode
    })
}

/// How long and how many: the limits on Apple's decoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Decodes running at once (Core Image uses the GPU; more at once only take more memory).
    pub slots: usize,
    /// Apple's decoder is turned off for the session after this many decodes in a row ran over their time.
    pub timeouts_before_off: usize,
    /// Time for a thumbnail (≤ 512 px), a preview (≤ 2,560 px), a full-size decode, and a description.
    pub thumb: Duration,
    pub preview: Duration,
    pub full: Duration,
    pub probe: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            slots: 2,
            timeouts_before_off: 3,
            thumb: Duration::from_secs(20),
            preview: Duration::from_secs(40),
            full: Duration::from_secs(120),
            probe: Duration::from_secs(20),
        }
    }
}

impl Limits {
    fn for_edge(&self, max_edge: usize) -> Duration {
        if max_edge <= SourceLevel::Thumb.max_edge() {
            self.thumb
        } else if max_edge <= SourceLevel::Preview.max_edge() {
            self.preview
        } else {
            self.full
        }
    }
}

/// Develops a raw ([`lightcraft_coreimage::decode`], or a stand-in in tests).
pub type DecodeFn = dyn Fn(&Path, &ci::Options) -> Result<ci::Decoded, String> + Send + Sync;
/// Describes a raw ([`lightcraft_coreimage::probe`], or a stand-in in tests).
pub type ProbeFn = dyn Fn(&Path, &ci::Options) -> Result<ci::Info, String> + Send + Sync;

/// Apple's raw decoder behind its guard (time limits, slots, crash markers).
pub struct Apple {
    decode: Arc<DecodeFn>,
    probe: Arc<ProbeFn>,
    identity: String,
    guard: Arc<Guard>,
}

impl Apple {
    /// `markers`: the folder for crash markers (`None`: none are left).
    pub fn new(decode: Arc<DecodeFn>, probe: Arc<ProbeFn>, identity: String, limits: Limits, markers: Option<PathBuf>) -> Apple {
        Apple { decode, probe, identity, guard: Arc::new(Guard::new(limits, markers)) }
    }

    /// What develops the pixels (Core Image's revision and the macOS version): part of cache keys.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    fn develop(&self, path: &Path, options: ci::Options, timeout: Duration) -> Result<ci::Decoded, String> {
        let (decode, file) = (self.decode.clone(), path.to_path_buf());
        self.guard.run(path, timeout, move || decode(&file, &options))
    }

    fn describe(&self, path: &Path, options: ci::Options) -> Result<ci::Info, String> {
        let (probe, file) = (self.probe.clone(), path.to_path_buf());
        self.guard.run(path, self.guard.limits.probe, move || probe(&file, &options))
    }
}

/// The folder for crash markers ([`system`]'s guard writes one while a decode runs): set by the desktop app before its
/// first session. Without it (CLI, MCP, tests) no markers are left. `false` when it was set already.
pub fn set_crash_markers(dir: PathBuf) -> bool {
    MARKERS.set(dir).is_ok()
}

static MARKERS: OnceLock<PathBuf> = OnceLock::new();

/// This Mac's Apple decoder, shared by every session (its guard counts for the whole process); `None` off macOS and
/// before macOS 12.
pub fn system() -> Option<Arc<Apple>> {
    static SYSTEM: OnceLock<Option<Arc<Apple>>> = OnceLock::new();
    SYSTEM
        .get_or_init(|| {
            ci::available().then(|| {
                Arc::new(Apple::new(
                    Arc::new(|p: &Path, o: &ci::Options| ci::decode(p, o)),
                    Arc::new(|p: &Path, o: &ci::Options| ci::probe(p, o)),
                    ci::identity(),
                    Limits::default(),
                    MARKERS.get().cloned(),
                ))
            })
        })
        .clone()
}

/// A session's raw decoders: the mode chosen (`app.rawDecoder`) and Apple's decoder, when there is one.
pub struct RawDecoders {
    mode: AtomicU8,
    apple: Option<Arc<Apple>>,
}

impl Default for RawDecoders {
    fn default() -> Self {
        RawDecoders::new(None)
    }
}

impl RawDecoders {
    pub fn new(apple: Option<Arc<Apple>>) -> RawDecoders {
        RawDecoders { mode: AtomicU8::new(Mode::default().to_u8()), apple }
    }

    /// The mode chosen (the preference; [`Self::effective`] is what is used).
    pub fn mode(&self) -> Mode {
        Mode::from_u8(self.mode.load(Relaxed))
    }

    /// Choose the mode; `true` when what develops raws changed.
    pub fn set_mode(&self, mode: Mode) -> bool {
        let before = self.effective();
        self.mode.store(mode.to_u8(), Relaxed);
        self.effective() != before
    }

    /// The mode in use: [`ENV`]'s over the preference; built-in only when there is no Apple decoder or it was turned off.
    pub fn effective(&self) -> Mode {
        match &self.apple {
            Some(a) if !a.guard.off.load(Relaxed) => env_mode().unwrap_or_else(|| self.mode()),
            _ => Mode::Builtin,
        }
    }

    pub fn available(&self) -> bool {
        self.apple.is_some()
    }

    /// What develops raws, for cache keys: `None` when only the built-in decoders do.
    pub fn identity(&self) -> Option<String> {
        let mode = self.effective();
        let apple = self.apple.as_ref().filter(|_| mode != Mode::Builtin)?;
        Some(format!("{}:{}", mode.name(), apple.identity))
    }

    /// Apple's decoder, when it is in use for `path` (not for a file it crashed on before).
    fn active_for(&self, path: &str) -> Option<&Apple> {
        let apple = self.apple.as_deref().filter(|_| self.effective() != Mode::Builtin)?;
        (!apple.guard.is_skipped(path)).then_some(apple)
    }

    /// Develop the file at `path` (its `bytes`) no larger than `max_edge`, with Apple's decoder as the mode says.
    pub fn load(&self, path: &str, bytes: Vec<u8>, max_edge: usize) -> Result<(Rgb32f, SourceInfo), String> {
        let Some(apple) = self.active_for(path) else { return crate::files::load_vec(bytes, max_edge) };
        let when = match self.effective() {
            Mode::Apple if lightcraft_raw::probe(&bytes) != Some(RawFormat::Dng) => AltWhen::First,
            _ => AltWhen::OnFallback,
        };
        let file = Path::new(path);
        let decode = |b: &[u8], edge: usize| develop(apple, file, b, edge);
        let probe = |b: &[u8]| describe(apple, file, b);
        crate::files::load_vec_with(bytes, max_edge, Some(&AltRaw { when, name: "Apple", decode: &decode, probe: &probe }))
    }

    /// Describe the file at `path` (its `bytes`) for import, asking Apple's decoder about the raws the built-in one
    /// can't read.
    pub fn probe(&self, path: &str, bytes: &[u8]) -> Result<ProbeInfo, String> {
        let Some(apple) = self.active_for(path) else { return crate::files::probe_bytes(path, bytes) };
        let file = Path::new(path);
        let decode = |b: &[u8], edge: usize| develop(apple, file, b, edge);
        let probe = |b: &[u8]| describe(apple, file, b);
        crate::files::probe_bytes_with(path, bytes, Some(&AltRaw { when: AltWhen::OnFallback, name: "Apple", decode: &decode, probe: &probe }))
    }

    /// Forget the files Apple's decoder crashed on (they get it again).
    pub fn clear_skipped(&self) {
        if let Some(a) = &self.apple {
            a.guard.clear_skipped();
        }
    }

    /// `{mode, effective, available, reason, overriddenBy, identity, lastFallback, skipped}` (`app.rawDecoder`).
    pub fn status(&self) -> Value {
        let reason = match &self.apple {
            None if cfg!(target_os = "macos") => Some("Apple's raw decoder needs macOS 12 or later".to_string()),
            None => Some("Apple's raw decoder is only on macOS".to_string()),
            Some(a) if a.guard.off.load(Relaxed) => Some(a.guard.off_reason()),
            Some(_) => None,
        };
        let guard = self.apple.as_ref().map(|a| &a.guard);
        json!({
            "mode": self.mode().name(),
            "effective": self.effective().name(),
            "available": self.available(),
            "reason": reason,
            "overriddenBy": env_mode().map(|_| ENV),
            "identity": self.apple.as_ref().map(|a| a.identity.clone()),
            "lastFallback": guard.and_then(|g| lock(&g.last_fallback).clone()),
            "skipped": guard.map(|g| lock(&g.skipped).clone()).unwrap_or_default(),
        })
    }
}

/// Develop the raw at `path` with Apple's decoder, as the built-in decoder would hand it to the pipeline.
fn develop(apple: &Apple, path: &Path, bytes: &[u8], max_edge: usize) -> Result<(Rgb32f, SourceInfo), String> {
    let format = lightcraft_raw::probe(bytes).ok_or("not a raw file")?;
    let header = lightcraft_raw::probe_info(bytes).ok();
    let options = ci::Options {
        max_edge,
        draft: max_edge <= SourceLevel::Thumb.max_edge(),
        // with the file's own frame known, its lens corrections are LightCraft's ("Enable Profile Corrections");
        // without it, Apple's (Olympus and Panasonic raws need theirs)
        lens_correction: header.is_none(),
        ..Default::default()
    };
    let decoded = apple.develop(path, options, apple.guard.limits.for_edge(max_edge))?;
    adapt(decoded, format, header.as_ref(), max_edge)
}

/// Describe the raw at `path` for import with Apple's decoder (only asked when the built-in decoder can't read its
/// headers, so with Apple's lens corrections, as [`develop`] will develop it).
fn describe(apple: &Apple, path: &Path, bytes: &[u8]) -> Result<AltFacts, String> {
    let format = lightcraft_raw::probe(bytes).ok_or("not a raw file")?;
    let info = apple.describe(path, ci::Options { lens_correction: true, ..Default::default() })?;
    let side = |v: usize| u32::try_from(v).ok().filter(|&v| v > 0).ok_or("Apple's raw decoder gave no size");
    let white = info.white_xy.map(|(x, y)| Xy { x, y });
    Ok(AltFacts { width: side(info.width)?, height: side(info.height)?, as_shot_wb: Some(as_shot(format, white)) })
}

/// The as-shot white balance LightCraft records for a raw developed for `white`: the relative scale's 6500 K / 0 for
/// the formats the catalog gives relative white balance (`camera_preview::file_local_look`), else the white as
/// Kelvin and tint.
fn as_shot(format: RawFormat, white: Option<Xy>) -> (f64, f64) {
    if crate::camera_preview::file_local_look(format) {
        return (6500.0, 0.0);
    }
    white
        .map(xy_to_temp_tint)
        .filter(|(t, tint)| t.is_finite() && tint.is_finite())
        .map(|(t, tint)| (t.round(), tint.round()))
        .unwrap_or((6500.0, 0.0))
}

/// Apple's picture as the pipeline takes a raw: fitted to the file's own frame (`header`, when the built-in decoder
/// reads it), no larger than `max_edge`, with its source facts.
pub(crate) fn adapt(d: ci::Decoded, format: RawFormat, header: Option<&RawInfo>, max_edge: usize) -> Result<(Rgb32f, SourceInfo), String> {
    if d.width == 0 || d.height == 0 || d.width.checked_mul(d.height) != Some(d.rgb.len()) {
        return Err("Apple's raw decoder returned a picture of the wrong size".into());
    }
    let white = d.white_xy.map(|(x, y)| Xy { x, y }).or_else(|| header.map(lightcraft_raw::color::as_shot_white_xy_of));
    let mut img = Rgb32f { width: d.width, height: d.height, data: d.rgb };
    if let Some(h) = header {
        img = fit_frame(img, frame(h))?;
    }
    if img.width.max(img.height) > max_edge {
        img = fit(&img, max_edge, max_edge, Filter::Box);
    }
    let (temp, tint) = as_shot(format, white);
    let info = SourceInfo {
        raw: true,
        as_shot_temp: temp,
        as_shot_tint: tint,
        relative_wb: crate::camera_preview::file_local_look(format),
        lens: header.and_then(crate::files::embedded_lens),
        ..Default::default()
    };
    Ok((img, info))
}

/// The developed frame the file's headers give: its default crop (else its active area), upright.
fn frame(raw: &RawInfo) -> (usize, usize) {
    let (mut w, mut h) = (raw.crop.width, raw.crop.height);
    if w <= 1 || h <= 1 {
        (w, h) = (raw.active_area.width, raw.active_area.height);
    }
    if raw.orientation.swaps_axes() { (h, w) } else { (w, h) }
}

/// How far Apple's frame may differ from the file's (its default crop can be a little larger or smaller): up to this,
/// the picture is cropped to the file's shape; beyond it, it is refused.
const FRAME_TOLERANCE: f64 = 0.05;

/// `img` centre-cropped to the shape of a `fw` × `fh` frame, so normalised coordinates (crops, masks, region renders)
/// mean the same as with the built-in decoder. An error when the shapes differ by more than [`FRAME_TOLERANCE`] (a
/// picture turned 90° from the file's frame, say).
fn fit_frame(img: Rgb32f, (fw, fh): (usize, usize)) -> Result<Rgb32f, String> {
    if fw == 0 || fh == 0 {
        return Ok(img);
    }
    let want = fw as f64 / fh as f64;
    let have = img.width as f64 / img.height as f64;
    let off = have / want - 1.0;
    if off.abs() <= 0.002 {
        return Ok(img);
    }
    if off.abs() > FRAME_TOLERANCE {
        let turned = ((1.0 / have) / want - 1.0).abs() <= FRAME_TOLERANCE;
        let how = if turned { "is turned from" } else { "does not fit" };
        return Err(format!("Apple's picture ({} × {}) {how} the file's frame ({fw} × {fh})", img.width, img.height));
    }
    let (w, h) = if have > want {
        (((img.height as f64 * want).round() as usize).clamp(1, img.width), img.height)
    } else {
        (img.width, ((img.width as f64 / want).round() as usize).clamp(1, img.height))
    };
    let (x0, y0) = ((img.width - w) / 2, (img.height - h) / 2);
    Ok(img.into_crop(x0, y0, w, h))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Time limits, slots and crash markers around Apple's decoder.
struct Guard {
    limits: Limits,
    busy: Mutex<usize>,
    freed: Condvar,
    /// Decodes in a row that ran over their time.
    timeouts: AtomicUsize,
    /// Decodes that ran over their time and are still running (each holds a slot).
    hung: AtomicUsize,
    off: AtomicBool,
    last_fallback: Mutex<Option<String>>,
    markers: Option<PathBuf>,
    /// Files Apple's decoder crashed on (markers a previous run left), most recent last.
    skipped: Mutex<Vec<String>>,
    seq: AtomicU64,
}

/// Most files remembered as crashing Apple's decoder.
const MAX_SKIPPED: usize = 1000;
/// The list of files Apple's decoder crashed on, in the marker folder (one path per line).
const SKIP_FILE: &str = "skip.txt";
const MARKER_PREFIX: &str = "inflight-";

impl Guard {
    fn new(limits: Limits, markers: Option<PathBuf>) -> Guard {
        let skipped = markers.as_deref().map(collect_crashes).unwrap_or_default();
        Guard {
            limits,
            busy: Mutex::new(0),
            freed: Condvar::new(),
            timeouts: AtomicUsize::new(0),
            hung: AtomicUsize::new(0),
            off: AtomicBool::new(false),
            last_fallback: Mutex::new(None),
            markers,
            skipped: Mutex::new(skipped),
            seq: AtomicU64::new(0),
        }
    }

    fn off_reason(&self) -> String {
        "Apple's raw decoder was turned off for this session: its decodes kept taking too long".into()
    }

    fn turn_off(&self) {
        if !self.off.swap(true, Relaxed) {
            log::warn!("{}", self.off_reason());
        }
    }

    fn is_skipped(&self, path: &str) -> bool {
        lock(&self.skipped).iter().any(|p| p == path)
    }

    fn clear_skipped(&self) {
        lock(&self.skipped).clear();
        if let Some(dir) = &self.markers {
            let _ = std::fs::remove_file(dir.join(SKIP_FILE));
        }
    }

    /// Run `f` (a call into Core Image for `path`) on its own thread, in a slot, within `timeout`.
    fn run<T: Send + 'static>(
        self: &Arc<Self>,
        path: &Path,
        timeout: Duration,
        f: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let r = self.run_guarded(path, timeout, f);
        if let Err(e) = &r {
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            *lock(&self.last_fallback) = Some(format!("{name}: {e}"));
        }
        r
    }

    fn run_guarded<T: Send + 'static>(
        self: &Arc<Self>,
        path: &Path,
        timeout: Duration,
        f: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        if self.off.load(Relaxed) {
            return Err(self.off_reason());
        }
        // wait for a slot (as long as a decode may take: falling back here would show a different picture)
        let slot = self.acquire(timeout)?;
        let marker = self.mark(path);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let thread_marker = marker.clone();
        let state = Arc::new(AtomicU8::new(RUNNING));
        let thread_state = state.clone();
        let spawned = std::thread::Builder::new().name("apple-raw".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .unwrap_or_else(|_| Err("Apple's raw decoder failed unexpectedly".to_string()));
            // finished (in time or not): no crash to remember, and the slot is free
            if let Some(m) = thread_marker {
                let _ = std::fs::remove_file(m);
            }
            if thread_state.swap(DONE, AcqRel) == TIMED_OUT {
                slot.0.hung.fetch_sub(1, AcqRel);
            }
            drop(slot);
            let _ = tx.send(r);
        });
        if let Err(e) = spawned {
            if let Some(m) = marker {
                let _ = std::fs::remove_file(m);
            }
            return Err(format!("could not start Apple's raw decoder: {e}"));
        }
        match rx.recv_timeout(timeout) {
            Ok(r) => {
                self.timeouts.store(0, Relaxed);
                r
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // counted as hung before the thread can see it timed out (it uncounts itself when it finishes)
                self.hung.fetch_add(1, AcqRel);
                if state.compare_exchange(RUNNING, TIMED_OUT, AcqRel, Acquire).is_err() {
                    // it finished just now
                    self.hung.fetch_sub(1, AcqRel);
                    return rx.recv().unwrap_or_else(|_| Err("Apple's raw decoder stopped unexpectedly".into()));
                }
                if self.timeouts.fetch_add(1, Relaxed) + 1 >= self.limits.timeouts_before_off {
                    self.turn_off();
                }
                Err(format!("Apple's raw decoder took longer than {} s", timeout.as_secs_f64().round()))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err("Apple's raw decoder stopped unexpectedly".into()),
        }
    }

    /// A slot, waiting up to `wait` for one (and no longer once the decoder is turned off).
    fn acquire(self: &Arc<Self>, wait: Duration) -> Result<Slot, String> {
        let deadline = web_time::Instant::now() + wait;
        let mut busy = lock(&self.busy);
        while *busy >= self.limits.slots.max(1) {
            // every slot held by a decode that never finished: none ever will be free
            if self.hung.load(Acquire) >= self.limits.slots.max(1) {
                self.turn_off();
            }
            if self.off.load(Relaxed) {
                return Err(self.off_reason());
            }
            let now = web_time::Instant::now();
            if now >= deadline {
                return Err("Apple's raw decoder is busy".into());
            }
            busy = self.freed.wait_timeout(busy, (deadline - now).min(Duration::from_millis(500))).unwrap_or_else(PoisonError::into_inner).0;
        }
        *busy += 1;
        Ok(Slot(self.clone()))
    }

    /// Leave a marker naming `path` while Core Image works on it (`None`: no marker folder, or it can't be written).
    fn mark(&self, path: &Path) -> Option<PathBuf> {
        let dir = self.markers.as_ref()?;
        let m = dir.join(format!("{MARKER_PREFIX}{}-{}", std::process::id(), self.seq.fetch_add(1, Relaxed)));
        std::fs::write(&m, path.to_string_lossy().as_bytes()).ok()?;
        Some(m)
    }
}

/// A decode's state, between its thread and the caller waiting for it.
const RUNNING: u8 = 0;
const DONE: u8 = 1;
const TIMED_OUT: u8 = 2;

/// A decode slot, freed when dropped.
struct Slot(Arc<Guard>);

impl Drop for Slot {
    fn drop(&mut self) {
        let mut busy = lock(&self.0.busy);
        *busy = busy.saturating_sub(1);
        self.0.freed.notify_one();
    }
}

/// The files Apple's decoder crashed on: the list kept in `dir`, plus the files of markers a previous run left there
/// (which are then removed and the list saved).
fn collect_crashes(dir: &Path) -> Vec<String> {
    let _ = std::fs::create_dir_all(dir);
    let list = dir.join(SKIP_FILE);
    let mut skipped: Vec<String> = std::fs::read_to_string(&list).unwrap_or_default().lines().filter(|l| !l.is_empty()).map(str::to_string).collect();
    let mut found = false;
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if !entry.file_name().to_string_lossy().starts_with(MARKER_PREFIX) {
            continue;
        }
        if let Ok(p) = std::fs::read_to_string(entry.path())
            && !p.is_empty()
            && !p.contains('\n')
        {
            log::warn!("Apple's raw decoder stopped LightCraft while it developed {p}: that file is left to the built-in decoder");
            skipped.retain(|s| *s != p);
            skipped.push(p);
            found = true;
        }
        let _ = std::fs::remove_file(entry.path());
    }
    if skipped.len() > MAX_SKIPPED {
        skipped.drain(..skipped.len() - MAX_SKIPPED);
    }
    if found {
        let _ = std::fs::write(&list, skipped.join("\n"));
    }
    skipped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raw developed by a stand-in for Apple's decoder: `w` × `h`, a gradient, developed for D50.
    fn decoded(w: usize, h: usize) -> ci::Decoded {
        let rgb = (0..w * h).map(|i| [(i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.2]).collect();
        ci::Decoded { width: w, height: h, rgb, white_xy: Some((0.3457, 0.3585)), baseline_exposure: 0.0, native: (w, h) }
    }

    #[test]
    fn apple_pictures_fit_the_files_own_frame() {
        let img = |w, h| Rgb32f::from_fn(w, h, |x, y| [x as f32, y as f32, 0.0]);
        // within rounding: as is
        let same = fit_frame(img(6024, 4024), (6000, 4000)).unwrap();
        assert_eq!((same.width, same.height), (6024, 4024));
        // a little wider or taller: centre-cropped to the frame's shape
        let wide = fit_frame(img(6100, 4000), (6000, 4000)).unwrap();
        assert_eq!((wide.width, wide.height, wide.data[0]), (6000, 4000, [50.0, 0.0, 0.0]));
        let tall = fit_frame(img(3000, 2100), (6000, 4000)).unwrap();
        assert_eq!((tall.width, tall.height, tall.data[0]), (3000, 2000, [0.0, 50.0, 0.0]));
        // turned, or another shape: refused with the sizes
        let e = fit_frame(img(400, 600), (6000, 4000)).unwrap_err();
        assert!(e.contains("is turned from") && e.contains("400 × 600") && e.contains("6000 × 4000"), "{e}");
        assert!(fit_frame(img(600, 300), (6000, 4000)).unwrap_err().contains("does not fit"));
        // no frame to fit
        assert_eq!(fit_frame(img(7, 3), (0, 0)).unwrap().width, 7);

        // the source facts: relative white balance for the formats the catalog gives it, else the white in Kelvin
        let (pic, info) = adapt(decoded(100, 50), RawFormat::Cr3, None, 40).unwrap();
        assert_eq!((pic.width, pic.height), (40, 20), "fitted to max_edge");
        assert!(info.raw && info.relative_wb && info.camera_color.is_none() && info.camera_tone.is_none());
        assert_eq!((info.as_shot_temp, info.as_shot_tint), (6500.0, 0.0));
        let (_, info) = adapt(decoded(100, 50), RawFormat::Orf, None, usize::MAX).unwrap();
        assert!(info.raw && !info.relative_wb);
        assert!((4900.0..5100.0).contains(&info.as_shot_temp), "D50 is about 5000 K: {}", info.as_shot_temp);
        let no_white = ci::Decoded { white_xy: None, ..decoded(10, 5) };
        assert_eq!(adapt(no_white, RawFormat::Orf, None, 64).unwrap().1.as_shot_temp, 6500.0);
        let bad = ci::Decoded { rgb: vec![[0.0; 3]; 3], ..decoded(10, 5) };
        assert!(adapt(bad, RawFormat::Orf, None, 64).is_err());
        // with the file's headers (a DNG here): Apple's slightly wider picture is cropped to their frame, and the
        // headers' white stands in when Apple gives none
        let dng = crate::tests_xmp::synthetic_dng_with(None, Default::default());
        let header = lightcraft_raw::probe_info(&dng).unwrap();
        let (fw, fh) = frame(&header);
        let wider = ci::Decoded { white_xy: None, ..decoded(fw + fw / 50, fh) };
        let (pic, info) = adapt(wider, header.format, Some(&header), usize::MAX).unwrap();
        assert_eq!((pic.width, pic.height), (fw, fh));
        let (t, tint) = xy_to_temp_tint(lightcraft_raw::color::as_shot_white_xy_of(&header));
        assert_eq!((info.as_shot_temp, info.as_shot_tint), (t.round(), tint.round()));
    }

    /// A stand-in for Core Image that hangs while `hang` is set (for at most 5 s), panics for `panic.cr3`, and
    /// counts its calls.
    struct Stub {
        hang: Mutex<bool>,
        released: Condvar,
        calls: AtomicUsize,
    }

    impl Stub {
        fn new() -> Arc<Stub> {
            Arc::new(Stub { hang: Mutex::new(false), released: Condvar::new(), calls: AtomicUsize::new(0) })
        }
        fn set_hang(&self, on: bool) {
            *lock(&self.hang) = on;
            self.released.notify_all();
        }
        fn decode(&self, path: &Path, o: &ci::Options) -> Result<ci::Decoded, String> {
            self.calls.fetch_add(1, Relaxed);
            if path.ends_with("panic.cr3") {
                panic!("a nil from Objective-C");
            }
            let mut hang = lock(&self.hang);
            let deadline = web_time::Instant::now() + Duration::from_secs(5);
            while *hang && web_time::Instant::now() < deadline {
                hang = self.released.wait_timeout(hang, Duration::from_millis(50)).unwrap_or_else(PoisonError::into_inner).0;
            }
            let edge = o.max_edge.min(90);
            Ok(decoded(edge, edge * 2 / 3))
        }
        fn apple(self: &Arc<Self>, limits: Limits, markers: Option<PathBuf>) -> Arc<Apple> {
            let (d, p) = (self.clone(), self.clone());
            Arc::new(Apple::new(
                Arc::new(move |path: &Path, o: &ci::Options| d.decode(path, o)),
                Arc::new(move |path: &Path, o: &ci::Options| {
                    p.decode(path, o).map(|d| ci::Info {
                        width: d.width,
                        height: d.height,
                        native: d.native,
                        white_xy: d.white_xy,
                        baseline_exposure: 0.0,
                    })
                }),
                "stub-1".into(),
                limits,
                markers,
            ))
        }
    }

    fn quick() -> Limits {
        let t = Duration::from_millis(150);
        Limits { slots: 1, timeouts_before_off: 2, thumb: t, preview: t, full: t, probe: t }
    }

    /// Wait (up to 5 s) until `f` holds.
    fn until(f: impl Fn() -> bool) {
        let deadline = web_time::Instant::now() + Duration::from_secs(5);
        while !f() && web_time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(f(), "timed out waiting");
    }

    #[test]
    fn the_guard_times_out_waits_for_a_slot_and_turns_apple_off() {
        let stub = Stub::new();
        let limits = quick();
        let apple = stub.apple(limits, None);
        let raw = RawDecoders::new(Some(apple.clone()));
        assert_eq!((raw.mode(), raw.effective()), (Mode::Auto, env_mode().unwrap_or(Mode::Auto)));
        let opts = || ci::Options { max_edge: 64, ..Default::default() };
        assert_eq!(apple.develop(Path::new("a.cr3"), opts(), limits.thumb).unwrap().width, 64);
        // a panic in Core Image's call is an error
        let e = apple.develop(Path::new("panic.cr3"), opts(), limits.thumb).unwrap_err();
        assert!(e.contains("failed unexpectedly"), "{e}");
        // the one slot taken by a slow decode: the next one waits for it, then gives up (not counted against Apple)
        stub.set_hang(true);
        let calls = stub.calls.load(Relaxed);
        let slow = {
            let apple = apple.clone();
            std::thread::spawn(move || {
                apple.develop(Path::new("slow.cr3"), ci::Options { max_edge: 64, ..Default::default() }, Duration::from_secs(5))
            })
        };
        until(|| stub.calls.load(Relaxed) > calls);
        assert!(apple.develop(Path::new("c.cr3"), opts(), limits.thumb).unwrap_err().contains("busy"));
        assert!(raw.status()["lastFallback"].as_str().is_some_and(|f| f.starts_with("c.cr3: ")), "{}", raw.status());
        stub.set_hang(false);
        assert!(slow.join().unwrap().is_ok());
        // over time: an error; it keeps its slot until it finishes, so with every slot stuck Apple is turned off
        stub.set_hang(true);
        let e = apple.develop(Path::new("e.cr3"), opts(), limits.thumb).unwrap_err();
        assert!(e.contains("took longer than"), "{e}");
        let calls = stub.calls.load(Relaxed);
        let e = apple.develop(Path::new("f.cr3"), opts(), limits.thumb).unwrap_err();
        assert!(e.contains("turned off"), "{e}");
        assert_eq!(stub.calls.load(Relaxed), calls, "not asked any more");
        assert_eq!((raw.effective(), raw.identity()), (Mode::Builtin, None));
        assert!(raw.status()["reason"].as_str().is_some_and(|r| r.contains("turned off")));
        stub.set_hang(false);
    }

    #[test]
    fn apple_is_turned_off_after_decodes_in_a_row_run_over() {
        let stub = Stub::new();
        let limits = Limits { slots: 3, ..quick() };
        let apple = stub.apple(limits, None);
        let opts = || ci::Options { max_edge: 64, ..Default::default() };
        let over = |name: &str| apple.develop(Path::new(name), opts(), limits.thumb).unwrap_err();
        stub.set_hang(true);
        assert!(over("a.cr3").contains("took longer"));
        // it finishes; one in time resets the count
        stub.set_hang(false);
        until(|| apple.guard.hung.load(Acquire) == 0);
        assert!(apple.develop(Path::new("b.cr3"), opts(), Duration::from_secs(5)).is_ok());
        stub.set_hang(true);
        assert!(over("c.cr3").contains("took longer"));
        assert!(!apple.guard.off.load(Relaxed), "one over time");
        assert!(over("d.cr3").contains("took longer"));
        assert!(apple.guard.off.load(Relaxed), "two in a row");
        assert!(over("e.cr3").contains("turned off"));
        stub.set_hang(false);
        until(|| apple.guard.hung.load(Acquire) == 0);
    }

    #[test]
    fn a_file_apple_crashed_on_is_left_to_the_built_in_decoder() {
        let dir = std::env::temp_dir().join(format!("lc-apple-markers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let stub = Stub::new();
        // a decode that finishes leaves no marker
        let apple = stub.apple(Limits::default(), Some(dir.clone()));
        apple.develop(Path::new("/photos/ok.cr3"), ci::Options { max_edge: 32, ..Default::default() }, Duration::from_secs(5)).unwrap();
        let markers = || std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(MARKER_PREFIX)).count();
        assert_eq!(markers(), 0);
        // one left behind (the process died during the decode) puts its file on the list at the next start
        assert!(apple.guard.mark(Path::new("/photos/crash.cr3")).is_some());
        let raw = RawDecoders::new(Some(stub.apple(Limits::default(), Some(dir.clone()))));
        assert_eq!(markers(), 0);
        assert_eq!(raw.status()["skipped"], json!(["/photos/crash.cr3"]));
        if raw.effective() != Mode::Builtin {
            assert!(raw.active_for("/photos/crash.cr3").is_none() && raw.active_for("/photos/other.cr3").is_some());
        }
        // the list is kept, until cleared
        assert_eq!(RawDecoders::new(Some(stub.apple(Limits::default(), Some(dir.clone())))).status()["skipped"], json!(["/photos/crash.cr3"]));
        raw.clear_skipped();
        assert_eq!(RawDecoders::new(Some(stub.apple(Limits::default(), Some(dir.clone())))).status()["skipped"], json!([]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue #2 end to end with a stand-in for Apple's decoder: a raw only it develops imports and renders as a raw;
    /// one imported while only the built-in decoders were used stays preview-only (and is shown from its preview) until
    /// Reload from Disk; in built-in mode Apple is never asked.
    #[test]
    fn apple_develops_the_raws_the_built_in_decoder_cannot() {
        use serde_json::json;
        if env_mode().is_some() {
            eprintln!("skipped: {ENV} is set");
            return;
        }
        let dir = std::env::temp_dir().join(format!("lc-apple-raw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (lib, files) = (dir.join("lib"), dir.join("files"));
        std::fs::create_dir_all(&files).unwrap();
        let (a, b) = (files.join("IMG_0001.CR3"), files.join("IMG_0002.CR3"));
        // (different files: an import skips a file it has already)
        std::fs::write(&a, crate::files::tests::cr3_with_preview(96, 64)).unwrap();
        std::fs::write(&b, crate::files::tests::cr3_with_preview(98, 64)).unwrap();
        let stub = Stub::new();
        let mut s = crate::Session::new().with_fs_and(Some(stub.apple(Limits::default(), None)));
        s.open_library(&lib, false).unwrap();
        let import = |s: &mut crate::Session, p: &std::path::Path| {
            let r = s.execute("library.import", &json!({"paths": [p.to_string_lossy()]})).unwrap();
            lightcraft_catalog::PhotoId(r["imported"][0].as_u64().unwrap())
        };
        // built-in only: preview-only, as before, and Apple is never asked
        assert!(s.media.raw.set_mode(Mode::Builtin));
        let old = import(&mut s, &b);
        assert!(s.catalog.photo(old).unwrap().preview_only.is_some());
        let loader = s.media.file_loader.clone().unwrap();
        assert!(!loader(&b.to_string_lossy(), 512).unwrap().1.raw);
        assert_eq!(stub.calls.load(Relaxed), 0);
        // auto: a new import is a raw that develops, described and developed by Apple
        assert!(s.media.raw.set_mode(Mode::Auto));
        let id = import(&mut s, &a);
        let p = s.catalog.photo(id).unwrap().clone();
        assert_eq!((p.preview_only.as_deref(), p.width, p.height, p.develops_raw(), p.relative_wb()), (None, 90, 60, true, true));
        let (img, info) = loader(&a.to_string_lossy(), 64).unwrap();
        assert_eq!((img.width, img.height, info.raw, info.relative_wb), (64, 42, true, true));
        assert!(s.render_job(id, 48, 48, false, true).unwrap().run().rendered.is_ok());
        // the photo imported preview-only keeps its embedded preview until it is reloaded
        let calls = stub.calls.load(Relaxed);
        let job = s.render_job(old, 48, 48, false, true).unwrap().run();
        assert!(job.rendered.is_ok() && !job.loaded.unwrap().info_or(SourceInfo::default()).raw);
        assert_eq!(stub.calls.load(Relaxed), calls);
        s.execute("photo.reload", &json!({"ids": [old.0]})).unwrap();
        let p = s.catalog.photo(old).unwrap();
        assert_eq!((p.preview_only.as_deref(), p.width, p.height), (None, 90, 60));
        let job = s.render_job(old, 48, 48, false, true).unwrap().run();
        assert!(job.loaded.unwrap().info_or(SourceInfo::default()).raw);
        assert!(s.media.raw.status()["identity"] == json!("stub-1"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue #2: where Apple's decoder may develop a raw, its renders (thumbnail, view and variant caches, job keys) and
    /// sources are told apart from the built-in decoder's; JPEGs, raws shown from their embedded preview, and every
    /// photo in built-in mode keep the keys they had; a source loaded before the decoder changed is not taken.
    #[test]
    fn renders_and_sources_carry_the_raw_decoder() {
        use crate::media::{SourceLevel, content_key};
        use lightcraft_codecs::{ChromaSubsampling, EncodeImage, EncodeMeta, Samples, encode_jpeg};
        use lightcraft_preview::Hasher128;
        use serde_json::json;
        if env_mode().is_some() {
            eprintln!("skipped: {ENV} is set");
            return;
        }
        let dir = std::env::temp_dir().join(format!("lc-apple-keys-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (lib, files) = (dir.join("lib"), dir.join("files"));
        std::fs::create_dir_all(&files).unwrap();
        let (raw_path, jpeg_path) = (files.join("IMG_0001.CR3"), files.join("IMG_0002.jpg"));
        std::fs::write(&raw_path, crate::files::tests::cr3_with_preview(96, 64)).unwrap();
        let px = vec![120u8; 32 * 24 * 3];
        std::fs::write(
            &jpeg_path,
            encode_jpeg(&EncodeImage::new(32, 24, 3, Samples::U8(&px)), 90, ChromaSubsampling::S444, &EncodeMeta::default()).unwrap(),
        )
        .unwrap();
        let mut s = crate::Session::new().with_fs_and(Some(Stub::new().apple(Limits::default(), None)));
        s.open_library(&lib, false).unwrap();
        let r = s.execute("library.import", &json!({"paths": [raw_path.to_string_lossy(), jpeg_path.to_string_lossy()]})).unwrap();
        let ids: Vec<_> = r["imported"].as_array().unwrap().iter().map(|v| lightcraft_catalog::PhotoId(v.as_u64().unwrap())).collect();
        let (raw, jpeg) = (ids[0], ids[1]);
        let keys = |s: &mut crate::Session, id| {
            let t = s.thumb_job(id, 256).unwrap();
            let l = s.loupe_job(id, 300, 200, true).unwrap();
            let v = s.variant_job(id, &lightcraft_develop::DevelopSettings::default(), 128).unwrap();
            let q = s.quick_view_job(id, 300, true).unwrap();
            (t.key, t.cache.map(|c| c.1), l.view_cache.map(|c| c.1), v.cache.map(|c| c.1), q.key, t.source_key)
        };
        let (auto_raw, auto_jpeg) = (keys(&mut s, raw), keys(&mut s, jpeg));
        assert!(s.media.raw.set_mode(Mode::Builtin));
        let (built_raw, built_jpeg) = (keys(&mut s, raw), keys(&mut s, jpeg));
        assert_eq!(auto_jpeg, built_jpeg);
        assert!(
            auto_raw.0 != built_raw.0
                && auto_raw.1 != built_raw.1
                && auto_raw.2 != built_raw.2
                && auto_raw.3 != built_raw.3
                && auto_raw.4 != built_raw.4
                && auto_raw.5 != built_raw.5
        );
        // built-in only: the key of old
        let p = s.catalog.photo(raw).unwrap().clone();
        assert_eq!(s.media.decoder_salt(&p), 0);
        assert_eq!(built_raw.5, Some(Hasher128::new().str(&content_key(&p)).finish()));
        // a raw shown from its embedded preview looks the same in every mode; developed, it is another picture of the
        // same content
        assert!(s.media.raw.set_mode(Mode::Auto));
        let shown = lightcraft_catalog::Photo { preview_only: Some("CR3 variant".into()), ..(*p).clone() };
        assert!(s.media.decoder_salt(&p) != 0 && s.media.decoder_salt(&shown) == 0);
        // a source loaded by the built-in decoder is not taken once Apple's is in use (and the other way round)
        assert!(s.media.raw.set_mode(Mode::Builtin));
        let before = s.render_job(raw, 64, 64, false, true).unwrap().run();
        assert!(s.media.raw.set_mode(Mode::Auto));
        s.accept(&before);
        assert!(!s.media.has_source(raw, before.level));
        let now = s.render_job(raw, 64, 64, false, true).unwrap().run();
        s.accept(&now);
        assert!(s.media.has_source(raw, now.level) && now.level == SourceLevel::Thumb);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_raw_decoder_command_sets_the_mode_and_reports() {
        use serde_json::json;
        // without Apple's decoder (Linux, or macOS before 12): built-in, and why
        let mut s = crate::Session::new().with_fs_and(None);
        let r = s.execute("app.rawDecoder", &json!({})).unwrap();
        assert_eq!((r["mode"].as_str(), r["effective"].as_str(), r["available"].as_bool()), (Some("auto"), Some("builtin"), Some(false)), "{r}");
        assert!(r["reason"].as_str().is_some_and(|w| w.contains("macOS")), "{r}");
        assert_eq!(s.execute("app.rawDecoder", &json!({"mode": "builtin"})).unwrap()["mode"], "builtin");
        for bad in [json!({"mode": "photoshop"}), json!({"mode": 2})] {
            let e = s.execute("app.rawDecoder", &bad).unwrap_err().to_string();
            assert!(e.contains("auto, apple or builtin"), "{e}");
        }
        assert_eq!(s.execute("app.rawDecoder", &json!({"clearSkipped": true})).unwrap()["skipped"], json!([]));
        // with it: the mode chosen is the one used (unless the environment forces one)
        let mut s = crate::Session::new().with_fs_and(Some(Stub::new().apple(Limits::default(), None)));
        let r = s.execute("app.rawDecoder", &json!({"mode": "apple"})).unwrap();
        assert_eq!(
            (r["mode"].as_str(), r["available"].as_bool(), r["identity"].as_str(), r["reason"].as_str()),
            (Some("apple"), Some(true), Some("stub-1"), None),
            "{r}"
        );
        assert_eq!(r["effective"].as_str(), Some(env_mode().unwrap_or(Mode::Apple).name()));
    }
}
