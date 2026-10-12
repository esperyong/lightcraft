//! Apple's raw decoder on macOS: Core Image's `CIRAWFilter` (macOS 12+) develops a camera raw into scene-linear
//! Rec.2020, as LightCraft's own decoders do.
//!
//! Apple keeps decoders and colour for hundreds of cameras up to date with the system, including raw variants
//! LightCraft can't decode yet (compressed Olympus ORF, Nikon High Efficiency NEF…). LightCraft uses it as an optional
//! decoder: the engine falls back to its own (see `docs/apple-raw.md`).
//!
//! Everything Apple would do to the *look* is switched off, so the pixels stay scene-referred and LightCraft's
//! pipeline does tone, noise reduction and sharpening as for any raw: no tone curve ("boost"), no local tone mapping,
//! no contrast, sharpening, noise or moiré reduction, no gamut mapping; extended (unclipped) output; the camera's
//! baseline exposure applied here as a linear gain (Core Image's gives the same pixels on macOS 26, measured; here it
//! does not depend on how a later Core Image applies it). Highlight recovery and, where asked, the lens corrections
//! the raw carries stay Apple's.
//!
//! The Objective-C calls are `unsafe` and live in one module (`macos`), every block with its `SAFETY:` reasoning; the
//! API here is safe. Objective-C exceptions cannot be caught without a C helper, so what Core Image is given is
//! checked first (classes missing from an older macOS are looked up before use, sizes are bounded) and what it
//! returns (extents, buffer sizes, numbers) is checked before it is used.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(target_os = "macos")]
mod macos;

use std::path::Path;

/// Largest side Core Image is asked to render.
pub const MAX_EDGE: usize = 65_536;
/// Most pixels Core Image is asked to render (larger than any camera's raw: 150 MP medium format is the largest).
pub const MAX_PIXELS: usize = 250_000_000;
/// Bytes of one strip of Core Image's float output: the image is read in strips of rows, so its whole RGBA float
/// rendering is never in memory at once.
pub const STRIP_BYTES: usize = 16 << 20;
/// Bumped when the way this crate develops a raw changes (part of [`identity`], so cached renders are redone).
const REVISION: u32 = 1;

/// How to develop a raw.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Longest side wanted: Core Image scales the raw before it demosaics when this is smaller than the photo
    /// (`usize::MAX`: full size). The result is no larger than this, give or take a pixel of rounding.
    pub max_edge: usize,
    /// Apple's faster, lower-quality development (thumbnails).
    pub draft: bool,
    /// Apple's own lens corrections, for cameras whose raws carry them (geometry changes: the frame is cropped).
    pub lens_correction: bool,
    /// Develop for this white (CIE 1931 xy) instead of the camera's as-shot one (for comparisons).
    pub neutral_xy: Option<(f64, f64)>,
    /// Tuning (`examples/raw_decoders.rs --sweep` in the engine compares the choices). Core Image's shadow bias
    /// (`None`: Apple's default for the camera).
    pub shadow_bias: Option<f32>,
    /// Reconstruct clipped highlights from the channels that are not clipped.
    pub highlight_recovery: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { max_edge: usize::MAX, draft: false, lens_correction: false, neutral_xy: None, shadow_bias: Some(0.0), highlight_recovery: true }
    }
}

impl Options {
    fn check(&self) -> Result<(), String> {
        if self.max_edge == 0 {
            return Err("asked for an image of no size".into());
        }
        if let Some((x, y)) = self.neutral_xy
            && !valid_xy(x, y)
        {
            return Err(format!("({x}, {y}) is not a white"));
        }
        if self.shadow_bias.is_some_and(|v| !(v.is_finite() && (-10.0..=10.0).contains(&v))) {
            return Err("tuning out of range".into());
        }
        Ok(())
    }
}

/// What a raw is, as Apple develops it (without developing it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info {
    /// Size of the developed image at full size, upright (Core Image applies the file's orientation), after its
    /// default crop (and lens corrections, when asked for).
    pub width: usize,
    pub height: usize,
    /// The sensor's size as stored (not upright).
    pub native: (usize, usize),
    /// The as-shot white (CIE 1931 xy) the camera recorded, as Apple reads it.
    pub white_xy: Option<(f64, f64)>,
    /// The camera's baseline exposure (EV) Apple uses for this file.
    pub baseline_exposure: f32,
}

/// A developed raw.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    /// Scene-linear Rec.2020 (D65), row by row from the top, upright; white-balanced for `white_xy` (or the white
    /// asked for), the baseline exposure applied; finite and ≥ 0.
    pub rgb: Vec<[f32; 3]>,
    /// The as-shot white (CIE 1931 xy), as Apple reads it.
    pub white_xy: Option<(f64, f64)>,
    /// The baseline exposure (EV) applied.
    pub baseline_exposure: f32,
    /// The sensor's size as stored.
    pub native: (usize, usize),
}

/// Whether this Mac has Apple's raw decoder (macOS 12+; never elsewhere).
pub fn available() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// What develops the pixels: this crate's revision and the system's version (Apple updates its raw decoders with the
/// system). Changes when the pixels may change; part of cache keys.
pub fn identity() -> String {
    #[cfg(target_os = "macos")]
    {
        format!("ciraw{REVISION}:{}", macos::os_version())
    }
    #[cfg(not(target_os = "macos"))]
    {
        format!("ciraw{REVISION}:unavailable")
    }
}

/// Size, white and baseline exposure of the raw at `path` as Apple would develop it with `options` (only
/// `lens_correction` changes the size), without developing it.
pub fn probe(path: &Path, options: &Options) -> Result<Info, String> {
    options.check()?;
    #[cfg(target_os = "macos")]
    {
        if !available() {
            return Err(UNAVAILABLE.into());
        }
        let info = guarded(|| macos::probe(path, options))?;
        Ok(Info { white_xy: info.white_xy.filter(|&(x, y)| valid_xy(x, y)), baseline_exposure: baseline(info.baseline_exposure), ..info })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        Err(NOT_MACOS.into())
    }
}

/// Develop the raw at `path`. Runs on the calling thread and waits for Core Image (which uses the GPU); several
/// threads may call it at once.
pub fn decode(path: &Path, options: &Options) -> Result<Decoded, String> {
    options.check()?;
    #[cfg(target_os = "macos")]
    {
        if !available() {
            return Err(UNAVAILABLE.into());
        }
        let d = guarded(|| macos::decode(path, options))?;
        if d.width == 0 || d.height == 0 || d.width.checked_mul(d.height) != Some(d.rgb.len()) {
            return Err("Apple's raw decoder returned an image of the wrong size".into());
        }
        if d.rgb.iter().all(|p| *p == [0.0; 3]) {
            return Err("Apple's raw decoder returned a black image".into());
        }
        Ok(Decoded { white_xy: d.white_xy.filter(|&(x, y)| valid_xy(x, y)), baseline_exposure: baseline(d.baseline_exposure), ..d })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        Err(NOT_MACOS.into())
    }
}

#[cfg(not(target_os = "macos"))]
const NOT_MACOS: &str = "Apple's raw decoder is only on macOS";
#[cfg(target_os = "macos")]
const UNAVAILABLE: &str = "Apple's raw decoder needs macOS 12 or later";

/// A chromaticity a white can have.
fn valid_xy(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 && x + y < 1.0
}

/// A baseline exposure Core Image reports, sanitised (0 when it is not a number; cameras use about −1..+2 EV).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn baseline(ev: f32) -> f32 {
    if ev.is_finite() { ev.clamp(-8.0, 8.0) } else { 0.0 }
}

/// The whole pixels of a Core Image extent (`x`, `y`: bottom-left corner, y up; `w`, `h`: size): the left edge, the top
/// edge (in Core Image's y), width and height. Fractional edges are rounded inward; an extent that is not a finite,
/// non-empty rectangle within [`MAX_EDGE`] and [`MAX_PIXELS`] is an error.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn pixel_bounds(x: f64, y: f64, w: f64, h: f64) -> Result<(f64, f64, usize, usize), String> {
    if ![x, y, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
        return Err("Apple's raw decoder returned an image without a size".into());
    }
    let (x0, y0, x1, y1) = (x.ceil(), y.ceil(), (x + w).floor(), (y + h).floor());
    let (pw, ph) = (x1 - x0, y1 - y0);
    let max = MAX_EDGE as f64;
    if !(1.0..=max).contains(&pw) || !(1.0..=max).contains(&ph) {
        return Err(format!("Apple's raw decoder returned a {w:.0} × {h:.0} image"));
    }
    let (pw, ph) = (pw as usize, ph as usize);
    if pw.checked_mul(ph).is_none_or(|n| n > MAX_PIXELS) {
        return Err(format!("the developed image would be {pw} × {ph} pixels, more than LightCraft takes from Apple's raw decoder"));
    }
    Ok((x0, y1, pw, ph))
}

/// Rows per strip for an image `width` pixels wide: as many RGBA float rows as fit in `strip_bytes`, at least one.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn strip_rows(width: usize, strip_bytes: usize) -> usize {
    (strip_bytes / width.saturating_mul(16).max(1)).max(1)
}

/// Append RGBA float pixels to `out` as RGB times `gain`: what is not a number, and anything below zero, becomes 0.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn take_rgb(rgba: &[f32], gain: f32, out: &mut Vec<[f32; 3]>) {
    let clean = |v: f32| {
        let v = v * gain;
        if v.is_finite() && v > 0.0 { v } else { 0.0 }
    };
    out.extend(rgba.chunks_exact(4).map(|p| [clean(p[0]), clean(p[1]), clean(p[2])]));
}

/// Run `f`, turning a panic in it (an unexpected nil from Objective-C, say) into an error.
#[cfg(target_os = "macos")]
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|_| "Apple's raw decoder failed unexpectedly".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_and_extents_are_checked_everywhere() {
        assert!(Options::default().check().is_ok());
        assert!(Options { max_edge: 0, ..Default::default() }.check().is_err());
        for xy in [(0.0, 0.3), (0.6, 0.5), (f64::NAN, 0.3), (0.3, f64::INFINITY)] {
            assert!(Options { neutral_xy: Some(xy), ..Default::default() }.check().is_err(), "{xy:?}");
        }
        assert!(Options { neutral_xy: Some((0.3127, 0.329)), ..Default::default() }.check().is_ok());
        assert!(Options { shadow_bias: Some(f32::NAN), ..Default::default() }.check().is_err());
        assert!(Options { shadow_bias: None, ..Default::default() }.check().is_ok());
        // whole pixels, rounded inward; y is the top edge in Core Image's (upward) coordinates
        assert_eq!(pixel_bounds(0.0, 0.0, 6000.0, 4000.0).unwrap(), (0.0, 4000.0, 6000, 4000));
        assert_eq!(pixel_bounds(-0.25, 10.5, 2560.4, 1706.9).unwrap(), (0.0, 1717.0, 2560, 1706));
        for bad in [(0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, -1.0), (f64::NAN, 0.0, 1.0, 1.0), (0.0, 0.0, 0.4, 10.0)] {
            assert!(pixel_bounds(bad.0, bad.1, bad.2, bad.3).is_err(), "{bad:?}");
        }
        // Core Image's infinite extent, and images past the pixel cap
        assert!(pixel_bounds(-8.98e307, -8.98e307, 1.79e308, 1.79e308).is_err());
        assert!(pixel_bounds(0.0, 0.0, 60_000.0, 60_000.0).is_err());
        assert_eq!(strip_rows(6000, STRIP_BYTES), 174);
        assert_eq!(strip_rows(10_000_000, 16), 1);
        assert!(strip_rows(0, 16) >= 1);
        let mut out = Vec::new();
        take_rgb(&[0.5, 1.5, -0.25, 1.0, f32::NAN, f32::INFINITY, 2.0, 0.0], 2.0, &mut out);
        assert_eq!(out, vec![[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!available());
            assert!(identity().ends_with(":unavailable"));
            let p = Path::new("/nonexistent.cr3");
            assert!(decode(p, &Options::default()).err().unwrap_or_default().contains("only on macOS"));
            assert!(probe(p, &Options::default()).err().unwrap_or_default().contains("only on macOS"));
        }
    }

    /// On a Mac: the strip reader gives back a known float image row by row from the top, from an extent whose origin
    /// is not (0, 0), over several strips. Elsewhere it is skipped.
    #[test]
    fn a_mac_reads_core_image_floats_in_strips_from_the_top() {
        #[cfg(target_os = "macos")]
        {
            if !available() {
                eprintln!("skipped: no Core Image raw decoder here");
                return;
            }
            let (w, h) = (10usize, 8usize);
            // a distinct value per pixel, row 0 at the top
            let src: Vec<[f32; 4]> = (0..w * h).map(|i| [(i % w) as f32 / 16.0, (i / w) as f32 / 16.0, 0.5 + i as f32 / 256.0, 1.0]).collect();
            // the 6 × 5 pixels from column 2, row 2 (counted from the top): Core Image's y from 1 (bottom) up to 6
            let (got_w, got_h, got) = macos::roundtrip(w, h, &src, (2.0, 1.0, 6.0, 5.0), 3 * 6 * 16).unwrap();
            assert_eq!((got_w, got_h, got.len()), (6, 5, 30));
            for (j, row) in got.chunks_exact(6).enumerate() {
                for (i, p) in row.iter().enumerate() {
                    let s = src[(2 + j) * w + 2 + i];
                    for c in 0..3 {
                        assert!((p[c] - s[c]).abs() < 1e-4, "row {j} column {i}: {p:?} vs {s:?}");
                    }
                }
            }
            assert!(identity().starts_with("ciraw"));
        }
    }

    /// A real raw, with timings:
    /// `LC_APPLE_RAW=<file> cargo test --release -p lightcraft-coreimage --lib real_raw -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a raw file: see the doc comment"]
    fn real_raw() {
        use std::time::Instant;
        let Some(path) = std::env::var_os("LC_APPLE_RAW") else { return };
        let path = Path::new(&path);
        println!("{path:?}: available {}, {}", available(), identity());
        let t = Instant::now();
        let info = probe(path, &Options::default()).unwrap();
        println!("probe: {info:?} in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        for (edge, draft) in [(512, true), (2560, false), (usize::MAX, false)] {
            let t = Instant::now();
            let d = decode(path, &Options { max_edge: edge, draft, ..Default::default() }).unwrap();
            let max = d.rgb.iter().flatten().fold(0.0f32, |m, v| m.max(*v));
            let mean = d.rgb.iter().map(|p| (p[0] + p[1] + p[2]) / 3.0).sum::<f32>() / d.rgb.len() as f32;
            println!(
                "decode ≤{edge} draft {draft}: {} × {} in {:.0} ms; mean {mean:.4}, max {max:.3}, white {:?}, baseline {:+.2} EV",
                d.width,
                d.height,
                t.elapsed().as_secs_f64() * 1000.0,
                d.white_xy,
                d.baseline_exposure
            );
        }
    }
}
