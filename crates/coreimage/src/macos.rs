//! The Objective-C side: a `CIRAWFilter` for the file, its look switched off, its output rendered by a `CIContext`
//! working in extended linear Rec.2020 floats, read back strip by strip.

#![allow(unsafe_code)]

use std::ffi::{CStr, c_void};
use std::path::Path;
use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject};
use objc2_core_foundation::{CFRetained, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceExtendedLinearITUR_2020};
use objc2_core_image::{
    CIContext, CIContextOption, CIImage, CIRAWFilter, kCIContextCacheIntermediates, kCIContextWorkingColorSpace, kCIContextWorkingFormat,
    kCIFormatRGBAf,
};
use objc2_foundation::{NSDictionary, NSNumber, NSProcessInfo, NSURL};

use crate::{Decoded, Info, Options, STRIP_BYTES, pixel_bounds, strip_rows, take_rgb};

/// Is this Objective-C class in the running system? (Classes newer than the system are missing, and using one would
/// fail.)
fn has(class: &CStr) -> bool {
    AnyClass::get(class).is_some()
}

pub fn available() -> bool {
    has(c"CIRAWFilter") && has(c"CIContext")
}

pub fn os_version() -> String {
    let v = NSProcessInfo::processInfo().operatingSystemVersion();
    format!("{}.{}.{}", v.majorVersion, v.minorVersion, v.patchVersion)
}

/// Extended linear Rec.2020: Core Image's working space here, and the space the floats are read in.
fn linear_rec2020() -> Result<CFRetained<CGColorSpace>, String> {
    // SAFETY: an immutable static CoreGraphics defines (macOS 10.14.3+, older than any macOS LightCraft runs on).
    CGColorSpace::with_name(Some(unsafe { kCGColorSpaceExtendedLinearITUR_2020 })).ok_or_else(|| "CoreGraphics has no linear Rec.2020".into())
}

/// A context that works in `space` with 32-bit floats and keeps no intermediate images.
fn context(space: &CGColorSpace) -> Retained<CIContext> {
    // SAFETY (all three): immutable statics Core Image defines (macOS 10.12+).
    let keys: [&CIContextOption; 3] = unsafe { [kCIContextWorkingColorSpace, kCIContextWorkingFormat, kCIContextCacheIntermediates] };
    // SAFETY: an immutable static Core Image defines.
    let format = NSNumber::new_i32(unsafe { kCIFormatRGBAf });
    let cache = NSNumber::new_bool(false);
    let values: [&AnyObject; 3] = [AsRef::<AnyObject>::as_ref(space), AsRef::<AnyObject>::as_ref(&*format), AsRef::<AnyObject>::as_ref(&*cache)];
    let options = NSDictionary::<CIContextOption, AnyObject>::from_slices(&keys, &values);
    // SAFETY: each option has the type Core Image documents for it: a CGColorSpace for the working colour space, an
    // NSNumber holding a CIFormat for the working format, an NSNumber holding a boolean for caching intermediates.
    unsafe { CIContext::contextWithOptions(Some(&options)) }
}

/// A raw filter for the file, set up as `options` asks (scene-linear, Apple's look off), and the camera's baseline
/// exposure. An error when Core Image can't develop the file (not a raw, or a camera it does not know).
fn filter(path: &Path, options: &Options) -> Result<(Retained<CIRAWFilter>, f32), String> {
    let url = NSURL::from_file_path(path).ok_or("the file's path can't be given to Core Image")?;
    // SAFETY: a file URL; Core Image returns nil for a file it can't develop, checked here.
    let f = unsafe { CIRAWFilter::filterWithImageURL(&url) }.ok_or("Apple's raw decoder does not support this file")?;
    // SAFETY (this block): getters and setters of a live filter, with values in the ranges Core Image documents
    // (amounts are floats it clamps itself); features this camera lacks are only set when the filter says it has them.
    let baseline = unsafe {
        let baseline = f.baselineExposure();
        if options.baseline_in_rust {
            f.setBaselineExposure(0.0);
        }
        f.setExposure(0.0);
        f.setBoostAmount(0.0);
        f.setBoostShadowAmount(options.boost_shadow_amount);
        if let Some(bias) = options.shadow_bias {
            f.setShadowBias(bias);
        }
        f.setExtendedDynamicRangeAmount(options.edr_amount);
        f.setGamutMappingEnabled(false);
        if f.isHighlightRecoverySupported() {
            f.setHighlightRecoveryEnabled(options.highlight_recovery);
        }
        if f.isLensCorrectionSupported() {
            f.setLensCorrectionEnabled(options.lens_correction);
        }
        if f.isLocalToneMapSupported() {
            f.setLocalToneMapAmount(0.0);
        }
        if f.isContrastSupported() {
            f.setContrastAmount(0.0);
        }
        if f.isSharpnessSupported() {
            f.setSharpnessAmount(0.0);
        }
        if f.isLuminanceNoiseReductionSupported() {
            f.setLuminanceNoiseReductionAmount(0.0);
        }
        if f.isColorNoiseReductionSupported() {
            f.setColorNoiseReductionAmount(0.0);
        }
        if f.isMoireReductionSupported() {
            f.setMoireReductionAmount(0.0);
        }
        if let Some((x, y)) = options.neutral_xy {
            f.setNeutralChromaticity(CGPoint { x, y });
        }
        if options.draft {
            f.setDraftModeEnabled(true);
        }
        baseline
    };
    Ok((f, baseline))
}

/// The sensor's size as stored, and the white the filter develops for.
fn facts(f: &CIRAWFilter) -> ((usize, usize), Option<(f64, f64)>) {
    // SAFETY: getters of a live filter.
    let (size, white) = unsafe { (f.nativeSize(), f.neutralChromaticity()) };
    let side = |v: f64| if v.is_finite() && v >= 1.0 && v <= crate::MAX_EDGE as f64 { v.round() as usize } else { 0 };
    ((side(size.width), side(size.height)), Some((white.x, white.y)))
}

fn output(f: &CIRAWFilter) -> Result<Retained<CIImage>, String> {
    // SAFETY: the output of a live, configured filter; nil when Core Image can't develop the file, checked here.
    unsafe { f.outputImage() }.ok_or_else(|| "Apple's raw decoder could not develop this file".into())
}

pub fn probe(path: &Path, options: &Options) -> Result<Info, String> {
    autoreleasepool(|_| {
        let (f, baseline) = filter(path, options)?;
        let (native, white_xy) = facts(&f);
        let image = output(&f)?;
        // SAFETY: a getter of a live image.
        let e = unsafe { image.extent() };
        let (_, _, width, height) = pixel_bounds(e.origin.x, e.origin.y, e.size.width, e.size.height)?;
        Ok(Info { width, height, native, white_xy, baseline_exposure: baseline })
    })
}

pub fn decode(path: &Path, options: &Options) -> Result<Decoded, String> {
    autoreleasepool(|_| {
        let (f, baseline) = filter(path, options)?;
        let (native, white_xy) = facts(&f);
        let long = native.0.max(native.1);
        if options.max_edge < long {
            // the raw is scaled before it is demosaiced (much faster than developing it whole and shrinking it)
            let scale = (options.max_edge as f64 / long as f64).clamp(1e-4, 1.0) as f32;
            // SAFETY: a setter of a live filter, with a scale in (0, 1].
            unsafe { f.setScaleFactor(scale) };
        }
        let image = output(&f)?;
        let space = linear_rec2020()?;
        let ctx = context(&space);
        let gain = if options.baseline_in_rust { 2f32.powf(crate::baseline(baseline)) } else { 1.0 };
        let (width, height, rgb) = render_rgb(&ctx, &image, &space, STRIP_BYTES, gain)?;
        Ok(Decoded { width, height, rgb, white_xy, baseline_exposure: if options.baseline_in_rust { baseline } else { 0.0 }, native })
    })
}

/// Render `image` (its whole extent, rounded inward to whole pixels) as RGB floats in `space`, row by row from the top,
/// times `gain`, in strips of at most `strip_bytes` of RGBA floats.
fn render_rgb(
    ctx: &CIContext,
    image: &CIImage,
    space: &CGColorSpace,
    strip_bytes: usize,
    gain: f32,
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    // SAFETY: a getter of a live image.
    let e = unsafe { image.extent() };
    let (x0, top, width, height) = pixel_bounds(e.origin.x, e.origin.y, e.size.width, e.size.height)?;
    let rows = strip_rows(width, strip_bytes).min(height);
    let row_floats = width.checked_mul(4).ok_or("the developed image is too wide")?;
    let row_bytes = row_floats.checked_mul(4).and_then(|b| isize::try_from(b).ok()).ok_or("the developed image is too wide")?;
    let mut strip = vec![0f32; row_floats.checked_mul(rows).ok_or("the developed image is too large")?];
    let mut rgb = Vec::with_capacity(width.checked_mul(height).ok_or("the developed image is too large")?);
    let mut done = 0;
    while done < height {
        let n = rows.min(height - done);
        let buf = strip.get_mut(..row_floats * n).ok_or("the developed image is too large")?;
        // Core Image's y goes up: this strip's rows run from `top - done` down to `top - done - n`
        let bounds = CGRect { origin: CGPoint { x: x0, y: top - (done + n) as f64 }, size: CGSize { width: width as f64, height: n as f64 } };
        let data = NonNull::new(buf.as_mut_ptr().cast::<c_void>()).ok_or("no memory for the developed image")?;
        // SAFETY: `data` points at `buf`, `n` rows of `row_bytes` bytes (four f32 per pixel, `width` pixels), which is
        // what rendering `bounds` (`width` × `n` pixels) as RGBA 32-bit floats writes; `buf` lives and is not touched
        // until the call returns. `image`, `ctx` and `space` are live.
        unsafe { ctx.render_toBitmap_rowBytes_bounds_format_colorSpace(image, data, row_bytes, bounds, kCIFormatRGBAf, Some(space)) };
        take_rgb(buf, gain, &mut rgb);
        done += n;
    }
    Ok((width, height, rgb))
}

/// For tests: render a `w` × `h` float image (RGBA, row 0 at the top) cropped to `crop` (x, y, width, height in Core
/// Image's coordinates) through [`render_rgb`] with `strip_bytes` per strip.
#[cfg(test)]
pub fn roundtrip(
    w: usize,
    h: usize,
    px: &[[f32; 4]],
    crop: (f64, f64, f64, f64),
    strip_bytes: usize,
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    use objc2_foundation::NSData;
    autoreleasepool(|_| {
        let bytes: Vec<u8> = px.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect();
        let data = NSData::with_bytes(&bytes);
        let space = linear_rec2020()?;
        let size = CGSize { width: w as f64, height: h as f64 };
        // SAFETY: `data` holds `h` rows of `w * 16` bytes, RGBA 32-bit floats, as the call is told; `space` is live.
        let image = unsafe { CIImage::imageWithBitmapData_bytesPerRow_size_format_colorSpace(&data, w * 16, size, kCIFormatRGBAf, Some(&space)) };
        let rect = CGRect { origin: CGPoint { x: crop.0, y: crop.1 }, size: CGSize { width: crop.2, height: crop.3 } };
        // SAFETY: a live image and a finite rectangle.
        let image = unsafe { image.imageByCroppingToRect(rect) };
        render_rgb(&context(&space), &image, &space, strip_bytes, 1.0)
    })
}
