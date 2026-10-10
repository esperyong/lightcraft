//! The Objective-C side: an image becomes a CGImage, Vision's requests run on it in one handler, and their results
//! are read back (labels and scores as numbers, masks out of their CVPixelBuffers).

#![allow(unsafe_code)]

use std::ffi::CStr;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyClass;
use objc2_core_foundation::{CFData, CFRetained};
use objc2_core_graphics::{CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage, CGImageAlphaInfo, kCGColorSpaceSRGB};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetDataSize, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferIsPlanar, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_OneComponent8, kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
use objc2_foundation::{NSArray, NSDictionary, NSError};
use objc2_vision::{
    VNCalculateImageAestheticsScoresRequest, VNClassifyImageRequest, VNGenerateForegroundInstanceMaskRequest, VNImageRequestHandler, VNRequest,
};

use crate::{Aesthetics, Analysis, Features, Image, Label, MAX_EDGE, Matte};

/// Is this Objective-C class in the running system? (Classes newer than the system are missing, and creating one
/// would fail.)
fn has(class: &CStr) -> bool {
    AnyClass::get(class).is_some()
}

pub fn features() -> Features {
    Features {
        labels: has(c"VNClassifyImageRequest"),
        aesthetics: has(c"VNCalculateImageAestheticsScoresRequest"),
        subject: has(c"VNGenerateForegroundInstanceMaskRequest"),
    }
}

fn text(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

/// `image` as a CGImage: 8-bit RGBX in sRGB, its own copy of the pixels.
fn cg_image(image: Image) -> Result<CFRetained<CGImage>, String> {
    let bytes = image.pixels.as_flattened();
    let row = image.width.checked_mul(4).ok_or("the image is too large")?;
    let data = CFData::from_bytes(bytes);
    let provider = CGDataProvider::with_cf_data(Some(&data)).ok_or("CoreGraphics could not take the pixels")?;
    // SAFETY: an immutable static CoreGraphics defines.
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB })).ok_or("CoreGraphics has no sRGB colour space")?;
    // SAFETY: `provider` holds `row × height` bytes (the caller checked that `pixels` has `width × height` pixels of
    // four bytes), which is what an 8-bit, 32-bit-per-pixel image of `width × height` with that row length reads; the
    // decode array may be null (no remapping). CoreGraphics returns null for a layout it refuses, checked below.
    unsafe {
        CGImage::new(
            image.width,
            image.height,
            8,
            32,
            row,
            Some(&space),
            CGBitmapInfo(CGImageAlphaInfo::NoneSkipLast.0),
            Some(&provider),
            std::ptr::null(),
            false,
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    }
    .ok_or_else(|| "CoreGraphics refused the image".into())
}

fn handler(image: &CGImage) -> Retained<VNImageRequestHandler> {
    // SAFETY: `image` is a valid CGImage (the handler retains it) and no options are given.
    unsafe { VNImageRequestHandler::initWithCGImage_options(VNImageRequestHandler::alloc(), image, &NSDictionary::new()) }
}

pub fn analyze(image: Image, f: Features) -> Result<Analysis, String> {
    autoreleasepool(|_| {
        let cg = cg_image(image)?;
        let handler = handler(&cg);
        // SAFETY (both): plain constructors of classes `features` found in this system.
        let classify = f.labels.then(|| unsafe { VNClassifyImageRequest::new() });
        let aesthetics = f.aesthetics.then(|| unsafe { VNCalculateImageAestheticsScoresRequest::new() });
        let mut requests: Vec<&VNRequest> = Vec::new();
        if let Some(r) = &classify {
            requests.push(r);
        }
        if let Some(r) = &aesthetics {
            requests.push(r);
        }
        handler.performRequests_error(&NSArray::from_slice(&requests)).map_err(|e| text(&e))?;
        let mut out = Analysis::default();
        if let Some(r) = &classify {
            // SAFETY: the results of a request that has run: an array of classification observations, or nil.
            let results = unsafe { r.results() };
            for o in results.iter().flat_map(|a| a.iter()) {
                // SAFETY: getters of a live observation. At 90 % precision the label is kept when its recall is at least
                // 1 % (Apple's high-precision operating point for tagging, as in its Vision examples).
                let (precise, name, confidence) =
                    unsafe { (o.hasPrecisionRecallCurve() && o.hasMinimumRecall_forPrecision(0.01, 0.9), o.identifier(), o.confidence()) };
                if precise {
                    out.labels.push(Label { name: name.to_string(), confidence });
                }
            }
        }
        if let Some(r) = &aesthetics {
            // SAFETY: as above, for the aesthetics request.
            if let Some(o) = unsafe { r.results() }.and_then(|a| a.firstObject()) {
                // SAFETY: getters of a live observation.
                let (score, utility) = unsafe { (o.overallScore(), o.isUtility()) };
                out.aesthetics = Some(Aesthetics { score, utility });
            }
        }
        Ok(out)
    })
}

pub fn subject(image: Image) -> Result<Option<Matte>, String> {
    autoreleasepool(|_| {
        let cg = cg_image(image)?;
        let handler = handler(&cg);
        // SAFETY: a plain constructor of a class `features` found in this system.
        let request = unsafe { VNGenerateForegroundInstanceMaskRequest::new() };
        handler.performRequests_error(&NSArray::from_slice(&[&*request as &VNRequest])).map_err(|e| text(&e))?;
        // SAFETY: the results of a request that has run: instance-mask observations, or nil.
        let Some(found) = unsafe { request.results() }.and_then(|a| a.firstObject()) else { return Ok(None) };
        // SAFETY: a getter of a live observation.
        let instances = unsafe { found.allInstances() };
        if instances.count() == 0 {
            return Ok(None);
        }
        // SAFETY: `instances` came from this observation, and `handler` is the one that produced it (as the method
        // requires, to scale the mask to its image).
        let mask = unsafe { found.generateScaledMaskForImageForInstances_fromRequestHandler_error(&instances, &handler) }.map_err(|e| text(&e))?;
        read_matte(&mask).map(Some)
    })
}

/// Unlocks a pixel buffer's base address when dropped.
struct Locked<'a>(&'a CVPixelBuffer);

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        // SAFETY: the buffer was locked read-only by `read_matte`, which made this guard only once that succeeded.
        unsafe { CVPixelBufferUnlockBaseAddress(self.0, CVPixelBufferLockFlags::ReadOnly) };
    }
}

/// A one-component mask (8-bit or 32-bit float) as alpha 0..1, after checking its format and that its rows lie in
/// its memory.
fn read_matte(buffer: &CVPixelBuffer) -> Result<Matte, String> {
    if CVPixelBufferIsPlanar(buffer) {
        return Err("Vision's mask is planar".into());
    }
    let format = CVPixelBufferGetPixelFormatType(buffer);
    let size = match format {
        f if f == kCVPixelFormatType_OneComponent32Float => 4,
        f if f == kCVPixelFormatType_OneComponent8 => 1,
        other => return Err(format!("Vision's mask has an unexpected pixel format ({other:#x})")),
    };
    // SAFETY: a valid pixel buffer; read-only, as it is only read. Unlocked by `Locked` on every path below.
    if unsafe { CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly) } != kCVReturnSuccess {
        return Err("could not read Vision's mask".into());
    }
    let _locked = Locked(buffer);
    let (w, h, stride, total) =
        (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), CVPixelBufferGetBytesPerRow(buffer), CVPixelBufferGetDataSize(buffer));
    let base = CVPixelBufferGetBaseAddress(buffer).cast::<u8>().cast_const();
    let row = w.checked_mul(size).ok_or("Vision's mask is too large")?;
    if w == 0 || h == 0 || w > MAX_EDGE || h > MAX_EDGE || base.is_null() || stride < row {
        return Err("Vision's mask has an unexpected layout".into());
    }
    // the last row ends `row` bytes after it starts
    let span = stride.checked_mul(h - 1).and_then(|n| n.checked_add(row)).ok_or("Vision's mask is too large")?;
    if span > total {
        return Err("Vision's mask is smaller than its rows".into());
    }
    // SAFETY: `base` is the locked base address of a non-planar buffer whose data is `total` bytes, and `span ≤ total`
    // covers every row read below; the bytes stay valid and unchanged while `_locked` holds the read-only lock, and
    // they are read as bytes (no alignment needed).
    let bytes = unsafe { std::slice::from_raw_parts(base, span) };
    let mut alpha = Vec::with_capacity(w * h);
    for y in 0..h {
        let line = bytes.get(y * stride..y * stride + row).ok_or("Vision's mask is smaller than its rows")?;
        if size == 4 {
            alpha.extend(line.chunks_exact(4).filter_map(|b| b.try_into().ok()).map(f32::from_ne_bytes));
        } else {
            alpha.extend(line.iter().map(|&b| f32::from(b) / 255.0));
        }
    }
    Ok(Matte { width: w, height: h, alpha })
}
