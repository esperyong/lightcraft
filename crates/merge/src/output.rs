//! Merge results as linear DNGs (`PhotometricInterpretation = LinearRaw`, three samples per
//! pixel), written with LightCraft's own TIFF/DNG writer.
//!
//! - HDR results are scaled so the brightest pixel sits just below 1.0 (the white level); the
//!   scale is carried in `BaselineExposure`, so a default rendering looks like the reference frame.
//!   Normal-range results (16-bit) keep the sources' white level, so clipped highlights stay
//!   where the raw reader's highlight reconstruction looks for them.
//! - Raw sources: the reference frame's DNG colour tags (`ColorMatrix*`, `ForwardMatrix*`,
//!   `CameraCalibration*`, `AsShotNeutral`, illuminants) are carried over — the data is still
//!   camera RGB; a vendor raw's white balance and fallback colour model are written as DNG tags
//!   ([`crate::frame::camera_color`]).
//! - Other sources: the "camera" is the file's linear RGB space; `ColorMatrix1` = XYZ(D65) → RGB,
//!   `ForwardMatrix1` = RGB → XYZ(D50), `AsShotNeutral` = (1, 1, 1).
//! - Semantic masks (a focus stack's depth map) go in SubIFDs of the image (DNG 1.6), an
//!   application's own data in `DNGPrivateData` ([`DngExtra`]).

use lightcraft_color::{D50, D65, Mat3, bradford};
use lightcraft_geom::Orientation;
use lightcraft_meta::Metadata;
use lightcraft_raster::Rgb32f;
use lightcraft_raw::{BlackLevel, ColorData, DngCompression, DngWriteOptions, Opcode, OpcodeLists, RawData, RawFormat, RawImage, Rect, SemanticMask};

use crate::frame::FrameColor;
use crate::{MergeError, Result};

/// Sample encoding of the output DNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DngSamples {
    /// 16-bit floating point, Deflate + floating-point predictor (HDR).
    Half,
    /// 32-bit floating point, Deflate + floating-point predictor.
    Float,
    /// 16-bit integer, Deflate + horizontal predictor (panoramas of normal-range photos).
    U16,
}

/// The DNG colour tags for frames of `color`. A source's gain table map is dropped: it is
/// positioned on that one capture's active area, which a merged image (aligned, oriented,
/// stitched, cropped) no longer shares.
pub fn color_data(color: &FrameColor) -> ColorData {
    match color {
        FrameColor::Camera(c) => {
            let mut c = (**c).clone();
            c.profile.gain_table_map = None;
            c
        }
        FrameColor::Linear { to_xyz_d50 } => {
            let to_xyz_d65 = bradford(D50, D65).mul(to_xyz_d50);
            let cm = to_xyz_d65.inverse().unwrap_or(Mat3::IDENTITY);
            ColorData {
                illuminant: [21, 0],
                color_matrix: [Some(cm), None],
                forward_matrix: [Some(*to_xyz_d50), None],
                camera_calibration: [None, None],
                analog_balance: None,
                as_shot_neutral: Some([1.0, 1.0, 1.0]),
                as_shot_white_xy: None,
                baseline_exposure: 0.0,
                baseline_sharpness: None,
                profile: Default::default(),
            }
        }
    }
}

/// What a merged DNG carries besides its image.
#[derive(Clone, Debug, Default)]
pub struct DngExtra {
    /// Semantic masks (a focus stack's depth map), each spanning the whole image.
    pub masks: Vec<SemanticMask>,
    /// `DNGPrivateData`: the writing application's NUL-terminated name, then its data.
    pub private_data: Option<Vec<u8>>,
    /// `RawDataUniqueID`: what makes this merge's file its own (see [`lightcraft_raw::DngWriteOptions`]).
    pub unique_id: Option<[u8; 16]>,
    /// Lens corrections (`WarpRectilinear`, `FixVignetteRadial`; others are dropped) for the image as
    /// written: a merge that keeps its reference photo's geometry keeps its profile corrections.
    pub lens_opcodes: Vec<Opcode>,
}

/// Encode `img` (scene-linear, 1.0 = source white) as a LinearRaw DNG.
pub fn write_linear_dng(
    img: &Rgb32f,
    color: &FrameColor,
    orientation: Orientation,
    metadata: &Metadata,
    baseline_exposure: f64,
    samples: DngSamples,
) -> Result<Vec<u8>> {
    write_linear_dng_with(img, color, orientation, metadata, baseline_exposure, samples, DngExtra::default())
}

/// [`write_linear_dng`] with masks and private data stored alongside ([`DngExtra`]).
pub fn write_linear_dng_with(
    img: &Rgb32f,
    color: &FrameColor,
    orientation: Orientation,
    metadata: &Metadata,
    baseline_exposure: f64,
    samples: DngSamples,
    extra: DngExtra,
) -> Result<Vec<u8>> {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return Err(MergeError::Output("empty image".into()));
    }
    let peak = img.data.iter().flat_map(|p| p.iter()).copied().filter(|v| v.is_finite()).fold(0.0f32, f32::max);
    let target = 0.95f32;
    let scale = match samples {
        // a normal-range result (resampling overshoots a little) keeps the sources' white level:
        // clipped highlights stay clipped, at 1.0
        DngSamples::U16 if peak <= 1.0 / target => 1.0,
        // never brighten: values ≤ 1 keep their scale (so a plain panorama keeps its white level)
        _ if peak > target => target / peak,
        _ => 1.0,
    };
    let mut cd = color_data(color);
    cd.baseline_exposure = baseline_exposure + (1.0 / scale as f64).log2();
    let clean = |v: f32| if v.is_finite() { (v * scale).max(0.0) } else { 0.0 };
    let (data, bits, white) = match samples {
        DngSamples::Half | DngSamples::Float => {
            (RawData::F32(img.data.iter().flat_map(|p| p.map(clean)).collect()), if samples == DngSamples::Half { 16 } else { 32 }, 1.0)
        }
        DngSamples::U16 => {
            (RawData::U16(img.data.iter().flat_map(|p| p.map(|v| (clean(v) * 65535.0).round().min(65535.0) as u16)).collect()), 16, 65535.0)
        }
    };
    let mut meta = metadata.clone();
    meta.width = Some(w as u32);
    meta.height = Some(h as u32);
    meta.orientation = Some(orientation);
    let raw = RawImage {
        format: RawFormat::Dng,
        width: w,
        height: h,
        cpp: 3,
        data,
        cfa: None,
        bits,
        black: BlackLevel::uniform(0.0),
        white: vec![white],
        active_area: Rect::new(0, 0, w, h),
        crop: Rect::new(0, 0, w, h),
        orientation,
        color: cd,
        wb_multipliers: None,
        linearized: false,
        opcodes: OpcodeLists { list3: extra.lens_opcodes.into_iter().filter(Opcode::is_lens_correction).collect(), ..Default::default() },
        metadata: meta,
    };
    let compression = DngCompression::Deflate { tile: 256, half: samples == DngSamples::Half };
    let opts = DngWriteOptions {
        compression,
        semantic_masks: extra.masks,
        private_data: extra.private_data,
        raw_data_unique_id: extra.unique_id,
        ..Default::default()
    };
    lightcraft_raw::write_dng(&raw, &opts).map_err(|e| MergeError::Output(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_dng_round_trips_through_the_raw_decoder() {
        let img = Rgb32f::from_fn(80, 60, |x, y| [x as f32 / 20.0, y as f32 / 30.0, 0.3]);
        let color = FrameColor::Linear { to_xyz_d50: bradford(D65, D50).mul(&lightcraft_color::SRGB.to_xyz()) };
        let bytes = write_linear_dng(&img, &color, Orientation::Normal, &Metadata::default(), 0.0, DngSamples::Half).unwrap();
        let raw = lightcraft_raw::decode(&bytes).unwrap();
        assert_eq!((raw.width, raw.height, raw.cpp), (80, 60, 3));
        let gain = 2f32.powf(raw.color.baseline_exposure as f32);
        let back = raw.develop(lightcraft_raw::Method::Bilinear).unwrap();
        for (a, b) in img.data.iter().zip(&back.data) {
            for c in 0..3 {
                assert!((a[c] - b[c] * gain).abs() <= a[c] * 2e-3 + 1e-5, "{a:?} {b:?}");
            }
        }
        // the colour model maps the source white to neutral (no white-balance shift)
        let xy = lightcraft_raw::color::as_shot_white_xy(&raw);
        assert!((xy.x - D65.x).abs() < 2e-3 && (xy.y - D65.y).abs() < 2e-3, "{xy:?}");
        let t = lightcraft_raw::color::camera_transform(&raw, xy);
        let white = t.matrix.apply([t.wb[0] as f64, t.wb[1] as f64, t.wb[2] as f64]);
        assert!(white.iter().all(|v| (v - white[0]).abs() < 1e-3), "{white:?}");
        // sRGB red maps to the Rec.2020 coordinates of sRGB red
        let red = t.matrix.apply([t.wb[0] as f64, 0.0, 0.0]);
        let want = lightcraft_color::SRGB.to_space(&lightcraft_color::REC2020).apply([1.0, 0.0, 0.0]);
        let k = white[0];
        for c in 0..3 {
            assert!((red[c] / k - want[c]).abs() < 5e-3, "{red:?} vs {want:?}");
        }
        let u16 = write_linear_dng(&img, &color, Orientation::Normal, &Metadata::default(), 0.0, DngSamples::U16).unwrap();
        assert!(lightcraft_raw::decode(&u16).is_ok());
    }

    /// A vendor raw (no colour matrix, white balance as multipliers) as the decoder gives it.
    fn vendor_raw() -> lightcraft_raw::RawImage {
        use lightcraft_raw::*;
        RawImage {
            format: RawFormat::Arw,
            width: 8,
            height: 8,
            cpp: 1,
            data: RawData::U16(vec![1000; 64]),
            cfa: Cfa::bayer("RGGB"),
            bits: 14,
            black: BlackLevel::uniform(512.0),
            white: vec![16000.0],
            active_area: Rect::new(0, 0, 8, 8),
            crop: Rect::new(0, 0, 8, 8),
            orientation: Orientation::Normal,
            color: ColorData { baseline_exposure: 0.3, ..Default::default() },
            wb_multipliers: Some([2.35, 1.0, 1.62]),
            linearized: false,
            opcodes: OpcodeLists::default(),
            metadata: Metadata::default(),
        }
    }

    #[test]
    fn a_vendor_raw_merge_renders_with_its_white_balance_and_colour_model() {
        use lightcraft_raw::color::{as_shot_white_xy, camera_transform};
        let src = vendor_raw();
        let want = camera_transform(&src, as_shot_white_xy(&src));
        let color = FrameColor::Camera(Box::new(crate::frame::camera_color(&src)));
        // clipped highlights (1.0) and resampling overshoot just above them
        let img = Rgb32f::from_fn(64, 48, |x, y| {
            if x < 8 {
                [1.0, 1.0, 1.0]
            } else if x < 10 {
                [1.03, 1.0, 1.01]
            } else {
                [0.2, 0.3 + y as f32 / 200.0, 0.25]
            }
        });
        let bytes = write_linear_dng(&img, &color, Orientation::Normal, &Metadata::default(), 0.3, DngSamples::U16).unwrap();
        let raw = lightcraft_raw::decode(&bytes).unwrap();
        let got = camera_transform(&raw, as_shot_white_xy(&raw));
        assert!(!got.matrix_is_fallback);
        for c in 0..3 {
            assert!((got.wb[c] - want.wb[c]).abs() < 2e-3 * want.wb[c], "wb {:?} vs {:?}", got.wb, want.wb);
            for k in 0..3 {
                assert!((got.matrix.0[c][k] - want.matrix.0[c][k]).abs() < 2e-3, "{:?} vs {:?}", got.matrix, want.matrix);
            }
        }
        assert!((got.baseline_exposure - 0.3).abs() < 1e-3, "{}", got.baseline_exposure);
        // the white level is the sources': clipped pixels stay at 1.0, where highlight
        // reconstruction finds them
        let back = raw.develop(lightcraft_raw::Method::Bilinear).unwrap();
        assert!(back.get(2, 10).iter().all(|v| *v >= 0.999), "{:?}", back.get(2, 10));
        assert!((back.get(30, 20)[1] - 0.4).abs() < 1e-3, "{:?}", back.get(30, 20));
    }
}
