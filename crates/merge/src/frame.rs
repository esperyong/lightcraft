//! Merge inputs: scene-linear frames decoded from raw files (camera RGB, before white balance) or
//! standard images (linear light in the file's primaries), with their exposure from EXIF.

use lightcraft_color::Mat3;
use lightcraft_geom::Orientation;
use lightcraft_meta::Metadata;
use lightcraft_raster::Rgb32f;
use lightcraft_raster::resample::{Filter, fit};
use lightcraft_raw::ColorData;

use crate::{MergeError, Result};

/// What the RGB values of a frame mean.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameColor {
    /// Camera RGB of a raw file (not white balanced): the DNG colour tags of the source.
    Camera(Box<ColorData>),
    /// Linear RGB with these primaries: linear RGB → XYZ (D50-adapted, ICC PCS).
    Linear { to_xyz_d50: Mat3 },
}

/// One input photo.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Scene-linear pixels; 1.0 = the sensor/file white level.
    pub image: Rgb32f,
    /// Values at or above this (any channel) are saturated.
    pub clip: f32,
    /// Capture brightness from EXIF, in EV: log2(t · ISO / N²) (higher = brighter frame); `None`
    /// when the file doesn't say.
    pub exposure: Option<f64>,
    pub raw: bool,
    pub color: FrameColor,
    /// How `image` must be rotated/flipped for display (`Normal` once applied).
    pub orientation: Orientation,
    pub metadata: Metadata,
    /// Raw files: the source's `BaselineExposure` (EV), carried into the output.
    pub baseline_exposure: f64,
}

impl Frame {
    /// A frame from linear pixels in the develop working space (linear Rec.2020), e.g. a rendered
    /// procedural scene. `exposure` as in [`Frame::exposure`].
    pub fn from_linear_rec2020(image: Rgb32f, exposure: Option<f64>) -> Frame {
        let to_xyz_d50 = lightcraft_color::bradford(lightcraft_color::D65, lightcraft_color::D50).mul(&lightcraft_color::REC2020.to_xyz());
        let clip = detect_clip(&image);
        Frame {
            image,
            clip,
            exposure,
            raw: false,
            color: FrameColor::Linear { to_xyz_d50 },
            orientation: Orientation::Normal,
            metadata: Metadata::default(),
            baseline_exposure: 0.0,
        }
    }

    pub fn width(&self) -> usize {
        self.image.width
    }
    pub fn height(&self) -> usize {
        self.image.height
    }

    /// Apply the display orientation to the pixels.
    pub fn orient(&mut self) {
        if self.orientation != Orientation::Normal {
            self.image = self.image.oriented(self.orientation);
            self.orientation = Orientation::Normal;
        }
    }
}

/// EV of a capture from its EXIF: log2(t · ISO / N²). ISO defaults to 100 and N to 1 when absent
/// (only ratios between frames matter); `None` without an exposure time.
pub fn exif_exposure(m: &Metadata) -> Option<f64> {
    let t = m.exposure_time.filter(|t| *t > 0.0 && t.is_finite())?;
    let iso = m.iso.filter(|i| *i > 0).unwrap_or(100) as f64;
    let n = m.f_number.filter(|n| *n > 0.0 && n.is_finite()).unwrap_or(1.0);
    Some((t * iso / (n * n)).log2())
}

/// The saturation level of a frame: 1.0, or the (lower) level at which values pile up when the
/// file's white level is optimistic. Returned slightly below the level.
pub fn detect_clip(img: &Rgb32f) -> f32 {
    let maxv = img.data.iter().map(|p| p[0].max(p[1]).max(p[2])).fold(0.0f32, f32::max);
    if maxv.is_nan() || maxv <= 0.0 {
        return 1.0;
    }
    let level = maxv.min(1.0);
    // a pile-up: the top 0.5 % of the range is ≥ 4× denser than the 2 % just below it
    let (mut near, mut below) = (0usize, 0usize);
    for p in &img.data {
        let m = p[0].max(p[1]).max(p[2]);
        if m >= level * 0.995 {
            near += 1;
        } else if m >= level * 0.97 && m < level * 0.99 {
            below += 1;
        }
    }
    let piled = near as f64 > img.data.len() as f64 * 1e-4 && near >= 4 && near as f64 / 0.005 > 4.0 * below as f64 / 0.02;
    if maxv >= 1.0 || piled { level * 0.995 } else { 1.0 }
}

/// The largest mosaic binning factor that still leaves at least ~90 % of `max_edge` on the long
/// side (`None` when the raw can't be binned or is too small for it).
pub fn bin_factor(raw: &lightcraft_raw::RawImage, max_edge: usize) -> Option<usize> {
    let c = raw.crop.clipped(raw.active_area.width, raw.active_area.height);
    let long = if c.width > 1 && c.height > 1 { c.width.max(c.height) } else { raw.active_area.width.max(raw.active_area.height) };
    let need = (max_edge.saturating_mul(9) / 10).max(1);
    [8usize, 6, 4, 3, 2].into_iter().find(|&k| long / k >= need && raw.can_bin(k))
}

/// Decode a photo (raw or standard format) into a frame no larger than `max_edge` (when given).
/// Raw frames at most half the sensor size are binned from the mosaic rather than demosaiced.
/// `orient` applies the EXIF orientation to the pixels.
pub fn load_frame(bytes: &[u8], max_edge: Option<usize>, orient: bool) -> Result<Frame> {
    let metadata = lightcraft_meta::extract(bytes);
    let exposure = exif_exposure(&metadata);
    if lightcraft_raw::probe(bytes).is_some() {
        let mut raw = lightcraft_raw::decode(bytes).map_err(|e| MergeError::Decode(e.to_string()))?;
        // geometric lens corrections stay with the develop settings ("profile corrections")
        raw.opcodes.list3.retain(|op| !op.is_lens_correction());
        // a frame well below the sensor size bins the mosaic (no demosaicing: several times faster)
        let binned = match max_edge.and_then(|m| bin_factor(&raw, m)) {
            Some(k) => raw.develop_binned(k, 0.99).map_err(|e| MergeError::Decode(e.to_string()))?,
            None => None,
        };
        let small = max_edge.is_some_and(|m| m <= 1200);
        let method = if small { lightcraft_raw::Method::Bilinear } else { lightcraft_raw::Method::Ahd };
        let mut image = match binned {
            Some(img) => img,
            None => raw.develop(method).map_err(|e| MergeError::Decode(e.to_string()))?,
        };
        for p in image.data.iter_mut() {
            *p = p.map(|v| v.max(0.0));
        }
        let clip = detect_clip(&image);
        if let Some(m) = max_edge
            && image.width.max(image.height) > m
        {
            image = fit(&image, m, m, Filter::Box);
        }
        let mut f = Frame {
            image,
            clip,
            exposure,
            raw: true,
            color: FrameColor::Camera(Box::new(raw.color.clone())),
            orientation: raw.orientation,
            metadata: if raw.metadata == Metadata::default() { metadata } else { raw.metadata.clone() },
            baseline_exposure: raw.color.baseline_exposure,
        };
        if f.exposure.is_none() {
            f.exposure = exif_exposure(&f.metadata);
        }
        if orient {
            f.orient();
        }
        return Ok(f);
    }
    let opts = match max_edge {
        Some(m) => lightcraft_codecs::DecodeOptions::fit(m as u32, m as u32),
        None => lightcraft_codecs::DecodeOptions::default(),
    };
    let d = lightcraft_codecs::decode(bytes, opts).map_err(|e| MergeError::Decode(e.to_string()))?;
    let mut image = d.image;
    if let Some(m) = max_edge
        && image.width.max(image.height) > m
    {
        image = fit(&image, m, m, Filter::Box);
    }
    let clip = detect_clip(&image);
    let mut f = Frame {
        image,
        clip,
        exposure,
        raw: false,
        color: FrameColor::Linear { to_xyz_d50: d.space.to_xyz_d50 },
        orientation: Orientation::from_exif(d.orientation),
        metadata,
        baseline_exposure: 0.0,
    };
    if orient {
        f.orient();
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_from_exif() {
        let m = Metadata { exposure_time: Some(1.0 / 125.0), f_number: Some(8.0), iso: Some(100), ..Default::default() };
        let e = exif_exposure(&m).unwrap();
        let m2 = Metadata { exposure_time: Some(4.0 / 125.0), ..m.clone() };
        assert!((exif_exposure(&m2).unwrap() - e - 2.0).abs() < 1e-9);
        assert!(exif_exposure(&Metadata::default()).is_none());
    }

    /// A Bayer DNG of a smooth colour gradient.
    fn bayer_dng(w: usize, h: usize) -> Vec<u8> {
        use lightcraft_raw::*;
        let data: Vec<u16> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let c = (x % 2) + (y % 2); // R = 0, G = 1, B = 2 (RGGB)
                (512 + x * 20 + y * 10 + c * 1500) as u16
            })
            .collect();
        let raw = RawImage {
            format: RawFormat::Dng,
            width: w,
            height: h,
            cpp: 1,
            data: RawData::U16(data),
            cfa: Cfa::bayer("RGGB"),
            bits: 14,
            black: BlackLevel::uniform(256.0),
            white: vec![16000.0],
            active_area: Rect::new(0, 0, w, h),
            crop: Rect::new(0, 0, w, h),
            orientation: Orientation::Normal,
            color: ColorData {
                illuminant: [21, 0],
                color_matrix: [Some(lightcraft_color::Mat3([[0.9, 0.2, -0.15], [-0.3, 1.25, 0.08], [0.02, -0.12, 0.85]])), None],
                as_shot_neutral: Some([0.5, 1.0, 0.7]),
                ..Default::default()
            },
            wb_multipliers: None,
            linearized: false,
            opcodes: OpcodeLists::default(),
            metadata: Metadata::default(),
        };
        write_dng(&raw, &DngWriteOptions::default()).unwrap()
    }

    #[test]
    fn small_raw_frames_are_binned_like_demosaiced_ones() {
        let dng = bayer_dng(400, 300);
        let small = load_frame(&dng, Some(100), true).unwrap();
        assert_eq!((small.width(), small.height()), (100, 75));
        let full = load_frame(&dng, None, true).unwrap();
        assert_eq!((full.width(), full.height()), (400, 300));
        let fitted = fit(&full.image, 100, 100, Filter::Box);
        for (x, y) in [(10, 10), (50, 37), (90, 70)] {
            let (a, b) = (small.image.get(x, y), fitted.get(x, y));
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 0.01 + 0.02 * b[c], "({x}, {y}) {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn clip_detection() {
        let mut img = Rgb32f::from_fn(100, 100, |x, _| [x as f32 / 100.0; 3]);
        assert_eq!(detect_clip(&img), 1.0);
        for p in img.data.iter_mut().take(500) {
            *p = [0.9, 0.9, 0.9];
        }
        for p in img.data.iter_mut() {
            *p = p.map(|v| v.min(0.9));
        }
        assert!((detect_clip(&img) - 0.9 * 0.995).abs() < 1e-6);
    }
}
