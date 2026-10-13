//! A merge result takes the look of its sources (issue #24).
//!
//! Adjust a bracket first (white balance, exposure, profile, lens corrections, noise reduction…),
//! then merge: the result starts from the reference photo's settings instead of the as-shot white
//! and Auto Settings. What depends on the picture's geometry stays behind — crop, Upright and
//! transforms, masks, spot removal, red eye, lens blur — as the result is aligned and cropped
//! differently; so does Enhance (AI denoise, raw details, super resolution), which works on raw
//! sensor data the merged DNG no longer has.
//!
//! White balance needs care: most vendor raws (ARW, NEF, RW2, RAF, CR2/CR3…) and rendered files
//! keep it relative to the camera's as-shot white, while the merged DNG has an absolute one (its
//! `AsShotNeutral`). A relative setting is carried over as the white that makes the same
//! correction from the result's as-shot white ([`lightcraft_color::transfer_white`]: exact where
//! both render with Bradford adaptation).

use std::sync::Arc;

use lightcraft_color::cct::{temp_tint_to_xy, xy_to_temp_tint};
use lightcraft_develop::{DevelopSettings, SettingsGroup, WbMode, WhiteBalance};
use serde_json::Value;

/// A source photo's settings, with what reading its white balance takes.
#[derive(Clone, Debug)]
pub struct SourceLook {
    pub settings: Arc<DevelopSettings>,
    /// White balance relative to the photo's as-shot white (vendor raws, rendered files).
    pub relative_wb: bool,
    /// The as-shot white the settings are relative to (temperature, tint).
    pub as_shot: (f64, f64),
}

/// The settings groups a merge result takes (white balance aside, see [`inherited_wb`]).
const GROUPS: [SettingsGroup; 13] = [
    SettingsGroup::Profile,
    SettingsGroup::Treatment,
    SettingsGroup::Light,
    SettingsGroup::ToneCurve,
    SettingsGroup::Color,
    SettingsGroup::ColorMixer,
    SettingsGroup::ColorGrading,
    SettingsGroup::Effects,
    SettingsGroup::Vignette,
    SettingsGroup::Grain,
    SettingsGroup::Detail,
    SettingsGroup::Optics,
    SettingsGroup::Calibration,
];

/// Edit-panel sections whose on/off state goes with the settings.
const SECTIONS: [&str; 6] = ["light", "color", "detail", "effects", "optics", "calibration"];

/// Lens settings a panorama keeps its own of: its picture is a projection of all the frames.
const PANO_OWN_OPTICS: [&str; 6] = ["lens_profile", "profile_distortion", "profile_vignetting", "distortion", "vignetting", "vignetting_midpoint"];

/// `base` (the result's own settings) with the look of `look` carried over; `result_as_shot` is the
/// result's as-shot white. `pano`: lens geometry and vignetting are not carried over.
pub fn inherit(base: &DevelopSettings, look: &SourceLook, result_as_shot: (f64, f64), pano: bool) -> DevelopSettings {
    let mut partial = lightcraft_develop::extract_groups(&look.settings, &GROUPS);
    if let Some(o) = partial.as_object_mut() {
        // AI denoise, raw details and super resolution need the raw sensor data
        o.remove("enhance");
        if pano && let Some(Value::Object(optics)) = o.get_mut("optics") {
            for k in PANO_OWN_OPTICS {
                optics.remove(k);
            }
        }
    }
    let mut out = lightcraft_develop::apply_partial(base, &partial, 1.0);
    out.wb = inherited_wb(&look.settings.wb, look.relative_wb, look.as_shot, base.wb, result_as_shot);
    for section in SECTIONS {
        out.set_section_enabled(section, look.settings.section_enabled(section));
    }
    out
}

/// The result's white balance for a source's `wb`: As Shot stays the result's own (`own`); an
/// absolute setting is the same light on the result (same camera, same colour tags); a relative
/// one becomes the Custom white that makes the same correction from the result's as-shot white.
pub fn inherited_wb(wb: &WhiteBalance, relative: bool, as_shot: (f64, f64), own: WhiteBalance, result_as_shot: (f64, f64)) -> WhiteBalance {
    if wb.mode == WbMode::AsShot {
        return own;
    }
    if !relative {
        return *wb;
    }
    let to = lightcraft_color::transfer_white(
        temp_tint_to_xy(as_shot.0, as_shot.1),
        temp_tint_to_xy(wb.temp, wb.tint),
        temp_tint_to_xy(result_as_shot.0, result_as_shot.1),
    );
    match to.map(xy_to_temp_tint).filter(|(t, tint)| t.is_finite() && tint.is_finite()) {
        Some((t, tint)) => WhiteBalance { mode: WbMode::Custom, temp: t.clamp(2000.0, 50000.0), tint: tint.clamp(-150.0, 150.0) },
        None => own,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_pipeline::SourceInfo;
    use lightcraft_pipeline::local::wb_matrix_for;

    fn settings(wb: WhiteBalance) -> DevelopSettings {
        DevelopSettings { wb, ..Default::default() }
    }

    /// A relative source (a vendor raw or a JPEG) and the merged DNG render the carried-over white
    /// balance with the same matrix.
    #[test]
    fn a_relative_white_balance_renders_the_same_on_the_result() {
        let src = SourceInfo { raw: true, relative_wb: true, as_shot_temp: 6500.0, as_shot_tint: 0.0, ..Default::default() };
        let jpeg = SourceInfo::default();
        let res = SourceInfo { raw: true, relative_wb: false, as_shot_temp: 5180.0, as_shot_tint: 4.0, ..Default::default() };
        let own = settings(WhiteBalance { mode: WbMode::AsShot, temp: 5180.0, tint: 4.0 }).wb;
        for info in [&src, &jpeg] {
            for (t, tint) in [(4800.0, 12.0), (7400.0, -20.0), (6500.0, 30.0), (5600.0, 0.0)] {
                let wb = WhiteBalance { mode: WbMode::Custom, temp: t, tint };
                let a = wb_matrix_for(info, &settings(wb)).unwrap();
                let carried = inherited_wb(&wb, true, (info.as_shot_temp, info.as_shot_tint), own, (5180.0, 4.0));
                assert_eq!(carried.mode, WbMode::Custom);
                let b = wb_matrix_for(&res, &settings(carried)).unwrap();
                for i in 0..3 {
                    for j in 0..3 {
                        assert!((a[i][j] - b[i][j]).abs() < 1e-4, "{t}/{tint}: {a:?} vs {b:?} ({carried:?})");
                    }
                }
            }
        }
        // As Shot stays the result's own; an absolute setting is copied
        let shot = WhiteBalance { mode: WbMode::AsShot, temp: 6500.0, tint: 0.0 };
        assert_eq!(inherited_wb(&shot, true, (6500.0, 0.0), own, (5180.0, 4.0)), own);
        let daylight = WhiteBalance { mode: WbMode::Daylight, temp: 5500.0, tint: 10.0 };
        assert_eq!(inherited_wb(&daylight, false, (5000.0, 0.0), own, (5180.0, 4.0)), daylight);
        // a relative preset becomes Custom (on the absolute result a preset would mean absolute Kelvin)
        assert_eq!(inherited_wb(&daylight, true, (6500.0, 0.0), own, (5180.0, 4.0)).mode, WbMode::Custom);
    }

    #[test]
    fn the_look_goes_over_the_geometry_stays() {
        let mut s = DevelopSettings::default();
        s.light.exposure = 0.7;
        s.detail.sharpen_amount = 60.0;
        s.enhance.denoise = 50.0;
        s.optics.lens_profile = true;
        s.optics.distortion = 12.0;
        s.optics.remove_ca = true;
        s.crop.geometry.angle = 3.0;
        s.set_section_enabled("effects", false);
        let look = SourceLook { settings: Arc::new(s), relative_wb: false, as_shot: (5000.0, 0.0) };
        let base = DevelopSettings::default();
        let out = inherit(&base, &look, (5000.0, 0.0), false);
        assert_eq!((out.light.exposure, out.detail.sharpen_amount), (0.7, 60.0));
        assert_eq!(out.enhance, base.enhance, "no AI denoise on a merged DNG");
        assert_eq!(out.crop, base.crop);
        assert!(out.optics.lens_profile && out.optics.remove_ca && out.optics.distortion == 12.0);
        assert!(!out.section_enabled("effects"));
        // a panorama keeps its own lens geometry and vignetting, takes the rest
        let pano = inherit(&base, &look, (5000.0, 0.0), true);
        assert!(!pano.optics.lens_profile && pano.optics.distortion == 0.0 && pano.optics.remove_ca);
        assert_eq!(pano.light.exposure, 0.7);
    }
}
