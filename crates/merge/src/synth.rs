//! Synthetic merge inputs from procedural scenes (tests, demos, UI checks — no media files):
//! exposure brackets as linear DNGs, overlapping panorama views as PNGs, focus brackets with their
//! ground truth.

use lightcraft_geom::Orientation;
use lightcraft_meta::Metadata;
use lightcraft_raster::Rgb32f;

use crate::frame::FrameColor;
use crate::linalg::{self, M3};
use crate::output::{DngSamples, write_linear_dng};
use crate::{MergeError, Result};

fn scene(kind: lightcraft_scenes::Kind) -> Result<lightcraft_scenes::Scene> {
    lightcraft_scenes::demo_library()
        .into_iter()
        .find(|s| s.kind == kind)
        .ok_or_else(|| MergeError::Mismatch(format!("no {kind:?} scene in the demo library")))
}

fn srgb_color() -> FrameColor {
    FrameColor::Linear { to_xyz_d50: lightcraft_color::bradford(lightcraft_color::D65, lightcraft_color::D50).mul(&lightcraft_color::SRGB.to_xyz()) }
}

/// Exposure brackets (`evs`, e.g. `[-2, 0, 2]`) of a sunset scene as 16-bit linear DNGs, each
/// shifted by a few pixels (handheld) with EXIF exposure times; `w × h` pixels.
pub fn bracket_dngs(w: usize, h: usize, evs: &[f64]) -> Result<Vec<Vec<u8>>> {
    let m = 12usize;
    let truth = scene(lightcraft_scenes::Kind::OceanSunset)?.render(w + 2 * m, h + 2 * m);
    let mut l: Vec<f32> = truth.data.iter().map(|p| p[1]).collect();
    l.sort_by(|a, b| a.total_cmp(b));
    let s = 0.18 / l.get(l.len() / 2).copied().unwrap_or(0.0).max(1e-6);
    let to_srgb = lightcraft_color::REC2020.to_space(&lightcraft_color::SRGB).to_f32();
    evs.iter()
        .enumerate()
        .map(|(i, ev)| {
            let k = 2f32.powf(*ev as f32) * s;
            let (dx, dy) = ((i as f32 * 2.3) % 5.0 - 2.0, (i as f32 * 1.7) % 4.0 - 1.5);
            let img = Rgb32f::from_fn(w, h, |x, y| {
                let p = truth.sample_bilinear(x as f32 + 0.5 + m as f32 - dx, y as f32 + 0.5 + m as f32 - dy);
                let q = [0, 1, 2].map(|r| to_srgb[r][0] * p[0] + to_srgb[r][1] * p[1] + to_srgb[r][2] * p[2]);
                q.map(|v| (v * k).clamp(0.0, 1.0))
            });
            let meta = Metadata {
                make: Some("LightCraft".into()),
                model: Some("Synthetic Bracket".into()),
                exposure_time: Some(2f64.powf(*ev) / 125.0),
                f_number: Some(8.0),
                iso: Some(100),
                ..Default::default()
            };
            write_linear_dng(&img, &srgb_color(), Orientation::Normal, &meta, 0.0, DngSamples::U16)
        })
        .collect()
}

/// `yaws.len()` overlapping views (degrees of yaw, small pitch/roll wobble) of a canyon scene used
/// as a 200° × 70° spherical environment, `w × h` pixels with focal length `f`, as linear sRGB.
pub fn pano_views(w: usize, h: usize, f: f64, yaws: &[f64]) -> Result<Vec<Rgb32f>> {
    let env = scene(lightcraft_scenes::Kind::Canyon)?.render(2400, 840);
    let mut l: Vec<f32> = env.data.iter().map(|p| p[1]).collect();
    l.sort_by(|a, b| a.total_cmp(b));
    let s = 0.25 / l.get(l.len() / 2).copied().unwrap_or(0.0).max(1e-6);
    let (theta, phi) = (200.0f64, 70.0f64);
    Ok(yaws
        .iter()
        .enumerate()
        .map(|(i, yaw)| {
            let wob = (i as f64 * 1.3).sin();
            let ry = linalg::rodrigues([0.0, yaw.to_radians(), 0.0]);
            let rx = linalg::rodrigues([(wob * 1.5).to_radians(), 0.0, 0.0]);
            let rz = linalg::rodrigues([0.0, 0.0, (wob * 0.8).to_radians()]);
            let c2w: M3 = linalg::mul3(&linalg::mul3(&ry, &rx), &rz);
            Rgb32f::from_fn(w, h, |x, y| {
                let d = linalg::apply3(&c2w, [(x as f64 + 0.5 - w as f64 / 2.0) / f, (y as f64 + 0.5 - h as f64 / 2.0) / f, 1.0]);
                let t = d[0].atan2(d[2]).to_degrees();
                let p = (d[1] / linalg::norm(d)).asin().to_degrees();
                if t.abs() > theta / 2.0 || p.abs() > phi / 2.0 {
                    return [0.0; 3];
                }
                let px = (t / theta + 0.5) * env.width as f64;
                let py = (p / phi + 0.5) * env.height as f64;
                env.sample_bilinear(px as f32, py as f32).map(|v| (v * s).min(1.0))
            })
        })
        .collect())
}

/// Parameters of a synthetic focus bracket ([`focus_bracket`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FocusSynth {
    /// Frames, focused at depths 0, 1, … n−1.
    pub n: usize,
    /// Defocus blur (Gaussian σ, pixels) per frame of focus distance…
    pub blur_per_frame: f32,
    /// …up to this much.
    pub max_blur: f32,
    /// Magnification change per frame (focus breathing), e.g. 0.002.
    pub breathing: f64,
    /// Handheld drift per frame: shift (pixels) and rotation (degrees), random walks.
    pub shift: f64,
    pub rotation: f64,
    /// Noise (standard deviation, linear 0..1 units), then 14-bit quantisation.
    pub noise: f32,
    /// Brightness wobble between frames (relative, ±).
    pub flicker: f32,
    pub seed: u64,
}

impl Default for FocusSynth {
    fn default() -> Self {
        FocusSynth { n: 12, blur_per_frame: 1.2, max_blur: 6.0, breathing: 0.002, shift: 0.6, rotation: 0.05, noise: 0.002, flicker: 0.0, seed: 1 }
    }
}

/// Uniform in [0, 1) from a 64-bit key (splitmix64).
fn hash_unit(k: u64) -> f64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// A synthetic focus bracket and its ground truth.
#[derive(Clone, Debug)]
pub struct FocusBracket {
    /// Linear Rec.2020 frames, `w × h`, in focus order.
    pub frames: Vec<Rgb32f>,
    /// The scene sharp everywhere, in "truth" coordinates.
    pub truth: Rgb32f,
    /// Depth of each truth pixel, in frames (0..n−1): where the frame focused there is sharp.
    pub depth: lightcraft_raster::Plane,
    /// Per frame: truth pixel coordinates → frame pixel coordinates.
    pub transforms: Vec<lightcraft_geom::Homography>,
}

/// A focus bracket of a macro scene: a background whose depth slopes across the picture and a
/// textured disc in front (a depth step, where halos form). Each frame blurs every layer by its
/// distance from the frame's focus (layered: the front disc's blur spills over the background as
/// in a real lens), breathes, drifts, gets noise and 14-bit quantisation.
pub fn focus_bracket(w: usize, h: usize, p: &FocusSynth) -> Result<FocusBracket> {
    use lightcraft_geom::Homography;
    use lightcraft_raster::Plane;
    use lightcraft_raster::blur::gaussian;
    let n = p.n.max(2);
    let m = ((w.max(h) as f64) * 0.06).ceil() as usize + 8;
    let (cw, ch) = (w + 2 * m, h + 2 * m);
    let base = scene(lightcraft_scenes::Kind::Macro)?.render(cw, ch);
    // fine texture everywhere (so every area has focus information), brightness normalised
    let mut rng = crate::ransac::Rng::new(p.seed.wrapping_mul(0x9e37_79b9).max(1));
    let cells = 6usize;
    let noise_grid: Vec<f32> = (0..(cw / cells + 2) * (ch / cells + 2)).map(|_| rng.unit() as f32).collect();
    let gw = cw / cells + 2;
    let tex = |x: usize, y: usize| {
        let (gx, gy) = (x as f32 / cells as f32, y as f32 / cells as f32);
        let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
        let (tx, ty) = (gx - x0 as f32, gy - y0 as f32);
        let g = |a: usize, b: usize| noise_grid.get(b * gw + a).copied().unwrap_or(0.5);
        let top = g(x0, y0) * (1.0 - tx) + g(x0 + 1, y0) * tx;
        let bot = g(x0, y0 + 1) * (1.0 - tx) + g(x0 + 1, y0 + 1) * tx;
        0.75 + 0.5 * (top * (1.0 - ty) + bot * ty)
    };
    let mut lum: Vec<f32> = base.data.iter().map(|q| q[1]).collect();
    lum.sort_by(|a, b| a.total_cmp(b));
    let s = 0.2 / lum.get(lum.len() / 2).copied().unwrap_or(0.0).max(1e-6);
    let bg = Rgb32f::from_fn(cw, ch, |x, y| base.get(x, y).map(|v| (v * s * tex(x, y)).clamp(0.0, 0.8)));
    let fg = Rgb32f::from_fn(cw, ch, |x, y| {
        let q = base.get(cw - 1 - x, y);
        let t = tex(y % cw, x % ch);
        [(q[1] * s * t * 1.3).clamp(0.0, 0.9), (q[0] * s * t * 0.8).clamp(0.0, 0.9), (q[2] * s * t).clamp(0.0, 0.9)]
    });
    let last = (n - 1) as f32;
    // depth: background slope; disc in front
    let (dcx, dcy, dr) = (m as f32 + 0.32 * w as f32, m as f32 + 0.45 * h as f32, 0.2 * w.min(h) as f32);
    let alpha = Plane::from_fn(cw, ch, |x, y| {
        let d = ((x as f32 + 0.5 - dcx).powi(2) + (y as f32 + 0.5 - dcy).powi(2)).sqrt();
        (dr - d + 0.5).clamp(0.0, 1.0)
    });
    let z_bg = |x: usize, y: usize| {
        let (u, v) = ((x as f32 - m as f32) / w as f32, (y as f32 - m as f32) / h as f32);
        (last * (0.05 + 0.75 * (0.75 * u + 0.25 * v))).clamp(0.0, last)
    };
    let z_fg = last * 0.92;
    // pre-blurred background levels
    let levels = 8usize;
    let sig = |j: usize| j as f32 * p.max_blur / (levels - 1) as f32;
    let bg_levels: Vec<Rgb32f> = (0..levels).map(|j| gaussian(&bg, sig(j))).collect();
    let fg_pm = Rgb32f::from_fn(cw, ch, |x, y| fg.get(x, y).map(|v| v * alpha.get(x, y)));
    let truth_canvas = Rgb32f::from_fn(cw, ch, |x, y| {
        let a = alpha.get(x, y);
        let (f, b) = (fg_pm.get(x, y), bg.get(x, y));
        [0, 1, 2].map(|c| f[c] + (1.0 - a) * b[c])
    });
    let truth = truth_canvas.crop(m, m, w, h);
    let depth = Plane::from_fn(w, h, |x, y| if alpha.get(x + m, y + m) >= 0.5 { z_fg } else { z_bg(x + m, y + m) });
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (mut tx, mut ty, mut th) = (0.0f64, 0.0f64, 0.0f64);
    let mut frames = Vec::with_capacity(n);
    let mut transforms = Vec::with_capacity(n);
    for i in 0..n {
        let blur = |z: f32| (p.blur_per_frame * (z - i as f32).abs()).min(p.max_blur);
        // background with depth-dependent blur (interpolating the pre-blurred levels)
        let bgi = Rgb32f::from_fn(cw, ch, |x, y| {
            let sgm = blur(z_bg(x, y));
            let f = if p.max_blur > 0.0 { sgm / p.max_blur * (levels - 1) as f32 } else { 0.0 };
            let j = (f.floor() as usize).min(levels - 2);
            let t = (f - j as f32).clamp(0.0, 1.0);
            let (a, b) = (bg_levels[j].get(x, y), bg_levels[j + 1].get(x, y));
            [0, 1, 2].map(|c| a[c] * (1.0 - t) + b[c] * t)
        });
        let sf = blur(z_fg);
        let (fpm, fa) = (gaussian(&fg_pm, sf), gaussian(&alpha, sf));
        let gain = 1.0 + p.flicker * (2.0 * rng.unit() as f32 - 1.0);
        let comp = Rgb32f::from_fn(cw, ch, |x, y| {
            let (f, a, b) = (fpm.get(x, y), fa.get(x, y), bgi.get(x, y));
            [0, 1, 2].map(|c| (f[c] + (1.0 - a) * b[c]) * gain)
        });
        // geometry: truth coordinates → frame coordinates
        if i > 0 {
            tx += p.shift * (2.0 * rng.unit() - 1.0);
            ty += p.shift * (2.0 * rng.unit() - 1.0);
            th += p.rotation.to_radians() * (2.0 * rng.unit() - 1.0);
        }
        let sc = 1.0 + p.breathing * (i as f64 - (n - 1) as f64 / 2.0);
        let (c, sn) = (sc * th.cos(), sc * th.sin());
        let g = Homography([c, -sn, cx - (c * cx - sn * cy) + tx, sn, c, cy - (sn * cx + c * cy) + ty, 0.0, 0.0, 1.0]);
        let gi = g.inverse().ok_or_else(|| MergeError::Mismatch("degenerate synthetic transform".into()))?;
        let seed = p.seed ^ ((i as u64 + 1) << 40);
        let img = Rgb32f::from_fn(w, h, |x, y| {
            let (qx, qy) = gi.apply_real(x as f64 + 0.5, y as f64 + 0.5);
            let v = comp.sample_bilinear(qx as f32 + m as f32, qy as f32 + m as f32);
            std::array::from_fn(|c| {
                // noise: Irwin–Hall approximation of a Gaussian, from a per-sample hash
                let k = seed ^ (((y * w + x) as u64) << 2 | c as u64);
                let nz: f64 = (0..4u64).map(|j| hash_unit(k.wrapping_add(j.wrapping_mul(0x632b_e59b_d9b4_e019)))).sum::<f64>() - 2.0;
                let v = (v[c] + p.noise * nz as f32 * 1.732).clamp(0.0, 1.0);
                (v * 16383.0).round() / 16383.0
            })
        });
        frames.push(img);
        transforms.push(g);
    }
    Ok(FocusBracket { frames, truth, depth, transforms })
}

/// A focus bracket ([`focus_bracket`]) as 16-bit linear DNGs with capture times one second apart,
/// the first `first_second` seconds after 10:00.
pub fn focus_bracket_dngs(w: usize, h: usize, n: usize, first_second: usize) -> Result<Vec<Vec<u8>>> {
    let b = focus_bracket(w, h, &FocusSynth { n, ..Default::default() })?;
    let to_xyz = lightcraft_color::bradford(lightcraft_color::D65, lightcraft_color::D50).mul(&lightcraft_color::REC2020.to_xyz());
    let color = FrameColor::Linear { to_xyz_d50: to_xyz };
    b.frames
        .iter()
        .enumerate()
        .map(|(i, img)| {
            let meta = Metadata {
                make: Some("LightCraft".into()),
                model: Some("Synthetic Focus Bracket".into()),
                exposure_time: Some(1.0 / 125.0),
                f_number: Some(5.6),
                iso: Some(100),
                capture_time: Some(lightcraft_meta::DateTime {
                    year: 2026,
                    month: 10,
                    day: 12,
                    hour: 10,
                    minute: ((first_second + i) / 60 % 60) as u8,
                    second: ((first_second + i) % 60) as u8,
                    millis: 0,
                    offset_minutes: None,
                }),
                ..Default::default()
            };
            write_linear_dng(img, &color, Orientation::Normal, &meta, 0.0, DngSamples::U16)
        })
        .collect()
}
