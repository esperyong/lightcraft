//! Focus stacking on synthetic focus brackets (`synth::focus_bracket`): every method against
//! the ground truth, alignment against the known motion, the streaming contract, failure modes.

use lightcraft_geom::Homography;
use lightcraft_merge::focus::{AlignOptions, FocusMethod, FocusOptions, FocusOrder, FocusResult, FocusStacker, Pass, stack_focus};
use lightcraft_merge::synth::{FocusBracket, FocusSynth, focus_bracket};
use lightcraft_merge::{Frame, MergeError, no_progress};
use lightcraft_raster::Rgb32f;

const W: usize = 640;
const H: usize = 420;

fn frames(b: &FocusBracket) -> Vec<Frame> {
    b.frames.iter().map(|f| Frame::from_linear_rec2020(f.clone(), None)).collect()
}

fn bracket(p: FocusSynth) -> FocusBracket {
    focus_bracket(W, H, &p).unwrap()
}

/// Truth coordinates of a result pixel.
fn truth_xy(b: &FocusBracket, r: &FocusResult, u: usize, v: usize) -> (f64, f64) {
    let g = b.transforms[r.reference].inverse().unwrap();
    g.apply_real((u + r.crop.x) as f64 + 0.5, (v + r.crop.y) as f64 + 0.5)
}

fn inside(x: f64, y: f64, margin: f64) -> bool {
    x >= margin && y >= margin && x <= W as f64 - margin && y <= H as f64 - margin
}

/// PSNR (dB, peak 1) of `img` against the truth, `img` pixel (u, v) showing truth point `map(u, v)`.
fn psnr(b: &FocusBracket, img: &Rgb32f, map: impl Fn(usize, usize) -> (f64, f64)) -> f64 {
    let (mut se, mut n) = (0.0f64, 0usize);
    for v in 0..img.height {
        for u in 0..img.width {
            let (x, y) = map(u, v);
            if !inside(x, y, 16.0) {
                continue;
            }
            let t = b.truth.sample_bilinear(x as f32, y as f32);
            let p = img.get(u, v);
            for c in 0..3 {
                se += ((p[c] - t[c]) as f64).powi(2);
            }
            n += 3;
        }
    }
    10.0 * (n as f64 / se.max(1e-20)).log10()
}

fn result_psnr(b: &FocusBracket, r: &FocusResult) -> f64 {
    psnr(b, &r.image, |u, v| truth_xy(b, r, u, v))
}

fn best_frame_psnr(b: &FocusBracket) -> f64 {
    b.frames
        .iter()
        .zip(&b.transforms)
        .map(|(f, g)| {
            let gi = g.inverse().unwrap();
            psnr(b, f, |u, v| gi.apply_real(u as f64 + 0.5, v as f64 + 0.5))
        })
        .fold(f64::MIN, f64::max)
}

/// Mean |depth error| in frames where the confidence is > 0.5 (and the fraction of such pixels),
/// away from the disc's edge.
fn depth_error(b: &FocusBracket, r: &FocusResult) -> (f64, f64) {
    let last = (r.used.len() - 1) as f64;
    let (mut e, mut n, mut total) = (0.0, 0usize, 0usize);
    for v in 0..r.image.height {
        for u in 0..r.image.width {
            let (x, y) = truth_xy(b, r, u, v);
            if !inside(x, y, 16.0) {
                continue;
            }
            let (xi, yi) = (x as usize, y as usize);
            // skip the depth step (±8 px around the disc's edge)
            let z = b.depth.get(xi, yi);
            let near_step = (0..17).any(|k| {
                let (xx, yy) = ((xi + k).saturating_sub(8).min(W - 1), yi);
                (b.depth.get(xx, yy) - z).abs() > 1.5
            }) || (0..17).any(|k| {
                let (xx, yy) = (xi, (yi + k).saturating_sub(8).min(H - 1));
                (b.depth.get(xx, yy) - z).abs() > 1.5
            });
            if near_step {
                continue;
            }
            total += 1;
            let c = r.confidence.as_ref().map_or(1.0, |c| c.get(u, v));
            if c > 0.5 {
                e += (r.depth.get(u, v) as f64 * last - z as f64).abs();
                n += 1;
            }
        }
    }
    (e / n.max(1) as f64, n as f64 / total.max(1) as f64)
}

/// Largest corner error (px) of the estimated alignments against the synthetic motion.
fn alignment_error(b: &FocusBracket, r: &FocusResult) -> f64 {
    let gr = b.transforms[r.reference].inverse().unwrap();
    r.alignments
        .iter()
        .map(|a| {
            let truth: Homography = b.transforms[a.index].mul(&gr);
            [(40.0, 40.0), (600.0, 40.0), (40.0, 380.0), (600.0, 380.0)]
                .iter()
                .map(|&(x, y)| {
                    let (p, q) = (a.h.apply_real(x, y), truth.apply_real(x, y));
                    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn run(b: &FocusBracket, opts: &FocusOptions) -> FocusResult {
    stack_focus(&frames(b), opts, &no_progress).unwrap()
}

#[test]
fn every_method_beats_every_frame() {
    let b = bracket(FocusSynth::default());
    let single = best_frame_psnr(&b);
    let mut got = Vec::new();
    for m in [FocusMethod::Average, FocusMethod::Depth, FocusMethod::Pyramid] {
        let r = run(&b, &FocusOptions { method: m, ..Default::default() });
        assert_eq!(r.used.len(), 12);
        assert!(r.skipped.is_empty() && r.warnings.is_empty(), "{:?} {:?}", r.skipped, r.warnings);
        let p = result_psnr(&b, &r);
        eprintln!("{m:?}: {p:.2} dB (best single frame {single:.2} dB), crop {:?}", r.crop);
        got.push(p);
    }
    let (a, bb, c) = (got[0], got[1], got[2]);
    assert!(a > single + 4.0, "A {a:.2} vs frame {single:.2}");
    assert!(bb > single + 6.0, "B {bb:.2} vs frame {single:.2}");
    assert!(c > single + 6.0, "C {c:.2} vs frame {single:.2}");
    assert!(bb >= a, "B {bb:.2} vs A {a:.2}");
}

#[test]
fn depth_map_finds_the_focus_distance() {
    let b = bracket(FocusSynth::default());
    let r = run(&b, &FocusOptions::default());
    let (mae, frac) = depth_error(&b, &r);
    eprintln!("depth MAE {mae:.3} frames over {:.0} % confident pixels", frac * 100.0);
    assert!(mae < 1.0, "depth MAE {mae}");
    // the confidence is conservative (textured but weakly peaked pixels fall below 0.5)
    assert!(frac > 0.3, "confident fraction {frac}");
    assert_eq!((r.depth.width, r.depth.height), (r.image.width, r.image.height));
}

#[test]
fn pyramid_keeps_the_brightness() {
    let b = bracket(FocusSynth::default());
    let r = run(&b, &FocusOptions { method: FocusMethod::Pyramid, ..Default::default() });
    let (mut s, mut t) = (0.0f64, 0.0f64);
    for v in 0..r.image.height {
        for u in 0..r.image.width {
            let (x, y) = truth_xy(&b, &r, u, v);
            if inside(x, y, 16.0) {
                s += r.image.get(u, v)[1] as f64;
                t += b.truth.sample_bilinear(x as f32, y as f32)[1] as f64;
            }
        }
    }
    let ratio = s / t;
    eprintln!("C mean ratio {ratio:.4}");
    assert!((0.98..1.02).contains(&ratio), "{ratio}");
}

#[test]
fn alignment_recovers_breathing_drift_and_rotation() {
    let b = bracket(FocusSynth { breathing: 0.004, shift: 1.5, rotation: 0.15, ..Default::default() });
    let r = run(&b, &FocusOptions::default());
    let err = alignment_error(&b, &r);
    eprintln!("alignment error {err:.3} px, crop {:?}", r.crop);
    assert!(err < 0.3, "alignment error {err} px");
    assert!(r.crop.w < W && r.crop.h < H, "{:?}", r.crop);
    // without the scale degree of freedom the breathing can't be followed
    let frozen = FocusOptions { align: AlignOptions { scale: false, ..Default::default() }, ..Default::default() };
    let r2 = run(&b, &frozen);
    let err2 = alignment_error(&b, &r2);
    eprintln!("alignment error without scale {err2:.3} px");
    assert!(err2 > 3.0 * err.max(0.1), "{err2} vs {err}");
}

#[test]
fn brightness_flicker_is_evened_out() {
    let b = bracket(FocusSynth { flicker: 0.08, ..Default::default() });
    let r = run(&b, &FocusOptions::default());
    let off = run(&b, &FocusOptions { align: AlignOptions { brightness: false, ..Default::default() }, ..Default::default() });
    let (p, q) = (result_psnr(&b, &r), result_psnr(&b, &off));
    eprintln!("flicker: with gains {p:.2} dB, without {q:.2} dB");
    assert!(p > q, "{p} vs {q}");
}

#[test]
fn state_does_not_grow_with_the_number_of_frames() {
    let measure = |n: usize, method: FocusMethod| {
        let b = focus_bracket(320, 210, &FocusSynth { n, ..Default::default() }).unwrap();
        let fr = frames(&b);
        let mut st = FocusStacker::new(n, FocusOptions { method, ..Default::default() }).unwrap();
        let mut peak = 0;
        let mut passes = Vec::new();
        while let Some(req) = st.next_pass().unwrap() {
            for &i in &req.frames {
                st.push(i, &fr[i], &no_progress).unwrap();
                peak = peak.max(st.state_bytes());
            }
            st.end_pass(&no_progress).unwrap();
            passes.push(req.pass);
        }
        assert_eq!(passes[..2], [Pass::Align, Pass::Fuse]);
        let r = st.finish(&no_progress).unwrap();
        (peak, r.crop)
    };
    for m in [FocusMethod::Average, FocusMethod::Depth, FocusMethod::Pyramid] {
        let (a, ca) = measure(10, m);
        let (b, cb) = measure(40, m);
        eprintln!("{m:?}: peak state {a} B (10 frames) vs {b} B (40 frames)");
        // the same per-pixel state; the output area differs a little with the drift
        let (pa, pb) = ((ca.w * ca.h) as f64, (cb.w * cb.h) as f64);
        assert!((a as f64 / pa - b as f64 / pb).abs() < 0.05 * a as f64 / pa + 64.0, "{a} {b}");
    }
}

#[test]
fn every_nth_frame_and_reverse_order() {
    let b = bracket(FocusSynth { n: 13, ..Default::default() });
    let r = run(&b, &FocusOptions { every: 4, ..Default::default() });
    assert_eq!(r.order, vec![0, 4, 8, 12]);
    let rev = run(&b, &FocusOptions { order: FocusOrder::Reverse, every: 4, ..Default::default() });
    assert_eq!(rev.order, vec![12, 8, 4, 0]);
    // same frames, same result (the depth runs the other way)
    let (p, q) = (result_psnr(&b, &r), result_psnr(&b, &rev));
    assert!((p - q).abs() < 0.5, "{p} vs {q}");
    let d = (r.depth.get(100, 100) + rev.depth.get(100, 100) - 1.0).abs();
    assert!(d < 0.2, "{} {}", r.depth.get(100, 100), rev.depth.get(100, 100));
}

#[test]
fn auto_order_sorts_a_shuffled_bracket() {
    let b = bracket(FocusSynth { n: 10, ..Default::default() });
    let perm = [6, 2, 9, 0, 4, 7, 1, 5, 8, 3];
    let shuffled: Vec<Frame> = perm.iter().map(|&i| Frame::from_linear_rec2020(b.frames[i].clone(), None)).collect();
    let r = stack_focus(&shuffled, &FocusOptions::default(), &no_progress).unwrap();
    assert!(r.warnings.iter().any(|w| w.contains("order")), "{:?}", r.warnings);
    let r = stack_focus(&shuffled, &FocusOptions { order: FocusOrder::Auto, ..Default::default() }, &no_progress).unwrap();
    let got: Vec<usize> = r.order.iter().map(|&i| perm[i]).collect();
    let fwd: Vec<usize> = (0..10).collect();
    let back: Vec<usize> = (0..10).rev().collect();
    assert!(got == fwd || got == back, "{got:?}");
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
}

#[test]
fn a_foreign_frame_is_skipped() {
    let b = bracket(FocusSynth { n: 8, ..Default::default() });
    let mut fr = frames(&b);
    let other = lightcraft_merge::synth::pano_views(W, H, 600.0, &[0.0]).unwrap().remove(0);
    fr.insert(3, Frame::from_linear_rec2020(other, None));
    let r = stack_focus(&fr, &FocusOptions::default(), &no_progress).unwrap();
    assert_eq!(r.skipped, vec![3]);
    assert_eq!(r.used.len(), 8);
    assert!(r.warnings.iter().any(|w| w.contains("left out")));
}

#[test]
fn flat_areas_do_not_amplify_noise() {
    let flat = |k: usize| {
        Rgb32f::from_fn(200, 140, |x, y| {
            let h = ((x * 7_919 + y * 104_729 + k * 1_299_709) % 1_000) as f32 / 1_000.0 - 0.5;
            [0.3 + 0.01 * h; 3]
        })
    };
    let fr: Vec<Frame> = (0..6).map(|k| Frame::from_linear_rec2020(flat(k), None)).collect();
    let sd = |img: &Rgb32f| {
        let m = img.data.iter().map(|p| p[1] as f64).sum::<f64>() / img.data.len() as f64;
        (img.data.iter().map(|p| (p[1] as f64 - m).powi(2)).sum::<f64>() / img.data.len() as f64).sqrt()
    };
    let one = sd(&flat(0));
    for m in [FocusMethod::Average, FocusMethod::Depth, FocusMethod::Pyramid] {
        let r = stack_focus(
            &fr,
            &FocusOptions { method: m, align: AlignOptions { enabled: false, ..Default::default() }, ..Default::default() },
            &no_progress,
        )
        .unwrap();
        let s = sd(&r.image);
        eprintln!("{m:?}: noise {s:.5} vs one frame {one:.5}");
        assert!(s <= one * 1.15, "{m:?}: {s} vs {one}");
    }
}

#[test]
fn halo_at_the_depth_step_stays_narrow() {
    let b = bracket(FocusSynth::default());
    let r = run(&b, &FocusOptions::default());
    // mean abs error in rings around the disc's edge (truth coordinates), for the result and for
    // an oracle that takes each pixel from the frame focused at its true depth (whose error is
    // the front disc's blur spilling over the background: not recoverable from these frames)
    let (dcx, dcy, dr) = (0.32 * W as f64, 0.45 * H as f64, 0.2 * H as f64);
    let ring_of = |x: f64, y: f64| {
        let d = ((x - dcx).powi(2) + (y - dcy).powi(2)).sqrt() - dr;
        if d.abs() < 4.0 {
            Some(0)
        } else if (4.0..12.0).contains(&d) {
            Some(1)
        } else if (12.0..24.0).contains(&d) {
            Some(2)
        } else if d >= 40.0 {
            Some(3)
        } else {
            None
        }
    };
    let mut ours = [(0.0f64, 0usize); 4];
    let mut oracle = [(0.0f64, 0usize); 4];
    for v in 0..r.image.height {
        for u in 0..r.image.width {
            let (x, y) = truth_xy(&b, &r, u, v);
            let Some(k) = ring_of(x, y).filter(|_| inside(x, y, 16.0)) else { continue };
            let t = b.truth.sample_bilinear(x as f32, y as f32);
            let p = r.image.get(u, v);
            ours[k].0 += (0..3).map(|c| ((p[c] - t[c]) as f64).abs()).sum::<f64>() / 3.0;
            ours[k].1 += 1;
            let f = (b.depth.get(x as usize, y as usize).round() as usize).min(b.frames.len() - 1);
            let (fx, fy) = b.transforms[f].apply_real(x, y);
            let q = b.frames[f].sample_bilinear(fx as f32, fy as f32);
            oracle[k].0 += (0..3).map(|c| ((q[c] - t[c]) as f64).abs()).sum::<f64>() / 3.0;
            oracle[k].1 += 1;
        }
    }
    let ours: Vec<f64> = ours.iter().map(|(e, n)| e / (*n).max(1) as f64).collect();
    let oracle: Vec<f64> = oracle.iter().map(|(e, n)| e / (*n).max(1) as f64).collect();
    eprintln!("mean abs error (edge, 4–12 px out, 12–24 px, far): ours {ours:.4?}, oracle {oracle:.4?}");
    for k in 1..4 {
        assert!(ours[k] < 1.5 * oracle[k] + 0.003, "ring {k}: {ours:?} vs oracle {oracle:?}");
    }
}

#[test]
fn cancelling_and_bad_inputs_are_errors() {
    let b = focus_bracket(200, 140, &FocusSynth { n: 4, ..Default::default() }).unwrap();
    let fr = frames(&b);
    let cancel = |f: f32, _: &str| f < 0.5;
    assert_eq!(stack_focus(&fr, &FocusOptions::default(), &cancel).unwrap_err(), MergeError::Cancelled);
    assert!(matches!(stack_focus(&fr[..1].to_vec(), &FocusOptions::default(), &no_progress), Err(MergeError::TooFew(1, 2))));
    let mut odd = fr.clone();
    odd[2] = Frame::from_linear_rec2020(Rgb32f::filled(140, 200, [0.2; 3]), None);
    assert!(matches!(stack_focus(&odd, &FocusOptions::default(), &no_progress), Err(MergeError::Mismatch(_))));
    let no_align = FocusOptions { align: AlignOptions { enabled: false, brightness: false, ..Default::default() }, ..Default::default() };
    assert!(matches!(stack_focus(&odd, &no_align, &no_progress), Err(MergeError::Mismatch(_))));
}
