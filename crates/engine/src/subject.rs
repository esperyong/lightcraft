//! Select Subject / Background with Apple Vision: on a Mac (macOS 14+), `mask.add {kind: subject|background}`
//! asks Vision for the photo's subject (every foreground instance it finds) and stores it with the shape as a
//! segmentation (`MaskShape::Subject { seg }`, `Background { seg }`: the inverse), so the mask renders the same in the
//! preview and the export, at any size, and on every platform. Without Vision (other systems, older macOS) the shapes
//! keep no segmentation and the renderer estimates them as before.
//!
//! Vision sees the frame mask coordinates live in, as SAM 3 does: the photo as developed, uncropped and without masks,
//! at [`crate::segment::INPUT_EDGE`] px. Its matte is averaged onto the stored square grid and kept as logits. In the
//! app ([`crate::segment::Segmenter::background`]) the mask is added at once with the estimate and the subject replaces
//! it when Vision answers ([`Session::segment_poll`] applies it); elsewhere the command waits for it.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use lightcraft_catalog::PhotoId;
use lightcraft_develop::{MaskShape, SegMask};
use lightcraft_raster::Rgba8;
use lightcraft_vision::Matte;

use crate::Session;
use crate::segment::Polled;

/// What finds the subject of a picture: a matte the picture's size, or `None` when it has none.
pub type SubjectFinder = Arc<dyn Fn(&Rgba8) -> Result<Option<Matte>, String> + Send + Sync>;

/// Side of the stored segmentation grid (the photo stretched to a square; Vision's own mask is coarser).
pub const SEG_SIDE: usize = 512;
/// Most subject requests running at once (each renders the photo; more would only queue in Vision).
const MAX_RUNNING: usize = 2;

/// Apple Vision's subject, where this Mac has it.
pub fn system_finder() -> Option<SubjectFinder> {
    if !lightcraft_vision::features().subject {
        return None;
    }
    Some(Arc::new(|img: &Rgba8| lightcraft_vision::subject(lightcraft_vision::Image { width: img.width, height: img.height, pixels: &img.data })))
}

/// `m` on a `side × side` grid over the whole picture (stretched), as logits: each cell the mean coverage of the matte
/// pixels it covers. `invert`: the rest of the picture (Background). `None` for a matte without pixels.
pub fn seg_from_matte(m: &Matte, side: usize, invert: bool) -> Option<SegMask> {
    let (w, h, n) = (m.width, m.height, side.min(lightcraft_develop::segmask::MAX_SIDE));
    if w == 0 || h == 0 || n == 0 || w.checked_mul(h) != Some(m.alpha.len()) {
        return None;
    }
    // the matte pixels each cell row / column covers (at least one)
    let span = |i: usize, len: usize| {
        let a = i * len / n;
        let b = ((i + 1) * len).div_ceil(n).max(a + 1).min(len);
        (a.min(len - 1), b)
    };
    let mut logits = Vec::with_capacity(n * n);
    for y in 0..n {
        let (y0, y1) = span(y, h);
        for x in 0..n {
            let (x0, x1) = span(x, w);
            let mut sum = 0f32;
            for row in m.alpha.get(y0 * w..y1 * w).unwrap_or_default().chunks_exact(w) {
                sum += row.get(x0..x1).unwrap_or_default().iter().sum::<f32>();
            }
            let count = ((y1 - y0) * (x1 - x0)).max(1) as f32;
            let mut p = (sum / count).clamp(0.0, 1.0);
            if invert {
                p = 1.0 - p;
            }
            // (a tiny margin keeps the logit finite; the grid stores ±15.9 at most anyway)
            let p = p.clamp(1e-4, 1.0 - 1e-4);
            logits.push((p / (1.0 - p)).ln());
        }
    }
    Some(SegMask::from_logits(n, &logits))
}

/// A finished background request.
struct Found {
    photo: PhotoId,
    mask: u32,
    comp: usize,
    result: Result<Option<Matte>, String>,
}

/// The finder and the background requests in flight.
pub struct SubjectState {
    /// `None`: no subject finder here; Subject / Background masks are estimated.
    pub finder: Option<SubjectFinder>,
    tx: Sender<Found>,
    rx: Receiver<Found>,
    running: usize,
}

impl Default for SubjectState {
    fn default() -> Self {
        let (tx, rx) = channel();
        SubjectState { finder: system_finder(), tx, rx, running: 0 }
    }
}

impl SubjectState {
    /// A background request is running.
    pub fn busy(&self) -> bool {
        self.running > 0
    }
}

/// What to tell the user when Vision could not give a subject (the mask keeps the estimate).
pub(crate) fn fallback_note(result: &Result<Option<Matte>, String>) -> Option<String> {
    match result {
        Ok(Some(_)) => None,
        Ok(None) => Some("Apple Vision found no subject in this photo: the mask shows LightCraft's estimate".into()),
        Err(e) => Some(format!("Apple Vision could not select the subject ({e}): the mask shows LightCraft's estimate")),
    }
}

impl Session {
    /// The render Vision gets: the photo as developed, uncropped, without masks.
    fn subject_job(&mut self, id: PhotoId) -> Option<crate::RenderJob> {
        let mut d = (*self.develop_of(id)?).clone();
        d.masks.clear();
        let edge = crate::segment::INPUT_EDGE;
        self.preview_job(id, edge, edge, false, &d)
    }

    /// Find the subject of photo `id` now (waits for the render and Vision).
    pub(crate) fn subject_now(&mut self, id: PhotoId) -> Result<Option<Matte>, String> {
        let finder = self.ai.subject.finder.clone().ok_or("this system has no subject finder")?;
        let job = self.subject_job(id).ok_or("no such photo")?;
        let image = job.run().rendered?.image;
        finder(&image)
    }

    /// Find the subject of photo `id` on a thread, for component `comp` of mask `mask` ([`Session::segment_poll`]
    /// applies it). `false` when it could not be started (too many running, or no thread): the shape keeps the
    /// estimate.
    pub(crate) fn subject_later(&mut self, id: PhotoId, mask: u32, comp: usize) -> bool {
        if self.ai.subject.running >= MAX_RUNNING {
            return false;
        }
        let (Some(finder), Some(job)) = (self.ai.subject.finder.clone(), self.subject_job(id)) else { return false };
        let tx = self.ai.subject.tx.clone();
        let spawned = std::thread::Builder::new().name("lightcraft-subject".into()).spawn(move || {
            let result = crate::guard::catch("Select Subject", || {
                let image = job.run().rendered?.image;
                finder(&image)
            })
            .unwrap_or_else(Err);
            let _ = tx.send(Found { photo: id, mask, comp, result });
        });
        if spawned.is_ok() {
            self.ai.subject.running += 1;
        }
        spawned.is_ok()
    }

    /// Store the subjects found in the background into their shapes (those still waiting for one).
    pub(crate) fn subject_poll(&mut self, polled: &mut Polled) {
        let found: Vec<Found> = self.ai.subject.rx.try_iter().collect();
        for f in found {
            self.ai.subject.running = self.ai.subject.running.saturating_sub(1);
            if let Some(note) = fallback_note(&f.result) {
                polled.messages.push(note);
                continue;
            }
            let Ok(Some(matte)) = f.result else { continue };
            let Some(d) = self.develop_of(f.photo) else { continue };
            let mut d = (*d).clone();
            let Some(c) = d.masks.iter_mut().find(|m| m.id == f.mask).and_then(|m| m.components.get_mut(f.comp)) else { continue };
            // the shape changed or was removed meanwhile: nothing to fill in
            let (seg, invert) = match &mut c.shape {
                MaskShape::Subject { seg: seg @ None } => (seg, false),
                MaskShape::Background { seg: seg @ None } => (seg, true),
                _ => continue,
            };
            *seg = seg_from_matte(&matte, SEG_SIDE, invert);
            match self.set_develop(f.photo, d, "Select Subject") {
                Ok(()) => polled.changed = true,
                Err(e) => polled.messages.push(e.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(w: usize, h: usize) -> Matte {
        let alpha = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                if ((x - w as f32 * 0.5).powi(2) + (y - h as f32 * 0.5).powi(2)).sqrt() < h as f32 * 0.3 { 1.0 } else { 0.0 }
            })
            .collect();
        Matte { width: w, height: h, alpha }
    }

    fn alpha_at(seg: &SegMask, x: f64, y: f64) -> f32 {
        let side = seg.side as usize;
        let l = seg.logits().unwrap();
        let (i, j) = (((x * side as f64) as usize).min(side - 1), ((y * side as f64) as usize).min(side - 1));
        1.0 / (1.0 + (-l[j * side + i]).exp())
    }

    #[test]
    fn a_matte_becomes_a_stretched_grid_of_logits_and_background_is_its_inverse() {
        // a wide picture: the grid is square, so the disc stays at the same normalized place
        let m = disc(300, 200);
        let seg = seg_from_matte(&m, 64, false).unwrap();
        assert_eq!(seg.side, 64);
        assert!(alpha_at(&seg, 0.5, 0.5) > 0.99);
        assert!(alpha_at(&seg, 0.05, 0.05) < 0.01);
        // the disc's edge, normalized: 0.3 × 200 / 300 = 0.2 of the width from the centre
        assert!(alpha_at(&seg, 0.5 + 0.15, 0.5) > 0.9 && alpha_at(&seg, 0.5 + 0.25, 0.5) < 0.1);
        let bg = seg_from_matte(&m, 64, true).unwrap();
        assert!(alpha_at(&bg, 0.5, 0.5) < 0.01 && alpha_at(&bg, 0.05, 0.05) > 0.99);
        // a tiny matte on a larger grid, and nonsense
        assert!(seg_from_matte(&disc(5, 3), 16, false).is_some());
        assert!(seg_from_matte(&Matte { width: 4, height: 4, alpha: vec![0.5; 3] }, 16, false).is_none());
        assert!(seg_from_matte(&Matte { width: 0, height: 0, alpha: vec![] }, 16, false).is_none());
    }
}
