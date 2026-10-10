//! Apple's Vision framework on macOS: what a photo shows, how good it looks, and its subject.
//!
//! The Mac runs these models on its Neural Engine and ships them with the system, so LightCraft has no weights to
//! download or license:
//! - [`analyze`]: labels for what the photo shows (`VNClassifyImageRequest`, Apple's ~1,300 scene and object classes,
//!   kept at its high-precision operating point) and an aesthetics score (`VNCalculateImageAestheticsScoresRequest`,
//!   macOS 15+), for auto keywords and AI culling;
//! - [`subject`]: the photo's main subjects as one matte (`VNGenerateForegroundInstanceMaskRequest`, macOS 14+), for
//!   Select Subject / Background.
//!
//! Every request is optional: [`features`] says what this Mac offers (nothing off macOS, where the calls say so).
//!
//! The Objective-C calls are `unsafe` and live in one module (`macos`), every block with its `SAFETY:` reasoning; the
//! API here is safe. Objective-C exceptions cannot be caught without a C helper, so what Vision is given is checked
//! here first (the image's size against its pixels), classes missing from an older macOS are looked up before use,
//! and results (pixel formats, sizes, buffer lengths) are checked before they are read.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(target_os = "macos")]
mod macos;

/// Largest side of an image handed to Vision (its models look at well under 1,000 pixels; this bounds what a caller
/// can make it allocate).
pub const MAX_EDGE: usize = 8192;
/// Most labels [`analyze`] returns.
pub const MAX_LABELS: usize = 16;

/// An 8-bit sRGB image: `pixels` row by row from the top, RGBA (alpha is ignored).
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a [[u8; 4]],
}

impl Image<'_> {
    /// Vision gets only images whose size matches their pixels.
    fn check(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("the image is empty".into());
        }
        if self.width > MAX_EDGE || self.height > MAX_EDGE {
            return Err(format!("the image is larger than {MAX_EDGE} pixels on a side"));
        }
        if self.width.checked_mul(self.height) != Some(self.pixels.len()) {
            return Err(format!("{} pixels for a {} × {} image", self.pixels.len(), self.width, self.height));
        }
        Ok(())
    }
}

/// What this Mac's Vision framework can do (all `false` off macOS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// Labels for what a photo shows (macOS 10.15+).
    pub labels: bool,
    /// The aesthetics score (macOS 15+).
    pub aesthetics: bool,
    /// The subject matte (macOS 14+).
    pub subject: bool,
}

impl Features {
    /// Anything at all.
    pub fn any(self) -> bool {
        self.labels || self.aesthetics || self.subject
    }
}

/// What this Mac offers.
pub fn features() -> Features {
    #[cfg(target_os = "macos")]
    {
        macos::features()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Features::default()
    }
}

/// Something a photo shows: one of Vision's class identifiers (English, lower case, `_` between words: `dog`,
/// `sky`, `sports_car`) and how sure it is, 0..1.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub name: String,
    pub confidence: f32,
}

/// How good a photo looks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aesthetics {
    /// Vision's overall score, −1 (poor) .. 1 (excellent): composition, exposure, focus, subject…
    pub score: f32,
    /// A utility image (a screenshot, receipt, document…) rather than a memorable photo.
    pub utility: bool,
}

/// What [`analyze`] found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Analysis {
    /// Most confident first, at most [`MAX_LABELS`]; only labels at Vision's high-precision operating point (about 9 in
    /// 10 right), so an unusual photo can have none.
    pub labels: Vec<Label>,
    /// `None` where the Mac has no aesthetics model (before macOS 15).
    pub aesthetics: Option<Aesthetics>,
}

/// Label `image` and score its aesthetics (each where this Mac can). Runs on the calling thread and waits for Vision;
/// several threads may call it at once.
pub fn analyze(image: Image) -> Result<Analysis, String> {
    image.check()?;
    #[cfg(target_os = "macos")]
    {
        let f = features();
        if !f.labels && !f.aesthetics {
            return Err("this Mac's Vision framework can't classify photos".into());
        }
        let mut a = guarded(|| macos::analyze(image, f))?;
        a.labels.retain(|l| !l.name.is_empty() && l.confidence.is_finite());
        for l in &mut a.labels {
            l.confidence = l.confidence.clamp(0.0, 1.0);
        }
        a.labels.sort_by(|x, y| y.confidence.total_cmp(&x.confidence));
        a.labels.truncate(MAX_LABELS);
        a.aesthetics = a.aesthetics.filter(|s| s.score.is_finite()).map(|s| Aesthetics { score: s.score.clamp(-1.0, 1.0), ..s });
        Ok(a)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(NOT_MACOS.into())
    }
}

/// A selection over an image: `alpha` row by row from the top, 0 (outside) .. 1 (inside).
#[derive(Clone, Debug, PartialEq)]
pub struct Matte {
    pub width: usize,
    pub height: usize,
    pub alpha: Vec<f32>,
}

impl Matte {
    /// The share of the image selected, 0..1.
    pub fn coverage(&self) -> f32 {
        if self.alpha.is_empty() { 0.0 } else { self.alpha.iter().sum::<f32>() / self.alpha.len() as f32 }
    }
}

/// The photo's main subjects (every foreground instance Vision finds) as one matte the size of `image`; `None` when it
/// finds none.
pub fn subject(image: Image) -> Result<Option<Matte>, String> {
    image.check()?;
    #[cfg(target_os = "macos")]
    {
        if !features().subject {
            return Err("selecting the subject needs macOS 14 or later".into());
        }
        let Some(mut m) = guarded(|| macos::subject(image))? else { return Ok(None) };
        if m.width == 0 || m.height == 0 || m.width.checked_mul(m.height) != Some(m.alpha.len()) {
            return Err("Vision's subject mask has the wrong size".into());
        }
        for a in &mut m.alpha {
            *a = if a.is_finite() { a.clamp(0.0, 1.0) } else { 0.0 };
        }
        Ok(Some(m))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(NOT_MACOS.into())
    }
}

#[cfg(not(target_os = "macos"))]
const NOT_MACOS: &str = "Apple's Vision framework is only on macOS";

/// Run `f`, turning a panic in it (an unexpected nil from Objective-C, say) into an error.
#[cfg(target_os = "macos")]
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|_| "Apple's Vision framework failed unexpectedly".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sky over a field with a bright disc in front: something to classify and a subject to find.
    fn scene(w: usize, h: usize) -> Vec<[u8; 4]> {
        let (cx, cy, r) = (w as f32 * 0.5, h as f32 * 0.6, h as f32 * 0.22);
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < r {
                    [230, 70, 40, 255]
                } else if y < h as f32 * 0.45 {
                    [90, 150, (200.0 + 50.0 * (1.0 - y / h as f32)) as u8, 255]
                } else {
                    [60, (120.0 + 40.0 * (x / w as f32)) as u8, 50, 255]
                }
            })
            .collect()
    }

    #[test]
    fn images_whose_size_does_not_match_their_pixels_are_refused_everywhere() {
        let px = vec![[0u8; 4]; 12];
        for (w, h) in [(0, 12), (12, 0), (5, 3), (4, 4), (MAX_EDGE + 1, 1)] {
            let img = Image { width: w, height: h, pixels: &px };
            assert!(analyze(img).is_err(), "{w} × {h}");
            assert!(subject(img).is_err(), "{w} × {h}");
        }
        assert!(Image { width: 4, height: 3, pixels: &px }.check().is_ok());
        let m = Matte { width: 2, height: 1, alpha: vec![1.0, 0.0] };
        assert_eq!(m.coverage(), 0.5);
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!features().any());
            let img = Image { width: 4, height: 3, pixels: &px };
            assert!(analyze(img).err().unwrap_or_default().contains("only on macOS"));
            assert!(subject(img).err().unwrap_or_default().contains("only on macOS"));
        }
    }

    /// On a Mac: labels and a score for a synthetic scene, in range and in order (what they are depends on the
    /// models, so only their form is checked). Elsewhere it is skipped.
    #[test]
    fn a_mac_labels_and_scores_a_scene() {
        let f = features();
        eprintln!("Vision here: {f:?}");
        if !f.labels {
            eprintln!("skipped: no Vision classification here");
            return;
        }
        let (w, h) = (640, 427);
        let px = scene(w, h);
        let a = analyze(Image { width: w, height: h, pixels: &px }).unwrap();
        eprintln!("labels: {:?}", a.labels);
        eprintln!("aesthetics: {:?}", a.aesthetics);
        assert!(a.labels.len() <= MAX_LABELS);
        assert!(a.labels.windows(2).all(|p| p[0].confidence >= p[1].confidence));
        assert!(a.labels.iter().all(|l| (0.0..=1.0).contains(&l.confidence) && !l.name.is_empty()));
        assert_eq!(a.aesthetics.is_some(), f.aesthetics);
        if let Some(s) = a.aesthetics {
            assert!((-1.0..=1.0).contains(&s.score));
        }
    }

    /// On a Mac: the subject matte is the image's size, in range, and (when Vision finds the disc) more inside the
    /// disc than outside it.
    #[test]
    fn a_mac_finds_a_subject_matte_the_size_of_the_image() {
        if !features().subject {
            eprintln!("skipped: no Vision subject masks here");
            return;
        }
        let (w, h) = (600, 400);
        let px = scene(w, h);
        let Some(m) = subject(Image { width: w, height: h, pixels: &px }).unwrap() else {
            eprintln!("Vision found no subject in the synthetic scene");
            return;
        };
        assert_eq!((m.width, m.height, m.alpha.len()), (w, h, w * h));
        assert!(m.alpha.iter().all(|a| (0.0..=1.0).contains(a)));
        let at = |x: usize, y: usize| m.alpha[y * w + x];
        eprintln!("coverage {:.3}; disc centre {:.2}, corner {:.2}", m.coverage(), at(w / 2, h * 6 / 10), at(5, 5));
        assert!(at(w / 2, h * 6 / 10) > at(5, 5));
    }

    /// A real photo, with timings:
    /// `LC_VISION_IMAGE=<photo.jpg> cargo test --release -p lightcraft-vision --lib real_photo -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a photo: see the doc comment"]
    fn real_photo() {
        use std::time::Instant;
        let Some(path) = std::env::var_os("LC_VISION_IMAGE") else { return };
        let bytes = std::fs::read(&path).unwrap();
        let t = lightcraft_codecs::decode_thumbnail(&bytes, 1024).unwrap().image;
        let img = Image { width: t.width, height: t.height, pixels: &t.data };
        println!("{:?}: {} × {}; Vision here: {:?}", path, t.width, t.height, features());
        for round in 0..3 {
            let s = Instant::now();
            let a = analyze(img).unwrap();
            println!("analyze #{round}: {:.0} ms", s.elapsed().as_secs_f64() * 1000.0);
            if round == 0 {
                for l in &a.labels {
                    println!("  {:<24} {:.3}", l.name, l.confidence);
                }
                println!("  aesthetics: {:?}", a.aesthetics);
            }
        }
        for round in 0..3 {
            let s = Instant::now();
            let m = subject(img).unwrap();
            println!(
                "subject #{round}: {:.0} ms, {}",
                s.elapsed().as_secs_f64() * 1000.0,
                m.map_or("none found".to_string(), |m| format!("{} × {}, coverage {:.3}", m.width, m.height, m.coverage()))
            );
        }
    }
}
