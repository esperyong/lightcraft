//! AI analysis: what each photo shows and how good it looks (`ai.analyze`), from Apple's Vision framework on a Mac
//! (`lightcraft-vision`), kept in the catalog (`Photo.ai`) where search (`ai:dog`, `aesthetic:>70`), smart albums and
//! the aesthetic sort use it.
//!
//! Photos are analyzed on worker threads: the session prepares each photo's picture (a raw's embedded preview while it
//! is unedited, else its cached thumbnail render — what the grid shows, edits and crop included), a worker renders it
//! and runs the model, and [`Session::ai_pump`] stores the results (LightCraft's own bookkeeping, not an undo step). In
//! the app ([`AiState::background`]) `ai.analyze` returns at once and the frames pump; elsewhere (CLI, MCP) it waits,
//! for at most its `budgetMs`, and what is left keeps going and is collected by the next `ai.*` command.
//!
//! The model is swappable ([`AiState::analyzer`]), so the queue, the catalog and the commands are tested anywhere; it
//! is `None` (every `ai.*` command disabled) where there is none: off macOS, or on a Mac whose Vision lacks both
//! classification and aesthetics.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use lightcraft_catalog::{AiAnalysis, AiLabel, MediaKind, Op, PhotoId};
use lightcraft_raster::Rgba8;

use crate::Session;
use crate::media::{PreviewLoader, RenderJob};

/// Long edge of the picture the model sees (Vision's own models look at less).
pub const EDGE: usize = 512;
/// Most photos worked on at once.
const MAX_THREADS: usize = 4;
/// Photos handed to the workers beyond the ones they are working on, so a worker that finishes starts its next at once.
const PREFETCH: usize = 2;
/// Most failures kept to report.
const MAX_FAILED: usize = 100;

/// What analyzes one picture.
pub type Analyzer = Arc<dyn Fn(&Rgba8) -> Result<AiAnalysis, String> + Send + Sync>;

/// The analyzer this machine has: Apple's Vision where it can label or score photos.
pub fn system_analyzer() -> Option<Analyzer> {
    let f = lightcraft_vision::features();
    if !f.labels && !f.aesthetics {
        return None;
    }
    Some(Arc::new(|img: &Rgba8| {
        let a = lightcraft_vision::analyze(lightcraft_vision::Image { width: img.width, height: img.height, pixels: &img.data })?;
        Ok(from_vision(a))
    }))
}

/// Vision's answer as the catalog keeps it: the aesthetics score from −1..1 to 0..100.
pub fn from_vision(a: lightcraft_vision::Analysis) -> AiAnalysis {
    AiAnalysis {
        labels: a.labels.into_iter().map(|l| AiLabel { name: l.name, confidence: l.confidence }).collect(),
        aesthetic: a.aesthetics.map(|s| ((s.score + 1.0) * 50.0).clamp(0.0, 100.0)),
        utility: a.aesthetics.is_some_and(|s| s.utility),
        model: "Apple Vision".into(),
    }
}

/// Why `ai.*` is unavailable here.
pub const UNAVAILABLE: &str = "AI analysis needs a Mac with Apple's Vision framework (macOS 10.15 or later; aesthetics scores need macOS 15)";

/// One photo for a worker: its picture (the embedded preview, tried first, else the render job) and the model.
struct Prepared {
    id: PhotoId,
    job: RenderJob,
    preview: Option<(String, PreviewLoader)>,
    analyzer: Analyzer,
}

struct Done {
    id: PhotoId,
    result: Result<AiAnalysis, String>,
}

fn process(p: Prepared) -> Done {
    let Prepared { id, job, preview, analyzer } = p;
    // last-resort guard: a panic in a render or the model fails this photo, not the worker
    let result = crate::guard::catch("AI analysis", || {
        let image = match preview.and_then(|(path, load)| load(&path, EDGE)) {
            Some(img) => img,
            None => job.run().rendered?.image,
        };
        analyzer(&image)
    })
    .unwrap_or_else(Err);
    Done { id, result }
}

/// The worker threads (started on the first photo, ended with the session).
struct Worker {
    jobs: Sender<Prepared>,
    done: Receiver<Done>,
    threads: usize,
}

impl Worker {
    fn start(threads: usize) -> Option<Worker> {
        let (jobs, job_rx) = channel::<Prepared>();
        let (done_tx, done) = channel::<Done>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let mut started = 0;
        for i in 0..threads {
            let (rx, tx) = (job_rx.clone(), done_tx.clone());
            let spawned = std::thread::Builder::new().name(format!("lightcraft-ai-{i}")).spawn(move || {
                loop {
                    // the lock is held only while waiting for a photo; the threads end when the session (the only
                    // sender) is dropped
                    let next = rx.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    let Ok(prepared) = next else { break };
                    if tx.send(process(prepared)).is_err() {
                        break;
                    }
                }
            });
            if spawned.is_ok() {
                started += 1;
            }
        }
        (started > 0).then_some(Worker { jobs, done, threads: started })
    }
}

/// Progress of the analysis since the queue was last empty.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct AiProgress {
    /// Photos analyzed (stored) in this run.
    pub done: usize,
    /// Photos in this run (done, failed, in progress and waiting).
    pub total: usize,
    /// Photos being analyzed now.
    pub running: usize,
    /// Photos waiting.
    pub queued: usize,
    /// Photos that could not be analyzed in this run: (id, why); at most the last 100.
    pub failed: Vec<(u64, String)>,
}

/// The model, the queue and the work in progress.
pub struct AiState {
    /// The model; `None`: no AI analysis here (see [`system_analyzer`]).
    pub analyzer: Option<Analyzer>,
    /// The app: `ai.analyze` returns at once and the frames call `ai.status` (which pumps). Otherwise it waits.
    pub background: bool,
    queue: VecDeque<PhotoId>,
    in_flight: Vec<PhotoId>,
    worker: Option<Worker>,
    progress: AiProgress,
    /// Photos stored since the last command asked (each `ai.analyze` reports its own).
    finished: Vec<PhotoId>,
}

impl Default for AiState {
    fn default() -> Self {
        AiState {
            analyzer: system_analyzer(),
            background: false,
            queue: VecDeque::new(),
            in_flight: Vec::new(),
            worker: None,
            progress: AiProgress::default(),
            finished: Vec::new(),
        }
    }
}

impl AiState {
    /// Anything queued or running.
    pub fn busy(&self) -> bool {
        !self.queue.is_empty() || !self.in_flight.is_empty()
    }

    pub fn progress(&self) -> AiProgress {
        AiProgress { running: self.in_flight.len(), queued: self.queue.len(), ..self.progress.clone() }
    }

    fn threads() -> usize {
        std::thread::available_parallelism().map_or(2, |n| n.get() / 2).clamp(1, MAX_THREADS)
    }
}

/// A photo worth analyzing: a still in the library (not deleted, not a video).
fn analyzable(s: &Session, id: PhotoId) -> bool {
    s.catalog.photo(id).is_some_and(|p| !p.deleted && p.kind != MediaKind::Video)
}

impl Session {
    /// Queue photos for analysis (`force`: also ones analyzed before). Returns how many were queued.
    pub fn ai_enqueue(&mut self, ids: &[PhotoId], force: bool) -> usize {
        if !self.ai.busy() {
            self.ai.progress = AiProgress::default();
        }
        let mut added = 0;
        for &id in ids {
            let fresh = self.catalog.photo(id).is_some_and(|p| p.ai.is_none());
            if analyzable(self, id) && (force || fresh) && !self.ai.queue.contains(&id) && !self.ai.in_flight.contains(&id) {
                self.ai.queue.push_back(id);
                added += 1;
            }
        }
        self.ai.progress.total += added;
        added
    }

    /// Forget the photos still waiting (the ones being analyzed finish and are stored). Returns how many were dropped.
    pub fn ai_cancel(&mut self) -> usize {
        let n = self.ai.queue.len();
        self.ai.queue.clear();
        self.ai.progress.total = self.ai.progress.total.saturating_sub(n);
        n
    }

    /// Store what the workers finished and hand them more photos. Cheap when there is nothing to do; the app calls it
    /// (through `ai.status`, which also saves) every few frames while [`AiState::busy`]. Returns whether anything was
    /// stored.
    pub fn ai_pump(&mut self) -> bool {
        let finished: Vec<Done> = self.ai.worker.as_ref().map(|w| w.done.try_iter().collect()).unwrap_or_default();
        let stored = !finished.is_empty();
        for d in finished {
            self.ai_store(d);
        }
        let Some(analyzer) = self.ai.analyzer.clone() else {
            self.ai.queue.clear();
            return stored;
        };
        if self.ai.worker.is_none() && !self.ai.queue.is_empty() {
            self.ai.worker = Worker::start(AiState::threads());
            if self.ai.worker.is_none() {
                let ids: Vec<PhotoId> = self.ai.queue.drain(..).collect();
                for id in ids {
                    self.ai_failed(id, "could not start the analysis threads".into());
                }
                return stored;
            }
        }
        let room = self.ai.worker.as_ref().map_or(0, |w| w.threads + PREFETCH);
        while self.ai.in_flight.len() < room {
            let Some(id) = self.ai.queue.pop_front() else { break };
            let Some(prepared) = self.ai_prepare(id, analyzer.clone()) else {
                self.ai_failed(id, "the photo is not in the library".into());
                continue;
            };
            if self.ai.worker.as_ref().is_some_and(|w| w.jobs.send(prepared).is_ok()) {
                self.ai.in_flight.push(id);
            } else {
                // the threads are gone: start again on the next pump
                self.ai.worker = None;
                self.ai.queue.push_front(id);
                break;
            }
        }
        stored
    }

    /// Pump until everything queued is done, or `deadline`.
    pub fn ai_wait(&mut self, deadline: Instant) {
        loop {
            self.ai_pump();
            if !self.ai.busy() {
                return;
            }
            let now = Instant::now();
            if now >= deadline {
                return;
            }
            let Some(w) = &self.ai.worker else { return };
            if let Ok(d) = w.done.recv_timeout((deadline - now).min(Duration::from_millis(250))) {
                self.ai_store(d);
            }
        }
    }

    /// The photos stored since the last call.
    pub fn ai_take_finished(&mut self) -> Vec<PhotoId> {
        std::mem::take(&mut self.ai.finished)
    }

    fn ai_prepare(&mut self, id: PhotoId, analyzer: Analyzer) -> Option<Prepared> {
        let photo = self.catalog.photo(id)?.clone();
        let job = self.thumb_job(id, EDGE)?;
        // an unedited raw: the camera's own JPEG is what it looks like, at a fraction of the cost of decoding the raw
        #[cfg(not(target_arch = "wasm32"))]
        let preview = if photo.is_edited() { None } else { self.scan_preview_of(&photo) };
        #[cfg(target_arch = "wasm32")]
        let preview = {
            let _ = photo;
            None
        };
        Some(Prepared { id, job, preview, analyzer })
    }

    fn ai_store(&mut self, d: Done) {
        self.ai.in_flight.retain(|&id| id != d.id);
        match d.result {
            // (a photo removed meanwhile has nothing to store into)
            Ok(a) if analyzable(self, d.id) => match self.apply_system(Op::SetAi { id: d.id, ai: Some(Arc::new(a)) }) {
                Ok(()) => {
                    self.ai.progress.done += 1;
                    self.ai.finished.push(d.id);
                }
                Err(e) => self.ai_failed(d.id, e.to_string()),
            },
            Ok(_) => self.ai_failed(d.id, "the photo was removed".into()),
            Err(e) => self.ai_failed(d.id, e),
        }
    }

    fn ai_failed(&mut self, id: PhotoId, why: String) {
        log::warn!("AI analysis of photo {}: {why}", id.0);
        let failed = &mut self.ai.progress.failed;
        if failed.len() >= MAX_FAILED {
            failed.remove(0);
        }
        failed.push((id.0, why));
    }
}
