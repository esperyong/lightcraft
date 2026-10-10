//! Memory accounting: what each cache holds (`library.memory`, and `ui.inspect` → `memory` with the
//! frontend's own caches added).
//!
//! The numbers are the caches' own bookkeeping (pixels held), not the process's resident size: the
//! allocator keeps freed pages for reuse and the GPU driver maps device buffers, so `ps`/`time -l`
//! report more. A binary built with a heap profiler installs [`set_heap_stats`] (e.g.
//! `lightcraft-cli` with `--features dhat-heap`) to add live/peak heap bytes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use serde::Serialize;

/// The memory budget: one number from which every cache's share is derived.
///
/// - half for the engine's caches (decoded sources and rendered previews), evicted least recently
///   used *across* them ([`crate::media::MediaCache`]);
/// - a quarter for the working memory of decodes in flight ([`work_gate`]);
/// - an eighth for the GPU renderer's pool of recycled buffers (trimmed when the app is idle).
///
/// The rest is headroom for what isn't cached (renders in progress, textures, the UI). Default:
/// a quarter of the machine's RAM (on Linux, of the cgroup's memory limit when that is lower: a
/// container, a systemd scope), at most 1.5 GiB (`LIGHTCRAFT_MEMORY_MB` overrides it).
pub fn default_budget() -> usize {
    if let Some(mb) = std::env::var("LIGHTCRAFT_MEMORY_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|m| *m >= 64) {
        return mb << 20;
    }
    let cap = 3usize << 29; // 1.5 GiB
    match total_ram() {
        Some(ram) => (ram / 4).clamp(256 << 20, cap),
        None => cap,
    }
}

/// Physical memory (bytes), when the platform tells us without native calls; on Linux no more than
/// the memory limit of the process's cgroup ([`cgroup_memory`]).
fn total_ram() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let ram = meminfo_bytes(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemTotal:")?;
        Some(cgroup_memory().map_or(ram, |(limit, _)| ram.min(limit)))
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(target_os = "freebsd")]
    {
        let out = std::process::Command::new("/sbin/sysctl").args(["-n", "hw.physmem"]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "freebsd")))]
    {
        None
    }
}

/// Memory that can be taken now (bytes), when known: on Linux the kernel's estimate of available
/// memory, and no more than is left under the cgroup's limit ([`cgroup_memory`]). `None` elsewhere.
fn available_ram() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let free = meminfo_bytes(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemAvailable:")?;
        Some(cgroup_memory().map_or(free, |(limit, used)| free.min(limit.saturating_sub(used))))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// `field` (`MemTotal:`, `MemAvailable:`…) of `/proc/meminfo`, in bytes.
#[cfg(any(target_os = "linux", test))]
fn meminfo_bytes(meminfo: &str, field: &str) -> Option<usize> {
    let kb: usize = meminfo.lines().find(|l| l.starts_with(field))?.split_whitespace().nth(1)?.parse().ok()?;
    kb.checked_mul(1024)
}

/// The tightest memory limit of the process's cgroup and its ancestors, and the usage of the
/// cgroup that sets it (bytes); `None` without a limit. A process started in a container or a
/// memory-limited systemd scope is killed past that limit however much RAM the machine has.
#[cfg(target_os = "linux")]
fn cgroup_memory() -> Option<(usize, usize)> {
    cgroup_limit(&std::fs::read_to_string("/proc/self/cgroup").ok()?, |p| std::fs::read_to_string(p).ok())
}

/// [`cgroup_memory`] from the text of `/proc/self/cgroup`, reading the cgroup files with `read`:
/// cgroup v2 (`0::/path`: `memory.max`, `memory.current`) or v1 (`N:memory:/path`:
/// `memory.limit_in_bytes`, `memory.usage_in_bytes`, where no limit reads as a huge number).
#[cfg(any(target_os = "linux", test))]
fn cgroup_limit(proc_cgroup: &str, read: impl Fn(&str) -> Option<String>) -> Option<(usize, usize)> {
    let number = |path: &str| read(path).and_then(|s| s.trim().parse::<usize>().ok()); // v2 "max": none
    for line in proc_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next()) else { continue };
        let (root, max_file, used_file) = if id == "0" && controllers.is_empty() {
            ("/sys/fs/cgroup", "memory.max", "memory.current")
        } else if controllers.split(',').any(|c| c == "memory") {
            ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes")
        } else {
            continue;
        };
        let mut tightest: Option<(usize, usize)> = None;
        let mut dir = path.trim_end_matches('/');
        // the cgroup, then each ancestor up to the root (a path has far fewer than 64 levels)
        for _ in 0..64 {
            if let Some(limit) = number(&format!("{root}{dir}/{max_file}"))
                && limit < 1 << 60
                && tightest.is_none_or(|(l, _)| limit < l)
            {
                tightest = Some((limit, number(&format!("{root}{dir}/{used_file}")).unwrap_or(0)));
            }
            if dir.is_empty() {
                break;
            }
            dir = dir.rfind('/').and_then(|i| dir.get(..i)).unwrap_or("");
        }
        if tightest.is_some() {
            return tightest;
        }
    }
    None
}

static BUDGET: AtomicUsize = AtomicUsize::new(0);

/// The current budget (bytes).
pub fn budget() -> usize {
    match BUDGET.load(Ordering::Relaxed) {
        0 => {
            let b = default_budget();
            BUDGET.store(b, Ordering::Relaxed);
            apply(b);
            b
        }
        b => b,
    }
}

/// Change the budget (bytes, at least 64 MiB) for the process: the decode gate and the GPU pool
/// follow at once; sessions apply their cache share with [`crate::Session::set_memory_budget`].
pub fn set_budget(bytes: usize) -> usize {
    let b = bytes.max(64 << 20);
    BUDGET.store(b, Ordering::Relaxed);
    apply(b);
    b
}

fn apply(b: usize) {
    if let Some(g) = GATE.get() {
        g.set_limit(b / 4);
    }
    lightcraft_gpu::set_pool_limit((b / 8) as u64);
}

/// Share of the budget for the engine's caches.
pub fn cache_share(budget: usize) -> usize {
    budget / 2
}

/// The memory a batch export may fill with photos rendered side by side
/// ([`crate::export::batch_jobs`]): half the machine's RAM less the budget (what the caches and
/// other decodes may hold meanwhile), at least the budget; twice the budget when the RAM is unknown;
/// and never more than is available now ([`available_ram`], Linux).
pub fn export_room() -> usize {
    let b = budget();
    let room = match total_ram() {
        Some(ram) => (ram / 2).saturating_sub(b).max(b),
        None => b.saturating_mul(2),
    };
    available_ram().map_or(room, |free| room.min(free))
}

/// A weighted semaphore over bytes of working memory: [`WorkGate::acquire`] waits while other
/// holders already use the limit (one holder may always proceed, whatever its size).
pub struct WorkGate {
    state: Mutex<(usize, usize)>,
    cv: Condvar,
}

/// Bytes held from a [`WorkGate`] until dropped.
pub struct Permit<'a> {
    gate: &'a WorkGate,
    bytes: usize,
}

impl WorkGate {
    pub fn new(limit: usize) -> WorkGate {
        WorkGate { state: Mutex::new((0, limit)), cv: Condvar::new() }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (usize, usize)> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_limit(&self, limit: usize) {
        self.lock().1 = limit;
        self.cv.notify_all();
    }

    /// Bytes held now and the limit.
    pub fn usage(&self) -> (usize, usize) {
        *self.lock()
    }

    /// Hold `bytes`, waiting until they fit (or nothing else is held).
    pub fn acquire(&self, bytes: usize) -> Permit<'_> {
        let mut g = self.lock();
        while g.0 > 0 && g.0 + bytes > g.1 {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.0 += bytes;
        Permit { gate: self, bytes }
    }

    /// Hold `bytes` without waiting (interactive work): counted, so background work waits for it.
    pub fn acquire_urgent(&self, bytes: usize) -> Permit<'_> {
        self.lock().0 += bytes;
        Permit { gate: self, bytes }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut g = self.gate.lock();
        g.0 = g.0.saturating_sub(self.bytes);
        drop(g);
        self.gate.cv.notify_all();
    }
}

thread_local! {
    static BACKGROUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` as background work (grid thumbnails, neighbour prefetch): its decodes wait at the
/// [`work_gate`] while interactive work holds the memory.
pub fn in_background<R>(f: impl FnOnce() -> R) -> R {
    let was = BACKGROUND.with(|b| b.replace(true));
    let r = f();
    BACKGROUND.with(|b| b.set(was));
    r
}

/// Is this thread running background work ([`in_background`])?
pub fn is_background() -> bool {
    BACKGROUND.with(|b| b.get())
}

/// The process-wide gate for decodes and import probes (limit: a quarter of the budget).
pub fn work_gate() -> &'static WorkGate {
    GATE.get_or_init(|| WorkGate::new(budget() / 4))
}

static GATE: OnceLock<WorkGate> = OnceLock::new();

/// Entries and bytes held by one cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub count: usize,
    pub bytes: usize,
}

impl Usage {
    pub fn new(count: usize, bytes: usize) -> Usage {
        Usage { count, bytes }
    }
}

/// Heap bytes as counted by an instrumented allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeapUsage {
    /// Live heap bytes now.
    pub current: u64,
    /// Highest live heap bytes so far.
    pub peak: u64,
}

/// Device buffers of the GPU renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuUsage {
    /// Every device buffer that exists.
    pub allocated: u64,
    /// Of which recycled buffers waiting in the free pool.
    pub pooled: u64,
    /// Of which buffers released since their thread's last submit.
    pub retired: u64,
}

/// What the engine's caches hold.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryReport {
    /// Decoded thumbnail-level sources (≤ 512 px, linear float).
    pub thumb_sources: Usage,
    /// Decoded preview-level sources (≤ 2560 px, linear float) of the current and prefetched photos.
    pub preview_sources: Usage,
    /// The last full-resolution original (exports, 1:1).
    pub full_source: Usage,
    /// Rendered thumbnails and view renders (8-bit) in memory.
    pub rendered: Usage,
    /// Sum of the above.
    pub engine_bytes: usize,
    pub gpu: GpuUsage,
    /// The memory budget (see [`budget`]) and the engine caches' share of it.
    pub budget: usize,
    pub cache_budget: usize,
    /// Working memory of decodes / probes in flight and its limit ([`work_gate`]).
    pub work_bytes: usize,
    pub work_limit: usize,
    /// Live/peak heap when the binary counts allocations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heap: Option<HeapUsage>,
}

static HEAP: OnceLock<fn() -> HeapUsage> = OnceLock::new();
static RELEASE: OnceLock<fn()> = OnceLock::new();

/// Install a hook that hands memory the allocator keeps after frees back to the system (first
/// call wins). The macOS allocator keeps freed large blocks resident (reusable, but counted in
/// the resident size) until the system runs short; the apps install
/// `malloc_zone_pressure_relief` so the resident size follows what is actually held.
pub fn set_release_hook(f: fn()) {
    let _ = RELEASE.set(f);
}

/// Return freed memory to the system now (after large temporaries were dropped: a decode, an
/// import). No-op without a hook.
pub fn release() {
    if let Some(f) = RELEASE.get() {
        f();
    }
}

/// Install the heap counter of an instrumented binary (first call wins).
pub fn set_heap_stats(f: fn() -> HeapUsage) {
    let _ = HEAP.set(f);
}

/// Live/peak heap bytes, when the binary installed a counter.
pub fn heap_stats() -> Option<HeapUsage> {
    HEAP.get().map(|f| f())
}

/// The GPU renderer's device buffers.
pub fn gpu_usage() -> GpuUsage {
    let g = lightcraft_gpu::memory();
    GpuUsage { allocated: g.allocated, pooled: g.pooled, retired: g.retired }
}

impl crate::Session {
    /// What the engine's caches hold now.
    pub fn memory_report(&self) -> MemoryReport {
        let (thumb_sources, preview_sources, full_source) = self.media.usage();
        let (n, b) = self.media.rendered.mem_usage();
        let rendered = Usage::new(n, b);
        MemoryReport {
            thumb_sources,
            preview_sources,
            full_source,
            rendered,
            engine_bytes: thumb_sources.bytes + preview_sources.bytes + full_source.bytes + rendered.bytes,
            gpu: gpu_usage(),
            budget: budget(),
            cache_budget: self.media.budget(),
            work_bytes: work_gate().usage().0,
            work_limit: work_gate().usage().1,
            heap: heap_stats(),
        }
    }

    /// Set the memory budget (bytes; process-wide) and apply this session's cache share.
    pub fn set_memory_budget(&mut self, bytes: usize) -> usize {
        let b = set_budget(bytes);
        self.media.set_budget(cache_share(b));
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::HashMap<String, String> = entries.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |p| map.get(p).cloned()
    }

    /// The OOM that prompted this: a 2.5 GB systemd scope on a 7 GB machine killed an export sized by the machine's RAM.
    #[test]
    fn the_tightest_cgroup_limit_counts() {
        let cg = "0::/user.slice/app.slice/run-1.scope\n";
        let read = files(&[
            ("/sys/fs/cgroup/user.slice/app.slice/run-1.scope/memory.max", "2621440000\n"),
            ("/sys/fs/cgroup/user.slice/app.slice/run-1.scope/memory.current", "730112000\n"),
            ("/sys/fs/cgroup/user.slice/app.slice/memory.max", "max\n"),
            ("/sys/fs/cgroup/user.slice/memory.max", "8000000000\n"),
        ]);
        assert_eq!(cgroup_limit(cg, &read), Some((2_621_440_000, 730_112_000)));
        // a tighter ancestor wins, with its own usage
        let read = files(&[
            ("/sys/fs/cgroup/a/b/memory.max", "4000000000"),
            ("/sys/fs/cgroup/a/memory.max", "1000000000"),
            ("/sys/fs/cgroup/a/memory.current", "200"),
        ]);
        assert_eq!(cgroup_limit("0::/a/b", &read), Some((1_000_000_000, 200)));
        // the root cgroup itself
        let read = files(&[("/sys/fs/cgroup/memory.max", "3000")]);
        assert_eq!(cgroup_limit("0::/", &read), Some((3000, 0)));
    }

    #[test]
    fn no_limit_means_none() {
        let read = files(&[("/sys/fs/cgroup/user.slice/memory.max", "max")]);
        assert_eq!(cgroup_limit("0::/user.slice", &read), None);
        assert_eq!(cgroup_limit("", &read), None);
        assert_eq!(cgroup_limit("garbage without colons\n12:cpu:/x", &read), None);
        // cgroup v1 reports "no limit" as a huge number, and a real one as bytes
        let v1 = "12:cpu,cpuacct:/docker/1\n9:memory:/docker/1\n";
        let unlimited = files(&[("/sys/fs/cgroup/memory/docker/1/memory.limit_in_bytes", "9223372036854771712")]);
        assert_eq!(cgroup_limit(v1, &unlimited), None);
        let limited = files(&[
            ("/sys/fs/cgroup/memory/docker/1/memory.limit_in_bytes", "536870912"),
            ("/sys/fs/cgroup/memory/docker/1/memory.usage_in_bytes", "1024"),
        ]);
        assert_eq!(cgroup_limit(v1, &limited), Some((536_870_912, 1024)));
    }

    #[test]
    fn meminfo_fields_in_bytes() {
        let m = "MemTotal:        7444620 kB\nMemFree:         2177948 kB\nMemAvailable:    5352676 kB\n";
        assert_eq!(meminfo_bytes(m, "MemTotal:"), Some(7_444_620 * 1024));
        assert_eq!(meminfo_bytes(m, "MemAvailable:"), Some(5_352_676 * 1024));
        assert_eq!(meminfo_bytes(m, "SwapTotal:"), None);
        assert_eq!(meminfo_bytes("MemTotal: lots kB", "MemTotal:"), None);
        assert_eq!(meminfo_bytes("MemTotal: 99999999999999999999 kB", "MemTotal:"), None);
    }
}
