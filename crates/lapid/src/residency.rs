//! Where file-backed weights live at eval time (llama.cpp's `--mmap` / `--no-mmap`).
//!
//! * [`Residency::Upload`]: every weight of an opened `.lapid` file is copied to the device
//!   once and kept (pinned) for the device's lifetime. Fastest when the model fits.
//! * [`Residency::Mmap`]: weights are used in place from the read-only file mapping
//!   (`Backend::map_host`): nothing is copied or pinned; the OS page cache holds whatever
//!   was read recently and drops it under memory pressure, so a model larger than memory
//!   streams from disk. Backends that cannot read host memory in place upload as before.
//! * [`Residency::Auto`] (default): `Mmap` for a file whose weights (plus the other open
//!   files kept resident) exceed the available memory minus a safety headroom, decided when
//!   the file is opened; `Upload` otherwise, so models that fit run exactly as before.
//!
//! The process-wide mode comes from `LAPID_MMAP=0|1|auto` unless [`set_residency`] is
//! called. Page-cache hints ([`advise`], [`prefetch`]) let a model say which weights it
//! will read next (e.g. the experts the router picked).

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

use lapid_tensor::{Error, HostData, Op, Result, Tensor};

/// Residency policy of file-backed weights; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Residency {
    /// Copy weights to the device once and keep them.
    Upload,
    /// Read weights in place from the file mapping (no copy, nothing pinned).
    Mmap,
    /// `Mmap` for files that do not fit in available memory, else `Upload`.
    #[default]
    Auto,
}

impl Residency {
    pub fn name(self) -> &'static str {
        match self {
            Residency::Upload => "upload",
            Residency::Mmap => "mmap",
            Residency::Auto => "auto",
        }
    }
}

impl std::str::FromStr for Residency {
    type Err = Error;
    /// `0|off|no|false|upload`, `1|on|yes|true|mmap`, `auto` (case-insensitive).
    fn from_str(s: &str) -> Result<Residency> {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "no" | "false" | "upload" => Ok(Residency::Upload),
            "1" | "on" | "yes" | "true" | "mmap" => Ok(Residency::Mmap),
            "" | "auto" => Ok(Residency::Auto),
            other => Err(Error::new("invalid_argument", format!("LAPID_MMAP: expected 0, 1 or auto, got {other:?}"))),
        }
    }
}

/// 0 = not set yet (read `LAPID_MMAP` on first use), else `1 + Residency as u8`.
static MODE: AtomicU8 = AtomicU8::new(0);

fn encode(r: Residency) -> u8 {
    1 + match r {
        Residency::Upload => 0,
        Residency::Mmap => 1,
        Residency::Auto => 2,
    }
}

/// The process-wide policy: [`set_residency`], else `$LAPID_MMAP`, else `Auto`. An invalid
/// `LAPID_MMAP` is reported once and treated as `auto`.
pub fn residency() -> Residency {
    match MODE.load(Ordering::Relaxed) {
        1 => Residency::Upload,
        2 => Residency::Mmap,
        3 => Residency::Auto,
        _ => {
            let r = match std::env::var("LAPID_MMAP") {
                Ok(v) => v.parse().unwrap_or_else(|e: Error| {
                    crate::log::emit(&format!("[lapid] {}; using auto", e.message));
                    Residency::Auto
                }),
                Err(_) => Residency::Auto,
            };
            // A concurrent `set_residency` wins.
            let _ = MODE.compare_exchange(0, encode(r), Ordering::Relaxed, Ordering::Relaxed);
            residency()
        }
    }
}

/// Set the process-wide policy (call before opening models; `Auto` is decided per file
/// at open time, explicit modes apply to every later eval).
pub fn set_residency(r: Residency) {
    MODE.store(encode(r), Ordering::Relaxed);
}

/// Bytes of memory a new resident model may use: `MemAvailable` (which counts reclaimable
/// page cache), capped by the cgroup limit of this process (`memory.max` minus the
/// non-reclaimable part of `memory.current`, over every ancestor). `None` where unknown.
pub fn available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| {
            meminfo.lines().find_map(|l| l.strip_prefix(name)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
        };
        let mut avail = field("MemAvailable:")? * 1024;
        if let Some(c) = cgroup_available() {
            avail = avail.min(c);
        }
        Some(avail)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Smallest `max - (current - file cache)` over the cgroup v2 hierarchy of this process.
#[cfg(target_os = "linux")]
fn cgroup_available() -> Option<u64> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = cg.lines().find_map(|l| l.strip_prefix("0::"))?.trim();
    let mut dir = std::path::Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    let mut best: Option<u64> = None;
    loop {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
        if let (Some(max), Some(cur)) = (read("memory.max"), read("memory.current"))
            && let (Ok(max), Ok(cur)) = (max.trim().parse::<u64>(), cur.trim().parse::<u64>())
        {
            let file = read("memory.stat")
                .and_then(|s| s.lines().find_map(|l| l.strip_prefix("file ")?.trim().parse::<u64>().ok()))
                .unwrap_or(0);
            let free = max.saturating_sub(cur.saturating_sub(file));
            best = Some(best.map_or(free, |b| b.min(free)));
        }
        if !dir.pop() || !dir.starts_with("/sys/fs/cgroup") || dir == std::path::Path::new("/sys/fs/cgroup") {
            break;
        }
    }
    best
}

/// Memory left free for activations, KV cache, the OS and other processes when deciding
/// `Auto`: 4 GiB or a tenth of the available memory, whichever is larger.
fn headroom(avail: u64) -> u64 {
    (4u64 << 30).max(avail / 10)
}

/// `Auto` decision for a newly opened file of `bytes` weights, while `resident` bytes of
/// other open files are kept resident: stream when they would not fit.
pub(crate) fn auto_streams(bytes: u64, resident: u64) -> bool {
    match available_memory() {
        Some(avail) => bytes.saturating_add(resident) > avail.saturating_sub(headroom(avail)),
        None => false,
    }
}

/// How host bytes will be read, for the OS page cache (`madvise`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// Default readahead.
    Normal,
    /// Scattered reads: no readahead (each fault reads only its page).
    Random,
    /// Sequential reads: aggressive readahead.
    Sequential,
    /// Will be read soon: start reading it into the page cache now (asynchronous).
    WillNeed,
    /// Prefer huge pages (`MADV_HUGEPAGE`; Linux only, else no-op): file reads fill 2 MiB
    /// folios mapped by one entry, so mapping / locking costs drop ~500x and GPU address
    /// translation is cheaper. Needs transparent huge pages `enabled` = `madvise` or `always`.
    HugePage,
    /// Read it into memory and map it now, waiting for the I/O (Linux 5.14+
    /// `MADV_POPULATE_READ`; elsewhere `WillNeed`). Later reads, including a GPU's through
    /// the host page tables, then take no page faults.
    Populate,
    /// Done with it for now: unmap its pages from this process (`MADV_DONTNEED`); they stay
    /// in the page cache as clean unmapped pages, which the kernel reclaims cheaply (no
    /// reverse-map walk or TLB shootdown) when more is read under memory pressure. For data
    /// copied elsewhere, e.g. experts copied into a device cache. Only applied to bytes of a
    /// live read-only .lapid file mapping (a no-op elsewhere: it would discard anonymous
    /// memory) and only on Linux.
    Release,
}

/// Page-cache hint for bytes `range` (relative to the tensor's data; `None` = all of it)
/// of a host tensor, typically a weight of an mmap'd `.lapid` file. A no-op for tensors
/// that are not host leaves and where the OS has no `madvise`. Errors only on a range
/// outside the tensor.
pub fn advise(t: &Tensor, range: Option<Range<usize>>, advice: Advice) -> Result<()> {
    match t.op() {
        Op::Host(h) => advise_host(h, range, advice),
        _ => Ok(()),
    }
}

/// Start reading bytes `range` of a host tensor into the page cache (e.g. the experts a
/// router selected, before the graph that reads them runs). See [`advise`].
pub fn prefetch(t: &Tensor, range: Range<usize>) -> Result<()> {
    advise(t, Some(range), Advice::WillNeed)
}

/// Read bytes `range` of a host tensor into memory and map them now, in parallel, waiting
/// for the I/O (`Advice::Populate`); e.g. the routed experts of a stack (see
/// [`set_read_ranges`]) ahead of the eval that reads them.
pub fn populate(t: &Tensor, range: Range<usize>) -> Result<()> {
    populate_all(&[(t, range)])
}

/// [`populate`] several ranges at once (all reads in flight together: prefer this over a
/// loop of `populate` calls, e.g. for all routed experts of a layer).
pub fn populate_all(items: &[(&Tensor, Range<usize>)]) -> Result<()> {
    let mut ranges = Vec::with_capacity(items.len());
    for (t, range) in items {
        let Op::Host(h) = t.op() else { continue };
        let bytes = h.bytes();
        let Some(sub) = bytes.get(range.clone()) else {
            return Err(Error::new("invalid_argument", format!("populate: range {range:?} outside {} bytes", bytes.len())));
        };
        ranges.push(sub);
    }
    populate_many(&ranges);
    Ok(())
}

/// Byte ranges GPU evals lock of a mapped weight, by `HostData::key` (absent: all of it).
static READ_RANGES: Mutex<Vec<((usize, usize), Vec<Range<usize>>)>> = Mutex::new(Vec::new());

/// Restrict which bytes of a mapped weight the next GPU evals read (`None`: all of it, the
/// default). Before each GPU eval the runtime reads in and locks the pages of every mapped
/// weight the graph uses, because GPU page faults on file pages are slow and, on some
/// drivers, return wrong data; for a weight read only partially, such as an MoE expert
/// stack, set the ranges of the experts the graph will read (e.g. after routing) so only
/// those are read from disk. The graph must not read outside the ranges.
pub fn set_read_ranges(t: &Tensor, ranges: Option<Vec<Range<usize>>>) -> Result<()> {
    let Op::Host(h) = t.op() else { return Ok(()) };
    if let Some(r) = ranges.iter().flatten().find(|r| r.start > r.end || r.end > h.len()) {
        return Err(Error::new("invalid_argument", format!("read range {r:?} outside {} bytes", h.len())));
    }
    let mut g = crate::runtime::lock(&READ_RANGES);
    g.retain(|(k, _)| *k != h.key());
    if let Some(r) = ranges {
        g.push((h.key(), r));
    }
    Ok(())
}

/// The bytes of `h` a GPU eval reads (see [`set_read_ranges`]).
pub(crate) fn read_slices(h: &HostData) -> Vec<&[u8]> {
    let bytes = h.bytes();
    let g = crate::runtime::lock(&READ_RANGES);
    match g.iter().find(|(k, _)| *k == h.key()) {
        Some((_, ranges)) => ranges.iter().filter_map(|r| bytes.get(r.clone())).collect(),
        None => vec![bytes],
    }
}

/// Bytes per work item of [`populate_many`] / [`pin_many`].
const POPULATE_CHUNK: usize = 4 << 20;
/// Threads of [`populate_many`] / [`pin_many`]: each waits on one readahead window at a
/// time, so this is the I/O queue depth (GB10 NVMe, cold pages: 8 threads 2.5 GB/s, 16:
/// 4 GB/s, 32: 5 GB/s, 64: 5 GB/s).
const POPULATE_THREADS: usize = 32;

/// Run `f` on every chunk of `ranges`, in parallel when there is enough work.
fn for_chunks(ranges: &[&[u8]], f: impl Fn(&[u8]) + Sync) {
    let items: Vec<&[u8]> = ranges.iter().flat_map(|r| r.chunks(POPULATE_CHUNK)).collect();
    let total: usize = items.iter().map(|c| c.len()).sum();
    let threads = POPULATE_THREADS.min(items.len());
    if threads <= 1 || total <= 2 * POPULATE_CHUNK {
        items.into_iter().for_each(f);
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                while let Some(c) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
                    f(c);
                }
            });
        }
    });
}

/// Populate (`Advice::Populate`) every range, in parallel over chunks.
pub(crate) fn populate_many(ranges: &[&[u8]]) {
    for_chunks(ranges, |c| sys::advise(c, Advice::Populate));
}

/// Pages locked in memory (read in and mapped) until dropped; see [`pin_many`].
pub(crate) struct Pinned {
    ranges: Mutex<Vec<(usize, usize)>>,
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let ranges = std::mem::take(self.ranges.get_mut().unwrap_or_else(|e| e.into_inner()));
        for (lo, len) in ranges {
            sys::unlock(lo, len);
        }
    }
}

static PIN_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Re-map, read in and lock every range (`MADV_DONTNEED` + `mlock`, in parallel) so that a
/// GPU reading them through the host page tables sees fresh translations and never
/// page-faults until the guard is dropped. Both are needed on GB10 (driver 580, kernel
/// 6.17): GPU faults on file pages are slow (0.5 GB/s) and can return wrong data, and
/// translations of pages that reclaim evicted can stay stale on the GPU. Pages that cannot
/// be locked (`RLIMIT_MEMLOCK`) are only populated, with a one-time warning: correct while
/// nothing evicts them before the GPU reads them.
pub(crate) fn pin_many(ranges: &[&[u8]]) -> Pinned {
    let pinned = Pinned { ranges: Mutex::new(Vec::new()) };
    // Never lock more than fits: locked pages cannot be reclaimed (an OOM instead of a
    // slow eval). Beyond that, populate only.
    let total: u64 = ranges.iter().map(|r| r.len() as u64).sum();
    if available_memory().is_some_and(|a| total > a.saturating_sub(headroom(a))) {
        if !PIN_WARNED.swap(true, Ordering::Relaxed) {
            crate::log::emit(&format!(
                "[lapid] one eval reads {:.1} GiB of mapped weights, more than fits in memory: not locking them; \
                 a GPU reading them may page-fault",
                total as f64 / (1u64 << 30) as f64
            ));
        }
        populate_many(ranges);
        return pinned;
    }
    let failed = std::sync::atomic::AtomicUsize::new(0);
    for_chunks(ranges, |c| match {
        // Drop the page table entries first: the kernel then invalidates the GPU's cached
        // translations of these addresses, which reclaim may have left stale (observed on
        // GB10: after page-cache reclaim a GPU read returned other pages' data).
        sys::zap(c);
        sys::lock(c)
    } {
        Some(r) => crate::runtime::lock(&pinned.ranges).push(r),
        None => {
            sys::advise(c, Advice::Populate);
            failed.fetch_add(c.len(), Ordering::Relaxed);
        }
    });
    let failed = failed.into_inner();
    if failed > 0 && !PIN_WARNED.swap(true, Ordering::Relaxed) {
        crate::log::emit(&format!(
            "[lapid] could not lock {:.1} MiB of mapped weights in memory (RLIMIT_MEMLOCK, `ulimit -l`); \
             a GPU reading them may page-fault if they are evicted meanwhile",
            failed as f64 / (1 << 20) as f64
        ));
    }
    pinned
}

/// [`advise`] on raw host data.
pub fn advise_host(h: &HostData, range: Option<Range<usize>>, advice: Advice) -> Result<()> {
    let bytes = h.bytes();
    let r = range.unwrap_or(0..bytes.len());
    let Some(sub) = bytes.get(r.clone()) else {
        return Err(Error::new("invalid_argument", format!("advise: range {r:?} outside {} bytes", bytes.len())));
    };
    if advice == Advice::Release && !crate::file::in_file_mapping(sub) {
        return Ok(());
    }
    sys::advise(sub, advice);
    Ok(())
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
mod sys {
    use std::ffi::{c_int, c_void};

    use super::Advice;

    unsafe extern "C" {
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        fn mlock(addr: *const c_void, len: usize) -> c_int;
        fn munlock(addr: *const c_void, len: usize) -> c_int;
        fn getpagesize() -> c_int;
    }

    /// Page-aligned `(start, len)` covering `bytes`.
    fn pages(bytes: &[u8]) -> (usize, usize) {
        // SAFETY: no arguments.
        let page = (unsafe { getpagesize() }).max(1) as usize;
        let start = bytes.as_ptr() as usize;
        let lo = start / page * page;
        let hi = (start + bytes.len()).div_ceil(page) * page;
        (lo, hi - lo)
    }

    /// Lock the pages of `bytes` in memory (reading them in); `None` if refused.
    pub(super) fn lock(bytes: &[u8]) -> Option<(usize, usize)> {
        if bytes.is_empty() {
            return None;
        }
        let (lo, len) = pages(bytes);
        // SAFETY: the range covers pages of a live allocation; locking does not change memory.
        (unsafe { mlock(lo as *const c_void, len) } == 0).then_some((lo, len))
    }

    pub(super) fn zap(bytes: &[u8]) {
        let (lo, len) = pages(bytes);
        // SAFETY: a read-only file mapping: dropping its page table entries loses nothing.
        unsafe { madvise(lo as *mut c_void, len, 4) };
    }

    /// Undo `lock` (the mapping is still alive: the caller holds the host data).
    pub(super) fn unlock(lo: usize, len: usize) {
        // SAFETY: a range previously locked by `lock`.
        unsafe { munlock(lo as *const c_void, len) };
    }

    /// Best effort: hints never fail the caller (e.g. memory that is not a mapping).
    pub(super) fn advise(bytes: &[u8], advice: Advice) {
        if bytes.is_empty() {
            return;
        }
        // Same values on Linux, macOS and the BSDs (MADV_POPULATE_READ: Linux only).
        let a = match advice {
            Advice::Normal => 0,
            Advice::Random => 1,
            Advice::Sequential => 2,
            Advice::WillNeed => 3,
            Advice::Populate if cfg!(target_os = "linux") => 22,
            Advice::Populate => 3,
            Advice::HugePage if cfg!(target_os = "linux") => 14,
            Advice::HugePage => return,
            // Callers checked it is a read-only shared file mapping (`advise_host`).
            Advice::Release if cfg!(target_os = "linux") => 4,
            Advice::Release => return,
        };
        let (lo, len) = pages(bytes);
        // SAFETY: advisory only; the range covers pages of a live allocation (`bytes`), and
        // none of these advices changes the contents of memory (`Release` unmaps pages of a
        // read-only shared file mapping only, which refault with the same bytes).
        let r = unsafe { madvise(lo as *mut c_void, len, a) };
        if r != 0 && advice == Advice::Populate {
            // Older kernel (EINVAL): at least start the reads.
            // SAFETY: as above.
            unsafe { madvise(lo as *mut c_void, len, 3) };
        }
    }
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
mod sys {
    pub(super) fn advise(_bytes: &[u8], _advice: super::Advice) {}
    pub(super) fn lock(_bytes: &[u8]) -> Option<(usize, usize)> {
        None
    }
    pub(super) fn unlock(_lo: usize, _len: usize) {}
    pub(super) fn zap(_bytes: &[u8]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_budget() {
        assert_eq!("0".parse::<Residency>().unwrap(), Residency::Upload);
        assert_eq!("MMAP".parse::<Residency>().unwrap(), Residency::Mmap);
        assert_eq!("auto".parse::<Residency>().unwrap(), Residency::Auto);
        assert!("maybe".parse::<Residency>().is_err());
        assert_eq!(headroom(1 << 30), 4 << 30);
        assert_eq!(headroom(100 << 30), 10 << 30);
        if let Some(a) = available_memory() {
            assert!(a > 0);
            // A tiny file fits; a file beyond any machine's memory streams.
            assert!(!auto_streams(1 << 20, 0) || a < (5 << 30));
            assert!(auto_streams(u64::MAX / 2, 0));
        }
    }

    #[test]
    fn advise_checks_range() {
        let t = Tensor::from_host(HostData::from_vec(vec![0u8; 8192]), &[2048], lapid_tensor::DType::F32).unwrap();
        prefetch(&t, 0..4096).unwrap();
        advise(&t, None, Advice::Random).unwrap();
        advise(&t, None, Advice::Normal).unwrap();
        assert!(prefetch(&t, 4096..9000).is_err());
    }
}
