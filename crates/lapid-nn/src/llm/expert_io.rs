//! Reading expert bytes into memory for the [`ExpertCache`](super::expert_cache::ExpertCache):
//! a persistent pool of I/O threads that populate (`MADV_POPULATE_READ`) the pages of a batch
//! of byte ranges in parallel, in order, while the caller consumes finished ranges (copies
//! them to the device) as they complete: disk reads overlap the device copies, and no
//! thread is spawned per batch.
//!
//! Two queues: reads a forward waits for ([`IoPool::read_in_order`]) go before background
//! prefetches ([`IoPool::populate_background`]), so speculative reads keep the disk busy
//! without delaying the ones needed now.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use lapid::{Advice, HostData};

/// Bytes per populate request (one unit of work for one thread: one 2 MiB page-cache folio).
const CHUNK: usize = 2 << 20;
/// I/O threads: the request queue depth (GB10 NVMe: 32 reach ~9 GB/s cold with 2 MiB folios).
const THREADS: usize = 32;

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct Queues {
    urgent: VecDeque<Job>,
    background: VecDeque<Job>,
    closed: bool,
}

struct Shared {
    queues: Mutex<Queues>,
    ready: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queues> {
        self.queues.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Next job (urgent first); `None` once closed.
    fn next(&self) -> Option<Job> {
        let mut q = self.lock();
        loop {
            if q.closed {
                return None;
            }
            if let Some(j) = q.urgent.pop_front().or_else(|| q.background.pop_front()) {
                return Some(j);
            }
            q = self.ready.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Persistent I/O threads (exit when the pool is dropped).
pub(crate) struct IoPool {
    shared: Arc<Shared>,
    threads: usize,
}

/// Completion state of one batch: chunks left per range.
struct Batch {
    left: Vec<AtomicUsize>,
    lock: Mutex<()>,
    done: Condvar,
}

/// `range` split into populate requests.
fn chunks(range: &Range<usize>) -> impl Iterator<Item = Range<usize>> + '_ {
    range.clone().step_by(CHUNK).map(move |at| at..(at + CHUNK).min(range.end))
}

impl IoPool {
    pub fn new() -> IoPool {
        let shared = Arc::new(Shared { queues: Mutex::new(Queues::default()), ready: Condvar::new() });
        let mut threads = 0;
        for i in 0..THREADS {
            let s = shared.clone();
            let spawned = std::thread::Builder::new().name(format!("lapid-expert-io-{i}")).spawn(move || {
                while let Some(job) = s.next() {
                    job();
                }
            });
            if spawned.is_ok() {
                threads += 1; // fewer threads: still correct (none: the caller populates inline)
            }
        }
        IoPool { shared, threads }
    }

    fn push(&self, jobs: Vec<Job>, urgent: bool) {
        let mut q = self.shared.lock();
        let dst = if urgent { &mut q.urgent } else { &mut q.background };
        dst.extend(jobs);
        drop(q);
        self.shared.ready.notify_all();
    }

    /// Populate `ranges` (of host data) in parallel and call `each(i)` for every range in
    /// order as soon as its bytes are in memory. Returns the first error of `each`.
    pub fn read_in_order<E>(&self, ranges: &[(HostData, Range<usize>)], mut each: impl FnMut(usize) -> Result<(), E>) -> Result<(), E> {
        let pooled = self.threads > 0;
        if pooled {
            let batch = Arc::new(Batch {
                left: ranges.iter().map(|(_, r)| AtomicUsize::new(chunks(r).count().max(1))).collect(),
                lock: Mutex::new(()),
                done: Condvar::new(),
            });
            let mut jobs: Vec<Job> = Vec::new();
            for (i, (h, r)) in ranges.iter().enumerate() {
                if r.is_empty() {
                    batch.left[i].store(0, Ordering::Release);
                }
                for c in chunks(r) {
                    let (h, b) = (h.clone(), batch.clone());
                    jobs.push(Box::new(move || {
                        let _ = lapid::residency::advise_host(&h, Some(c), Advice::Populate);
                        if b.left[i].fetch_sub(1, Ordering::AcqRel) == 1 {
                            let _g = b.lock.lock();
                            b.done.notify_all();
                        }
                    }));
                }
            }
            self.push(jobs, true);
            for i in 0..ranges.len() {
                let mut g = batch.lock.lock().unwrap_or_else(|e| e.into_inner());
                while batch.left[i].load(Ordering::Acquire) > 0 {
                    g = batch.done.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                drop(g);
                each(i)?;
            }
        } else {
            for (i, (h, r)) in ranges.iter().enumerate() {
                // No pool: read inline (correct, slower).
                let _ = lapid::residency::advise_host(h, Some(r.clone()), Advice::Populate);
                each(i)?;
            }
        }
        Ok(())
    }

    /// Queue populating `ranges` without waiting, behind every read a forward waits for.
    pub fn populate_background(&self, ranges: Vec<(HostData, Range<usize>)>) {
        if self.threads == 0 {
            return;
        }
        let mut jobs: Vec<Job> = Vec::new();
        for (h, r) in ranges {
            for c in chunks(&r) {
                let h = h.clone();
                jobs.push(Box::new(move || {
                    let _ = lapid::residency::advise_host(&h, Some(c), Advice::Populate);
                }));
            }
        }
        self.push(jobs, false);
    }

    /// Release ([`Advice::Release`]) the pages of `ranges` on an I/O thread, ahead of
    /// queued prefetches (it frees memory for them) and off the caller's path.
    pub fn release(&self, ranges: Vec<(HostData, Range<usize>)>) {
        if ranges.is_empty() {
            return;
        }
        let job: Job = Box::new(move || {
            for (h, r) in ranges {
                let _ = lapid::residency::advise_host(&h, Some(r), Advice::Release);
            }
        });
        if self.threads == 0 {
            job();
        } else {
            self.push(vec![job], true);
        }
    }

    /// Drop the queued background reads (e.g. the guesses of a forward that has ended).
    pub fn cancel_background(&self) {
        self.shared.lock().background.clear();
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        // The threads finish their current job and exit.
        self.shared.lock().closed = true;
        self.shared.ready.notify_all();
    }
}
