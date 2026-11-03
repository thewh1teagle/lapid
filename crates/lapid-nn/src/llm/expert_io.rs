//! Reading expert bytes into memory for the [`ExpertCache`](super::expert_cache::ExpertCache):
//! a persistent pool of I/O threads that populate (`MADV_POPULATE_READ`) the pages of a batch
//! of byte ranges in parallel, in order, while the caller consumes finished ranges (copies
//! them to the device) as they complete: disk reads overlap the device copies, and no
//! thread is spawned per batch.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

use lapid::{Advice, HostData};

/// Bytes per populate request (one readahead-sized unit of work for one thread).
const CHUNK: usize = 4 << 20;
/// I/O threads: the request queue depth (GB10 NVMe: 32 reach ~9 GB/s cold with 2 MiB folios).
const THREADS: usize = 32;

type Job = Box<dyn FnOnce() + Send>;

/// Persistent I/O threads (exit when the pool is dropped).
pub(crate) struct IoPool {
    tx: Mutex<Option<Sender<Job>>>,
}

/// Completion state of one batch: chunks left per range.
struct Batch {
    left: Vec<AtomicUsize>,
    lock: Mutex<()>,
    done: Condvar,
}

impl IoPool {
    pub fn new() -> IoPool {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..THREADS {
            let rx: Arc<Mutex<Receiver<Job>>> = rx.clone();
            let spawned = std::thread::Builder::new().name(format!("lapid-expert-io-{i}")).spawn(move || {
                loop {
                    let job = match rx.lock() {
                        Ok(r) => r.recv(),
                        Err(_) => return,
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                }
            });
            if spawned.is_err() {
                break; // fewer threads: still correct (the caller populates inline below)
            }
        }
        IoPool { tx: Mutex::new(Some(tx)) }
    }

    /// Populate `ranges` (of host data) in parallel and call `each(i)` for every range in
    /// order as soon as its bytes are in memory. Returns the first error of `each`.
    pub fn read_in_order<E>(&self, ranges: &[(HostData, Range<usize>)], mut each: impl FnMut(usize) -> Result<(), E>) -> Result<(), E> {
        let batch = Arc::new(Batch {
            left: ranges.iter().map(|(_, r)| AtomicUsize::new(r.len().div_ceil(CHUNK).max(1))).collect(),
            lock: Mutex::new(()),
            done: Condvar::new(),
        });
        let sent = {
            let tx = self.tx.lock().unwrap_or_else(|e| e.into_inner());
            match tx.as_ref() {
                Some(tx) => {
                    let mut ok = true;
                    'all: for (i, (h, r)) in ranges.iter().enumerate() {
                        let mut at = r.start;
                        loop {
                            let end = (at + CHUNK).min(r.end);
                            let (h, b) = (h.clone(), batch.clone());
                            let job: Job = Box::new(move || {
                                let _ = lapid::residency::advise_host(&h, Some(at..end), Advice::Populate);
                                if b.left[i].fetch_sub(1, Ordering::AcqRel) == 1 {
                                    let _g = b.lock.lock();
                                    b.done.notify_all();
                                }
                            });
                            if tx.send(job).is_err() {
                                ok = false;
                                break 'all;
                            }
                            at = end;
                            if at >= r.end {
                                break;
                            }
                        }
                    }
                    ok
                }
                None => false,
            }
        };
        for (i, (h, r)) in ranges.iter().enumerate() {
            if sent {
                let mut g = batch.lock.lock().unwrap_or_else(|e| e.into_inner());
                while batch.left[i].load(Ordering::Acquire) > 0 {
                    g = batch.done.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            } else {
                // No pool: read inline (correct, slower).
                let _ = lapid::residency::advise_host(h, Some(r.clone()), Advice::Populate);
            }
            each(i)?;
        }
        Ok(())
    }
}

impl IoPool {
    /// Queue populating `ranges` without waiting (later batches queue behind them).
    pub fn populate_background(&self, ranges: Vec<(HostData, Range<usize>)>) {
        let tx = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        let Some(tx) = tx.as_ref() else { return };
        for (h, r) in ranges {
            let mut at = r.start;
            while at < r.end {
                let end = (at + CHUNK).min(r.end);
                let h = h.clone();
                let _ = tx.send(Box::new(move || {
                    let _ = lapid::residency::advise_host(&h, Some(at..end), Advice::Populate);
                }));
                at = end;
            }
        }
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        // Closing the channel ends the threads.
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}
