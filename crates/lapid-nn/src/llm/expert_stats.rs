//! Opt-in log of [`ExpertCache`] behavior (`LAPID_EXPERT_CACHE_STATS=1`): one line per
//! prompt chunk, one per window of decoded tokens (and the rest when the log is dropped).
//!
//! ```text
//! [lapid] expert cache: prefill 512 tok 41.2 s (12.4 tok/s): hits 3120 misses 12240 (20.3%), 225.0 GB loaded, read stall 30.1 s, write 9.8 s, eval 1.2 s, 0 extra passes, prefetched 9800 (used 8900)
//! [lapid] expert cache: decode 64 tok 9.1 s (7.0 tok/s): hits 14950 misses 410 (97.3%), ...
//! ```
//! `read stall` is time waiting for expert bytes beyond what overlapped with copying, `write`
//! the host side of the copies into the pool, `eval` the device graphs of the forward
//! (attention, routers, expert matmuls) including waiting for their results.

use std::sync::Mutex;
use std::time::Duration;

use super::expert_cache::{CacheStats, ExpertCache};

/// Decoded tokens per decode line.
const WINDOW: usize = 64;

/// One forward's numbers beyond the cache's own counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardTimes {
    /// Wall time of the forward.
    pub wall: Duration,
    /// Device evaluations (and reading their results back).
    pub eval: Duration,
}

#[derive(Default)]
struct Window {
    tokens: usize,
    times: ForwardTimes,
    stats: CacheStats,
}

/// Accumulates and prints expert-cache statistics when enabled (see the module docs).
pub struct StatsLog {
    enabled: bool,
    decode: Mutex<Window>,
}

impl StatsLog {
    /// Enabled by `LAPID_EXPERT_CACHE_STATS` set to anything but `0` / empty.
    pub fn from_env() -> StatsLog {
        let enabled = std::env::var("LAPID_EXPERT_CACHE_STATS").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
        StatsLog { enabled, decode: Mutex::new(Window::default()) }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Record a forward of `tokens` tokens; `before` is `cache.stats()` taken at its start.
    /// Prompt chunks (more than one token) are printed at once, decode steps per window.
    pub fn record(&self, cache: &ExpertCache, tokens: usize, before: CacheStats, times: ForwardTimes) {
        if !self.enabled {
            return;
        }
        let delta = cache.stats().since(&before);
        if tokens > 1 {
            // A prompt ends the decode window of the previous turn.
            self.flush();
            line("prefill", tokens, &times, &delta);
            return;
        }
        let mut w = self.decode.lock().unwrap_or_else(|e| e.into_inner());
        w.tokens += tokens;
        w.times.wall += times.wall;
        w.times.eval += times.eval;
        w.stats = w.stats.plus(&delta);
        if w.tokens >= WINDOW {
            line("decode", w.tokens, &w.times, &w.stats);
            *w = Window::default();
        }
    }

    /// Print the decode tokens not printed yet.
    pub fn flush(&self) {
        let mut w = self.decode.lock().unwrap_or_else(|e| e.into_inner());
        if w.tokens > 0 {
            line("decode", w.tokens, &w.times, &w.stats);
            *w = Window::default();
        }
    }
}

impl Drop for StatsLog {
    fn drop(&mut self) {
        if self.enabled {
            self.flush();
        }
    }
}

fn line(what: &str, tokens: usize, t: &ForwardTimes, s: &CacheStats) {
    let secs = t.wall.as_secs_f64();
    lapid::log_line(&format!(
        "[lapid] expert cache: {what} {tokens} tok {secs:.2} s ({:.2} tok/s): hits {} misses {} ({:.1}%), {:.2} GB loaded, read stall {:.2} s, write {:.2} s, eval {:.2} s, {} extra passes, prefetched {} (used {})",
        if secs > 0.0 { tokens as f64 / secs } else { 0.0 },
        s.hits,
        s.misses,
        100.0 * s.hit_rate(),
        s.loaded_bytes as f64 / 1e9,
        s.read_time.as_secs_f64(),
        s.write_time.as_secs_f64(),
        t.eval.as_secs_f64(),
        s.extra_passes,
        s.prefetched,
        s.prefetch_used,
    ));
}
