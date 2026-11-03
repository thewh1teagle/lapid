//! Device-resident cache of MoE experts for models whose experts do not fit in memory.
//!
//! The routed expert stacks stay in the mmap'd `.lapid` file and are never graph leaves.
//! The device holds one pool per stack kind (`[S, N, K]`, e.g. gate / up / down), shared by
//! every layer: a slot holds one `(layer, expert)` in all kinds. Before the expert matmuls of
//! a layer run, the model reads the router's choice to the host and calls
//! [`ExpertCache::ensure`]: missing experts are read by the host from the mapping (page
//! cache, else disk; all ranges in flight together) and copied into free or evicted slots
//! with [`Device::write`]; the matmuls then run `MatMulId` on the pools with expert ids
//! remapped to slot ids. The device never reads file-backed memory in place.
//!
//! Replacement: decayed access frequency over every `(layer, expert)` (also the ones not
//! resident), ties broken by recency; the slots the current pass needs are pinned. A
//! one-off expert thus replaces the previous one-off instead of a popular one, and the
//! steady state keeps each layer's most used experts.
//!
//! A pass needs at most `S` distinct experts: [`ExpertCache::forward`] splits a larger
//! union (long prefill chunks, tiny pools) into passes, evaluating all but the last.
//!
//! Size: [`budget_bytes`] (memory left after the model's other allocations minus a large
//! headroom, or `LAPID_EXPERT_CACHE_GB` / [`set_budget_gib`]).

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lapid::{Device, Error, Op, Result, Tensor};

use super::moe::{Experts, GateUp};
use crate::mlp::Activation;

const NONE: u32 = u32::MAX;

/// Usage statistics of an [`ExpertCache`] since creation (or [`ExpertCache::reset_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CacheStats {
    /// Distinct experts requested per `ensure` that were resident.
    pub hits: u64,
    /// ... that had to be loaded.
    pub misses: u64,
    /// Bytes copied into the pools.
    pub loaded_bytes: u64,
    /// Time spent reading misses into memory (page cache / disk).
    pub read_time: Duration,
    /// Time spent copying them to the device.
    pub write_time: Duration,
    /// Passes run by [`ExpertCache::forward`] beyond one per call (union larger than the pool).
    pub extra_passes: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 { 0.0 } else { self.hits as f64 / n as f64 }
    }
}

/// Replacement bookkeeping (host only).
struct State {
    /// Slot of key `layer * E + expert`, or `NONE`.
    slot_of: Vec<u32>,
    /// Key held by each slot, or `NONE` (empty).
    owner: Vec<u32>,
    /// Decayed access count per key (in units of the current `inc`).
    score: Vec<f32>,
    /// Weight of one access now; grows by `growth` per `ensure` (= exponential decay).
    inc: f32,
    growth: f32,
    /// Last use (ensure tick) per slot.
    used: Vec<u64>,
    /// Slots pinned for the current pass: `pinned[s] == tick`.
    pinned: Vec<u64>,
    tick: u64,
    stats: CacheStats,
}

impl State {
    fn new(keys: usize, slots: usize, half_life: f64) -> State {
        State {
            slot_of: vec![NONE; keys],
            owner: vec![NONE; slots],
            score: vec![0.0; keys],
            inc: 1.0,
            growth: 2f64.powf(1.0 / half_life.max(1.0)) as f32,
            used: vec![0; slots],
            pinned: vec![0; slots],
            tick: 0,
            stats: CacheStats::default(),
        }
    }

    /// Count one access of every key (decaying older ones).
    fn touch(&mut self, keys: &[u32]) {
        self.tick += 1;
        self.inc *= self.growth;
        if self.inc > 1e30 {
            let s = 1.0 / self.inc;
            self.score.iter_mut().for_each(|v| *v *= s);
            self.inc = 1.0;
        }
        for &k in keys {
            self.score[k as usize] += self.inc;
        }
    }

    /// Slots for `keys` (distinct), assigning `(key, slot)` pairs to load for the missing
    /// ones: empty slots first, then the unpinned slots with the lowest score (oldest first).
    fn assign(&mut self, keys: &[u32]) -> (Vec<u32>, Vec<(u32, u32)>) {
        let tick = self.tick;
        let mut out = vec![NONE; keys.len()];
        let mut missing = Vec::new();
        for (i, &k) in keys.iter().enumerate() {
            let s = self.slot_of[k as usize];
            if s != NONE {
                out[i] = s;
                self.pinned[s as usize] = tick;
                self.used[s as usize] = tick;
            } else {
                missing.push(i);
            }
        }
        self.stats.hits += (keys.len() - missing.len()) as u64;
        self.stats.misses += missing.len() as u64;
        if missing.is_empty() {
            return (out, Vec::new());
        }
        // Victims: (empty first, score, last use, slot) smallest.
        let mut cand: Vec<(bool, f32, u64, u32)> = (0..self.owner.len())
            .filter(|&s| self.pinned[s] != tick)
            .map(|s| {
                let o = self.owner[s];
                let score = if o == NONE { 0.0 } else { self.score[o as usize] };
                (o != NONE, score, self.used[s], s as u32)
            })
            .collect();
        let need = missing.len();
        let cmp = |a: &(bool, f32, u64, u32), b: &(bool, f32, u64, u32)| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2));
        if cand.len() > need {
            cand.select_nth_unstable_by(need - 1, cmp);
            cand.truncate(need);
        }
        let mut loads = Vec::with_capacity(need);
        for (i, c) in missing.into_iter().zip(cand) {
            let (k, s) = (keys[i], c.3);
            let old = self.owner[s as usize];
            if old != NONE {
                self.slot_of[old as usize] = NONE;
            }
            self.owner[s as usize] = k;
            self.slot_of[k as usize] = s;
            self.pinned[s as usize] = tick;
            self.used[s as usize] = tick;
            out[i] = s;
            loads.push((k, s));
        }
        (out, loads)
    }

    /// Forget the slots of `loads` (their copy failed: contents unknown).
    fn drop_loads(&mut self, loads: &[(u32, u32)]) {
        for &(k, s) in loads {
            self.slot_of[k as usize] = NONE;
            self.owner[s as usize] = NONE;
        }
    }
}

/// One stack kind: the device pool and the host stack of every layer.
struct Kind {
    pool: Tensor,
    /// Bytes of one expert.
    bytes: usize,
    /// Per layer: the host stack `[E, N, K]` (`None`: layer without experts).
    layers: Vec<Option<Tensor>>,
}

/// Device pool of MoE experts shared by all layers; see the module docs.
pub struct ExpertCache {
    device: Device,
    kinds: Vec<Kind>,
    experts: usize,
    slots: usize,
    state: Mutex<State>,
}

/// Bytes of `t`'s host storage (a host leaf, e.g. a weight of an mmap'd file).
fn host_bytes(t: &Tensor) -> Result<&[u8]> {
    match t.op() {
        Op::Host(h) => Ok(h.bytes()),
        _ => Err(Error::new("invalid_argument", "expert cache: expert stacks must be host tensors (Loader::host_tensor_checked)")),
    }
}

impl ExpertCache {
    /// A cache of `slots` experts on `device` over `layers[l][kind]` (host stacks
    /// `[E, N_kind, K_kind]` with the same shape and dtype for every layer; `None` for
    /// layers without routed experts). Allocates the pools now.
    pub fn new(device: &Device, layers: Vec<Option<Vec<Tensor>>>, slots: usize) -> Result<ExpertCache> {
        let first = layers.iter().flatten().next().ok_or_else(|| Error::new("invalid_argument", "expert cache: no layer has experts"))?.clone();
        if slots == 0 {
            return Err(Error::new("invalid_argument", "expert cache: zero slots"));
        }
        let experts = first.first().map(|t| t.shape()[0]).unwrap_or(0);
        let mut kinds = Vec::with_capacity(first.len());
        for (k, t) in first.iter().enumerate() {
            let shape = t.shape().to_vec();
            if shape.len() != 3 || shape[0] != experts {
                return Err(Error::new("shape_mismatch", format!("expert cache: stack {k} has shape {shape:?}, expected [{experts}, N, K]")));
            }
            let total = host_bytes(t)?.len();
            if total % experts != 0 {
                return Err(Error::new("format", format!("expert cache: stack {k}: {total} bytes are not {experts} equal experts")));
            }
            let mut per_layer = Vec::with_capacity(layers.len());
            for (l, ks) in layers.iter().enumerate() {
                per_layer.push(match ks {
                    None => None,
                    Some(ks) => {
                        let s = ks.get(k).ok_or_else(|| Error::new("invalid_argument", format!("expert cache: layer {l} lacks stack {k}")))?;
                        if s.shape() != shape.as_slice() || s.dtype() != t.dtype() || host_bytes(s)?.len() != total {
                            return Err(Error::new("shape_mismatch", format!("expert cache: layer {l} stack {k} differs from the first layer's")));
                        }
                        Some(s.clone())
                    }
                });
            }
            let pool = device.alloc(&[slots, shape[1], shape[2]], t.dtype())?;
            kinds.push(Kind { pool, bytes: total / experts, layers: per_layer });
        }
        let keys = layers.len() * experts;
        let n_moe = layers.iter().flatten().count().max(1);
        // Half-life of the access counts: ~64 tokens' worth of lookups.
        let state = Mutex::new(State::new(keys, slots, 64.0 * n_moe as f64));
        Ok(ExpertCache { device: device.clone(), kinds, experts, slots, state })
    }

    /// Experts the pools hold.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Experts per layer.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes of one expert over all kinds (one slot).
    pub fn slot_bytes(&self) -> usize {
        self.kinds.iter().map(|k| k.bytes).sum()
    }

    /// The pool of each kind, `[S, N, K]` (read them with slot ids from [`ensure`](Self::ensure)).
    pub fn pools(&self) -> Vec<Tensor> {
        self.kinds.iter().map(|k| k.pool.clone()).collect()
    }

    pub fn stats(&self) -> CacheStats {
        self.lock().stats
    }

    pub fn reset_stats(&self) {
        self.lock().stats = CacheStats::default();
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn key(&self, layer: usize, expert: u32) -> Result<u32> {
        if expert as usize >= self.experts || self.kinds.first().and_then(|k| k.layers.get(layer)).is_none_or(Option::is_none) {
            return Err(Error::new("invalid_argument", format!("expert cache: no expert {expert} in layer {layer}")));
        }
        Ok((layer * self.experts + expert as usize) as u32)
    }

    /// Byte range of expert `e` in a stack of kind `k`.
    fn range(&self, k: usize, e: usize) -> Range<usize> {
        let b = self.kinds[k].bytes;
        e * b..(e + 1) * b
    }

    /// Source bytes of key `key` in kind `k`.
    fn source(&self, k: usize, key: u32) -> Result<&[u8]> {
        let (l, e) = (key as usize / self.experts, key as usize % self.experts);
        let t = self.kinds[k].layers[l].as_ref().ok_or_else(|| Error::new("internal", "expert cache: layer without experts"))?;
        let r = self.range(k, e);
        host_bytes(t)?.get(r).ok_or_else(|| Error::new("internal", "expert cache: range outside the stack"))
    }

    /// Make `experts` (distinct, at most [`slots`](Self::slots)) of `layer` resident and
    /// return their slots, in order. The slots stay valid until the next `ensure`; graphs
    /// reading them must be evaluated before it.
    pub fn ensure(&self, layer: usize, experts: &[u32]) -> Result<Vec<u32>> {
        if experts.len() > self.slots {
            return Err(Error::new("invalid_argument", format!("expert cache: {} experts in one pass, pool has {} slots", experts.len(), self.slots)));
        }
        let keys = experts.iter().map(|&e| self.key(layer, e)).collect::<Result<Vec<u32>>>()?;
        let mut st = self.lock();
        st.touch(&keys);
        let (slots, loads) = st.assign(&keys);
        if loads.is_empty() {
            return Ok(slots);
        }
        match self.load(&loads) {
            Ok((read, write, bytes)) => {
                st.stats.read_time += read;
                st.stats.write_time += write;
                st.stats.loaded_bytes += bytes;
                Ok(slots)
            }
            Err(e) => {
                st.drop_loads(&loads);
                Err(e)
            }
        }
    }

    /// Read the sources of `loads` into memory (all in flight together), then copy them
    /// into their slots. Returns (read time, write time, bytes).
    fn load(&self, loads: &[(u32, u32)]) -> Result<(Duration, Duration, u64)> {
        let t0 = Instant::now();
        let mut ranges = Vec::with_capacity(loads.len() * self.kinds.len());
        for &(key, _) in loads {
            for k in 0..self.kinds.len() {
                let (l, e) = (key as usize / self.experts, key as usize % self.experts);
                if let Some(t) = &self.kinds[k].layers[l] {
                    ranges.push((t, self.range(k, e)));
                }
            }
        }
        let refs: Vec<(&Tensor, Range<usize>)> = ranges.iter().map(|(t, r)| (*t, r.clone())).collect();
        lapid::populate_all(&refs)?;
        let read = t0.elapsed();
        let t1 = Instant::now();
        let mut bytes = 0u64;
        for &(key, slot) in loads {
            for (k, kind) in self.kinds.iter().enumerate() {
                let src = self.source(k, key)?;
                self.device.write(&kind.pool, slot as usize * kind.bytes, src)?;
                bytes += src.len() as u64;
            }
        }
        Ok((read, t1.elapsed(), bytes))
    }

    /// Start reading `experts` of `layer` that are not resident into the page cache
    /// (asynchronous; e.g. the likely experts of the next layer).
    pub fn prefetch(&self, layer: usize, experts: &[u32]) {
        let st = self.lock();
        for &e in experts {
            let Ok(key) = self.key(layer, e) else { continue };
            if st.slot_of[key as usize] != NONE {
                continue;
            }
            for (k, kind) in self.kinds.iter().enumerate() {
                if let Some(t) = &kind.layers[layer] {
                    let _ = lapid::prefetch(t, self.range(k, e as usize));
                }
            }
        }
    }

    /// Routed SwiGLU-style experts through the cache (kinds `[gate, up, down]`): like
    /// [`Experts::forward`] with host expert `ids` (`n * k`, row-major) for `x [n, D]` and
    /// weights `w [n, k]`. When the distinct experts exceed the pool, all passes but the
    /// last are evaluated here; the result must be evaluated before the next `ensure`.
    /// `x` and `w` should be realized (each pass re-reads them).
    pub fn forward(&self, layer: usize, act: Activation, limit: Option<f32>, act_quant: bool, x: &Tensor, ids: &[i32], w: &Tensor) -> Result<Tensor> {
        let pools = self.pools();
        let [gate, up, down] = pools.as_slice() else {
            return Err(Error::new("invalid_argument", "expert cache: forward needs [gate, up, down] stacks"));
        };
        let ex = Experts { gate_up: GateUp::Split { gate: gate.clone(), up: up.clone() }, down: down.clone(), act, limit, act_quant };
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if ids.len() != n * k {
            return Err(Error::new("shape_mismatch", format!("expert cache: {} ids for weights {:?}", ids.len(), w.shape())));
        }
        let mut uniq: Vec<u32> = Vec::new();
        let mut seen = vec![false; self.experts];
        for &e in ids {
            let e = u32::try_from(e).ok().filter(|&e| (e as usize) < self.experts).ok_or_else(|| Error::new("invalid_argument", format!("expert cache: expert id {e} out of range")))?;
            if !std::mem::replace(&mut seen[e as usize], true) {
                uniq.push(e);
            }
        }
        let groups: Vec<&[u32]> = uniq.chunks(self.slots).collect();
        let mut acc: Option<Tensor> = None;
        let last = groups.len() - 1;
        for (gi, group) in groups.iter().enumerate() {
            let slots = self.ensure(layer, group)?;
            let mut slot_of = vec![NONE; self.experts];
            for (&e, &s) in group.iter().zip(&slots) {
                slot_of[e as usize] = s;
            }
            let fill = slots[0] as i32;
            let sid: Vec<i32> = ids.iter().map(|&e| slot_of[e as usize]).map(|s| if s == NONE { fill } else { s as i32 }).collect();
            let sid = Tensor::from_i32(&sid, &[n, k])?;
            let y = if groups.len() == 1 {
                ex.forward(x, &sid, w)?
            } else {
                let mask: Vec<f32> = ids.iter().map(|&e| if slot_of[e as usize] == NONE { 0.0 } else { 1.0 }).collect();
                ex.forward(x, &sid, &w.mul(&Tensor::from_f32(&mask, &[n, k])?)?)?
            };
            let y = match acc.take() {
                Some(a) => a.add(&y)?,
                None => y,
            };
            if gi < last {
                // The next pass rewrites slots this one reads.
                acc = Some(self.device.eval_owned(vec![y])?.remove(0));
                self.lock().stats.extra_passes += 1;
            } else {
                acc = Some(y);
            }
        }
        acc.ok_or_else(|| Error::new("internal", "expert cache: no pass"))
    }
}

/// Explicit cache size in bytes (`set_budget_gib`); 0 = not set.
static BUDGET_OVERRIDE: AtomicU64 = AtomicU64::new(0);

/// Use `gib` GiB for expert caches (CLI `--expert-cache`; wins over `LAPID_EXPERT_CACHE_GB`).
/// Still capped so that at least [`MIN_FREE`] stays available.
pub fn set_budget_gib(gib: f64) {
    BUDGET_OVERRIDE.store((gib.max(0.0) * (1u64 << 30) as f64) as u64, Ordering::Relaxed);
}

const GIB: u64 = 1 << 30;
/// Memory always left available besides the cache, even with an explicit size.
pub const MIN_FREE: u64 = 8 * GIB;

/// Memory left available by the default size: the larger of 16 GiB and an eighth of the
/// available memory (page cache for the experts that do not fit, the OS, other processes).
pub fn default_headroom(avail: u64) -> u64 {
    (16 * GIB).max(avail / 8)
}

/// Requested size: [`set_budget_gib`], else `LAPID_EXPERT_CACHE_GB` (GiB, fractional ok).
fn requested() -> Option<u64> {
    match BUDGET_OVERRIDE.load(Ordering::Relaxed) {
        0 => std::env::var("LAPID_EXPERT_CACHE_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).map(|g| (g.max(0.0) * GIB as f64) as u64),
        b => Some(b),
    }
}

/// Bytes an expert cache may take now, with `scratch` more bytes still to be allocated
/// by the model (KV cache, activations): the requested size capped at `available -
/// scratch - MIN_FREE`, else `available - scratch - default_headroom`. `available` is
/// [`lapid::residency::available_memory`] (call it after the dense weights are loaded);
/// unknown -> only an explicit size. Returns (bytes, how it was chosen).
pub fn budget_bytes(scratch: u64) -> (u64, String) {
    let avail = lapid::residency::available_memory();
    let gib = |b: u64| b as f64 / GIB as f64;
    match (requested(), avail) {
        (Some(r), Some(a)) => {
            let cap = a.saturating_sub(scratch + MIN_FREE);
            if r > cap {
                (cap, format!("{:.1} GiB requested, capped to {:.1} GiB ({:.1} GiB available, {:.1} GiB scratch, {:.0} GiB kept free)", gib(r), gib(cap), gib(a), gib(scratch), gib(MIN_FREE)))
            } else {
                (r, format!("{:.1} GiB requested ({:.1} GiB available)", gib(r), gib(a)))
            }
        }
        (Some(r), None) => (r, format!("{:.1} GiB requested (available memory unknown)", gib(r))),
        (None, Some(a)) => {
            let h = default_headroom(a);
            let b = a.saturating_sub(scratch + h);
            (b, format!("{:.1} GiB = {:.1} GiB available - {:.1} GiB scratch - {:.1} GiB headroom (LAPID_EXPERT_CACHE_GB / --expert-cache to change)", gib(b), gib(a), gib(scratch), gib(h)))
        }
        (None, None) => (0, "available memory unknown: set LAPID_EXPERT_CACHE_GB".to_string()),
    }
}

static FORCED_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Explicit slot count ([`set_forced_slots`], else `LAPID_EXPERT_CACHE_SLOTS`; for tests and
/// benchmarks): models use the cache on any device, with exactly this many slots.
pub fn forced_slots() -> Option<usize> {
    match FORCED_SLOTS.load(Ordering::Relaxed) {
        0 => std::env::var("LAPID_EXPERT_CACHE_SLOTS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n: &usize| n > 0),
        n => Some(n as usize),
    }
}

/// Force the cache with `slots` slots for models loaded afterwards (`None`: back to
/// `LAPID_EXPERT_CACHE_SLOTS` / the automatic choice).
pub fn set_forced_slots(slots: Option<usize>) {
    FORCED_SLOTS.store(slots.map_or(0, |n| n as u64), Ordering::Relaxed);
}

#[cfg(test)]
#[path = "expert_cache_tests.rs"]
mod tests;
