//! Schedule cache: compiled schedules keyed by the exact *structure* of the graph.
//!
//! The key encodes every node in topological order (op + parameters, dtype, shape,
//! operand positions) and the requested outputs; leaves are encoded by kind, byte length
//! and *alias class* (the ordinal of the first leaf with the same identity), never by
//! identity itself. `lapid_compiler::compile` is a pure function of exactly this
//! information, so two graphs with equal keys compile to the same schedule up to which
//! leaf buffers they read. A cached `Plan` therefore stores the schedule with its leaf
//! buffers replaced by leaf ordinals, and each eval rebinds them to the leaves of the
//! graph at hand (e.g. a new token-id host leaf, a new KV-cache device buffer).
//!
//! The integer parameters of the graph (shapes, slice / write positions, leaf sizes) are
//! kept apart from the structure (`GraphKey::dims`): an exact (structure, dims) match reuses
//! a plan, and within one structure `predict` re-derives the plan for new dims from earlier
//! compiles (a decode step whose KV length grows every token) without recompiling.
//! The key walk costs a few microseconds per hundred nodes.

use std::collections::VecDeque;
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

use lapid_compiler::{BufferSource, Launch, OutputSpec, Schedule};
use lapid_tensor::fxhash::{FxHashMap, FxHasher};
use lapid_tensor::walk::{MARK_MAX, Marker};
use lapid_tensor::{DeviceData, HostData, Op, Tensor};

use super::predict::{Line, Sample};

/// Max exact plans cached over all structures.
const MAX_PLANS: usize = 512;
/// Max total key words kept (bounds memory: 8 bytes per word).
const MAX_KEY_WORDS: usize = 4 << 20;

/// A distinct leaf of the graph, in canonical (first use) order.
pub(crate) enum LeafRef {
    Host(HostData),
    Device(DeviceData),
}

/// Structural key of a graph, its integer parameters, and its distinct leaves.
pub(crate) struct GraphKey {
    /// Structure: ops, structural params, dtypes, ranks, operand positions, alias classes.
    pub words: Vec<u64>,
    /// Shapes, slice / pad / write positions, top-k sizes and leaf byte lengths.
    pub dims: Vec<u64>,
    /// Hash of `words`.
    pub hash: u64,
    /// Hash of `dims`.
    pub dims_hash: u64,
    pub leaves: Vec<LeafRef>,
    pub nodes: usize,
}

/// Hash of a word slice: four independent FxHash lanes (no serial multiply chain).
fn hash_words(w: &[u64]) -> u64 {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    let mut lanes = [1u64, 2, 3, 4];
    let chunks = w.chunks_exact(4);
    let rest = chunks.remainder();
    for c in chunks {
        for (h, &x) in lanes.iter_mut().zip(c) {
            *h = (h.rotate_left(5) ^ x).wrapping_mul(K);
        }
    }
    let mut h = FxHasher::default();
    lanes.iter().chain(rest).for_each(|&x| h.write_u64(x));
    h.write_u64(w.len() as u64);
    h.finish()
}

thread_local! {
    /// Word buffers of dropped keys, reused by the next key (no regrowth per eval).
    static SPARE: std::cell::RefCell<Vec<Vec<u64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_buf() -> Vec<u64> {
    SPARE.try_with(|s| s.borrow_mut().pop()).ok().flatten().unwrap_or_else(|| Vec::with_capacity(16 << 10))
}

impl Drop for GraphKey {
    fn drop(&mut self) {
        for mut b in [std::mem::take(&mut self.words), std::mem::take(&mut self.dims)] {
            b.clear();
            let _ = SPARE.try_with(|s| {
                let mut s = s.borrow_mut();
                if s.len() < 4 && b.capacity() <= 1 << 22 {
                    s.push(b);
                }
            });
        }
    }
}

fn op_code(op: &Op) -> u64 {
    match op {
        Op::Host(_) => 1,
        Op::Device(_) => 2,
        Op::Const(_) => 3,
        Op::Cast(_) => 4,
        Op::Unary(_) => 5,
        Op::Binary(_) => 6,
        Op::Where => 7,
        Op::Reduce { .. } => 8,
        Op::Reshape => 9,
        Op::Permute(_) => 10,
        Op::Expand => 11,
        Op::Slice { .. } => 12,
        Op::Pad { .. } => 13,
        Op::Cat { .. } => 14,
        Op::Contiguous => 15,
        Op::MatMul { .. } => 16,
        Op::Gather => 17,
        Op::Softmax => 18,
        Op::LayerNorm { .. } => 19,
        Op::Attention { .. } => 20,
        Op::GatherAxis { .. } => 21,
        Op::TopK { .. } => 22,
        Op::WriteSlice { .. } => 23,
        Op::MatMulId => 24,
        Op::GatedDeltaNet { .. } => 25,
        Op::RmsNorm { .. } => 26,
        Op::Sinkhorn { .. } => 27,
    }
}

/// Walk positions of visited nodes: node marks (the common case) or a map when another
/// thread is walking at the same time.
trait Positions {
    fn get(&self, t: &Tensor) -> Option<u64>;
    fn set(&mut self, t: &Tensor, v: u64);
}

/// Mark value of a node pushed but not finished yet.
const ON_STACK: u32 = MARK_MAX;

impl Positions for Marker {
    #[inline]
    fn get(&self, t: &Tensor) -> Option<u64> {
        Marker::get(self, t).map(|v| if v == ON_STACK { u64::MAX } else { v as u64 })
    }
    #[inline]
    fn set(&mut self, t: &Tensor, v: u64) {
        Marker::set(self, t, if v == u64::MAX { ON_STACK } else { v as u32 })
    }
}

impl Positions for FxHashMap<u64, u64> {
    #[inline]
    fn get(&self, t: &Tensor) -> Option<u64> {
        FxHashMap::get(self, &t.id()).copied()
    }
    #[inline]
    fn set(&mut self, t: &Tensor, v: u64) {
        self.insert(t.id(), v);
    }
}

impl GraphKey {
    pub(crate) fn new(outputs: &[&Tensor]) -> GraphKey {
        if let Some(mut m) = Marker::try_new() {
            if let Some(k) = GraphKey::walk(outputs, &mut m, (ON_STACK - 1) as u64) {
                return k;
            }
        }
        let mut pos: FxHashMap<u64, u64> = FxHashMap::default();
        pos.reserve(1024);
        GraphKey::walk(outputs, &mut pos, u64::MAX - 1).expect("unbounded walk")
    }

    /// Encode the graph; `None` when it has more than `max_nodes` nodes.
    fn walk(outputs: &[&Tensor], pos: &mut impl Positions, max_nodes: u64) -> Option<GraphKey> {
        // Borrowed post-order DFS (same order as `lapid_tensor::toposort`), encoding each
        // node as it is finished. `pos` doubles as the visited set (u64::MAX = on stack).
        let mut host_alias: FxHashMap<(usize, usize), u64> = FxHashMap::default();
        let mut dev_alias: FxHashMap<usize, u64> = FxHashMap::default();
        // Sized for a model's weights: no rehashing while the leaves are collected.
        host_alias.reserve(512);
        dev_alias.reserve(512);
        let mut leaves = Vec::with_capacity(512);
        let mut w = take_buf();
        let mut dims = take_buf();
        let mut nodes = 0u64;
        // Post-order DFS with (node, next operand) entries: every node is pushed once.
        let mut stack: Vec<(&Tensor, u32)> = Vec::with_capacity(256);
        for out in outputs {
            if pos.get(out).is_some() {
                continue;
            }
            pos.set(out, u64::MAX);
            stack.push((*out, 0));
            while let Some(top) = stack.last_mut() {
                let (t, i) = *top;
                if let Some(s) = t.src().get(i as usize) {
                    top.1 += 1;
                    if pos.get(s).is_none() {
                        pos.set(s, u64::MAX);
                        stack.push((s, 0));
                    }
                    continue;
                }
                stack.pop();
                if nodes >= max_nodes {
                    return None;
                }
                pos.set(t, nodes);
                nodes += 1;
                let op = t.op();
                w.push(op_code(op) | (t.dtype() as u64) << 8 | (t.rank() as u64) << 16 | (t.src().len() as u64) << 32);
                dims.extend(t.shape().iter().map(|&d| d as u64));
                for s in t.src() {
                    // Operands always precede their users in topological order.
                    w.push(pos.get(s).unwrap_or(u64::MAX));
                }
                match op {
                    Op::Host(h) => {
                        let next = leaves.len() as u64;
                        let a = *host_alias.entry(h.key()).or_insert_with(|| {
                            leaves.push(LeafRef::Host(h.clone()));
                            next
                        });
                        dims.push(h.len() as u64);
                        w.push(a);
                    }
                    Op::Device(d) => {
                        let next = leaves.len() as u64;
                        let key = Arc::as_ptr(d) as *const () as usize;
                        let a = *dev_alias.entry(key).or_insert_with(|| {
                            leaves.push(LeafRef::Device(d.clone()));
                            next
                        });
                        dims.push(d.len() as u64);
                        w.push(a);
                    }
                    Op::Const(c) => w.push(c.to_bits()),
                    Op::Cast(dt) => w.push(*dt as u64),
                    Op::Unary(u) => w.push(*u as u64),
                    Op::Binary(b) => w.push(*b as u64),
                    Op::Reduce { op, axis, keepdim } => w.extend([*op as u64, *axis as u64, *keepdim as u64]),
                    Op::Permute(p) => w.extend(p.iter().map(|&a| a as u64)),
                    Op::Slice { axis, start, end } => {
                        w.push(*axis as u64);
                        dims.extend([*start as u64, *end as u64]);
                    }
                    Op::Pad { axis, before, after, value } => {
                        w.extend([*axis as u64, value.to_bits()]);
                        dims.extend([*before as u64, *after as u64]);
                    }
                    Op::Cat { axis } => w.push(*axis as u64),
                    Op::MatMul { b_transposed } => w.push(*b_transposed as u64),
                    Op::LayerNorm { eps } | Op::RmsNorm { eps } => w.push(eps.to_bits() as u64),
                    Op::Attention { scale, causal, has_mask } => {
                        w.extend([scale.to_bits() as u64, *causal as u64, *has_mask as u64])
                    }
                    Op::GatherAxis { axis } => w.push(*axis as u64),
                    Op::TopK { k, axis } => {
                        w.push(*axis as u64);
                        dims.push(*k as u64);
                    }
                    Op::WriteSlice { axis, start } => {
                        w.push(*axis as u64);
                        dims.push(*start as u64);
                    }
                    Op::GatedDeltaNet { scale } => w.push(scale.to_bits() as u64),
                    Op::Sinkhorn { iters, eps } => w.extend([*iters as u64, eps.to_bits() as u64]),
                    Op::Where | Op::Reshape | Op::Expand | Op::Contiguous | Op::Gather | Op::Softmax | Op::MatMulId => {}
                }
            }
        }
        w.push(u64::MAX - outputs.len() as u64);
        w.extend(outputs.iter().map(|t| pos.get(t).unwrap_or(u64::MAX)));
        Some(GraphKey { hash: hash_words(&w), dims_hash: hash_words(&dims), words: w, dims, leaves, nodes: nodes as usize })
    }

    fn leaf_index(&self) -> FxHashMap<(u8, usize, usize), usize> {
        self.leaves
            .iter()
            .enumerate()
            .map(|(i, l)| match l {
                LeafRef::Host(h) => ((0u8, h.key().0, h.key().1), i),
                LeafRef::Device(d) => ((1u8, Arc::as_ptr(d) as *const () as usize, 0), i),
            })
            .collect()
    }
}

/// Where a schedule buffer comes from, with graph leaves abstracted to ordinals.
#[derive(Clone)]
pub(crate) enum PlanSrc {
    Leaf(usize),
    Temp(usize),
    Output,
    /// A leaf the key walk did not see (never cached).
    Fixed(BufferSource),
}

pub(crate) struct PlanBuf {
    pub bytes: usize,
    pub src: PlanSrc,
}

/// A schedule with rebindable leaves.
pub(crate) struct Plan {
    pub launches: Vec<Launch>,
    pub bufs: Vec<PlanBuf>,
    pub slot_bytes: Vec<usize>,
    pub outputs: Vec<OutputSpec>,
    /// True when every leaf buffer maps to a key leaf (safe to cache).
    pub cacheable: bool,
    /// Device copies of file-backed host leaves seen by the last eval, by leaf ordinal
    /// (the host data is kept alive, so a matching key is the same data): weights skip the
    /// upload-cache lookup on later evals. Shared by the plans of one graph structure.
    pub memo: Arc<Mutex<Vec<Option<(HostData, DeviceData)>>>>,
}

impl Plan {
    pub(crate) fn new(s: Schedule, key: &GraphKey) -> Plan {
        let index = key.leaf_index();
        let mut cacheable = true;
        let bufs = s
            .buffers
            .into_iter()
            .map(|b| {
                let found = match &b.source {
                    BufferSource::Host(h) => index.get(&(0, h.key().0, h.key().1)).copied(),
                    BufferSource::Device(d) => index.get(&(1, Arc::as_ptr(d) as *const () as usize, 0)).copied(),
                    _ => None,
                };
                let src = match (b.source, found) {
                    (_, Some(i)) => PlanSrc::Leaf(i),
                    (BufferSource::Temp { slot }, _) => PlanSrc::Temp(slot),
                    (BufferSource::Output, _) => PlanSrc::Output,
                    (other, None) => {
                        cacheable = false;
                        PlanSrc::Fixed(other)
                    }
                };
                PlanBuf { bytes: b.bytes, src }
            })
            .collect();
        Plan { launches: s.launches, bufs, slot_bytes: s.slot_bytes, outputs: s.outputs, cacheable, memo: Arc::default() }
    }

    /// Copy of the schedule (sharing the leaf memo).
    pub(crate) fn clone_shallow(&self) -> Plan {
        Plan {
            launches: self.launches.clone(),
            bufs: self.bufs.iter().map(|b| PlanBuf { bytes: b.bytes, src: b.src.clone() }).collect(),
            slot_bytes: self.slot_bytes.clone(),
            outputs: self.outputs.clone(),
            cacheable: self.cacheable,
            memo: self.memo.clone(),
        }
    }
}

/// Plans of one graph structure.
struct Entry {
    words: Vec<u64>,
    /// Exactly compiled plans by (dims hash, dims) (most recent last).
    exact: VecDeque<(u64, Vec<u64>, Arc<Plan>)>,
    line: Option<Line>,
}

/// How a plan was obtained.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Lookup {
    Exact,
    Predicted,
}

/// Plans by graph structure, bounded by the total number of exact plans and key memory
/// (FIFO eviction of structures; each keeps its last `MAX_EXACT` exact plans).
#[derive(Default)]
pub(crate) struct PlanCache {
    map: FxHashMap<u64, Vec<Entry>>,
    /// Structure hashes in insertion order.
    order: VecDeque<u64>,
    words: usize,
    /// Exact plans held over all structures.
    count: usize,
}

/// Exact plans kept per structure (repeated runs over the same lengths hit them).
const MAX_EXACT: usize = 128;

impl PlanCache {
    /// An exact plan for `key`, else one predicted from earlier compiles of its structure.
    pub(crate) fn get(&mut self, key: &GraphKey) -> Option<(Arc<Plan>, Lookup)> {
        let e = self.map.get_mut(&key.hash)?.iter_mut().find(|e| e.words == key.words)?;
        if let Some((_, _, p)) = e.exact.iter().rev().find(|(h, d, _)| *h == key.dims_hash && *d == key.dims) {
            return Some((p.clone(), Lookup::Exact));
        }
        if predict_disabled() {
            return None;
        }
        let p = e.line.as_mut()?.predict(&key.dims)?;
        Some((p, Lookup::Predicted))
    }

    /// Store a freshly compiled plan (also as a sample for prediction).
    pub(crate) fn insert(&mut self, key: &GraphKey, plan: Arc<Plan>) {
        if !plan.cacheable {
            return;
        }
        let found = self.map.get(&key.hash).is_some_and(|v| v.iter().any(|e| e.words == key.words));
        while (self.count >= MAX_PLANS || (!found && self.words + key.words.len() > MAX_KEY_WORDS)) && self.evict_one() {}
        let found = self.map.get(&key.hash).is_some_and(|v| v.iter().any(|e| e.words == key.words));
        if !found {
            self.words += key.words.len();
            self.map.entry(key.hash).or_default().push(Entry { words: key.words.clone(), exact: VecDeque::new(), line: None });
            self.order.push_back(key.hash);
        }
        let Some(e) = self.map.get_mut(&key.hash).and_then(|v| v.iter_mut().find(|e| e.words == key.words)) else {
            return;
        };
        if e.exact.len() >= MAX_EXACT {
            e.exact.pop_front();
            self.count -= 1;
        }
        self.count += 1;
        e.exact.push_back((key.dims_hash, key.dims.clone(), plan.clone()));
        e.line = Some(Line::record(e.line.take(), Sample::new(key.dims.clone(), plan)));
    }

    /// Drop the oldest structure. Returns false when empty.
    fn evict_one(&mut self) -> bool {
        let Some(h) = self.order.pop_front() else { return false };
        if let Some(v) = self.map.get_mut(&h) {
            if !v.is_empty() {
                let e = v.remove(0);
                self.words -= e.words.len();
                self.count -= e.exact.len();
            }
            if v.is_empty() {
                self.map.remove(&h);
            }
        }
        true
    }
}

/// `LAPID_NO_PREDICT=1`: always compile when the dims change (A/B checks).
fn predict_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("LAPID_NO_PREDICT").is_some_and(|v| v != "0"))
}

#[cfg(test)]
mod tests {
    use super::GraphKey;
    use crate::Device;
    use lapid_tensor::Tensor;

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_f32(v, &[v.len()]).unwrap()
    }

    #[test]
    fn key_ignores_leaf_identity_but_not_aliasing() {
        let (a, b, c) = (t(&[1.0, 2.0]), t(&[3.0, 4.0]), t(&[5.0, 6.0]));
        let k1 = GraphKey::new(&[&a.add(&b).unwrap()]);
        let k2 = GraphKey::new(&[&c.add(&a).unwrap()]);
        assert_eq!(k1.words, k2.words);
        assert_eq!(k1.leaves.len(), 2);
        // x + x reads one buffer: a different schedule.
        let k3 = GraphKey::new(&[&a.add(&a).unwrap()]);
        assert_ne!(k1.words, k3.words);
        // Shapes, ops and constants are part of the key.
        // Shapes are dims (same structure), ops and constants are structure.
        let k4 = GraphKey::new(&[&t(&[1.0, 2.0, 3.0]).add(&t(&[1.0, 2.0, 3.0])).unwrap()]);
        assert_eq!(k1.words, k4.words);
        assert_ne!(k1.dims, k4.dims);
        assert_ne!(k1.words, GraphKey::new(&[&a.mul(&b).unwrap()]).words);
        let s1 = GraphKey::new(&[&a.add_scalar(1.0).unwrap()]);
        let s2 = GraphKey::new(&[&a.add_scalar(2.0).unwrap()]);
        assert_ne!(s1.words, s2.words);
    }

    #[test]
    fn cached_plans_rebind_leaves() {
        let dev = Device::cpu();
        let w = dev.upload(&Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap()).unwrap();
        // The same structure evaluated with different host and device leaves.
        let mut prev: Option<Tensor> = None;
        for step in 0..4 {
            let x = Tensor::from_f32(&[step as f32, 1.0], &[1, 2]).unwrap();
            let mut y = x.matmul_t(&w).unwrap().add_scalar(0.5).unwrap();
            if let Some(p) = &prev {
                y = y.add(p).unwrap();
            }
            let out = dev.eval(&[&y]).unwrap().pop().unwrap();
            let got = dev.to_vec_f32(&out).unwrap();
            let base = |s: f32| [s + 2.0 + 0.5, 3.0 * s + 4.0 + 0.5, 5.0 * s + 6.0 + 0.5];
            let mut want = base(step as f32);
            if let Some(p) = &prev {
                let pv = dev.to_vec_f32(p).unwrap();
                for (w, p) in want.iter_mut().zip(pv) {
                    *w += p;
                }
            }
            assert_eq!(got, want, "step {step}");
            prev = Some(out);
        }
        // Two outputs that are the same node, and a leaf output, through the cache twice.
        for _ in 0..2 {
            let x = t(&[1.0, -1.0]);
            let y = x.mul_scalar(3.0).unwrap();
            let outs = dev.eval(&[&y, &y, &x]).unwrap();
            assert_eq!(dev.to_vec_f32(&outs[0]).unwrap(), [3.0, -3.0]);
            assert_eq!(dev.to_vec_f32(&outs[1]).unwrap(), [3.0, -3.0]);
            assert_eq!(dev.to_vec_f32(&outs[2]).unwrap(), [1.0, -1.0]);
        }
    }
}
