//! Graph → `Schedule` lowering (agent "compiler"). Op semantics: docs/spec/ops.md. Design notes:
//! plans/lapid-v1/notes/compiler.md.
//!
//! Pipeline: `toposort` → every node gets a lazy value (`LExpr`, see `lexpr`) over its
//! shape. Movement ops only rewrite leaf views (`view`); elementwise ops build expression
//! trees (fusion); a value is *materialized* — i.e. kernels are emitted (`emit`) — only
//! when a consumer needs a buffer (matmul operands, composites, multi-user values,
//! outputs). Matmuls with a single user are deferred ("pending") so that the elementwise
//! chain consuming them becomes the matmul epilogue. Finally temps get memory slots by
//! linear-scan liveness (`memory`).

mod composite;
mod dump;
mod emit;
mod index;
mod lexpr;
mod memory;
mod moe;
mod ops;
mod view;

#[cfg(test)]
mod tests;

use lapid_tensor::fxhash::FxHashMap as HashMap;

use lapid_tensor::{DType, DeviceData, Error, HostData, Op, Result, Tensor, UnaryOp};

use crate::CompileOptions;
use crate::ir::{BufId, IntUnaryOp, Launch, MatMulDims, OutputSpec, Schedule, View};

use lexpr::LExpr;

pub(crate) use dump::dump;

/// Fused expressions larger than this are materialized (keeps kernels and specs small).
const MAX_FUSED_OPS: usize = 48;
/// Max distinct inputs of a fused elementwise kernel.
const MAX_FUSED_LOADS: usize = 12;
/// A value read by two fusing consumers (e.g. a normalized q read by both RoPE halves) is
/// recomputed in each instead of materialized when it has at most this many ops.
const CHEAP_RECOMPUTE_OPS: usize = 8;
/// Expressions up to this many ops are fused through a numel-growing `Expand` (recomputed
/// per broadcast element, e.g. `rsqrt(mean + eps)` of a norm) instead of materialized.
const CHEAP_EXPAND_OPS: usize = 4;

pub(crate) fn compile(outputs: &[Tensor], opts: &CompileOptions) -> Result<Schedule> {
    let order = creation_order(lapid_tensor::toposort(outputs));
    let mut lw = Lowerer::new(opts, outputs);
    lw.users.reserve(order.len());
    lw.vals.reserve(order.len());
    for t in &order {
        let heavy = materializes_operands(t.op());
        for s in t.src() {
            *lw.users.entry(s.id()).or_insert(0) += 1;
            if heavy {
                lw.heavy_users.insert(s.id());
            }
        }
    }
    for t in &order {
        lw.lower_node(t)?;
    }
    let outs = lw.finish_outputs(outputs)?;
    Ok(memory::plan(lw.bufs, lw.launches, outs))
}

/// Lower nodes in creation order (ids grow with creation, so it is topological): a side
/// output such as a recurrent state slice is then materialized right after its producer,
/// while the producer's buffer is still hot in cache and before it would otherwise stay
/// alive (pinning a temp slot) until the end of the graph. Every leaf also precedes every
/// op. Falls back to the DFS order if some node has a src with a larger id.
fn creation_order(mut order: Vec<Tensor>) -> Vec<Tensor> {
    if order.iter().all(|t| t.src().iter().all(|s| s.id() < t.id())) {
        order.sort_unstable_by_key(|t| t.id());
    }
    order
}

// ------------------------------------------------------------------ buffers / pending

pub(crate) enum BufKind {
    Host(HostData),
    Device(DeviceData),
    Temp,
    Output,
}

pub(crate) struct Buf {
    pub bytes: usize,
    pub kind: BufKind,
}

/// Everything needed to emit a matmul except output and epilogue.
#[derive(Clone, Debug)]
pub(crate) struct MatMulPlan {
    pub a: BufId,
    pub b: BufId,
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub b_transposed: bool,
    pub dims: MatMulDims,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendState {
    Waiting,
    /// Emitted without epilogue into this temp.
    Flushed(BufId),
    /// Emitted with an epilogue; the raw product exists nowhere.
    Consumed,
}

/// What a pending value computes.
pub(crate) enum PendKind {
    MatMul(MatMulPlan),
    /// `Cat` along `axis` of these (value, shape) pieces; written straight into the
    /// consumer's region when possible (see `emit::try_cat_into`).
    Cat { axis: usize, pieces: Vec<(LExpr, Vec<usize>)> },
}

/// A deferred single-user node: a matmul (its consumer may become the epilogue, see
/// `emit::try_epilogue`) or a concatenation.
pub(crate) struct Pending {
    pub kind: PendKind,
    pub out_shape: Vec<usize>,
    pub out_dtype: DType,
    pub state: PendState,
}

// ------------------------------------------------------------------ errors / helpers

pub(crate) fn internal(msg: impl Into<String>) -> Error {
    Error::new("internal", msg)
}

pub(crate) fn unsupported(msg: impl Into<String>) -> Error {
    Error::new("unsupported", msg)
}

pub(crate) fn shape_mismatch(msg: impl Into<String>) -> Error {
    Error::new("shape_mismatch", msg)
}

pub(crate) fn src_of(t: &Tensor, i: usize) -> Result<&Tensor> {
    t.src()
        .get(i)
        .ok_or_else(|| internal(format!("node #{} ({}) is missing operand {i}", t.id(), op_name(t.op()))))
}

/// Bytes of a dense non-quantized buffer.
pub(crate) fn bytes_of(dtype: DType, shape: &[usize]) -> Result<usize> {
    if dtype.is_quant() {
        return Err(unsupported(format!(
            "cannot materialize a {} tensor (quantized tensors are read-only weights)",
            dtype.name()
        )));
    }
    Ok(view::numel(shape) * dtype.size())
}

/// Ops that read their operands from buffers (a lazy operand is materialized for them), as
/// opposed to fusing consumers (elementwise, movement, reduce prologues, cat / pad pieces).
fn materializes_operands(op: &Op) -> bool {
    matches!(
        op,
        Op::MatMul { .. }
            | Op::MatMulId
            | Op::Softmax
            | Op::LayerNorm { .. }
            | Op::RmsNorm { .. }
            | Op::Attention { .. }
            | Op::GatedDeltaNet { .. }
            | Op::Sinkhorn { .. }
            | Op::Gather
            | Op::GatherAxis { .. }
            | Op::TopK { .. }
            | Op::Contiguous
    )
}

pub(crate) fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Host(_) => "host",
        Op::Device(_) => "device",
        Op::Const(_) => "const",
        Op::Cast(_) => "cast",
        Op::Unary(_) => "unary",
        Op::Binary(_) => "binary",
        Op::Where => "where",
        Op::Reduce { .. } => "reduce",
        Op::Reshape => "reshape",
        Op::Permute(_) => "permute",
        Op::Expand => "expand",
        Op::Slice { .. } => "slice",
        Op::Pad { .. } => "pad",
        Op::Cat { .. } => "cat",
        Op::Contiguous => "contiguous",
        Op::MatMul { .. } => "matmul",
        Op::Gather => "gather",
        Op::Softmax => "softmax",
        Op::LayerNorm { .. } => "layer_norm",
        Op::RmsNorm { .. } => "rms_norm",
        Op::Attention { .. } => "attention",
        Op::GatherAxis { .. } => "gather_axis",
        Op::TopK { .. } => "topk",
        Op::WriteSlice { .. } => "write_slice",
        Op::MatMulId => "matmul_id",
        Op::GatedDeltaNet { .. } => "gated_delta_net",
        Op::Sinkhorn { .. } => "sinkhorn",
    }
}

// ------------------------------------------------------------------ lowerer

pub(crate) struct Lowerer {
    opts: CompileOptions,
    vals: HashMap<u64, LExpr>,
    users: HashMap<u64, usize>,
    /// Nodes with at least one user that reads them from a buffer (`materializes_operands`).
    heavy_users: lapid_tensor::fxhash::FxHashSet<u64>,
    /// node id → index of its first occurrence in `outputs`.
    out_claim: HashMap<u64, usize>,
    out_bufs: Vec<Option<BufId>>,
    bufs: Vec<Buf>,
    launches: Vec<Launch>,
    pendings: Vec<Pending>,
    host_ids: HashMap<(usize, usize), BufId>,
    dev_ids: HashMap<usize, BufId>,
    /// Device buffer → number of distinct leaf nodes reading it (in-place writes need 1).
    dev_nodes: HashMap<BufId, usize>,
}

impl Lowerer {
    fn new(opts: &CompileOptions, outputs: &[Tensor]) -> Lowerer {
        let mut out_claim = HashMap::default();
        for (i, t) in outputs.iter().enumerate() {
            out_claim.entry(t.id()).or_insert(i);
        }
        Lowerer {
            opts: opts.clone(),
            vals: HashMap::default(),
            users: HashMap::default(),
            heavy_users: Default::default(),
            out_claim,
            out_bufs: vec![None; outputs.len()],
            bufs: Vec::new(),
            launches: Vec::new(),
            pendings: Vec::new(),
            host_ids: HashMap::default(),
            dev_ids: HashMap::default(),
            dev_nodes: HashMap::default(),
        }
    }

    fn users_of(&self, t: &Tensor) -> usize {
        self.users.get(&t.id()).copied().unwrap_or(0)
    }

    fn is_output(&self, t: &Tensor) -> bool {
        self.out_claim.contains_key(&t.id())
    }

    fn val(&self, t: &Tensor) -> Result<LExpr> {
        self.vals
            .get(&t.id())
            .cloned()
            .ok_or_else(|| internal(format!("node #{} used before it was lowered", t.id())))
    }

    fn lower_node(&mut self, t: &Tensor) -> Result<()> {
        if self.vals.contains_key(&t.id()) {
            return Ok(());
        }
        let val = match t.op() {
            Op::Host(h) => self.host_leaf(h, t),
            Op::Device(d) => self.device_leaf(d, t),
            Op::Const(c) if t.dtype() == DType::I32 => LExpr::ConstInt(*c as i32),
            Op::Const(c) => LExpr::Const(*c as f32),
            Op::Cast(_) | Op::Unary(_) | Op::Binary(_) | Op::Where => self.lower_elementwise(t)?,
            Op::Reduce { op, axis, .. } => self.lower_reduce(t, *op, *axis)?,
            Op::Reshape | Op::Permute(_) | Op::Expand | Op::Slice { .. } => self.lower_movement(t)?,
            Op::Pad { axis, before, after, value } => self.lower_pad(t, *axis, *before, *after, *value)?,
            Op::Cat { axis } => self.lower_cat(t, *axis)?,
            Op::Contiguous => self.lower_contiguous(t)?,
            Op::MatMul { b_transposed } => self.lower_matmul(t, *b_transposed)?,
            Op::Gather => self.lower_gather(t)?,
            Op::GatherAxis { axis } => self.lower_gather_axis(t, *axis)?,
            Op::TopK { k, axis } => self.lower_topk(t, *k, *axis)?,
            Op::WriteSlice { axis, start } => self.lower_write_slice(t, *axis, *start)?,
            Op::Softmax
            | Op::LayerNorm { .. }
            | Op::RmsNorm { .. }
            | Op::Attention { .. }
            | Op::MatMulId
            | Op::GatedDeltaNet { .. }
            | Op::Sinkhorn { .. } => {
                self.lower_composite(t)?
            }
        };
        let val = self.claim_output(t, val)?;
        self.vals.insert(t.id(), val);
        Ok(())
    }

    // ---------------------------------------------------------------- leaves

    fn host_leaf(&mut self, h: &HostData, t: &Tensor) -> LExpr {
        let key = h.key();
        let id = match self.host_ids.get(&key) {
            Some(&b) => b,
            None => {
                let b = self.new_buf(h.bytes().len(), BufKind::Host(h.clone()));
                self.host_ids.insert(key, b);
                b
            }
        };
        LExpr::buf(id, t.dtype(), View::contiguous(t.shape()))
    }

    fn device_leaf(&mut self, d: &DeviceData, t: &Tensor) -> LExpr {
        let key = std::sync::Arc::as_ptr(d) as *const () as usize;
        let id = match self.dev_ids.get(&key) {
            Some(&b) => b,
            None => {
                let b = self.new_buf(d.len(), BufKind::Device(d.clone()));
                self.dev_ids.insert(key, b);
                b
            }
        };
        *self.dev_nodes.entry(id).or_insert(0) += 1;
        LExpr::buf(id, t.dtype(), View::contiguous(t.shape()))
    }

    // ---------------------------------------------------------------- elementwise

    fn lower_elementwise(&mut self, t: &Tensor) -> Result<LExpr> {
        let arity = match t.op() {
            Op::Cast(_) | Op::Unary(_) => 1,
            Op::Binary(_) => 2,
            _ => 3,
        };
        if t.src().len() != arity {
            return Err(internal(format!("node #{} ({}) has {} operands", t.id(), op_name(t.op()), t.src().len())));
        }
        let mut srcs = Vec::with_capacity(arity);
        for s in t.src() {
            if s.dtype().is_quant() {
                return Err(unsupported(format!(
                    "elementwise {} on quantized ({}) node #{}",
                    op_name(t.op()),
                    s.dtype().name(),
                    s.id()
                )));
            }
            if s.shape() != t.shape() {
                return Err(shape_mismatch(format!(
                    "elementwise node #{}: operand shape {:?} != output shape {:?}",
                    t.id(),
                    s.shape(),
                    t.shape()
                )));
            }
            srcs.push(self.val(s)?);
        }
        let int = t.dtype() == DType::I32;
        let mut it = srcs.into_iter();
        let mut next = || it.next().ok_or_else(|| internal("missing elementwise operand"));
        let e = match t.op() {
            // i32 results are integer words; float results read i32 values numerically.
            Op::Cast(_) if int => lexpr::as_int(next()?),
            Op::Cast(_) => lexpr::as_f32(next()?),
            Op::Unary(op) if int => match op {
                UnaryOp::Neg => lexpr::int_unary(IntUnaryOp::Neg, next()?),
                UnaryOp::Abs => lexpr::int_unary(IntUnaryOp::Abs, next()?),
                _ => lexpr::as_int(lexpr::unary(*op, lexpr::as_f32(next()?))),
            },
            Op::Unary(op) => lexpr::unary(*op, lexpr::as_f32(next()?)),
            Op::Binary(op) => {
                let a = next()?;
                let b = next()?;
                match (int, lexpr::int_op(*op)) {
                    (true, Some(iop)) => lexpr::int_binary(iop, a, b),
                    (true, None) => lexpr::as_int(lexpr::binary(*op, lexpr::as_f32(a), lexpr::as_f32(b))),
                    (false, _) if op.is_int_only() => {
                        return Err(unsupported(format!("{op:?} on {} node #{}", t.dtype().name(), t.id())));
                    }
                    (false, _) => lexpr::binary(*op, lexpr::as_f32(a), lexpr::as_f32(b)),
                }
            }
            Op::Where => {
                let c = next()?;
                let a = next()?;
                let b = next()?;
                if int {
                    lexpr::where_(c, lexpr::as_int(a), lexpr::as_int(b))
                } else {
                    lexpr::where_(c, lexpr::as_f32(a), lexpr::as_f32(b))
                }
            }
            _ => return Err(internal("lower_elementwise on a non-elementwise op")),
        };
        if self.should_materialize(t, &e) {
            return self.materialize_node(t, &e);
        }
        Ok(e)
    }

    /// Fusion policy for a lazy value of node `t`.
    fn should_materialize(&self, t: &Tensor, e: &LExpr) -> bool {
        if matches!(e, LExpr::Leaf(_) | LExpr::Const(_) | LExpr::ConstInt(_)) {
            return false;
        }
        if !self.opts.fuse {
            return true;
        }
        let ops = e.op_count();
        if ops > MAX_FUSED_OPS || e.distinct_leaves() > MAX_FUSED_LOADS {
            return true;
        }
        let u = self.users_of(t);
        if u <= 1 {
            return false;
        }
        // A consumer reads it from a buffer anyway (e.g. a norm of `x + attn` whose sum is also
        // the residual): write it once (often as a matmul epilogue) and let every user read it,
        // instead of recomputing it — and a pending matmul inside it — in the other users.
        if self.heavy_users.contains(&t.id()) {
            return true;
        }
        if ops <= 1 && u <= 2 {
            return false;
        }
        // Two fusing consumers: recompute a cheap expression in both. Not with a pending
        // matmul (it would run twice) or when a consumer materializes it anyway.
        let cheap = u == 2 && ops <= CHEAP_RECOMPUTE_OPS && !e.has_pending() && !self.heavy_users.contains(&t.id());
        !cheap
    }

    // ---------------------------------------------------------------- movement

    fn lower_movement(&mut self, t: &Tensor) -> Result<LExpr> {
        let s = src_of(t, 0)?;
        let e = self.val(s)?;
        let quant = s.dtype().is_quant();
        if quant && !matches!(t.op(), Op::Reshape) {
            return Err(unsupported(format!(
                "{} of quantized node #{} (only same-shape reshape is allowed)",
                op_name(t.op()),
                s.id()
            )));
        }
        let mut out = match t.op() {
            Op::Reshape => {
                if quant {
                    if s.shape() != t.shape() {
                        return Err(unsupported(format!(
                            "reshape of quantized node #{} from {:?} to {:?}",
                            s.id(),
                            s.shape(),
                            t.shape()
                        )));
                    }
                    e
                } else {
                    match e.map_views(&mut |v| view::reshape(v, s.shape(), t.shape())) {
                        Some(r) => r,
                        None => {
                            // Not expressible as strides: materialize row-major, then the
                            // reshape is free. Written straight into the node's buffer.
                            let out = self.alloc_node(t)?;
                            let cv = View::contiguous(s.shape());
                            self.write_dense(&e, s.shape(), s.dtype(), out, &cv)?;
                            LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))
                        }
                    }
                }
            }
            Op::Permute(perm) => e
                .map_views(&mut |v| view::permute(v, perm))
                .ok_or_else(|| shape_mismatch(format!("permute {perm:?} of rank-{} node #{}", s.rank(), s.id())))?,
            Op::Expand => {
                let mut e = e;
                // Don't recompute a non-trivial expression for every broadcast copy.
                if self.opts.fuse && t.numel() > s.numel() && (e.op_count() > CHEAP_EXPAND_OPS || e.has_pending()) {
                    e = self.materialize_tmp(&e, s.shape(), s.dtype())?;
                }
                e.map_views(&mut |v| view::broadcast_to(v, s.shape(), t.shape())).ok_or_else(|| {
                    shape_mismatch(format!("expand {:?} -> {:?} (node #{})", s.shape(), t.shape(), t.id()))
                })?
            }
            Op::Slice { axis, start, .. } => e
                .map_views(&mut |v| view::slice(v, *axis, *start))
                .ok_or_else(|| shape_mismatch(format!("slice axis {axis} of node #{}", s.id())))?,
            _ => return Err(internal("lower_movement on a non-movement op")),
        };
        if self.users_of(t) > 1 && out.op_count() == 0 && out.has_pending() {
            out = self.flush(&out)?;
        }
        if self.should_materialize(t, &out) {
            return self.materialize_node(t, &out);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- buffers / outputs

    fn new_buf(&mut self, bytes: usize, kind: BufKind) -> BufId {
        self.bufs.push(Buf { bytes, kind });
        self.bufs.len() - 1
    }

    fn new_temp(&mut self, dtype: DType, shape: &[usize]) -> Result<BufId> {
        let bytes = bytes_of(dtype, shape)?;
        Ok(self.new_buf(bytes, BufKind::Temp))
    }

    /// Output index whose buffer node `t` may still write directly.
    fn claimable(&self, t: &Tensor) -> Option<usize> {
        self.out_claim.get(&t.id()).copied().filter(|&i| self.out_bufs.get(i).is_some_and(|b| b.is_none()))
    }

    fn new_output(&mut self, i: usize, t: &Tensor) -> Result<BufId> {
        let bytes = bytes_of(t.dtype(), t.shape())?;
        let id = self.new_buf(bytes, BufKind::Output);
        match self.out_bufs.get_mut(i) {
            Some(slot) => *slot = Some(id),
            None => return Err(internal("output index out of range")),
        }
        Ok(id)
    }

    /// Buffer for a kernel producing node `t` row-major: its Output buffer if `t` is a
    /// requested output not yet written, else a fresh temp.
    fn alloc_node(&mut self, t: &Tensor) -> Result<BufId> {
        match self.claimable(t) {
            Some(i) => self.new_output(i, t),
            None => self.new_temp(t.dtype(), t.shape()),
        }
    }

    /// Requested outputs not written by their producing kernel get a copy now.
    fn claim_output(&mut self, t: &Tensor, val: LExpr) -> Result<LExpr> {
        let Some(i) = self.claimable(t) else {
            return Ok(val);
        };
        let out = self.new_output(i, t)?;
        let cv = View::contiguous(t.shape());
        self.write_dense(&val, t.shape(), t.dtype(), out, &cv)?;
        Ok(LExpr::buf(out, t.dtype(), cv))
    }

    fn finish_outputs(&mut self, outputs: &[Tensor]) -> Result<Vec<OutputSpec>> {
        // Duplicated outputs: every occurrence after the first gets its own copy.
        for (i, t) in outputs.iter().enumerate() {
            if self.out_bufs.get(i).is_some_and(|b| b.is_none()) {
                let e = self.val(t)?;
                let out = self.new_output(i, t)?;
                self.write_dense(&e, t.shape(), t.dtype(), out, &View::contiguous(t.shape()))?;
            }
        }
        outputs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let buf = self
                    .out_bufs
                    .get(i)
                    .copied()
                    .flatten()
                    .ok_or_else(|| internal(format!("output {i} was never materialized")))?;
                Ok(OutputSpec { buf, shape: t.shape().to_vec(), dtype: t.dtype() })
            })
            .collect()
    }
}
