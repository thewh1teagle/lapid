//! MSL code generation and launch planning. A `Key` fully determines the shader source
//! (spec + structural variant); `plan` turns a `Kernel` into one or more `Stage`s: key,
//! launch params (`constant uint* P` at buffer 0, passed with `setBytes`), dispatch size and
//! buffer bindings. Sizes never enter the source, so one pipeline serves every shape of a
//! spec. Compile-time exceptions (documented on the keys): the GEMV row count and the
//! attention head dim, which keep accumulators and Q/O fragments in registers.
//!
//! Binding convention: 0 = params, then the stage's `binds` in order (buffer 1, 2, ...).
//!
//! This module is plain Rust (no Metal calls) and builds on every platform, so the golden
//! shader tests run anywhere.

mod attention;
mod elementwise;
pub(crate) mod index;
mod matmul;
pub(crate) mod matmul_id;
pub(crate) mod msl;
mod reduce;
mod rows;
pub(crate) mod template;

use std::sync::Arc;

use lapid_compiler::{
    AttentionSpec, EwSpec, GatedDeltaNetSpec, GatherAxisSpec, GatherSpec, Kernel, LayerNormSpec, MatMulSpec, ReduceSpec, SoftmaxSpec,
    TopKSpec, View,
};
use lapid_tensor::{DType, Error, Result};

/// How a view indexes the flat iteration index of an elementwise kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Idx {
    /// Contiguous over the iteration space: `offset + i`.
    Flat,
    /// Broadcast scalar: `offset`.
    Scalar,
    /// General: `offset + sum(coord * stride)`.
    Strided,
}

/// Pipeline cache key: everything that influences the generated code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    /// Elementwise: per-input index mode, output index mode.
    Ew(Arc<EwSpec>, Vec<Idx>, Idx),
    /// Reduce: `true` = one threadgroup per output row, `false` = one thread per output.
    Reduce(Arc<ReduceSpec>, bool),
    /// Tiled simdgroup-matrix GEMM (64x64 tiles).
    Gemm(Arc<MatMulSpec>),
    /// Matrix-vector for m rows (1..=4, compile-time), transposed B.
    Gemv(Arc<MatMulSpec>, u32),
    Gather(Arc<GatherSpec>),
    Softmax(Arc<SoftmaxSpec>),
    LayerNorm(Arc<LayerNormSpec>),
    /// RMS norm (layer_norm template, `RMS` flag).
    RmsNorm(Arc<LayerNormSpec>),
    /// Flash attention on simdgroup matrices; head dim (compile-time, multiple of 8, <= 128).
    AttnTile(Arc<AttentionSpec>, u32),
    /// Attention for few queries split over 128-key chunks; head dim (compile-time) and
    /// `direct` (single split: writes the output, no merge stage).
    AttnVec(Arc<AttentionSpec>, u32, bool),
    /// Merge stage of `AttnVec(.., false)`; output dtype.
    AttnMerge(DType),
    GatherAxis(Arc<GatherAxisSpec>),
    /// Top-k; `true` = the k = 1 (argmax) kernel.
    TopK(Arc<TopKSpec>, bool),
    /// MatMulId per-pair GEMV (MR = 1); the matmul spec derived from the MatMulId spec.
    GemvId(Arc<MatMulSpec>),
    /// MatMulId grouped GEMM over the route stage's expert lists.
    GemmId(Arc<MatMulSpec>),
    /// MatMulId route stage (bucket pairs by expert).
    MoeRoute,
    /// Gated delta rule; dk chunks of 32 per lane (compile-time, 1..=8).
    DeltaNet(Arc<GatedDeltaNetSpec>, u32),
}

impl Key {
    /// Short label (profile bucket).
    pub fn label(&self) -> &'static str {
        match self {
            Key::Ew(..) => "elementwise",
            Key::Reduce(..) => "reduce",
            Key::Gemm(_) => "gemm",
            Key::Gemv(..) => "gemv",
            Key::Gather(_) => "gather",
            Key::Softmax(_) => "softmax",
            Key::LayerNorm(_) => "layer_norm",
            Key::RmsNorm(_) => "rms_norm",
            Key::AttnTile(..) => "attention_tile",
            Key::AttnVec(..) => "attention_vec",
            Key::AttnMerge(_) => "attention_merge",
            Key::GatherAxis(_) => "gather_axis",
            Key::TopK(..) => "topk",
            Key::GemvId(_) => "gemv_id",
            Key::GemmId(_) => "gemm_id",
            Key::MoeRoute => "moe_route",
            Key::DeltaNet(..) => "gated_delta_net",
        }
    }
}

/// What a stage binds at buffer index `1 + position`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bind {
    /// `launch.inputs[i]`
    Input(usize),
    /// `launch.output`
    Output,
    /// Backend scratch memory (shared by the stages of one launch).
    Scratch,
    /// Backend status word (`device uint*`): kernels set it to non-zero on an out-of-range
    /// index; `execute` checks and clears it after the command buffer completes.
    Status,
}

/// One dispatch.
#[derive(Clone, Debug)]
pub(crate) struct Stage {
    pub key: Key,
    pub params: Vec<u32>,
    /// Threadgroups per grid; any zero means "nothing to do".
    pub groups: [u32; 3],
    /// Threads per threadgroup (1-D).
    pub threads: u32,
    pub binds: Vec<Bind>,
}

impl Stage {
    pub fn is_empty(&self) -> bool {
        self.groups.contains(&0)
    }
}

/// A planned launch: stages run in order with a barrier in between.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub stages: Vec<Stage>,
    /// Bytes of scratch memory the launch needs.
    pub scratch: usize,
}

impl Plan {
    fn single(stage: Stage) -> Plan {
        Plan { stages: vec![stage], scratch: 0 }
    }
}

/// Binds `Input(0..n)` followed by `Output`.
pub(crate) fn io_binds(n: usize) -> Vec<Bind> {
    let mut v: Vec<Bind> = (0..n).map(Bind::Input).collect();
    v.push(Bind::Output);
    v
}

pub(crate) fn plan(kernel: &Kernel) -> Result<Plan> {
    // MXFP4 and i-quant weights have no Metal kernels (CPU / CUDA only).
    let weights = match kernel {
        Kernel::MatMul(s, _) => Some(s.b_dtype),
        Kernel::Gather(s, _) => Some(s.table_dtype),
        Kernel::MatMulId(s, _) => Some(s.w_dtype),
        _ => None,
    };
    if let Some(w) = weights.filter(|w| *w == DType::Mxfp4 || w.is_iquant()) {
        return Err(Error::new("unsupported_dtype", format!("metal: {} weights are not supported", w.name())));
    }
    match kernel {
        Kernel::Elementwise(s, d) => elementwise::plan(s, d).map(Plan::single),
        Kernel::Reduce(s, d) => reduce::plan(s, d).map(Plan::single),
        Kernel::MatMul(s, d) => matmul::plan(s, d).map(Plan::single),
        Kernel::Gather(s, d) => rows::plan_gather(s, d).map(Plan::single),
        Kernel::Softmax(s, d) => rows::plan_softmax(s, d).map(Plan::single),
        Kernel::LayerNorm(s, d) => rows::plan_layer_norm(s, d).map(Plan::single),
        Kernel::Attention(s, d) => attention::plan(s, d),
        Kernel::GatherAxis(s, d) => index::plan_gather_axis(s, d).map(Plan::single),
        Kernel::TopK(s, d) => index::plan_topk(s, d).map(Plan::single),
        Kernel::MatMulId(s, d) => matmul_id::plan_matmul_id(s, d),
        Kernel::GatedDeltaNet(s, d) => matmul_id::plan_delta_net(s, d),
        Kernel::RmsNorm(s, d) => rows::plan_rms_norm(s, d).map(Plan::single),
        // opt-in kernel (`CompileOptions::native_sinkhorn`): never lowered for this backend
        Kernel::Sinkhorn(_) => Err(Error::new("unsupported", "metal: no native sinkhorn kernel")),
    }
}

/// Generate the complete Metal source for `key`.
pub(crate) fn source(key: &Key) -> String {
    match key {
        Key::Ew(s, modes, out) => elementwise::source(s, modes, *out),
        Key::Reduce(s, row) => reduce::source(s, *row),
        Key::Gemm(s) => matmul::source_gemm(s, matmul::Mode::Plain),
        Key::Gemv(s, m) => matmul::source_gemv(s, *m, matmul::Mode::Plain),
        Key::GemvId(s) => matmul::source_gemv(s, 1, matmul::Mode::IdGemv),
        Key::GemmId(s) => matmul::source_gemm(s, matmul::Mode::IdGemm),
        Key::MoeRoute => matmul_id::source_route(),
        Key::DeltaNet(s, dkc) => matmul_id::source_delta_net(s, *dkc),
        Key::Gather(s) => rows::source_gather(s),
        Key::Softmax(s) => rows::source_softmax(s),
        Key::LayerNorm(s) => rows::source_layer_norm(s),
        Key::RmsNorm(s) => rows::source_rms_norm(s),
        Key::AttnTile(s, d) => attention::source_tile(s, *d),
        Key::AttnVec(s, d, direct) => attention::source_vec(s, *d, *direct),
        Key::AttnMerge(out) => attention::source_merge(*out),
        Key::GatherAxis(s) => index::source_gather_axis(s),
        Key::TopK(s, k1) => index::source_topk(s, *k1),
    }
}

// ------------------------------------------------------------------ host-side helpers

/// Builder for the u32 params array.
#[derive(Default)]
pub(crate) struct Params(pub Vec<u32>);

impl Params {
    pub fn u(&mut self, v: usize) -> Result<&mut Self> {
        let x = u32::try_from(v).map_err(|_| Error::new("too_large", format!("size {v} exceeds 32-bit indexing")))?;
        self.0.push(x);
        Ok(self)
    }

    /// Signed stride, stored as two's complement (index math wraps mod 2^32).
    pub fn i(&mut self, v: isize) -> Result<&mut Self> {
        if v < i32::MIN as isize || v > u32::MAX as isize {
            return Err(Error::new("too_large", format!("stride {v} exceeds 32-bit indexing")));
        }
        self.0.push(v as i64 as u32);
        Ok(self)
    }

    pub fn bits(&mut self, v: u32) -> &mut Self {
        self.0.push(v);
        self
    }

    /// offset, strides[rank]
    pub fn view(&mut self, v: &View, rank: usize) -> Result<&mut Self> {
        if v.strides.len() != rank {
            return Err(bad(format!("view rank {} != {rank}", v.strides.len())));
        }
        self.u(v.offset)?;
        for &s in &v.strides {
            self.i(s)?;
        }
        Ok(self)
    }
}

/// Threadgroups for a grid-stride loop over `n` items with `threads` per group. ~4M threads
/// fill any Apple GPU; the loop covers the rest.
pub(crate) fn grid_threads(n: usize, threads: u32) -> [u32; 3] {
    if n == 0 {
        return [0, 0, 0];
    }
    let groups = n.div_ceil(threads as usize).min(16384);
    [groups as u32, 1, 1]
}

/// Threadgroups for `items` independent per-threadgroup work items (grid-stride capped).
pub(crate) fn grid_items(items: usize) -> [u32; 3] {
    if items == 0 {
        return [0, 0, 0];
    }
    [items.min(65535) as u32, 1, 1]
}

/// Checked u32 grid dimension.
pub(crate) fn dim(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::new("too_large", format!("grid dimension {v} exceeds 32 bits")))
}

pub(crate) fn check_plain(d: DType, what: &str) -> Result<()> {
    if d.is_quant() {
        return Err(Error::new("unsupported_dtype", format!("metal {what}: quantized dtype {} not allowed here", d.name())));
    }
    Ok(())
}

pub(crate) fn bad(msg: impl Into<String>) -> Error {
    Error::new("bad_launch", msg)
}
