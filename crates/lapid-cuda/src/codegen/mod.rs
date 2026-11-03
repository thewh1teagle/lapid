//! PTX code generation. A `Key` fully determines a kernel's source (spec + structural
//! variant); `plan` turns a `Kernel` into a key, the launch params and the launch shape.
//! Sizes never enter the source (except documented perf-critical constants such as the
//! attention head dim), so one compiled kernel serves every shape of a spec.
//!
//! Kernel ABI: `.entry lapid_main(.param .u64 A_<buf>..., .param .align 4 .b8 P[512])` —
//! one pointer per bound buffer (launch inputs in order, the output, then the scratch
//! buffer when `Key::uses_scratch`), followed by the params block: `PARAM_WORDS` u32 words
//! (sizes, strides, offsets; layout documented per kernel) passed by value.

mod attention;
mod elementwise;
mod expr;
mod flash;
mod gdn;
mod gemv;
mod index;
mod kquant;
mod mxfp4;
mod matmul;
mod mma;
mod moe;
mod ptx;
mod reduce;
mod rows;
mod sinkhorn;
mod template;

use std::sync::Arc;

use lapid_compiler::{
    AttentionSpec, EwSpec, GatedDeltaNetSpec, GatherAxisSpec, GatherSpec, Kernel, LayerNormSpec,
    MatMulSpec, ReduceSpec, SoftmaxSpec, TopKSpec, View,
};
use lapid_tensor::{DType, Error, Result};

use crate::context::Caps;

pub(crate) use elementwise::IdxMode;

/// u32 words in every kernel's params block.
pub(crate) const PARAM_WORDS: usize = 128;

/// Kernel cache key: everything that influences the generated code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    /// Elementwise: per-input index mode, output index mode.
    Ew(Arc<EwSpec>, Vec<IdxMode>, IdxMode),
    /// Reduce: `true` = one block per output row, `false` = one thread per output.
    Reduce(Arc<ReduceSpec>, bool),
    /// Scalar tiled GEMM (64x64 tiles, exact f32 math): any dtypes / devices.
    Gemm(Arc<MatMulSpec>),
    /// Tensor-core GEMM (`mma.sync` f16 x f16 + f32); (A f32 readable as float4,
    /// f16 transposed B readable as 16-byte vectors).
    GemmMma(Arc<MatMulSpec>, bool, bool),
    /// Matrix-vector (small m, transposed B), warp per column; (A f32 readable as float4,
    /// B f16/f32 readable as 16-byte vectors, warps per column (k split, 1/2/4/8)).
    Gemv(Arc<MatMulSpec>, bool, bool, u32),
    Gather(Arc<GatherSpec>),
    Softmax(Arc<SoftmaxSpec>),
    /// Layer norm; `true` = RMS norm (inputs [x, w]); last flag: 1024-thread blocks (few
    /// long rows).
    LayerNorm(Arc<LayerNormSpec>, bool, bool),
    /// Few queries, split over key chunks; `true` = one chunk, writes the output directly
    /// (no merge stage); head dim baked in (<= 256).
    /// Last field: queries per block (they share the K / V loads).
    AttnSplit(Arc<AttentionSpec>, bool, u32, u32),
    /// Merge stage of `AttnSplit(_, false, ..)`.
    AttnMerge(Arc<AttentionSpec>),
    /// Flash attention on tensor cores (many queries); head dim baked in (multiple of 16,
    /// <= 128); `true` = K/V rows readable as 16-byte vectors.
    /// Last flag: precise variant (hi/lo f16 splits, ~f32 accuracy; f32 / bf16 inputs).
    AttnMma(Arc<AttentionSpec>, u32, bool, bool),
    GatherAxis(Arc<GatherAxisSpec>),
    /// `true` = k == 1 (argmax reduction), `false` = radix-select top-k.
    TopK(Arc<TopKSpec>, bool),
    /// Merge stage of the split argmax (`TopK(_, true)` over several blocks per row).
    TopKMerge,
    /// Top-k (k > 1) of short rows: rank per column by one scan of the row in shared memory.
    TopKSmall(Arc<TopKSpec>),
    /// MatMulId as GEMV rows (few tokens): `Gemv` with per-row expert pointers.
    GemvId(Arc<MatMulSpec>, bool, bool, u32),
    /// MatMulId routing stage (rows grouped by expert into scratch).
    MoeRoute,
    /// MatMulId grouped tensor-core GEMM over the routed rows (after `MoeRoute`);
    /// last flag: 32-row tiles (few rows per expert).
    GemmMmaId(Arc<MatMulSpec>, bool, bool, bool),
    /// Gated delta rule; Dk baked in.
    Gdn(Arc<GatedDeltaNetSpec>, u32),
    /// Sinkhorn normalization of n x n matrices; n baked in.
    Sinkhorn(u32),
}

impl Key {
    /// Number of launch inputs bound by this kernel.
    pub fn num_inputs(&self) -> usize {
        match self {
            Key::Ew(s, ..) => s.in_dtypes.len(),
            Key::Reduce(s, _) => s.in_dtypes.len(),
            Key::Gemm(s) | Key::GemmMma(s, ..) | Key::Gemv(s, ..) => 2 + s.extra_dtypes.len(),
            Key::Gather(_) | Key::GatherAxis(_) => 2,
            Key::TopK(..) | Key::TopKMerge | Key::TopKSmall(_) => 1,
            Key::GemvId(..) | Key::MoeRoute | Key::GemmMmaId(..) => 3,
            Key::Gdn(..) => 6,
            Key::Softmax(_) | Key::Sinkhorn(_) => 1,
            Key::LayerNorm(_, rms, _) => 3 - *rms as usize,
            Key::AttnSplit(s, ..) | Key::AttnMerge(s) | Key::AttnMma(s, ..) => {
                3 + s.mask_dtype.is_some() as usize
            }
        }
    }

    /// Binds the backend scratch buffer after the output.
    pub fn uses_scratch(&self) -> bool {
        matches!(
            self,
            Key::AttnSplit(_, false, ..)
                | Key::AttnMerge(_)
                | Key::TopK(_, true)
                | Key::TopKMerge
                | Key::MoeRoute
                | Key::GemmMmaId(..)
        )
    }

    /// Binds the backend error flag (a host-visible u32) last: kernels that check indices
    /// and set it to 1 on an out-of-range index.
    pub fn uses_flag(&self) -> bool {
        matches!(
            self,
            Key::Gather(_) | Key::GatherAxis(_) | Key::GemvId(..) | Key::MoeRoute | Key::GemmMmaId(..)
        )
    }

    /// Profile label: kind plus the dtypes that select the code path.
    pub fn name(&self) -> String {
        match self {
            Key::Gemm(s) => format!("gemm {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::GemmMma(s, ..) => format!("gemm_mma {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::Gemv(s, ..) => format!("gemv {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::GemvId(s, ..) => format!("gemv_id {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::GemmMmaId(s, ..) => {
                format!("gemm_mma_id {}x{}", s.a_dtype.name(), s.b_dtype.name())
            }
            Key::Gdn(_, dk) => format!("gated_delta_net dk{dk}"),
            Key::AttnSplit(s, direct, d, qb) => {
                format!(
                    "attention_split{} d{d} q{qb} {}",
                    if *direct { "_direct" } else { "" },
                    s.dtype.name()
                )
            }
            Key::AttnMma(s, d, _, precise) => {
                format!("attention_mma{} d{d} {}", if *precise { "_precise" } else { "" }, s.dtype.name())
            }
            k => k.label().to_string(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Key::Ew(..) => "elementwise",
            Key::Reduce(..) => "reduce",
            Key::Gemm(_) => "gemm",
            Key::GemmMma(..) => "gemm_mma",
            Key::Gemv(..) => "gemv",
            Key::Gather(_) => "gather",
            Key::Softmax(_) => "softmax",
            Key::LayerNorm(_, false, _) => "layer_norm",
            Key::LayerNorm(_, true, _) => "rms_norm",
            Key::AttnSplit(..) => "attention_split",
            Key::AttnMerge(_) => "attention_merge",
            Key::AttnMma(..) => "attention_mma",
            Key::GatherAxis(_) => "gather_axis",
            Key::TopK(..) | Key::TopKSmall(_) => "topk",
            Key::TopKMerge => "topk_merge",
            Key::GemvId(..) => "gemv_id",
            Key::MoeRoute => "moe_route",
            Key::GemmMmaId(..) => "gemm_mma_id",
            Key::Gdn(..) => "gated_delta_net",
            Key::Sinkhorn(_) => "sinkhorn",
        }
    }
}

/// A planned launch.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub key: Key,
    pub params: Vec<u32>,
    /// Grid size in blocks; any zero means "nothing to do".
    pub grid: [u32; 3],
    /// Threads per block.
    pub block: u32,
    /// Bytes of backend scratch the launch needs (bound after the output, shared by all
    /// stages of the launch).
    pub scratch: usize,
    /// Next stage, e.g. the merge pass of split attention (stream order makes it see all
    /// writes of this stage).
    pub next: Option<Box<Plan>>,
}

impl Plan {
    pub fn new(key: Key, params: Vec<u32>, grid: [u32; 3], block: u32) -> Plan {
        Plan {
            key,
            params,
            grid,
            block,
            scratch: 0,
            next: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.grid.contains(&0)
    }
}

pub(crate) fn plan(kernel: &Kernel, caps: &Caps) -> Result<Plan> {
    let p = match kernel {
        Kernel::Elementwise(s, d) => elementwise::plan(s, d, caps),
        Kernel::Reduce(s, d) => reduce::plan(s, d, caps),
        Kernel::MatMul(s, d) => matmul::plan(s, d, caps),
        Kernel::Gather(s, d) => rows::plan_gather(s, d, caps),
        Kernel::Softmax(s, d) => rows::plan_softmax(s, d, caps),
        Kernel::LayerNorm(s, d) => rows::plan_layer_norm(s, d, caps, false),
        Kernel::RmsNorm(s, d) => rows::plan_layer_norm(s, d, caps, true),
        Kernel::Attention(s, d) => attention::plan(s, d, caps),
        Kernel::GatherAxis(s, d) => index::plan_gather_axis(s, d, caps),
        Kernel::TopK(s, d) => index::plan_topk(s, d, caps),
        Kernel::MatMulId(s, d) => moe::plan(s, d, caps),
        Kernel::GatedDeltaNet(s, d) => gdn::plan(s, d, caps),
        Kernel::Sinkhorn(d) => sinkhorn::plan(d, caps),
    }?;
    let mut stage = Some(&p);
    while let Some(s) = stage {
        if s.params.len() > PARAM_WORDS {
            return Err(Error::new(
                "too_large",
                format!(
                    "{}: {} params words > {PARAM_WORDS}",
                    kernel.kind(),
                    s.params.len()
                ),
            ));
        }
        stage = s.next.as_deref();
    }
    Ok(p)
}

/// Generate the PTX body (without `.version` / `.target` header) for `key`.
pub(crate) fn source(key: &Key, _caps: &Caps) -> String {
    match key {
        Key::Ew(s, modes, out) => elementwise::source(s, modes, *out),
        Key::Reduce(s, row) => reduce::source(s, *row),
        Key::Gemm(s) => matmul::source(s),
        Key::GemmMma(s, va, vb) => mma::source(s, *va, *vb, false, false),
        Key::Gemv(s, va, vb, w) => gemv::source(s, *va, *vb, false, *w),
        Key::GemvId(s, va, vb, w) => gemv::source(s, *va, *vb, true, *w),
        Key::GemmMmaId(s, va, vb, small) => mma::source(s, *va, *vb, true, *small),
        Key::MoeRoute => moe::source_route(),
        Key::Gdn(s, dk) => gdn::source(s, *dk as usize),
        Key::Gather(s) => rows::source_gather(s),
        Key::Softmax(s) => rows::source_softmax(s),
        Key::LayerNorm(s, rms, wide) => rows::source_layer_norm(s, *rms, *wide),
        Key::AttnSplit(s, direct, d, qb) => attention::source_split(s, *direct, *d, *qb),
        Key::AttnMerge(s) => attention::source_merge(s),
        Key::AttnMma(s, d, vec, precise) => flash::source(s, *d, *vec, *precise),
        Key::GatherAxis(s) => index::source_gather_axis(s),
        Key::TopK(s, k1) => index::source_topk(s, *k1),
        Key::TopKMerge => index::source_topk_merge(),
        Key::TopKSmall(s) => index::source_topk_small(s),
        Key::Sinkhorn(n) => sinkhorn::source(*n),
    }
}

// ------------------------------------------------------------------ host-side helpers

/// Builder for the u32 params block.
#[derive(Default)]
pub(crate) struct Params(pub Vec<u32>);

impl Params {
    pub fn u(&mut self, v: usize) -> Result<&mut Self> {
        let x = u32::try_from(v)
            .map_err(|_| Error::new("too_large", format!("size {v} exceeds 32-bit indexing")))?;
        self.0.push(x);
        Ok(self)
    }

    /// Signed stride, stored as two's complement (index math wraps mod 2^32).
    pub fn i(&mut self, v: isize) -> Result<&mut Self> {
        if v < i32::MIN as isize || v > u32::MAX as isize {
            return Err(Error::new(
                "too_large",
                format!("stride {v} exceeds 32-bit indexing"),
            ));
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
            return Err(Error::new(
                "bad_launch",
                format!("view rank {} != {}", v.strides.len(), rank),
            ));
        }
        self.u(v.offset)?;
        for &s in &v.strides {
            self.i(s)?;
        }
        Ok(self)
    }
}

/// Threads per block of most kernels.
pub(crate) const BLOCK: u32 = 256;

/// 1-D grid of `blocks` blocks (x up to 2^31 - 1 on every sm_30+ device).
pub(crate) fn grid1(blocks: u64, caps: &Caps) -> Result<[u32; 3]> {
    if blocks == 0 {
        return Ok([0, 0, 0]);
    }
    if blocks > caps.max_grid[0] as u64 {
        return Err(Error::new(
            "too_large",
            format!("launch needs {blocks} blocks (max {})", caps.max_grid[0]),
        ));
    }
    Ok([blocks as u32, 1, 1])
}

/// Grid for one thread per element over `n` elements in blocks of `BLOCK`.
pub(crate) fn grid_threads(n: u64, caps: &Caps) -> Result<[u32; 3]> {
    grid1(n.div_ceil(BLOCK as u64), caps)
}

pub(crate) fn check_plain(d: DType, what: &str) -> Result<()> {
    if d.is_quant() {
        return Err(Error::new(
            "unsupported_dtype",
            format!("{what}: quantized dtype {} not allowed here", d.name()),
        ));
    }
    Ok(())
}

pub(crate) fn bad(msg: impl Into<String>) -> Error {
    Error::new("bad_launch", msg)
}

/// Scales-plane offset (bytes) of a quantized `[rows, cols]` matrix.
/// 0 for the AoS K-quant formats (no scales plane).
pub(crate) fn scales_offset(d: DType, rows: usize, cols: usize) -> usize {
    match d.quant_format() {
        Some(f) if !f.is_aos() => lapid_quant::layout(f, rows, cols).scales_offset,
        _ => 0,
    }
}

/// Elements per quantization block along a row (1 for plain dtypes).
pub(crate) fn quant_block(d: DType) -> usize {
    d.quant_format().map(|f| f.block_elems()).unwrap_or(1)
}
