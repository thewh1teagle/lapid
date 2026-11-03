//! lapid-wgsl: WGSL kernel templates and code generation shared by the Vulkan
//! (naga -> SPIR-V) and WebGPU (wgpu) backends. Pure: no graphics API dependency.
//!
//! A `Key` fully determines the shader source (spec + structural variant); `plan` turns a
//! `Kernel` into a key, the launch params (`array<u32>` at binding 0) and the dispatch
//! size. Sizes never enter the source, so one pipeline serves every shape of a spec.
//! `render` does both in one call.
//!
//! Binding convention: 0 = params, 1..=n = launch inputs (in order), n+1 = output,
//! n+2 = backend extra buffer (`Key::extra`: scratch or the error-status word).
//!
//! Kernels are real `.wgsl` files under `src/kernels/` with `{{SLOT}}` placeholders and
//! `// #if FLAG` line conditionals (see `template`).

mod attention;
mod common;
mod elementwise;
mod expr;
mod gather;
mod gemv;
mod kquant;
mod moe;
mod delta;
mod matmul;
mod reduce;
mod rows;
mod split;
mod template;
mod topk;
#[cfg(test)]
mod tests_webgpu;

use std::sync::Arc;

use lapid_compiler::{
    AttentionSpec, EwSpec, GatedDeltaNetSpec, GatherAxisSpec, GatherSpec, Kernel, MatMulIdSpec, LayerNormSpec, MatMulSpec, ReduceSpec, SoftmaxSpec,
    TopKSpec, View,
};
use lapid_tensor::{DType, Error, Result};

pub use elementwise::IdxMode;
pub use split::{Ranges, SubLaunch, plan_split};
pub use template::fill;

/// What the target device can do, as far as code generation is concerned.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Caps {
    /// `enable f16;` + `array<f16>` storage.
    pub f16: bool,
    /// Subgroup arithmetic/ballot/shuffle usable from compute shaders.
    pub subgroup: bool,
    pub subgroup_size: u32,
    /// Full subgroups can be required (256-thread workgroups = whole subgroups).
    pub full_subgroups: bool,
    pub max_groups: [u32; 3],
    /// Workgroup shared memory bytes.
    pub max_shared: u32,
    /// Cooperative matrices (naga `wgpu_cooperative_matrix`, 16x16x16 f16 -> f32).
    pub coopmat: bool,
    /// Emit WebGPU-portable WGSL: `enable subgroups;`, no `subgroup_id` /
    /// `num_subgroups` builtins (derived from the local index instead).
    pub webgpu: bool,
}

impl Caps {
    /// Plain WGSL with no optional features (any WebGPU device).
    pub fn baseline() -> Caps {
        Caps { f16: false, subgroup: false, subgroup_size: 32, full_subgroups: false, max_groups: [65535; 3], max_shared: 16384, coopmat: false, webgpu: true }
    }
}

/// A kernel ready to dispatch: final WGSL source plus how to bind and launch it.
#[derive(Clone, Debug)]
pub struct RenderedKernel {
    /// Pipeline cache key (the source is a pure function of key + caps).
    pub key: Key,
    pub wgsl: String,
    /// Total storage bindings at group 0: params + inputs + output [+ extra].
    pub bindings: usize,
    /// Workgroup counts; any zero means "nothing to do".
    pub workgroups: [u32; 3],
    /// Launch params (binding 0).
    pub params: Vec<u32>,
    /// Backend scratch bytes (bound as the extra buffer).
    pub scratch: usize,
    /// Next stage (after a barrier), e.g. the merge pass of split attention.
    pub next: Option<Box<RenderedKernel>>,
}

/// Plan `kernel` and generate its source (all stages).
pub fn render(kernel: &Kernel, caps: &Caps) -> Result<RenderedKernel> {
    Ok(render_plan(plan(kernel, caps)?, caps))
}

/// Generate the sources for an existing plan (all stages).
pub fn render_plan(p: Plan, caps: &Caps) -> RenderedKernel {
    let wgsl = source(&p.key, caps);
    RenderedKernel {
        bindings: p.key.num_bindings(),
        wgsl,
        workgroups: p.groups,
        params: p.params,
        scratch: p.scratch,
        next: p.next.map(|n| Box::new(render_plan(*n, caps))),
        key: p.key,
    }
}

/// Pipeline cache key: everything that influences the generated code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// Elementwise: per-input index mode, output index mode.
    Ew(Arc<EwSpec>, Vec<IdxMode>, IdxMode),
    /// Reduce: `true` = one workgroup per output row, `false` = one thread per output.
    Reduce(Arc<ReduceSpec>, bool),
    /// Tiled GEMM (64x64 tiles).
    Gemm(Arc<MatMulSpec>),
    /// Tiled GEMM on cooperative matrices (f16 in, f32 accumulate).
    GemmCoop(Arc<MatMulSpec>),
    /// Matrix-vector (small m, transposed B); (A read as vec4, f16 B read as 8-half vectors).
    Gemv(Arc<MatMulSpec>, bool, bool),
    Gather(Arc<GatherSpec>),
    Softmax(Arc<SoftmaxSpec>),
    LayerNorm(Arc<LayerNormSpec>),
    /// RMS norm (layer norm without centering and bias).
    RmsNorm(Arc<LayerNormSpec>),
    /// Flash attention, one thread per query, head dim baked in (multiple of 4, <= 128).
    AttnTile(Arc<AttentionSpec>, u32),
    /// Attention for few queries, split over key chunks; `true` = one chunk, writes the
    /// output directly (no merge stage); head dim baked in (<= 256).
    AttnSplit(Arc<AttentionSpec>, bool, u32),
    /// Flash attention on cooperative matrices (head dim 64, many queries).
    AttnCoop(Arc<AttentionSpec>),
    /// Flash attention on cooperative matrices for wide heads (head dim 256 / 512 baked in).
    AttnCoopWide(Arc<AttentionSpec>, u32),
    /// Merge stage of `AttnSplit(_, false)`.
    AttnMerge(Arc<AttentionSpec>),
    /// take_along_axis (exact element copy).
    GatherAxis(Arc<GatherAxisSpec>),
    /// Top-k indices per row; `true` = argmax variant (k = 1).
    TopK(Arc<TopKSpec>, bool),
    /// Top-k for small k and rows (k rounds of a workgroup argmax; needs subgroups).
    TopKSmall(Arc<TopKSpec>),
    /// Mixture-of-experts matmul (one lane group per output column).
    MatMulId(Arc<MatMulIdSpec>),
    /// MoE prefill stage 1: bucket (token, slot) pairs by expert (one workgroup; experts
    /// rounded up to a power of two for the shared counters).
    MoeRoute(u32),
    /// MoE prefill stage 2: grouped cooperative-matrix GEMM over the routed pairs.
    GemmIdCoop(Arc<MatMulIdSpec>),
    /// Gated delta rule (one lane group per state column; state rows per lane baked in).
    GatedDeltaNet(Arc<GatedDeltaNetSpec>, u32),
    /// Gated delta rule, few lanes per column (32-lane full subgroups): (rows per lane,
    /// lanes per column).
    GatedDeltaNet8(Arc<GatedDeltaNetSpec>, u32, u32),
}

/// Backend buffer bound after a kernel's output.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Extra {
    /// Per-launch scratch (split kernels).
    Scratch,
    /// The device error flag (`status[0] = 1` on an out-of-range index).
    Status,
    /// Scratch, then the error flag (two bindings).
    ScratchStatus,
}

impl Key {
    /// Number of launch inputs bound by this kernel.
    pub fn num_inputs(&self) -> usize {
        match self {
            Key::Ew(s, ..) => s.in_dtypes.len(),
            Key::Reduce(s, _) => s.in_dtypes.len(),
            Key::Gemm(s) | Key::GemmCoop(s) | Key::Gemv(s, ..) => 2 + s.extra_dtypes.len(),
            Key::Gather(_) | Key::GatherAxis(_) => 2,
            Key::TopK(..) | Key::TopKSmall(_) => 1,
            Key::MatMulId(_) | Key::MoeRoute(_) | Key::GemmIdCoop(_) => 3,
            Key::GatedDeltaNet(..) | Key::GatedDeltaNet8(..) => 6,
            Key::Softmax(_) => 1,
            Key::LayerNorm(_) => 3,
            Key::RmsNorm(_) => 2,
            Key::AttnTile(s, _) | Key::AttnCoop(s) | Key::AttnCoopWide(s, _) | Key::AttnSplit(s, ..) | Key::AttnMerge(s) => 3 + s.mask_dtype.is_some() as usize,
        }
    }

    /// Total descriptor bindings (params + inputs + output [+ extra]).
    pub fn num_bindings(&self) -> usize {
        self.num_inputs()
            + 2
            + match self.extra() {
                None => 0,
                Some(Extra::ScratchStatus) => 2,
                Some(_) => 1,
            }
    }

    /// Backend buffer bound after the output, if any.
    pub fn extra(&self) -> Option<Extra> {
        match self {
            Key::AttnSplit(_, false, _) | Key::AttnMerge(_) => Some(Extra::Scratch),
            Key::Gather(_) | Key::GatherAxis(_) | Key::MatMulId(_) => Some(Extra::Status),
            Key::MoeRoute(_) | Key::GemmIdCoop(_) => Some(Extra::ScratchStatus),
            _ => None,
        }
    }

    /// Whether the generated code uses subgroup operations (when the device has them).
    pub fn uses_subgroups(&self, caps: &Caps) -> bool {
        caps.subgroup
            && match self {
                Key::Reduce(_, row) => *row,
                Key::TopK(_, argmax) => *argmax,
                Key::TopKSmall(_) => true,
                Key::Gemv(..) | Key::MatMulId(_) | Key::GatedDeltaNet(..) | Key::GatedDeltaNet8(..) | Key::GemmCoop(_) | Key::GemmIdCoop(_) | Key::Softmax(_) | Key::LayerNorm(_) | Key::RmsNorm(_) | Key::AttnSplit(..) | Key::AttnCoop(_) | Key::AttnCoopWide(..) => true,
                _ => false,
            }
    }

    /// Profile label: kind plus the dtypes that select the code path.
    pub fn name(&self) -> String {
        match self {
            Key::Gemm(s) => format!("gemm {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::GemmCoop(s) => format!("gemm_coop {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::Gemv(s, ..) => format!("gemv {}x{}", s.a_dtype.name(), s.b_dtype.name()),
            Key::MatMulId(s) => format!("matmul_id {}x{}", s.x_dtype.name(), s.w_dtype.name()),
            Key::GemmIdCoop(s) => format!("gemm_id_coop {}x{}", s.x_dtype.name(), s.w_dtype.name()),
            Key::AttnTile(s, d) => format!("attention_tile d{d} {}", s.dtype.name()),
            Key::AttnSplit(s, direct, _) => format!("attention_split{} {}", if *direct { " direct" } else { "" }, s.dtype.name()),
            k => k.label().to_string(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Key::Ew(..) => "elementwise",
            Key::Reduce(..) => "reduce",
            Key::Gemm(_) => "gemm",
            Key::GemmCoop(_) => "gemm_coop",
            Key::Gemv(..) => "gemv",
            Key::Gather(_) => "gather",
            Key::Softmax(_) => "softmax",
            Key::LayerNorm(_) => "layer_norm",
            Key::RmsNorm(_) => "rms_norm",
            Key::AttnTile(..) => "attention_tile",
            Key::AttnSplit(..) => "attention_split",
            Key::AttnMerge(_) => "attention_merge",
            Key::AttnCoop(_) => "attention_coop",
            Key::AttnCoopWide(..) => "attention_coop_wide",
            Key::GatherAxis(_) => "gather_axis",
            Key::TopK(..) | Key::TopKSmall(_) => "topk",
            Key::MatMulId(_) => "matmul_id",
            Key::MoeRoute(_) => "moe_route",
            Key::GemmIdCoop(_) => "gemm_id_coop",
            Key::GatedDeltaNet(..) | Key::GatedDeltaNet8(..) => "gated_delta_net",
        }
    }
}

/// A planned launch.
#[derive(Clone, Debug)]
pub struct Plan {
    pub key: Key,
    pub params: Vec<u32>,
    /// Workgroup counts; any zero means "nothing to do".
    pub groups: [u32; 3],
    /// Bytes of backend scratch memory the launch needs (bound after the output, shared
    /// by all stages of the launch).
    pub scratch: usize,
    /// Next stage (runs after a barrier), e.g. the reduce pass of a split kernel.
    pub next: Option<Box<Plan>>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.groups.contains(&0)
    }
}

pub fn plan(kernel: &Kernel, caps: &Caps) -> Result<Plan> {
    match kernel {
        Kernel::Elementwise(s, d) => elementwise::plan(s, d, caps),
        Kernel::Reduce(s, d) => reduce::plan(s, d, caps),
        Kernel::MatMul(s, d) => matmul::plan(s, d, caps),
        Kernel::Gather(s, d) => gather::plan(s, d, caps),
        Kernel::Softmax(s, d) => rows::plan_softmax(s, d, caps),
        Kernel::LayerNorm(s, d) => rows::plan_layer_norm(s, d, caps),
        Kernel::Attention(s, d) => attention::plan(s, d, caps),
        Kernel::GatherAxis(s, d) => gather::plan_axis(s, d, caps),
        Kernel::TopK(s, d) => topk::plan(s, d, caps),
        Kernel::MatMulId(s, d) => moe::plan(s, d, caps),
        Kernel::GatedDeltaNet(s, d) => delta::plan(s, d, caps),
        Kernel::RmsNorm(s, d) => rows::plan_rms_norm(s, d, caps),
        // opt-in kernel (`CompileOptions::native_sinkhorn`): never lowered for this backend
        Kernel::Sinkhorn(_) => Err(Error::new("unsupported", "wgsl: no native sinkhorn kernel")),
    }
}

/// Generate the WGSL source for `key`.
pub fn source(key: &Key, caps: &Caps) -> String {
    match key {
        Key::Ew(s, modes, out) => elementwise::source(s, modes, *out, caps),
        Key::Reduce(s, row) => reduce::source(s, *row, caps),
        Key::Gemm(s) => matmul::source(s, caps),
        Key::GemmCoop(s) => matmul::source_coop(s, caps),
        Key::Gemv(s, va, vb) => gemv::source(s, *va, *vb, caps),
        Key::Gather(s) => gather::source(s, caps),
        Key::Softmax(s) => rows::source_softmax(s, caps),
        Key::LayerNorm(s) => rows::source_layer_norm(s, caps),
        Key::RmsNorm(s) => rows::source_rms_norm(s, caps),
        Key::AttnTile(s, d) => attention::source_tile(s, *d, caps),
        Key::AttnSplit(s, direct, d) => attention::source_split(s, *direct, *d, caps),
        Key::AttnMerge(s) => attention::source_merge(s, caps),
        Key::AttnCoop(s) => attention::source_coop(s, caps),
        Key::AttnCoopWide(s, d) => attention::source_coop_wide(s, *d, caps),
        Key::GatherAxis(s) => gather::source_axis(s, caps),
        Key::TopK(s, argmax) => topk::source(s, *argmax, caps),
        Key::TopKSmall(s) => topk::source_small(s, caps),
        Key::MatMulId(s) => moe::source(s, caps),
        Key::MoeRoute(emax) => moe::source_route(*emax, caps),
        Key::GemmIdCoop(s) => matmul::source_coop_id(s, caps),
        Key::GatedDeltaNet(s, np) => delta::source(s, *np, caps),
        Key::GatedDeltaNet8(s, np, lpc) => delta::source8(s, *np, *lpc, caps),
    }
}

// ------------------------------------------------------------------ host-side helpers

/// Builder for the u32 params array.
#[derive(Default)]
pub struct Params(pub Vec<u32>);

impl Params {
    pub fn u(&mut self, v: usize) -> Result<&mut Self> {
        let x = u32::try_from(v).map_err(|_| Error::new("too_large", format!("size {v} exceeds 32-bit indexing")))?;
        self.0.push(x);
        Ok(self)
    }

    /// Signed stride, stored as two's complement (index math wraps mod 2^32).
    pub fn i(&mut self, v: isize) -> Result<&mut Self> {
        // i64 compare: on 32-bit targets (wasm32) `u32::MAX as isize` is -1
        if (v as i64) < i32::MIN as i64 || (v as i64) > u32::MAX as i64 {
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
            return Err(Error::new("bad_launch", format!("view rank {} != {}", v.strides.len(), rank)));
        }
        self.u(v.offset)?;
        for &s in &v.strides {
            self.i(s)?;
        }
        Ok(self)
    }
}

/// Workgroup threads used by most kernels.
pub const WG: u32 = 256;

/// Dispatch size for `items` independent workgroup-sized work items. Kernels iterate
/// `item = wg_lin; item < items; item += wg_total`, so capping is always safe.
pub fn grid(items: u64, caps: &Caps) -> [u32; 3] {
    if items == 0 {
        return [0, 0, 0];
    }
    let max_x = caps.max_groups[0].clamp(1, 65535) as u64;
    let max_y = caps.max_groups[1].clamp(1, 65535) as u64;
    let x = items.min(max_x);
    let y = items.div_ceil(x).min(max_y);
    [x as u32, y as u32, 1]
}

/// Dispatch size for a grid-stride loop over `n` elements with `WG` threads per group.
pub fn grid_threads(n: u64, caps: &Caps) -> [u32; 3] {
    // ~1M threads is plenty to fill any GPU; the loop covers the rest.
    grid(n.div_ceil(WG as u64).min(4096), caps)
}

pub fn check_plain(d: DType, what: &str) -> Result<()> {
    if d.is_quant() {
        return Err(Error::new("unsupported_dtype", format!("{what}: quantized dtype {} not allowed here", d.name())));
    }
    Ok(())
}

/// Quantized weight dtypes the WGSL kernels implement: MXFP4 and the i-quants are not (CPU
/// and CUDA only), and
/// is rejected at plan time instead of producing a shader without its loaders.
pub fn check_quant_supported(d: DType, what: &str) -> Result<()> {
    if d == DType::Mxfp4 || d.is_iquant() {
        return Err(Error::new("unsupported_dtype", format!("{what}: {} is not supported by the WGSL backends", d.name())));
    }
    Ok(())
}

pub fn bad(msg: impl Into<String>) -> Error {
    Error::new("bad_launch", msg)
}
