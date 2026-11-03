//! The operation set. CONTRACT FILE — owned by the coordinator.
//!
//! ~25 primitives plus 3 composites. Composites (`Softmax`, `LayerNorm`, `RmsNorm`, `Attention`) have
//! a reference decomposition into primitives (see `lapid_tensor::decompose`) and backends
//! may run them natively. Semantics are defined in docs/spec/ops.md.

use crate::buffer::{DeviceData, HostData};
use crate::dtype::DType;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Recip,
    Sin,
    Cos,
    Tanh,
    Erf,
    Abs,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Pow,
    /// 1.0 if a < b else 0.0
    CmpLt,
    /// 1.0 if a == b else 0.0
    CmpEq,
    // ---- integer-only (i32 operands; exact 32-bit integer math, see docs/spec/ops.md) ----
    /// Truncated remainder (sign of `a`); `a % 0` is unspecified (no trap).
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    /// `a << (b & 31)`
    Shl,
    /// Arithmetic `a >> (b & 31)`
    Shr,
}

impl BinaryOp {
    /// Defined only for i32 tensors.
    pub fn is_int_only(self) -> bool {
        matches!(self, BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
}

#[derive(Clone, Debug)]
pub enum Op {
    // ---- leaves ----
    /// Host bytes, contiguous row-major in the node's dtype.
    Host(HostData),
    /// Already-realized device buffer, contiguous row-major in the node's dtype.
    Device(DeviceData),
    /// Scalar broadcast to the node shape.
    Const(f64),

    // ---- elementwise (all srcs have the node's shape; broadcasting is done with Expand
    //      by the builder before the op is created) ----
    Cast(DType),
    Unary(UnaryOp),
    Binary(BinaryOp),
    /// srcs = [cond, a, b]; cond != 0 ? a : b
    Where,

    // ---- reduction over one axis ----
    Reduce { op: ReduceOp, axis: usize, keepdim: bool },

    // ---- movement (views; no compute) ----
    Reshape,
    Permute(Vec<usize>),
    /// Broadcast size-1 dims to the node shape (rank must match).
    Expand,
    /// Take [start, end) along axis.
    Slice { axis: usize, start: usize, end: usize },
    /// Pad along axis with `value`.
    Pad { axis: usize, before: usize, after: usize, value: f64 },
    /// srcs concatenated along axis.
    Cat { axis: usize },
    /// Materialize as contiguous row-major.
    Contiguous,

    // ---- linear algebra ----
    /// srcs = [a, b]. a: [..., m, k]. If `b_transposed`, b: [..., n, k] else [..., k, n].
    /// Batch dims broadcast (b may be 2-D). Quantized b requires `b_transposed` and 2-D b.
    MatMul { b_transposed: bool },

    // ---- indexing ----
    /// srcs = [table [V, D], indices (I32, any shape S)] -> [S..., D]
    Gather,
    /// take_along_axis: srcs = [x, indices (I32, same rank as x, same dims except `axis`)]
    /// -> indices' shape, dtype of x. out[.., j, ..] = x[.., indices[.., j, ..], ..].
    GatherAxis { axis: usize },
    /// Indices (I32) of the `k` largest values along `axis` (sorted descending, ties →
    /// lower index first). Output shape = x's shape with `axis` set to `k`.
    TopK { k: usize, axis: usize },

    // ---- in-place update ----
    /// srcs = [dest, src]: dest with `[start, start + src.shape[axis])` along `axis` replaced
    /// by `src` (same dtype and rank, same dims except `axis`). When `dest` is a realized
    /// device buffer used by nothing else in the graph, the write happens in place: the
    /// result aliases dest's buffer (no copy of the rest).
    WriteSlice { axis: usize, start: usize },

    // ---- composites ----
    /// Softmax over the last axis.
    Softmax,
    /// srcs = [x, weight, bias]; normalize over the last axis.
    LayerNorm { eps: f32 },
    /// srcs = [q, k, v] (+ [mask] if has_mask); q [B, H, Sq, D], k/v [B, H, Sk, D],
    /// mask additive f32 broadcastable to [B, H, Sq, Sk]. out [B, H, Sq, D].
    /// causal: query i (0-based) may attend key j iff j <= i + (Sk - Sq).
    Attention { scale: f32, causal: bool, has_mask: bool },

    // ---- round 3 ----
    /// Mixture-of-experts matmul (ggml MUL_MAT_ID). srcs = [x [T, K] (float),
    /// w [E, N, K] (expert weights, float or quantized; each expert in b_transposed form,
    /// quantized stacks are E matrices back to back), ids I32 [T, R]] -> [T, R, N] with
    /// out[t, r, :] = w[ids[t, r]] @ x[t, :]. ids out of range -> error "index_out_of_range".
    MatMulId,
    /// Gated delta rule (Gated DeltaNet linear attention; semantics of ggml
    /// GATED_DELTA_NET in llama.cpp — the reference implementation is normative).
    /// srcs = [q [B,T,H,Dk], k [B,T,H,Dk], v [B,T,H,Dv], g [B,T,H] (log decay),
    /// beta [B,T,H], state [B,H,Dk,Dv]] (all f32). Per (b, h), sequentially over t:
    ///   S = exp(g_t) * S ;  S = S + k_t ⊗ (beta_t * (v_t - Sᵀ k_t)) ;  o_t = Sᵀ (q_t * scale)
    /// Output [B, H, T + Dk, Dv]: rows [0, T) are o_t, rows [T, T + Dk) the final state.
    GatedDeltaNet { scale: f32 },

    // ---- round 5 ----
    /// RMS norm over the last axis: srcs = [x, weight [D]];
    /// out = x * rsqrt(mean(x^2) + eps) * weight (f32 math, x's dtype).
    RmsNorm { eps: f32 },

    // ---- round 6 ----
    /// Sinkhorn normalization of square matrices (DeepSeek mHC `comb`): src = x [..., n, n]
    /// f32 logits -> f32 [..., n, n]. Per matrix: `c = softmax(x) + eps` (over the last
    /// axis), `c = c / (sum_rows(c) + eps)` (each column over the second-to-last axis), then
    /// `iters - 1` times: `c = c / (sum_cols(c) + eps)` (each row), `c = c / (sum_rows(c) + eps)`.
    Sinkhorn { iters: usize, eps: f32 },
}

impl Op {
    pub fn is_leaf(&self) -> bool {
        matches!(self, Op::Host(_) | Op::Device(_) | Op::Const(_))
    }

    pub fn is_movement(&self) -> bool {
        matches!(
            self,
            Op::Reshape | Op::Permute(_) | Op::Expand | Op::Slice { .. }
        )
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, Op::Cast(_) | Op::Unary(_) | Op::Binary(_) | Op::Where)
    }
}
