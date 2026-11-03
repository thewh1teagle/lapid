//! DeepSeek V4.1 MoE FFN (upstream `MoE` / `Gate`; llama.cpp `build_moe_ffn`, sqrt-softplus):
//!
//! ```text
//! s   = sqrt(softplus(x Wr^T))                         [n, E]   (f32)
//! ids = topk(s + bias, k)                              the bias picks, it does not weigh
//! w   = s[ids] / (sum(s[ids]) + 1e-20) * route_scale   (normalized when k > 1)
//! y   = sum_r w[r] down_e(silu(min(gate_e(x), l)) * clamp(up_e(x), -l, l))   (MatMulId)
//!     + shared(x)                                      (same clamped SwiGLU, always on)
//! ```
//! V4.1 has no hash-routed layers; the vision routing bias (`bias_vl`) is not used.

use lapid::{BinaryOp, DType, Result, Tensor};
use lapid_nn::llm::expert_cache::ExpertCache;
use lapid_nn::llm::moe::Experts;
use lapid_nn::llm::softplus;
use lapid_nn::{Activation, Linear};

/// Where the routed experts live.
pub(crate) enum Routed {
    /// Graph leaves (uploaded, or mapped on the CPU): one graph per forward.
    Direct(Experts),
    /// In the model's [`ExpertCache`] (layer `layer`): the forward evaluates layer by layer.
    Cached { layer: usize, act_quant: bool },
}

pub(crate) struct Moe {
    /// Router `[E, D]` f32.
    pub router: Tensor,
    /// Selection bias `[E]` f32.
    pub bias: Tensor,
    pub routed: Routed,
    /// Shared expert w1 (gate), w3 (up), w2 (down).
    pub shared: [Linear; 3],
    pub top_k: usize,
    pub norm_topk: bool,
    pub route_scale: f32,
    /// SwiGLU clamp (`swiglu_limit`).
    pub limit: Option<f32>,
}

impl Moe {
    /// Expert ids `[n, k]` (i32) and their weights `[n, k]` for `x [n, D]` (f32).
    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let scores = softplus(&x.matmul_t(&self.router)?)?.sqrt()?; // [n, E]
        let ids = scores.add(&self.bias)?.topk_indices(self.top_k, 1)?;
        let mut w = scores.gather_axis(1, &ids)?;
        if self.norm_topk && self.top_k > 1 {
            w = w.div(&w.sum(1, true)?.add_scalar(1e-20)?)?;
        }
        Ok((ids, w.mul_scalar(f64::from(self.route_scale))?)) // [n, k]
    }

    /// The shared expert (same clamped SwiGLU, always on).
    fn shared(&self, x: &Tensor) -> Result<Tensor> {
        let [w1, w3, w2] = &self.shared;
        let (gate, up) = (w1.forward(x)?, w3.forward(x)?);
        let act = match self.limit {
            Some(l) => {
                let l = f64::from(l);
                let up = up.binary_scalar(BinaryOp::Min, l)?.binary_scalar(BinaryOp::Max, -l)?;
                Activation::Silu.apply(&gate.binary_scalar(BinaryOp::Min, l)?)?.mul(&up)?
            }
            None => Activation::Silu.apply(&gate)?.mul(&up)?,
        };
        w2.forward(&act)
    }

    /// `x [n, D]` (f32) -> `[n, D]` with graph-leaf experts.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let Routed::Direct(experts) = &self.routed else {
            return Err(lapid::Error::new("internal", "deepseek41: cached experts in a one-graph forward"));
        };
        let (ids, w) = self.route(x)?;
        experts.forward(x, &ids, &w)?.add(&self.shared(x)?)?.cast(DType::F32)
    }

    /// `x [n, D]` (f32, realized) with host expert `ids` (`[n, k]` row-major) and weights `w`
    /// from [`route`](Self::route), through the cache. Evaluate before the cache's next use.
    pub fn forward_cached(&self, cache: &ExpertCache, x: &Tensor, ids: &[i32], w: &Tensor) -> Result<Tensor> {
        let Routed::Cached { layer, act_quant } = self.routed else {
            return Err(lapid::Error::new("internal", "deepseek41: graph-leaf experts in a cached forward"));
        };
        let routed = cache.forward(layer, Activation::Silu, self.limit, act_quant, x, ids, w)?;
        routed.add(&self.shared(x)?)?.cast(DType::F32)
    }
}
