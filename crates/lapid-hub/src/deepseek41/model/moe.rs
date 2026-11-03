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

use lapid::{DType, Result, Tensor};
use lapid_nn::Linear;
use lapid_nn::llm::moe::Experts;
use lapid_nn::llm::softplus;

pub(crate) struct Moe {
    /// Router `[E, D]` f32.
    pub router: Tensor,
    /// Selection bias `[E]` f32.
    pub bias: Tensor,
    pub experts: Experts,
    /// Shared expert w1 (gate), w3 (up), w2 (down).
    pub shared: [Linear; 3],
    pub top_k: usize,
    pub norm_topk: bool,
    pub route_scale: f32,
}

impl Moe {
    /// `x [n, D]` (f32) -> `[n, D]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scores = softplus(&x.matmul_t(&self.router)?)?.sqrt()?; // [n, E]
        let ids = scores.add(&self.bias)?.topk_indices(self.top_k, 1)?;
        let mut w = scores.gather_axis(1, &ids)?;
        if self.norm_topk && self.top_k > 1 {
            w = w.div(&w.sum(1, true)?.add_scalar(1e-20)?)?;
        }
        let w = w.mul_scalar(f64::from(self.route_scale))?; // [n, k]
        let routed = self.experts.forward(x, &ids, &w)?;
        let [w1, w3, w2] = &self.shared;
        let shared = w2.forward(&self.experts.glu(&w1.forward(x)?, &w3.forward(x)?)?)?;
        routed.add(&shared)?.cast(DType::F32)
    }
}
