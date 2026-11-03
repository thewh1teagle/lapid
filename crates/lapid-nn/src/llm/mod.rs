//! Building blocks of decoder LLMs shared by several hub models (Llama / Qwen families,
//! Qwen3.5, Qwen4-Exp, DeepSeek V4 / V4.1, Qwen3-VL; Gemma 4 has the same MoE shape).
//!
//! * [`moe`]: softmax top-k routing and SwiGLU / GeGLU expert stacks applied with `MatMulId`,
//!   the Qwen-style sparse MoE block (optional sigmoid-gated shared expert).
//! * [`gated_attention`]: GQA attention with a sigmoid output gate interleaved in `q_proj`
//!   (Qwen3.5 / Qwen4-Exp).
//! * [`delta_net`]: the Gated DeltaNet linear-attention mixer (Qwen3.5 / Qwen4-Exp).
//! * [`hyper`]: hyper-connection residual streams (low-rank gated: Qwen4-Exp; Sinkhorn mHC:
//!   DeepSeek V4).
//! * [`rope`]: inverse-frequency variants (linear, llama3, per-frequency factors, YaRN), cos /
//!   sin tables, `rotate_half`, M-RoPE (Qwen-VL) and the interleaved tail RoPE (DeepSeek V4).
//! * [`engram`]: n-gram hashed lookup tables gated into the hyper-connection streams
//!   (DeepSeek V4.1): host hashing, host FP8 row gather, the gate.
//! * [`expert_cache`]: a device pool of MoE experts filled from the mmap'd file on demand,
//!   for models whose experts do not fit in memory; [`expert_stats`]: its opt-in log.
//! * [`head`]: token embedding + logits projection (`lm_head` or the tied embedding).
//!
//! Tensor names in the `load` functions are the ones the block is defined by (HF Qwen /
//! upstream DeepSeek names under a caller-given prefix).

pub mod delta_net;
pub mod engram;
pub mod expert_cache;
mod expert_io;
pub mod expert_stats;
pub mod gated_attention;
pub mod head;
pub mod hyper;
pub mod moe;
pub mod rope;
#[cfg(test)]
mod tests;

use lapid::{Result, Tensor, UnaryOp};

/// softplus(x) = max(x, 0) + log(1 + exp(-|x|)) (stable for large |x|).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = x.maximum(&Tensor::zeros(x.shape(), x.dtype()))?;
    pos.add(&x.unary(UnaryOp::Abs)?.neg()?.exp()?.add_scalar(1.0)?.log()?)
}

/// x * rsqrt(sum(x^2, last) + eps) (HF `l2norm`, FLA convention).
pub fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    x.mul(&x.mul(x)?.sum(-1, true)?.add_scalar(eps)?.unary(UnaryOp::Rsqrt)?)
}
