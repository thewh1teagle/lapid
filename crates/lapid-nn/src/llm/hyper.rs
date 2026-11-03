//! Hyper-connections: the residual is `hc` parallel streams, collapsed into each block's input
//! and updated from its output by learned, input-dependent weights. Two variants:
//!
//! [`LowRankHc`] (HF `Qwen4ExpTextGatedResidual`, llama.cpp `build_hc_mix` / `build_hc_combine`),
//! residual `x [1, n, hc, D]`:
//!
//! ```text
//! xn    = rmsnorm_per_stream(x) * w_norm            (w_norm [hc, D], effective scale)
//! lo    = silu(down(flat(xn)) / hc)                 (rank hc_lowrank)
//! mixed = mean_s(sigmoid(up(lo))_s * xn_s)          -> block input [1, n, D]
//! w     = 2 * sigmoid(inject(flat(xn)) / hc)        -> [1, n, hc, 1]
//! x'    = x + block(mixed) * w                       (combine)
//! ```
//! The final mixer (no inject) replaces the output norm.
//!
//! [`SinkhornHc`]: manifold-constrained hyper-connections (DeepSeek V4 mHC, llama.cpp
//! `DSV4_HC_PRE` / `DSV4_HC_COMB` / `DSV4_HC_POST`), residual `x [n, hc, D]`:
//!
//! ```text
//! mix  = rmsnorm(flatten(x)) @ fn^T                      [n, (2 + hc) hc]
//! pre  = sigmoid(mix[:hc] s0 + b[:hc]) + eps             collapse weights
//! post = 2 sigmoid(mix[hc:2hc] s1 + b[hc:2hc])           expand weights
//! comb = sinkhorn(softmax_row(mix[2hc:] s2 + b[2hc:]) + eps)   [n, src, dst], doubly stochastic
//! block input  = sum_i pre[i] x[i]
//! block output = post[j] y + sum_i comb[i, j] x[i]       for every stream j
//! ```
//! Its final head ([`SinkhornHead`]) collapses the streams with `pre = sigmoid(mix s + b) + eps`.
//! [`SinkhornHc::mix`] / [`collapse`] / [`expand`] are the separate steps, for wirings that
//! collapse with another block's `pre` (DeepSeek V4.1's lagged "single-pass" mHC).

use lapid::{DType, Result, Tensor, bail};

use crate::layers::{Linear, RmsNorm};
use crate::ops;
use crate::weights::Loader;

/// RMS norm over the last dim of every stream, scaled by a `[hc, D]` weight.
pub fn grouped_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.shape().last().copied().unwrap_or(0);
    RmsNorm::new(Tensor::full(1.0, &[d], DType::F32), eps).forward(x)?.mul(weight)
}

pub struct LowRankHc {
    /// [hc, D] effective scale.
    pub norm: Tensor,
    pub down: Linear,
    pub up: Linear,
    /// `None` for the final mixer.
    pub inject: Option<Linear>,
    pub hc: usize,
    pub eps: f32,
}

/// Output of [`LowRankHc::mix`].
pub struct Mixed {
    /// Block input [1, n, D].
    pub input: Tensor,
    /// Scatter weights [1, n, hc, 1] (`None` for the final mixer).
    pub weights: Option<Tensor>,
}

impl LowRankHc {
    /// `{p}.hc_norm.weight` `[hc, d]`, `{p}.input_mix_weight_{down,up}`, `{p}.block_inject_weight`
    /// (when `inject`).
    pub fn load(l: &Loader<'_>, p: &str, [hc, d, rank]: [usize; 3], inject: bool, eps: f32) -> Result<LowRankHc> {
        Ok(LowRankHc {
            norm: l.tensor_checked(&format!("{p}.hc_norm.weight"), &[hc, d])?.cast(DType::F32)?,
            down: l.linear_checked(&format!("{p}.input_mix_weight_down"), rank, hc * d, false)?,
            up: l.linear_checked(&format!("{p}.input_mix_weight_up"), hc * d, rank, false)?,
            inject: inject.then(|| l.linear_checked(&format!("{p}.block_inject_weight"), hc, hc * d, false)).transpose()?,
            hc,
            eps,
        })
    }

    pub fn mix(&self, x: &Tensor) -> Result<Mixed> {
        if x.rank() != 4 || x.shape()[2] != self.hc {
            bail!("shape_mismatch", "hyper-connection expects [1, n, {}, D], got {:?}", self.hc, x.shape());
        }
        let (n, hc, d) = (x.shape()[1], self.hc, x.shape()[3]);
        let inv = 1.0 / hc as f64;
        let xn = grouped_rms_norm(x, &self.norm, self.eps)?; // [1, n, hc, D]
        let flat = xn.reshape(&[1, n, hc * d])?;
        let lo = ops::silu(&self.down.forward(&flat)?.cast(DType::F32)?.mul_scalar(inv)?)?;
        let gate = ops::sigmoid(&self.up.forward(&lo)?.cast(DType::F32)?)?.reshape(&[1, n, hc, d])?;
        let input = gate.mul(&xn)?.sum(2, false)?.mul_scalar(inv)?; // [1, n, D]
        let weights = match &self.inject {
            Some(inj) => Some(ops::sigmoid(&inj.forward(&flat)?.cast(DType::F32)?.mul_scalar(inv)?)?.mul_scalar(2.0)?.reshape(&[1, n, hc, 1])?),
            None => None,
        };
        Ok(Mixed { input, weights })
    }

    /// `x + block(mix(x).input) * mix(x).weights` (layer mixers only).
    pub fn around(&self, x: &Tensor, block: impl FnOnce(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
        let m = self.mix(x)?;
        let Some(w) = m.weights else { bail!("internal", "hyper-connection: final mixer used around a block") };
        x.add(&block(&m.input)?.cast(DType::F32)?.unsqueeze(2)?.mul(&w)?)
    }
}

/// Host copy of a small f32 parameter.
fn host_f32(l: &Loader<'_>, name: &str) -> Result<Vec<f32>> {
    l.device.to_vec_f32(&l.tensor_f32(name)?)
}

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = l.tensor_f32(name)?;
    l.expect_shape(&t, name, shape)?;
    Ok(t)
}

/// `rmsnorm(flatten(x)) @ func^T` of x [n, hc, D].
fn mixes(x: &Tensor, func: &Tensor, ones: &Tensor, norm_eps: f32) -> Result<Tensor> {
    let (n, hc, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    RmsNorm::new(ones.clone(), norm_eps).forward(&x.reshape(&[n, hc * d])?)?.matmul_t(func)
}

pub struct SinkhornHc {
    /// `[mix, hc * D]` f32.
    func: Tensor,
    /// `[mix]` f32.
    base: Tensor,
    scale: [f32; 3],
    hc: usize,
    iters: usize,
    eps: f32,
    norm_eps: f32,
    ones: Tensor,
}

/// Mixing weights of [`SinkhornHc::mix`] (`pre` collapses, `post` / `comb` expand).
pub struct HcMix {
    pub pre: Tensor,
    pub post: Tensor,
    pub comb: Tensor,
}

/// Output of [`SinkhornHc::pre`].
pub struct HcPre {
    /// Block input [n, D].
    pub collapsed: Tensor,
    pub post: Tensor,
    pub comb: Tensor,
}

impl SinkhornHc {
    /// `{p}_fn`, `{p}_base`, `{p}_scale` (DeepSeek names, e.g. `layers.3.hc_attn`); `iters`
    /// Sinkhorn iterations, `eps` added to the weights, `norm_eps` of the mixing norm.
    pub fn load(l: &Loader<'_>, p: &str, [hc, d, iters]: [usize; 3], eps: f32, norm_eps: f32) -> Result<SinkhornHc> {
        let mix = (2 + hc) * hc;
        let s = host_f32(l, &format!("{p}_scale"))?;
        let [s0, s1, s2] = s[..] else { bail!("shape_mismatch", "{p}_scale: expected 3 values, got {}", s.len()) };
        Ok(SinkhornHc {
            func: f32_param(l, &format!("{p}_fn"), &[mix, hc * d])?,
            base: f32_param(l, &format!("{p}_base"), &[mix])?,
            scale: [s0, s1, s2],
            hc,
            iters,
            eps,
            norm_eps,
            ones: l.device.upload(&Tensor::full(1.0, &[hc * d], DType::F32))?,
        })
    }

    /// Mixing weights of `x [n, hc, D]`: `pre` / `post` `[n, hc]`, `comb` `[n, hc, hc]`.
    pub fn mix(&self, x: &Tensor) -> Result<HcMix> {
        let (n, hc) = (x.shape()[0], self.hc);
        let m = mixes(x, &self.func, &self.ones, self.norm_eps)?; // [n, mix]
        let eps = f64::from(self.eps);
        let part = |i: usize, a: usize, b: usize| -> Result<Tensor> { m.slice(1, a, b)?.mul_scalar(f64::from(self.scale[i]))?.add(&self.base.slice(0, a, b)?) };
        let pre = ops::sigmoid(&part(0, 0, hc)?)?.add_scalar(eps)?; // [n, hc]
        let post = ops::sigmoid(&part(1, hc, 2 * hc)?)?.mul_scalar(2.0)?;
        // comb[n, src, dst]: softmax over dst, then Sinkhorn (column = over src, row = over dst).
        let comb = part(2, 2 * hc, 2 * hc + hc * hc)?.reshape(&[n, hc, hc])?.sinkhorn(self.iters, self.eps)?;
        Ok(HcMix { pre, post, comb })
    }

    pub fn pre(&self, x: &Tensor) -> Result<HcPre> {
        let m = self.mix(x)?;
        Ok(HcPre { collapsed: collapse(&m.pre, x)?, post: m.post, comb: m.comb })
    }

    /// `y [n, D]`, `residual [n, hc, D]` -> `[n, hc, D]`.
    pub fn post(&self, y: &Tensor, residual: &Tensor, pre: &HcPre) -> Result<Tensor> {
        expand(y, residual, &pre.post, &pre.comb)
    }
}

/// Block input `sum_i pre[i] x[i]`: `pre [n, hc]`, `x [n, hc, D]` -> `[n, D]`.
pub fn collapse(pre: &Tensor, x: &Tensor) -> Result<Tensor> {
    pre.unsqueeze(2)?.mul(x)?.sum(1, false)
}

/// Block output `post[j] y + sum_i comb[i, j] residual[i]` for every stream j: `y [n, D]`,
/// `residual [n, hc, D]`, `post [n, hc]`, `comb [n, hc, hc]` -> `[n, hc, D]`.
pub fn expand(y: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
    let out = post.unsqueeze(2)?.mul(&y.unsqueeze(1)?)?; // [n, hc, D]
    let mixed = comb.unsqueeze(3)?.mul(&residual.unsqueeze(2)?)?.sum(1, false)?; // [n, dst, D]
    out.add(&mixed)
}

/// Final collapse of the [`SinkhornHc`] streams.
pub struct SinkhornHead {
    func: Tensor,
    base: Tensor,
    scale: f32,
    eps: f32,
    norm_eps: f32,
    ones: Tensor,
}

impl SinkhornHead {
    /// `{p}_fn`, `{p}_base`, `{p}_scale` (e.g. `hc_head`).
    pub fn load(l: &Loader<'_>, p: &str, hc: usize, d: usize, eps: f32, norm_eps: f32) -> Result<SinkhornHead> {
        let s = host_f32(l, &format!("{p}_scale"))?;
        Ok(SinkhornHead {
            func: f32_param(l, &format!("{p}_fn"), &[hc, hc * d])?,
            base: f32_param(l, &format!("{p}_base"), &[hc])?,
            scale: s.first().copied().unwrap_or(1.0),
            eps,
            norm_eps,
            ones: l.device.upload(&Tensor::full(1.0, &[hc * d], DType::F32))?,
        })
    }

    /// `[n, hc, D]` -> `[n, D]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let m = mixes(x, &self.func, &self.ones, self.norm_eps)?;
        let pre = ops::sigmoid(&m.mul_scalar(f64::from(self.scale))?.add(&self.base)?)?.add_scalar(f64::from(self.eps))?;
        pre.unsqueeze(2)?.mul(x)?.sum(1, false)
    }
}
