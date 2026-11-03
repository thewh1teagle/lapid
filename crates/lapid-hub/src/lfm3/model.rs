//! LFM3 network (HF `Lfm3ForCausalLM` layout). Decoder layer (sandwich norms):
//! `x += post_op_norm(op(operator_norm(x)))`, `x += post_ffn_norm(mlp(ffn_norm(x)))`, where
//! `op` is a gated short convolution (`conv` layers) or GQA attention with q/k RMSNorm,
//! zero-tail RoPE, a per-head sigmoid output gate and an optional sliding window. Final
//! `embedding_norm`, logits through the tied embedding. State: KV cache (sliding layers keep
//! their window) + one conv window per conv layer (`LmCache` slots).

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::attention::{merge_heads, scaled_dot_product_attention};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Attention, Conv1d, Embedding, GatedMlp, KvCache, Linear, Loader, Mask, RmsNorm, ZeroTailRope, ops};
use serde_json::Value;

use super::config::{LayerType, Lfm3Config};
use crate::runtime::{CausalLm, Hook, LmCache, LmDims, LmState, Rows, Taps};

/// Token embedding; also the tied output projection.
pub const EMBED: &str = "model.embed_tokens.weight";

/// Gated short conv: `out_proj(C * conv(B * x))` with `B, C, x = in_proj(h)` split by channel.
struct ShortConv {
    in_proj: Linear,
    conv: Conv1d,
    out_proj: Linear,
    d: usize,
}

impl ShortConv {
    /// h [1, n, D] -> [1, n, D]; `state` is the conv window (replaced by the new window).
    fn forward(&self, h: &Tensor, state: &mut Option<Tensor>) -> Result<Tensor> {
        let d = self.d;
        let bcx = self.in_proj.forward(h)?;
        let (b, c, x) = (bcx.slice(2, 0, d)?, bcx.slice(2, d, 2 * d)?, bcx.slice(2, 2 * d, 3 * d)?);
        let conv = self.conv.forward_stream(&b.mul(&x)?.transpose(1, 2)?, state)?.transpose(1, 2)?;
        self.out_proj.forward(&c.mul(&conv)?)
    }
}

/// Attention with q/k norm, zero-tail RoPE and a sigmoid output gate per head.
struct GatedAttention {
    attn: Attention,
    gate: Linear,
    rope: ZeroTailRope,
    mask: Mask,
}

impl GatedAttention {
    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, pos: usize) -> Result<Tensor> {
        let q = self.rope.apply(&self.attn.project_q(h)?, pos)?;
        let (k, v) = self.attn.project_kv(h)?;
        let (k, v) = cache.append(layer, &self.rope.apply(&k, pos)?, &v)?;
        let a = scaled_dot_product_attention(&q, &k, &v, self.attn.scale, &self.mask)?; // [1, H, n, Dh]
        let g = ops::sigmoid(&self.gate.forward(h)?)?.permute(&[0, 2, 1])?.unsqueeze(3)?; // [1, H, n, 1]
        self.attn.o.forward(&merge_heads(&a.mul(&g)?)?)
    }
}

enum Operator {
    Conv(ShortConv),
    Attention(GatedAttention),
}

struct Layer {
    op: Operator,
    op_norm: RmsNorm,
    post_op_norm: Option<RmsNorm>,
    ffn_norm: RmsNorm,
    post_ffn_norm: Option<RmsNorm>,
    mlp: GatedMlp,
}

pub struct Lfm3Model {
    pub cfg: Lfm3Config,
    embed: Tensor,
    logits_act_quant: bool,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

fn layer(l: &Loader<'_>, cfg: &Lfm3Config, i: usize, kind: LayerType) -> Result<Layer> {
    let (d, eps, hd, p) = (cfg.hidden_size, cfg.norm_eps, cfg.head_dim, format!("model.layers.{i}"));
    let lin = |n: &str, o: usize, inp: usize, bias: bool| l.linear_checked(&format!("{p}.{n}"), o, inp, bias);
    let norm = |n: &str, dim: usize| l.rms_norm_checked(&format!("{p}.{n}"), dim, eps);
    let op = if kind == LayerType::Conv {
        let w = l.tensor_checked(&format!("{p}.conv.conv.weight"), &[d, 1, cfg.conv_kernel])?;
        let bias = if cfg.conv_bias { Some(l.tensor(&format!("{p}.conv.conv.bias"))?) } else { None };
        Operator::Conv(ShortConv {
            in_proj: lin("conv.in_proj", 3 * d, d, cfg.conv_bias)?,
            conv: Conv1d::depthwise(w, bias),
            out_proj: lin("conv.out_proj", d, d, cfg.conv_bias)?,
            d,
        })
    } else {
        let (h, kv) = (cfg.n_head, cfg.n_kv_head);
        let attn = Attention::new(
            lin("self_attn.q_proj", h * hd, d, false)?,
            lin("self_attn.k_proj", kv * hd, d, false)?,
            lin("self_attn.v_proj", kv * hd, d, false)?,
            lin("self_attn.o_proj", d, h * hd, false)?,
            h,
            kv,
        )?
        .with_qk_norm(norm("self_attn.q_norm", hd)?, norm("self_attn.k_norm", hd)?);
        let (rp, mask) = match kind {
            LayerType::SlidingAttention => (cfg.rope_sliding, Mask::Sliding(cfg.sliding_window)),
            _ => (cfg.rope_full, Mask::Causal),
        };
        Operator::Attention(GatedAttention {
            attn,
            gate: lin("self_attn.gate_proj", h, d, false)?,
            rope: ZeroTailRope::new(hd, rp.theta, rp.partial),
            mask,
        })
    };
    let sandwich = |n: &str| if cfg.sandwich_norm { norm(n, d).map(Some) } else { Ok(None) };
    let ff = cfg.intermediate_size;
    Ok(Layer {
        op,
        op_norm: norm("operator_norm", d)?,
        post_op_norm: sandwich("post_operator_norm")?,
        ffn_norm: norm("ffn_norm", d)?,
        post_ffn_norm: sandwich("post_ffn_norm")?,
        mlp: GatedMlp::swiglu(lin("feed_forward.w1", ff, d, false)?, lin("feed_forward.w3", ff, d, false)?, lin("feed_forward.w2", d, ff, false)?),
    })
}

impl CausalLm for Lfm3Model {
    const ARCH: &'static str = "lfm3";
    type State = LmCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Lfm3Model> {
        let cfg = Lfm3Config::from_json(meta)?;
        let layers = cfg.layer_types.iter().enumerate().map(|(i, &k)| layer(l, &cfg, i, k)).collect::<Result<_>>()?;
        Ok(Lfm3Model {
            embed: l.tensor_checked(EMBED, &[cfg.vocab_size, cfg.hidden_size])?,
            logits_act_quant: l.act_quant(EMBED)?,
            norm: l.rms_norm_checked("model.embedding_norm", cfg.hidden_size, cfg.norm_eps)?,
            layers,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size, hidden: self.cfg.hidden_size, layers: self.layers.len() }
    }

    fn new_state(&self) -> LmCache {
        LmCache::new(KvCache::with_windows(self.cfg.kv_windows()), self.layers.len())
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>"]
    }

    /// Taps: `embed`, `layers.{i}.op`, `layers.{i}`, `norm` (all rows only).
    fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            bail!("invalid_argument", "lfm3: forward of zero tokens");
        }
        let pos = st.offset();
        let mut x = Embedding::new(self.embed.clone()).lookup(ids)?.cast(DType::F32)?;
        taps.tap("embed", &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.op_norm.forward(&x)?;
            let mut y = match &layer.op {
                Operator::Conv(c) => st.with_slot(i, |s| c.forward(&h, s))?,
                Operator::Attention(a) => a.forward(&h, &mut st.kv, i, pos)?,
            };
            taps.tap(format_args!("layers.{i}.op"), &y);
            if let Some(n) = &layer.post_op_norm {
                y = n.forward(&y)?;
            }
            x = x.add(&taps.steer(Hook::AttnOut, i, &y)?)?;
            let mut f = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            if let Some(n) = &layer.post_ffn_norm {
                f = n.forward(&f)?;
            }
            x = taps.steer(Hook::Resid, i, &x.add(&taps.steer(Hook::FfnOut, i, &f)?)?)?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        let n = ids.len();
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(1, n - 1, n)?,
            _ => x,
        };
        let mut h = self.norm.forward(&x)?;
        if rows == Rows::All {
            taps.tap("norm", &h);
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        Ok(Some(h.matmul_t(&self.embed)?))
    }
}
