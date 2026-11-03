//! Qwen3.5 network (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` text decoder).
//!
//! Decoder layer (pre-norm): `x += mixer(input_layernorm(x))`, `x += ffn(post_attention_layernorm(x))`.
//! `mixer` is Gated DeltaNet (3 of 4 layers) or gated full attention: q (+ a per-head output
//! gate interleaved in q_proj), q/k RMSNorm, partial RoPE (half-split; text-only MRoPE = plain
//! RoPE), GQA, causal; output `o_proj(attn * sigmoid(gate))`. `ffn` is SwiGLU (dense) or the
//! sparse MoE block with a sigmoid-gated shared expert. Final norm, logits through the tied
//! embedding or `lm_head`.
//!
//! File conventions (see `convert`): RMSNorms store the effective scale (HF `1 + w`),
//! DeltaNet `A = -exp(A_log)`, v heads tiled. State (`LmCache`): KV cache for attention
//! layers; slot `i` = conv window, slot `L + i` = delta-rule state of DeltaNet layer `i`.
//! The blocks are `lapid_nn::llm`'s (`gated_attention`, `delta_net`, `moe`); their loaders
//! here are reused by Qwen4-Exp.

use lapid::{Result, Tensor, bail};
use lapid_nn::llm::delta_net::{DeltaNet, DeltaNetDims};
use lapid_nn::llm::gated_attention::GatedAttention;
use lapid_nn::llm::head::LmHead;
use lapid_nn::llm::moe::SparseMoe;
use lapid_nn::{GatedMlp, KvCache, Loader, Mask, RmsNorm, Rope};
use serde_json::Value;

use super::config::{LayerType, Qwen35Config};
use crate::llama::model::{lm_head, logits, qwen_moe};
use crate::runtime::{CausalLm, Hook, LmCache, LmDims, LmState, Rows, Taps};

/// Gated attention of layer `p` (`{p}.self_attn.*`).
pub(crate) fn gated_attention(l: &Loader<'_>, cfg: &Qwen35Config, p: &str) -> Result<GatedAttention> {
    let rope = Rope::new(cfg.head_dim, cfg.rope_theta).with_rotary_dim(cfg.rotary_dim());
    GatedAttention::load(l, &format!("{p}.self_attn"), cfg.hidden_size, [cfg.n_head, cfg.n_kv_head, cfg.head_dim], cfg.norm_eps, rope)
}

/// Gated DeltaNet of layer `p` (`{p}.linear_attn.*`); `sigmoid_gate` for Qwen4-Exp's output gate.
pub(crate) fn delta_net(l: &Loader<'_>, cfg: &Qwen35Config, p: &str, sigmoid_gate: bool) -> Result<DeltaNet> {
    let dims = DeltaNetDims {
        hidden: cfg.hidden_size,
        k_heads: cfg.linear_k_heads,
        v_heads: cfg.linear_v_heads,
        k_dim: cfg.linear_k_dim,
        v_dim: cfg.linear_v_dim,
        conv_kernel: cfg.conv_kernel,
        eps: cfg.norm_eps,
    };
    DeltaNet::load(l, &format!("{p}.linear_attn"), dims, sigmoid_gate)
}

/// Sparse MoE of layer `p` (`{p}.mlp.*`): renormalized top-k plus the sigmoid-gated shared expert.
pub(crate) fn moe(l: &Loader<'_>, cfg: &Qwen35Config, p: &str) -> Result<SparseMoe> {
    let Some(m) = &cfg.moe else { bail!("internal", "qwen35: moe layer without moe config") };
    qwen_moe(l, p, cfg.hidden_size, [m.n_experts, m.expert_intermediate, m.top_k], true, Some(m.shared_intermediate))
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Attention(GatedAttention),
}

enum Ffn {
    Dense(GatedMlp),
    Moe(Box<SparseMoe>),
}

struct Layer {
    mixer: Mixer,
    norm1: RmsNorm,
    norm2: RmsNorm,
    ffn: Ffn,
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    head: LmHead,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

fn layer(l: &Loader<'_>, cfg: &Qwen35Config, i: usize, kind: LayerType) -> Result<Layer> {
    let (d, eps, p) = (cfg.hidden_size, cfg.norm_eps, format!("model.layers.{i}"));
    let ffn = if cfg.layer_is_moe(i) {
        Ffn::Moe(Box::new(moe(l, cfg, &p)?))
    } else {
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{p}.mlp.{n}"), o, i, false);
        let f = cfg.intermediate_size;
        Ffn::Dense(GatedMlp::swiglu(lin("gate_proj", f, d)?, lin("up_proj", f, d)?, lin("down_proj", d, f)?))
    };
    Ok(Layer {
        mixer: match kind {
            LayerType::FullAttention => Mixer::Attention(gated_attention(l, cfg, &p)?),
            LayerType::LinearAttention => Mixer::Linear(Box::new(delta_net(l, cfg, &p, false)?)),
        },
        norm1: l.rms_norm_checked(&format!("{p}.input_layernorm"), d, eps)?,
        norm2: l.rms_norm_checked(&format!("{p}.post_attention_layernorm"), d, eps)?,
        ffn,
    })
}

impl CausalLm for Qwen35Model {
    const ARCH: &'static str = "qwen35";
    type State = LmCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Qwen35Model> {
        let cfg = Qwen35Config::from_json(meta)?;
        let layers = cfg.layer_types.iter().enumerate().map(|(i, &k)| layer(l, &cfg, i, k)).collect::<Result<_>>()?;
        Ok(Qwen35Model {
            head: lm_head(l, cfg.vocab_size, cfg.hidden_size, !cfg.tie_embeddings)?,
            norm: l.rms_norm_checked("model.norm", cfg.hidden_size, cfg.norm_eps)?,
            layers,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size, hidden: self.cfg.hidden_size, layers: self.layers.len() }
    }

    fn new_state(&self) -> LmCache {
        let n = self.layers.len();
        LmCache::new(KvCache::new(n), 2 * n)
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>", "<|endoftext|>"]
    }

    /// Taps: `embed`, `layers.{i}.op` (mixer output), `layers.{i}`, `norm`.
    fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let (offset, nl) = (st.offset(), self.layers.len());
        let mut x = self.head.embed(ids)?;
        taps.tap("embed", &x);
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.norm1.forward(&x)?;
            let y = match &layer.mixer {
                Mixer::Attention(a) => a.forward(&h, &mut st.kv, i, offset, &Mask::Causal)?,
                Mixer::Linear(dn) => deltanet_step(dn, &h, st, i, nl + i)?,
            };
            let y = taps.steer(Hook::AttnOut, i, &y)?;
            taps.tap(format_args!("layers.{i}.op"), &y);
            x = x.add(&y)?;
            let h = layer.norm2.forward(&x)?;
            let f = match &layer.ffn {
                Ffn::Dense(m) => m.forward(&h)?,
                Ffn::Moe(m) => m.forward(&h)?,
            };
            x = taps.steer(Hook::Resid, i, &x.add(&taps.steer(Hook::FfnOut, i, &f)?)?)?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        logits(&self.head, &x, rows, taps, |x| self.norm.forward(x))
    }
}

/// DeltaNet over `h` with its conv window in slot `conv` and delta-rule state in slot `ssm`.
pub(crate) fn deltanet_step(dn: &DeltaNet, h: &Tensor, st: &mut LmCache, conv: usize, ssm: usize) -> Result<Tensor> {
    let (mut c, mut s) = (st.slot(conv), st.slot(ssm));
    let y = dn.forward(h, &mut c, &mut s)?;
    for (i, t) in [(conv, c), (ssm, s)] {
        if let Some(t) = t {
            st.stage_slot(i, t);
        }
    }
    Ok(y)
}
