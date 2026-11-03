//! Qwen4-Exp network (`Qwen4ExpForCausalLM` text decoder). The residual is `hc` streams
//! `x [1, n, hc, D]` (start: hc copies of the embedding). Layer i:
//!
//! ```text
//! x += ple(x)                                  (the PLE layer only, see `ple`)
//! m = attn_hc.mix(x);  x = combine(x, mixer(m.input), m.weights)
//! m = mlp_hc.mix(x);   x = combine(x, moe(m.input),   m.weights)
//! ```
//! `mixer` is Qwen3.5's Gated DeltaNet (sigmoid output gate) or its gated attention with the
//! QSA indexer; `moe` is Qwen3.5's; the hyper-connections are `lapid_nn::llm::hyper::LowRankHc`. The final mixer (`hyper_connection_mixer`) collapses the
//! streams and replaces the output norm; logits through `lm_head`. File conventions as Qwen3.5
//! (effective-scale norms, grouped `[hc, D]` norms, `A = -exp(A_log)`, tiled v heads).
//!
//! State ([`State`]): `LmCache` (attention KV of layer `i`, raw indexer keys as layer `L + i`;
//! slots: conv `i`, delta-rule `L + i`, PLE conv `2L`) plus the last n-gram tokens (host).

use lapid::{DType, Result, Tensor, bail};
use lapid_nn::llm::delta_net::DeltaNet;
use lapid_nn::llm::head::LmHead;
use lapid_nn::llm::hyper::LowRankHc;
use lapid_nn::llm::moe::SparseMoe;
use lapid_nn::{Conv1d, KvCache, Loader, Rope};
use serde_json::Value;

use super::attn::{Indexer, QsaAttention};
use super::config::{GateAct, Qwen4ExpConfig};
use super::ple::{NgramTable, Ple};
use crate::llama::model::{lm_head, logits};
use crate::qwen35::config::LayerType;
use crate::qwen35::model::{delta_net, deltanet_step, gated_attention, moe};
use crate::runtime::{CausalLm, Hook, LmCache, LmDims, LmState, Rows, Taps};

pub const MIXER: &str = "model.hyper_connection_mixer";

enum Mixer {
    Linear(Box<DeltaNet>),
    Attention(Box<QsaAttention>),
}

struct Layer {
    mixer: Mixer,
    attn_hc: LowRankHc,
    mlp_hc: LowRankHc,
    moe: SparseMoe,
    ple: Option<Box<Ple>>,
}

/// Per-sequence state: the cache plus the PLE n-gram context (not rewindable).
pub struct State {
    cache: LmCache,
    /// Last `ngram - 1` consumed tokens.
    hist: Vec<u32>,
    staged_hist: Option<Vec<u32>>,
}

impl LmState for State {
    fn offset(&self) -> usize {
        self.cache.offset()
    }

    fn staged(&self) -> Vec<&Tensor> {
        self.cache.staged()
    }

    fn commit(&mut self, realized: Vec<Tensor>, n_tokens: usize) -> Result<()> {
        self.cache.commit(realized, n_tokens)?;
        if let Some(h) = self.staged_hist.take() {
            self.hist = h;
        }
        Ok(())
    }

    fn discard(&mut self) {
        self.cache.discard();
        self.staged_hist = None;
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.hist.clear();
        self.staged_hist = None;
    }
}

pub struct Qwen4ExpModel {
    pub cfg: Qwen4ExpConfig,
    head: LmHead,
    layers: Vec<Layer>,
    final_hc: LowRankHc,
}

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    l.tensor_checked(name, shape)?.cast(DType::F32)
}

fn hyper(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str, inject: bool) -> Result<LowRankHc> {
    LowRankHc::load(l, p, [cfg.hc, cfg.hidden_size(), cfg.hc_lowrank], inject, cfg.base.norm_eps)
}

fn attention(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<QsaAttention> {
    let b = &cfg.base;
    let indexer = match &cfg.qsa {
        Some(q) => {
            let ix = format!("{p}.self_attn.indexer");
            let norm = |n: &str| Ok::<_, lapid::Error>(lapid_nn::RmsNorm::new(f32_param(l, &format!("{ix}.{n}.weight"), &[q.head_dim])?, b.norm_eps));
            Some(Indexer {
                q: l.linear_checked(&format!("{ix}.q_proj"), q.n_heads * q.head_dim, b.hidden_size, false)?,
                k: l.linear_checked(&format!("{ix}.k_proj"), q.head_dim, b.hidden_size, false)?,
                q_norm: norm("q_norm")?,
                k_norm: norm("k_norm")?,
                rope: Rope::new(q.head_dim, b.rope_theta).with_rotary_dim(b.rotary_dim()),
                n_head: q.n_heads,
                head_dim: q.head_dim,
                budget: q.budget,
                ratio: q.ratio,
            })
        }
        None => None,
    };
    Ok(QsaAttention { attn: gated_attention(l, b, p)?, indexer })
}

fn ple(l: &Loader<'_>, cfg: &Qwen4ExpConfig, p: &str) -> Result<Ple> {
    let Some(pc) = cfg.ple.clone() else { bail!("internal", "qwen4exp: PLE layer without PLE config") };
    let (hc, d, q) = (cfg.hc, cfg.hidden_size(), format!("{p}.ple"));
    // host only: rows are gathered on the CPU (see `ple`)
    let table = l.file.get(&format!("{q}.ngram_embedding.weight"))?;
    if table.shape() != [pc.rows, pc.row_dim] {
        bail!("shape_mismatch", "{q}.ngram_embedding.weight: {:?}, expected [{}, {}]", table.shape(), pc.rows, pc.row_dim);
    }
    let lin = |n: &str, o: usize, i: usize| l.linear_checked(&format!("{q}.{n}"), o, i, false);
    let grouped = |n: &str| f32_param(l, &format!("{q}.{n}.weight"), &[hc, d]);
    Ok(Ple {
        table: NgramTable { table, rows: pc.rows, cols: pc.row_dim },
        key: lin("key_proj", hc * d, pc.embed_dim())?,
        value: lin("value_proj", d, pc.embed_dim())?,
        norm_key: grouped("norm_key")?,
        norm_query: grouped("norm_query")?,
        norm_conv: grouped("norm_conv")?,
        conv: Conv1d::depthwise(f32_param(l, &format!("{q}.conv1d.weight"), &[hc * d, 1, pc.conv_kernel])?, None).with_dilation(pc.ngram),
        hc,
        eps: cfg.base.norm_eps,
        cfg: pc,
    })
}

impl CausalLm for Qwen4ExpModel {
    const ARCH: &'static str = "qwen4exp";
    type State = State;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Qwen4ExpModel> {
        let cfg = Qwen4ExpConfig::from_json(meta)?;
        let mut layers = Vec::with_capacity(cfg.n_layers());
        for (i, &kind) in cfg.base.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                mixer: match kind {
                    LayerType::FullAttention => Mixer::Attention(Box::new(attention(l, &cfg, &p)?)),
                    LayerType::LinearAttention => Mixer::Linear(Box::new(delta_net(l, &cfg.base, &p, cfg.output_gate == GateAct::Sigmoid)?)),
                },
                attn_hc: hyper(l, &cfg, &format!("{p}.attn_hyper_connection"), true)?,
                mlp_hc: hyper(l, &cfg, &format!("{p}.mlp_hyper_connection"), true)?,
                moe: moe(l, &cfg.base, &p)?,
                ple: if cfg.ple.as_ref().is_some_and(|q| q.layer == i) { Some(Box::new(ple(l, &cfg, &p)?)) } else { None },
            });
        }
        Ok(Qwen4ExpModel {
            head: lm_head(l, cfg.vocab_size(), cfg.hidden_size(), true)?,
            final_hc: hyper(l, &cfg, MIXER, false)?,
            layers,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size(), hidden: self.cfg.hidden_size(), layers: self.layers.len() }
    }

    fn new_state(&self) -> State {
        let n = self.layers.len();
        State { cache: LmCache::new(KvCache::new(2 * n), 2 * n + 1), hist: Vec::new(), staged_hist: None }
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>", "<|endoftext|>"]
    }

    /// Taps: `embed`, `layers.{i}.ple` (injection), `layers.{i}.op` (mixer output),
    /// `layers.{i}` (hc-wide residual), `norm` (final mixer).
    fn forward(&self, ids: &[u32], st: &mut State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let (n, d, hc, nl) = (ids.len(), self.cfg.hidden_size(), self.cfg.hc, self.layers.len());
        let offset = st.offset();
        let e = self.head.embed(ids)?;
        taps.tap("embed", &e);
        let mut x = e.reshape(&[1, n, 1, d])?.expand(&[1, n, hc, d])?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &layer.ple {
                let emb = ple.embed(&st.hist, ids)?;
                let inj = st.cache.with_slot(2 * nl, |c| ple.forward(&x, &emb, c))?;
                taps.tap(format_args!("layers.{i}.ple"), &inj);
                x = x.add(&inj)?;
            }
            x = layer.attn_hc.around(&x, |h| {
                let y = match &layer.mixer {
                    Mixer::Attention(a) => a.forward(h, &mut st.cache.kv, i, nl + i, offset)?,
                    Mixer::Linear(dn) => deltanet_step(dn, h, &mut st.cache, i, nl + i)?,
                };
                let y = taps.steer(Hook::AttnOut, i, &y)?;
                taps.tap(format_args!("layers.{i}.op"), &y);
                Ok(y)
            })?;
            x = layer.mlp_hc.around(&x, |h| taps.steer(Hook::FfnOut, i, &layer.moe.forward(h)?))?;
            x = taps.steer(Hook::Resid, i, &x)?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        if let Some(p) = &self.cfg.ple {
            let mut h: Vec<u32> = st.hist.iter().chain(ids).copied().collect();
            h.drain(..h.len().saturating_sub(p.ngram - 1));
            st.staged_hist = Some(h);
        }
        logits(&self.head, &x, rows, taps, |x| Ok(self.final_hc.mix(x)?.input))
    }
}
