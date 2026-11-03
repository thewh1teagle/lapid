//! DeepSeek V4 network as lazy lapid graphs (HF `DeepseekV4ForCausalLM`, llama.cpp `deepseek4`).
//!
//! ```text
//! x = embed(ids) repeated to hc streams                 [n, hc, D]
//! per layer:  x = hc_attn.post(attention(attn_norm(hc_attn.pre(x))), x)
//!             x = hc_ffn.post(moe(ffn_norm(hc_ffn.pre(x))), x)
//! logits = head(norm(hc_head(x)))
//! ```
//! State ([`State`]): per layer the raw sliding window and the compressor / indexer states
//! (see `compress.rs`); not rewindable. Tensor names follow the upstream checkpoints
//! (`layers.{i}.ffn.experts.w1` `[E, I, D]` stacks, `attn.wo_a.weight` `[G, o_lora, H / G *
//! head_dim]`, ...; see `convert.rs`).

pub(crate) mod attention;
pub(crate) mod compress;
pub(crate) mod moe;

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::llm::hyper::{SinkhornHc, SinkhornHead};
use lapid_nn::llm::moe::Experts;
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Embedding, Linear, Loader, RmsNorm};
use serde_json::Value;

use super::config::Deepseek4Config;
use super::encoding;
use crate::runtime::{CausalLm, ChatOptions, Hook, LmDims, LmState, Rows, Taps};
use attention::{Attention, Indexer, LayerState};
use compress::Compressor;
use moe::Moe;

pub(crate) const EMBED: &str = "embed.weight";
pub(crate) const HEAD: &str = "head.weight";

struct Layer {
    attn_hc: SinkhornHc,
    ffn_hc: SinkhornHc,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn: Attention,
    moe: Moe,
}

/// Per-sequence state: committed per-layer states, plus the ones staged by the last forward
/// (`(layer, slot, tensor)` to realize).
#[derive(Default)]
pub struct State {
    layers: Vec<LayerState>,
    next: Option<Vec<LayerState>>,
    staged: Vec<(usize, usize, Tensor)>,
    pos: usize,
}

impl LmState for State {
    fn offset(&self) -> usize {
        self.pos
    }

    fn staged(&self) -> Vec<&Tensor> {
        self.staged.iter().map(|(_, _, t)| t).collect()
    }

    fn commit(&mut self, realized: Vec<Tensor>, n_tokens: usize) -> Result<()> {
        let mut next = self.next.take().ok_or_else(|| Error::new("internal", "deepseek4: commit without a forward"))?;
        if realized.len() != self.staged.len() {
            self.discard();
            return Err(Error::new("backend", "deepseek4: wrong number of realized state tensors"));
        }
        for ((i, j, _), t) in self.staged.drain(..).zip(realized) {
            if let Some(slot) = next[i].slots_mut().into_iter().nth(j) {
                *slot = Some(t);
            }
        }
        self.layers = next;
        self.pos += n_tokens;
        Ok(())
    }

    fn discard(&mut self) {
        self.next = None;
        self.staged.clear();
    }

    fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = LayerState::default());
        self.discard();
        self.pos = 0;
    }
}

pub struct Deepseek4Model {
    pub cfg: Deepseek4Config,
    embed: Embedding,
    head: Linear,
    norm: RmsNorm,
    hc_head: SinkhornHead,
    layers: Vec<Layer>,
}

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    l.tensor_checked(name, shape)?.cast(DType::F32)
}

fn load_compressor(l: &Loader<'_>, p: &str, d: usize, ratio: usize, hd: usize, eps: f32) -> Result<Compressor> {
    let c = Deepseek4Config::compress_coff(ratio) * hd;
    Ok(Compressor {
        wkv: l.linear_checked(&format!("{p}.wkv"), c, d, false)?,
        wgate: l.linear_checked(&format!("{p}.wgate"), c, d, false)?,
        ape: f32_param(l, &format!("{p}.ape"), &[ratio, c])?,
        norm: l.rms_norm_checked(&format!("{p}.norm"), hd, eps)?,
        ratio,
        head_dim: hd,
        overlap: ratio == 4,
    })
}

fn load_attention(l: &Loader<'_>, cfg: &Deepseek4Config, i: usize) -> Result<Attention> {
    let p = format!("layers.{i}.attn");
    let (d, h, hd, eps) = (cfg.hidden_size, cfg.n_head, cfg.head_dim, cfg.norm_eps);
    let (ql, g, ol) = (cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank);
    let lin = |n: &str, o: usize, inp: usize| l.linear_checked(&format!("{p}.{n}"), o, inp, false);
    let ratio = cfg.compress_ratios[i];
    let comp_rope = TailRope::new(cfg.rope_dim, cfg.compress_rope_theta, cfg.yarn.as_ref());
    let rope = if ratio == 0 { TailRope::new(cfg.rope_dim, cfg.rope_theta, None) } else { comp_rope.clone() };
    let comp = if ratio > 0 { Some(load_compressor(l, &format!("{p}.compressor"), d, ratio, hd, eps)?) } else { None };
    let indexer = if ratio == 4 {
        let (ih, id) = (cfg.index_n_heads, cfg.index_head_dim);
        Some(Indexer {
            wq_b: lin("indexer.wq_b", ih * id, ql)?,
            weights_proj: lin("indexer.weights_proj", ih, d)?,
            comp: load_compressor(l, &format!("{p}.indexer.compressor"), d, ratio, id, eps)?,
            heads: ih,
            dim: id,
            topk: cfg.index_topk,
        })
    } else {
        None
    };
    let wo_a = format!("{p}.wo_a.weight");
    Ok(Attention {
        wq_a: lin("wq_a", ql, d)?,
        q_norm: l.rms_norm_checked(&format!("{p}.q_norm"), ql, eps)?,
        wq_b: lin("wq_b", h * hd, ql)?,
        wkv: lin("wkv", hd, d)?,
        kv_norm: l.rms_norm_checked(&format!("{p}.kv_norm"), hd, eps)?,
        wo_a: l.tensor_checked(&wo_a, &[g, ol, h / g * hd])?,
        wo_a_act_quant: l.act_quant(&wo_a)?,
        wo_b: lin("wo_b", d, g * ol)?,
        sinks: f32_param(l, &format!("{p}.attn_sink"), &[h])?.reshape(&[1, h, 1])?,
        comp,
        indexer,
        rope,
        comp_rope,
        ones: l.device.upload(&Tensor::full(1.0, &[hd], DType::F32))?,
        n_head: h,
        head_dim: hd,
        groups: g,
        window: cfg.sliding_window,
        ratio,
        eps,
    })
}

/// Hash routing table `[vocab, k]`: i32 (round-5 files) or f32 (exact small integers).
fn tid2eid(l: &Loader<'_>, name: &str, shape: [usize; 2]) -> Result<Vec<i32>> {
    let t = l.file.get(name)?;
    let b = l.file.bytes(name)?;
    let v: Vec<i32> = match t.dtype() {
        DType::I32 => b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect(),
        DType::F32 => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c) as i32).collect(),
        other => return Err(Error::new("format", format!("{name}: hash table dtype {}", other.name()))),
    };
    if t.shape() != shape {
        return Err(Error::new("format", format!("{name}: expected [{}, {}], got {:?}", shape[0], shape[1], t.shape())));
    }
    Ok(v)
}

fn load_moe(l: &Loader<'_>, cfg: &Deepseek4Config, i: usize) -> Result<Moe> {
    let p = format!("layers.{i}.ffn");
    let (d, e, inter, k) = (cfg.hidden_size, cfg.n_experts, cfg.moe_intermediate, cfg.top_k);
    let hash = i < cfg.n_hash_layers;
    let stacks = ["w1", "w3", "w2"].map(|w| format!("{p}.experts.{w}"));
    let shared = |w: &str, o: usize, inp: usize| l.linear_checked(&format!("{p}.shared_experts.{w}"), o, inp, false);
    Ok(Moe {
        router: f32_param(l, &format!("{p}.gate.weight"), &[e, d])?,
        bias: if hash { None } else { Some(f32_param(l, &format!("{p}.gate.bias"), &[e])?) },
        tid2eid: if hash { Some(tid2eid(l, &format!("{p}.gate.tid2eid"), [cfg.vocab_size, k])?) } else { None },
        experts: Experts::load(l, stacks.each_ref().map(String::as_str), e, inter, d)?.with_limit(cfg.swiglu_limit),
        shared: [shared("w1", inter, d)?, shared("w3", inter, d)?, shared("w2", d, inter)?],
        top_k: k,
        n_experts: e,
        route_scale: cfg.route_scale,
    })
}

impl CausalLm for Deepseek4Model {
    const ARCH: &'static str = "deepseek4";
    type State = State;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Deepseek4Model> {
        let cfg = Deepseek4Config::from_json(meta)?;
        let (d, eps, hc) = (cfg.hidden_size, cfg.norm_eps, cfg.hc_mult);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let conn = |kind: &str| SinkhornHc::load(l, &format!("{p}.hc_{kind}"), [hc, d, cfg.hc_sinkhorn_iters], cfg.hc_eps, eps);
            layers.push(Layer {
                attn_hc: conn("attn")?,
                ffn_hc: conn("ffn")?,
                attn_norm: l.rms_norm_checked(&format!("{p}.attn_norm"), d, eps)?,
                ffn_norm: l.rms_norm_checked(&format!("{p}.ffn_norm"), d, eps)?,
                attn: load_attention(l, &cfg, i)?,
                moe: load_moe(l, &cfg, i)?,
            });
        }
        Ok(Deepseek4Model {
            embed: Embedding::new(l.tensor_checked(EMBED, &[cfg.vocab_size, d])?),
            head: l.linear_checked("head", cfg.vocab_size, d, false)?,
            norm: l.rms_norm_checked("norm", d, eps)?,
            hc_head: SinkhornHead::load(l, "hc_head", hc, d, cfg.hc_eps, eps)?,
            layers,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size, hidden: self.cfg.hidden_size, layers: self.layers.len() }
    }

    fn new_state(&self) -> State {
        State { layers: vec![LayerState::default(); self.layers.len()], ..Default::default() }
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &[encoding::EOS]
    }

    /// V4 ships no Jinja template: prompts are the upstream `encoding_dsv4` format.
    fn render_prompt(&self, messages_json: &str, tools_json: Option<&str>, _add_generation_prompt: bool, opts: &ChatOptions) -> Option<Result<String>> {
        Some(encoding::render_json(messages_json, tools_json, opts.enable_thinking.unwrap_or(false)))
    }

    /// Taps: `embed`, `layers.{i}.op` (attention block), `layers.{i}` (the hc streams
    /// flattened to `[n, hc * D]`), `norm` (all rows only).
    fn forward(&self, ids: &[u32], st: &mut State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "deepseek4: forward of zero tokens"));
        }
        let (n, d, hc) = (ids.len(), self.cfg.hidden_size, self.cfg.hc_mult);
        let pos = st.pos;
        let e = self.embed.lookup(ids)?.cast(DType::F32)?.reshape(&[n, d])?;
        taps.tap("embed", &e);
        let mut x = e.unsqueeze(1)?.expand(&[n, hc, d])?.contiguous()?;
        let mut next = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.attn_hc.pre(&x)?;
            let (a, s) = layer.attn.forward(&layer.attn_norm.forward(&pre.collapsed)?, pos, &st.layers[i])?;
            next.push(s);
            let a = taps.steer(Hook::AttnOut, i, &a)?;
            taps.tap(format_args!("layers.{i}.op"), &a);
            x = layer.attn_hc.post(&a, &x, &pre)?;
            let pre = layer.ffn_hc.pre(&x)?;
            let m = layer.moe.forward(&layer.ffn_norm.forward(&pre.collapsed)?, ids)?;
            x = taps.steer(Hook::Resid, i, &layer.ffn_hc.post(&taps.steer(Hook::FfnOut, i, &m)?, &x, &pre)?)?;
            if taps.enabled() {
                taps.tap(format_args!("layers.{i}"), &x.reshape(&[n, hc * d])?);
            }
        }
        st.staged.clear();
        for (i, (old, new)) in st.layers.iter().zip(&next).enumerate() {
            for (j, (o, nw)) in old.slots().into_iter().zip(new.slots()).enumerate() {
                if let Some(t) = nw
                    && o.as_ref().is_none_or(|o| o.id() != t.id())
                {
                    st.staged.push((i, j, t.clone()));
                }
            }
        }
        st.next = Some(next);
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(0, n - 1, n)?,
            _ => x,
        };
        let h = self.norm.forward(&self.hc_head.forward(&x)?)?;
        if rows == Rows::All {
            taps.tap("norm", &h);
        }
        Ok(Some(self.head.forward(&h)?))
    }
}
