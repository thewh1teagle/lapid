//! DeepSeek V4.1 network as lazy lapid graphs (upstream `inference/model.py` `Transformer`,
//! llama.cpp `deepseek41`).
//!
//! ```text
//! x = embed(ids) repeated to hc streams                       [n, hc, D]
//! pre = one-hot(0)
//! per layer:  x = engram(x)                                   (Engram layers)
//!             a = hc_attn.mix(x);  x = a.post(attention(attn_norm(collapse(pre, x))), x)
//!             f = hc_ffn.mix(x);   x = f.post(moe(ffn_norm(collapse(a.pre, x))), x)
//!             pre = f.pre                                     (lagged "single-pass" mHC)
//! logits = head(norm(collapse(pre, x)))                       (no hc_head)
//! ```
//! State ([`State`]): per layer the raw sliding window, per kv-source layer the compressor
//! state and its rows, and the last `N - 1` compressed token ids for the Engram hashes; not
//! rewindable. Tensor names follow the upstream checkpoint (see `convert.rs`).

pub(crate) mod attention;
pub(crate) mod compress;
pub(crate) mod index;
mod load;
pub(crate) mod moe;
mod staged;

use lapid::{DType, Device, Error, Result, Tensor};
use lapid_nn::llm::engram::{EngramGate, Fp8Rows, NgramHash};
use lapid_nn::llm::expert_cache::ExpertCache;
use lapid_nn::llm::hyper::{SinkhornHc, collapse, expand};
use lapid_nn::{Embedding, Linear, Loader, RmsNorm};
use serde_json::Value;

use super::config::Deepseek41Config;
use super::encoding;
use crate::runtime::{CausalLm, ChatOptions, LmDims, LmState, Rows, Taps};
use attention::{Attention, LayerState, Shared};
use moe::Moe;

pub(crate) const EMBED: &str = "embed.weight";
pub(crate) const HEAD: &str = "head.weight";

/// Engram of one layer: its hash module, FP8 table and gate.
struct Engram {
    module: usize,
    table: Fp8Rows,
    gate: EngramGate,
}

struct Layer {
    engram: Option<Engram>,
    attn_hc: SinkhornHc,
    ffn_hc: SinkhornHc,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn: Attention,
    moe: Moe,
}

/// Per-sequence state: committed per-layer states and Engram look-back, plus the ones staged
/// by the last forward (`(layer, slot, tensor)` to realize).
#[derive(Default)]
pub struct State {
    layers: Vec<LayerState>,
    history: Vec<i64>,
    next: Option<(Vec<LayerState>, Vec<i64>)>,
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
        let (mut next, history) = self.next.take().ok_or_else(|| Error::new("internal", "deepseek41: commit without a forward"))?;
        if realized.len() != self.staged.len() {
            self.discard();
            return Err(Error::new("backend", "deepseek41: wrong number of realized state tensors"));
        }
        for ((i, j, _), t) in self.staged.drain(..).zip(realized) {
            if let Some(slot) = next[i].slots_mut().into_iter().nth(j) {
                *slot = Some(t);
            }
        }
        self.layers = next;
        self.history = history;
        self.pos += n_tokens;
        Ok(())
    }

    fn discard(&mut self) {
        self.next = None;
        self.staged.clear();
    }

    fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = LayerState::default());
        self.history.clear();
        self.discard();
        self.pos = 0;
    }
}

pub struct Deepseek41Model {
    pub cfg: Deepseek41Config,
    embed: Embedding,
    head: Linear,
    norm: RmsNorm,
    hash: Option<NgramHash>,
    layers: Vec<Layer>,
    /// Routed experts that do not fit in memory (`None`: graph leaves).
    cache: Option<ExpertCache>,
    device: Device,
    /// Held by a cached forward: one at a time uses the cache's slots.
    lock: std::sync::Mutex<()>,
}

impl Deepseek41Model {
    /// The device expert cache, when the routed experts do not fit in memory.
    pub fn expert_cache(&self) -> Option<&ExpertCache> {
        self.cache.as_ref()
    }

    /// Start reading the Engram rows of `ids` into the page cache before the layers run.
    fn prefetch_engram(&self, ids: &[u32], history: &[i64]) {
        let Some(hash) = &self.hash else { return };
        for e in self.layers.iter().filter_map(|l| l.engram.as_ref()) {
            e.table.prefetch(&hash.rows(e.module, history, ids));
        }
    }

    /// Streams after layer `i`'s Engram: the hashed rows of `ids` gathered from its table.
    fn engram(&self, e: &Engram, x: &Tensor, ids: &[u32], history: &[i64]) -> Result<Tensor> {
        let hash = self.hash.as_ref().ok_or_else(|| Error::new("internal", "deepseek41: engram layer without hash constants"))?;
        let rows = hash.rows(e.module, history, ids);
        let mut v = Vec::with_capacity(rows.len() * e.table.dim());
        e.table.gather(&rows, &mut v)?;
        let rows = Tensor::from_f32(&v, &[ids.len(), hash.cols() * e.table.dim()])?;
        e.gate.forward(x, &rows)
    }
}

/// Stage the layer states of a forward (the slots that changed) and the Engram look-back.
fn stage(st: &mut State, next: Vec<LayerState>, ids: &[u32], hash: Option<&NgramHash>) {
    st.staged.clear();
    for (i, (old, new)) in st.layers.iter().zip(&next).enumerate() {
        for (j, (o, nw)) in old.slots().into_iter().zip(new.slots()).enumerate() {
            if let Some(t) = nw
                && o.is_none_or(|o| o.id() != t.id())
            {
                st.staged.push((i, j, t.clone()));
            }
        }
    }
    let history = hash.map_or_else(Vec::new, |h| h.next_history(&st.history, ids));
    st.next = Some((next, history));
}

impl CausalLm for Deepseek41Model {
    const ARCH: &'static str = "deepseek41";
    type State = State;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Deepseek41Model> {
        load::model(l, meta)
    }

    fn dims(&self) -> LmDims {
        LmDims { vocab: self.cfg.vocab_size, hidden: self.cfg.hidden_size, layers: self.layers.len() }
    }

    fn new_state(&self) -> State {
        State { layers: vec![LayerState::default(); self.layers.len()], ..Default::default() }
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &[encoding::EOS]
    }

    /// V4.1 ships no Jinja template: prompts are the upstream `encoding/encoding.py` format.
    fn render_prompt(&self, messages_json: &str, tools_json: Option<&str>, _add_generation_prompt: bool, opts: &ChatOptions) -> Option<Result<String>> {
        Some(encoding::render_json(messages_json, tools_json, opts.enable_thinking.unwrap_or(false), None))
    }

    /// Taps: `embed`, `layers.{i}.op` (attention block), `layers.{i}` (the hc streams
    /// flattened to `[n, hc * D]`), `norm` (all rows only).
    fn forward(&self, ids: &[u32], st: &mut State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "deepseek41: forward of zero tokens"));
        }
        self.prefetch_engram(ids, &st.history);
        if let Some(cache) = &self.cache {
            return self.forward_cached(cache, ids, st, rows, taps);
        }
        let (n, d, hc) = (ids.len(), self.cfg.hidden_size, self.cfg.hc_mult);
        let pos = st.pos;
        let e = self.embed.lookup(ids)?.cast(DType::F32)?.reshape(&[n, d])?;
        taps.tap("embed", &e);
        let mut x = e.unsqueeze(1)?.expand(&[n, hc, d])?.contiguous()?;
        let mut pre: Option<Tensor> = None;
        let mut shared = Shared::default();
        let mut next = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(eg) = &layer.engram {
                x = self.engram(eg, &x, ids, &st.history)?;
            }
            let am = layer.attn_hc.mix(&x)?;
            let xa = match &pre {
                Some(p) => collapse(p, &x)?,
                None => x.slice(1, 0, 1)?.reshape(&[n, d])?,
            };
            let (a, s) = layer.attn.forward(&layer.attn_norm.forward(&xa)?, pos, &st.layers[i], &mut shared)?;
            next.push(s);
            taps.tap(format_args!("layers.{i}.op"), &a);
            x = expand(&a, &x, &am.post, &am.comb)?;
            let fm = layer.ffn_hc.mix(&x)?;
            let m = layer.moe.forward(&layer.ffn_norm.forward(&collapse(&am.pre, &x)?)?.cast(DType::F32)?)?;
            x = expand(&m, &x, &fm.post, &fm.comb)?;
            pre = Some(fm.pre);
            if taps.enabled() {
                taps.tap(format_args!("layers.{i}"), &x.reshape(&[n, hc * d])?);
            }
        }
        stage(st, next, ids, self.hash.as_ref());
        let pre = pre.ok_or_else(|| Error::new("internal", "deepseek41: no layers"))?;
        let (x, pre) = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => (x.slice(0, n - 1, n)?, pre.slice(0, n - 1, n)?),
            _ => (x, pre),
        };
        let h = self.norm.forward(&collapse(&pre, &x)?)?;
        if rows == Rows::All {
            taps.tap("norm", &h);
        }
        Ok(Some(self.head.forward(&h)?))
    }
}
