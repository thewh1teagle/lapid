//! DeepSeek V4 attention (HF `DeepseekV4Attention`; llama.cpp `build_attention_impl`):
//! shared-KV MQA with a sliding window over raw tokens, plus compressed long-range entries.
//!
//! ```text
//! qr = rmsnorm(x Wq_a^T)                      low-rank query
//! q  = rope(rmsnorm_unweighted(qr Wq_b^T))    [n, H, 512]
//! kv = rope(rmsnorm(x Wkv^T))                 one head, K = V
//! keys = raw window (last `sliding_window` tokens) ++ compressed entries
//!     sliding layer: raw only
//!     HCA (ratio 128): every closed window
//!     CSA (ratio 4):   the lightning indexer's top-k closed windows per query
//! o  = rope^-1(softmax([q k^T * scale + mask, sink_h]) v)   (sink column dropped)
//! out = Wo_b (grouped Wo_a (o))               groups of H / G heads, low-rank each
//! ```
//! The attention itself is a composition (matmul + mask + per-head sink column + softmax):
//! the head dim is 512 and the sink needs a per-head extra logit, which the fused attention
//! kernels do not take (see the round-5 notes, "Requests").

use lapid::{DType, Result, Tensor};
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Linear, RmsNorm, ops};

use super::compress::{CompState, Compressor, NEG};
use lapid_nn::llm::rope::TailRope;

/// Lightning indexer (HF `DeepseekV4Indexer`, llama.cpp `LIGHTNING_INDEXER` + `top_k`):
/// `score[t, s] = sum_h w[t, h] relu(q[t, h] . k_s)`, causal, top-k → attention mask.
pub(crate) struct Indexer {
    pub wq_b: Linear,
    pub weights_proj: Linear,
    pub comp: Compressor,
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
}

#[derive(Clone, Default)]
pub(crate) struct LayerState {
    /// Raw K = V rows of the last `sliding_window - 1` tokens `[m, head_dim]`.
    pub raw: Option<Tensor>,
    pub comp: CompState,
    pub idx: CompState,
}

impl LayerState {
    pub fn slots_mut(&mut self) -> Vec<&mut Option<Tensor>> {
        let mut v = vec![&mut self.raw];
        v.extend(self.comp.slots_mut());
        v.extend(self.idx.slots_mut());
        v
    }

    pub fn slots(&self) -> Vec<&Option<Tensor>> {
        let mut v = vec![&self.raw];
        v.extend(self.comp.slots());
        v.extend(self.idx.slots());
        v
    }
}

pub(crate) struct Attention {
    pub wq_a: Linear,
    pub q_norm: RmsNorm,
    pub wq_b: Linear,
    pub wkv: Linear,
    pub kv_norm: RmsNorm,
    /// Grouped low-rank output `[G, o_lora, H / G * head_dim]`.
    pub wo_a: Tensor,
    pub wo_a_act_quant: bool,
    pub wo_b: Linear,
    /// Per-head sink logits `[1, H, 1]` f32.
    pub sinks: Tensor,
    pub comp: Option<Compressor>,
    pub indexer: Option<Indexer>,
    /// Main RoPE (sliding layers) or the compressed-layer RoPE (theta + YaRN).
    pub rope: TailRope,
    /// Indexer / compressor RoPE (always the compressed-layer one).
    pub comp_rope: TailRope,
    pub ones: Tensor,
    pub n_head: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub window: usize,
    pub ratio: usize,
    pub eps: f32,
}

/// `[n, len]` host mask: 0 where `visible(query position, key index)`, else NEG.
pub(crate) fn host_mask(n: usize, len: usize, visible: impl Fn(usize, usize) -> bool) -> Result<Tensor> {
    let mut m = Vec::with_capacity(n * len);
    for i in 0..n {
        for j in 0..len {
            m.push(if visible(i, j) { 0.0 } else { NEG });
        }
    }
    Tensor::from_f32(&m, &[n, len])
}

impl Indexer {
    /// Attention mask `[n, T]` over the CSA entries: the top-k of the indexer scores among
    /// the causally visible entries (`(w + 1) ratio <= t + 1`).
    fn select(&self, x: &Tensor, qr: &Tensor, entries: &Tensor, pos: usize, rope: &TailRope, ratio: usize) -> Result<Tensor> {
        let n = x.shape()[0];
        let t = entries.shape()[0];
        let (h, d) = (self.heads, self.dim);
        let positions: Vec<usize> = (pos..pos + n).collect();
        let (cos, sin) = rope.tables(&positions)?;
        let q = self.wq_b.forward(qr)?.cast(DType::F32)?.reshape(&[n, h, d])?;
        let q = rope.apply(&q, &cos, &sin, false)?.reshape(&[n * h, d])?;
        let w = self.weights_proj.forward(x)?.cast(DType::F32)?.mul_scalar(1.0 / ((d * h) as f64).sqrt())?; // [n, h]
        let s = ops::relu(&q.matmul_t(entries)?)?.reshape(&[n, h, t])?;
        let s = s.mul(&w.unsqueeze(2)?)?.sum(1, false)?; // [n, T]
        let visible = |i: usize, j: usize| (j + 1) * ratio <= pos + i + 1;
        let causal = host_mask(n, t, visible)?;
        let k = self.topk.min(t);
        let top = s.add(&causal)?.topk_indices(k, 1)?; // [n, k]
        let iota: Vec<f32> = (0..t).map(|j| j as f32).collect();
        let iota = Tensor::from_f32(&iota, &[1, 1, t])?;
        let hit = top.cast(DType::F32)?.reshape(&[n, k, 1])?.binary(lapid::BinaryOp::CmpEq, &iota)?.sum(1, false)?; // [n, T]
        let valid: Vec<f32> = (0..n).flat_map(|i| (0..t).map(move |j| if visible(i, j) { 1.0 } else { 0.0 })).collect();
        let keep = hit.mul(&Tensor::from_f32(&valid, &[n, t])?)?;
        Tensor::where_(&keep, &Tensor::full(0.0, &[n, t], DType::F32), &Tensor::full(f64::from(NEG), &[n, t], DType::F32))
    }
}

/// Attention with per-head sinks over shared K = V rows: `q [n, H, hd]`, `keys [S, hd]`,
/// additive `mask [n, S]`, `sinks [1, H, 1]` -> `[n, H, hd]`. `[n, H, S]` logits plus the sink
/// column, softmax, the sink dropped.
pub(crate) fn sink_attention(q: &Tensor, keys: &Tensor, mask: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    let (n, h, hd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let s = keys.shape()[0];
    let logits = q.reshape(&[n * h, hd])?.matmul_t(keys)?.reshape(&[n, h, s])?;
    let logits = logits.mul_scalar(1.0 / (hd as f64).sqrt())?.add(&mask.unsqueeze(1)?)?;
    let sink = sinks.expand(&[n, h, 1])?;
    let p = Tensor::cat(&[&logits, &sink], 2)?.softmax()?.slice(2, 0, s)?;
    p.reshape(&[n * h, s])?.matmul(keys)?.reshape(&[n, h, hd])
}

/// Grouped low-rank output projection `o [n, H, hd]` -> `[n, D]`: one MatMulId "expert" of
/// `wo_a [G, o_lora, H / G * hd]` per group, then `wo_b`.
pub(crate) fn grouped_output(o: &Tensor, wo_a: &Tensor, act_quant: bool, wo_b: &Linear, g: usize) -> Result<Tensor> {
    let (n, h, hd) = (o.shape()[0], o.shape()[1], o.shape()[2]);
    let rows = o.reshape(&[n * g, h / g * hd])?;
    let rows = if act_quant { fake_quant_act(&rows)? } else { rows };
    let ids: Vec<i32> = (0..n * g).map(|i| (i % g) as i32).collect();
    let oa = rows.matmul_id(wo_a, &Tensor::from_i32(&ids, &[n * g, 1])?)?; // [n*g, 1, o_lora]
    let ol = oa.shape()[2];
    wo_b.forward(&oa.reshape(&[n, g * ol])?)?.cast(DType::F32)
}

impl Attention {
    /// `x [n, D]` (normed block input), absolute positions `pos..pos + n`. Returns the block
    /// output `[n, D]` and the layer's next state.
    pub fn forward(&self, x: &Tensor, pos: usize, st: &LayerState) -> Result<(Tensor, LayerState)> {
        let n = x.shape()[0];
        let (h, hd, g) = (self.n_head, self.head_dim, self.groups);
        let positions: Vec<usize> = (pos..pos + n).collect();
        let (cos, sin) = self.rope.tables(&positions)?;

        let qr = self.q_norm.forward(&self.wq_a.forward(x)?)?.cast(DType::F32)?;
        let q = self.wq_b.forward(&qr)?.cast(DType::F32)?.reshape(&[n, h, hd])?;
        let q = RmsNorm::new(self.ones.clone(), self.eps).forward(&q)?;
        let q = self.rope.apply(&q, &cos, &sin, false)?;
        let kv = self.kv_norm.forward(&self.wkv.forward(x)?)?.cast(DType::F32)?.reshape(&[n, 1, hd])?;
        let kv = self.rope.apply(&kv, &cos, &sin, false)?.reshape(&[n, hd])?;

        // raw sliding window
        let m = st.raw.as_ref().map_or(0, |r| r.shape()[0]);
        let raw = match &st.raw {
            Some(r) => Tensor::cat(&[r, &kv], 0)?,
            None => kv,
        };
        let s_raw = m + n;
        let first = pos - m; // absolute position of raw row 0
        let win = self.window;
        let raw_mask = host_mask(n, s_raw, |i, j| {
            let (qp, kp) = (pos + i, first + j);
            kp <= qp && qp - kp < win
        })?;
        let mut next = st.clone();
        let keep = (win - 1).min(s_raw);
        next.raw = if keep > 0 { Some(raw.slice(0, s_raw - keep, s_raw)?) } else { None };

        // compressed entries
        let (keys, mask) = match &self.comp {
            None => (raw, raw_mask),
            Some(c) => {
                let (entries, cs) = c.step(x, pos, &self.comp_rope, &st.comp)?;
                next.comp = cs;
                let comp_mask = match (&self.indexer, &entries) {
                    (Some(ix), Some(e)) => {
                        let (ie, is) = ix.comp.step(x, pos, &self.comp_rope, &st.idx)?;
                        next.idx = is;
                        match ie {
                            Some(ie) => Some(ix.select(x, &qr, &ie, pos, &self.comp_rope, self.ratio)?),
                            None => None,
                        }
                        .map(|mk| (e.clone(), mk))
                    }
                    (Some(ix), None) => {
                        let (_, is) = ix.comp.step(x, pos, &self.comp_rope, &st.idx)?;
                        next.idx = is;
                        None
                    }
                    (None, Some(e)) => {
                        let t = e.shape()[0];
                        let r = self.ratio;
                        Some((e.clone(), host_mask(n, t, |i, j| (j + 1) * r <= pos + i + 1)?))
                    }
                    (None, None) => None,
                };
                match comp_mask {
                    Some((e, mk)) => (Tensor::cat(&[&raw, &e], 0)?, Tensor::cat(&[&raw_mask, &mk], 1)?),
                    None => (raw, raw_mask),
                }
            }
        };

        let o = sink_attention(&q, &keys, &mask, &self.sinks)?;
        let o = self.rope.apply(&o, &cos, &sin, true)?;
        let out = grouped_output(&o, &self.wo_a, self.wo_a_act_quant, &self.wo_b, g)?;
        Ok((out, next))
    }
}
