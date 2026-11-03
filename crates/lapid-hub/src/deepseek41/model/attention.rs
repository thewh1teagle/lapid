//! DeepSeek V4.1 attention (upstream `Attention`; llama.cpp `build_attention41`): shared-KV
//! MQA over a sliding window of raw tokens plus selected compressed rows.
//!
//! ```text
//! qr = rmsnorm(x Wq_a^T)                      low-rank query
//! q  = rope(qr Wq_b^T)                        [n, H, 512]   (no per-head q norm, unlike V4)
//! kv = rope(rmsnorm(x Wkv^T))                 one head, K = V
//! keys = raw window ++ the compressed rows the indexer selected (every compressed layer)
//! o  = rope^-1(softmax([q k^T * scale + mask, sink_h]) v)   (sink column dropped)
//! out = Wo_b (grouped Wo_a (o))
//! ```
//! Compressed rows and selections are shared down the stack ([`Shared`]): a kv-source layer
//! compresses (`compress.rs`), the compressed layers after it read its rows; an index-source
//! layer selects (`index.rs`), the compressed layers after it reuse the selection.

use lapid::{DType, Result, Tensor};
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Linear, RmsNorm};

use super::compress::{CompState, Compressed, Compressor};
use super::index::{Candidates, Indexer, Selection, visible_lens};
use crate::deepseek4::model::attention::{grouped_output, host_mask};

#[derive(Clone, Default)]
pub(crate) struct LayerState {
    /// Raw K = V rows of the last `sliding_window - 1` tokens `[m, head_dim]`.
    pub raw: Option<Tensor>,
    /// kv-source layers: the compressor state and its rows.
    pub comp: CompState,
}

impl LayerState {
    /// Drop the state of the last `drop` tokens (now at `len` tokens): raw rows kept beyond
    /// the window by a rewindable forward, and the compressor's rows / open group.
    pub fn truncate(&mut self, drop: usize, len: usize, ratio: usize) -> Result<bool> {
        if let Some(r) = &self.raw {
            let m = r.shape()[0];
            self.raw = if m > drop { Some(r.slice(0, 0, m - drop)?) } else { None };
        }
        if ratio > 0 { self.comp.truncate(len, ratio) } else { Ok(true) }
    }

    pub fn slots(&self) -> Vec<Option<&Tensor>> {
        let mut v = vec![self.raw.as_ref()];
        v.extend(self.comp.slots());
        v
    }

    pub fn slots_mut(&mut self) -> Vec<&mut Option<Tensor>> {
        let mut v = vec![&mut self.raw];
        v.extend(self.comp.slots_mut());
        v
    }
}

/// What the attention layers of one forward hand down the stack (upstream
/// `SharedAttentionRuntime`): the latest kv source's rows, index source's selection and the
/// candidate source's blocks.
#[derive(Default)]
pub(crate) struct Shared {
    pub comp: Option<Compressed>,
    pub selection: Option<Selection>,
    pub candidates: Option<Tensor>,
}

impl Shared {
    /// Every tensor held (to realize them between evaluations).
    pub fn tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v = Vec::new();
        if let Some(c) = &mut self.comp {
            v.extend([&mut c.rows, &mut c.keys]);
        }
        match &mut self.selection {
            Some(Selection::Mask(m)) => v.push(m),
            Some(Selection::Gather { idx, mask }) => v.extend([idx, mask]),
            Some(Selection::Visible) | None => {}
        }
        v.extend(self.candidates.as_mut());
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
    /// kv-source layers.
    pub comp: Option<Compressor>,
    /// Index-source layers.
    pub indexer: Option<Indexer>,
    /// The candidate source layer.
    pub candidates: Option<Candidates>,
    /// Index source after the candidate source: selects inside its blocks.
    pub uses_candidates: bool,
    /// Main RoPE (sliding layers) or the compressed-layer RoPE (theta + YaRN).
    pub rope: TailRope,
    pub n_head: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub window: usize,
    pub ratio: usize,
}

/// One set of keys of [`sink_attention`]: shared rows `[S, hd]`, per-query rows `[n, k, hd]`,
/// or per-block rows `[nb, k, hd]` shared by each block of `block` consecutive queries
/// (a prompt chunk's sliding window), with an additive mask `[n, S]` / `[n, k]`.
struct Keys {
    rows: Tensor,
    mask: Tensor,
    block: usize,
}

/// `[n, h, x]` viewed as `[nb, block * h, x]` (queries padded to whole blocks) and back.
fn to_blocks(t: &Tensor, block: usize) -> Result<Tensor> {
    let (n, h, x) = (t.shape()[0], t.shape()[1], t.shape()[2]);
    let nb = n.div_ceil(block);
    t.pad(0, 0, nb * block - n, 0.0)?.reshape(&[nb, block * h, x])
}

fn from_blocks(t: &Tensor, n: usize, h: usize) -> Result<Tensor> {
    let (nb, bh, x) = (t.shape()[0], t.shape()[1], t.shape()[2]);
    t.reshape(&[nb * bh / h, h, x])?.slice(0, 0, n)
}

/// Attention with per-head sinks over several key sets (K = V): `q [n, H, hd]` -> `[n, H, hd]`.
fn sink_attention(q: &Tensor, parts: &[Keys], sinks: &Tensor) -> Result<Tensor> {
    let (n, h, hd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut logits = Vec::with_capacity(parts.len() + 1);
    for k in parts {
        let l = if k.block > 0 {
            from_blocks(&to_blocks(q, k.block)?.matmul_t(&k.rows)?, n, h)?
        } else if k.rows.rank() == 2 {
            q.reshape(&[n * h, hd])?.matmul_t(&k.rows)?.reshape(&[n, h, k.rows.shape()[0]])?
        } else {
            q.matmul_t(&k.rows)? // [n, h, k]
        };
        logits.push(l.mul_scalar(scale)?.add(&k.mask.unsqueeze(1)?)?);
    }
    logits.push(sinks.expand(&[n, h, 1])?);
    let p = Tensor::cat(&logits.iter().collect::<Vec<_>>(), 2)?.softmax()?;
    let mut out: Option<Tensor> = None;
    let mut at = 0;
    for k in parts {
        let len = k.mask.shape()[1];
        let pk = p.slice(2, at, at + len)?;
        at += len;
        let o = if k.block > 0 {
            from_blocks(&to_blocks(&pk, k.block)?.matmul(&k.rows)?, n, h)?
        } else if k.rows.rank() == 2 {
            pk.reshape(&[n * h, len])?.matmul(&k.rows)?.reshape(&[n, h, hd])?
        } else {
            pk.matmul(&k.rows)?
        };
        out = Some(match out {
            Some(acc) => acc.add(&o)?,
            None => o,
        });
    }
    out.ok_or_else(|| lapid::Error::new("internal", "deepseek41 attention: no keys"))
}

impl Attention {
    /// The compressed keys of this layer's queries (`None`: window only), updating `shared`
    /// (and the kv source's state in `next`).
    fn compressed_keys(&self, x: &Tensor, qr: &Tensor, pos: usize, st: &LayerState, next: &mut LayerState, shared: &mut Shared) -> Result<Option<Keys>> {
        let n = x.shape()[0];
        let r = self.ratio;
        if let Some(c) = &self.comp {
            let (rows, cs) = c.step(x, pos, &self.rope, &st.comp, n <= super::REWIND_MAX)?;
            next.comp = cs;
            shared.comp = rows;
        }
        let total = (pos + n) / r;
        let Some(comp) = shared.comp.clone().filter(|_| total > 0) else {
            if self.indexer.is_some() {
                shared.selection = None;
            }
            return Ok(None);
        };
        let lens = visible_lens(pos, n, r);
        if let Some(ix) = &self.indexer {
            let score = ix.scores(x, qr, &comp.keys, pos, &lens, &self.rope)?;
            if let Some(c) = &self.candidates {
                shared.candidates = Some(c.select(&score, &lens)?);
            }
            let cand = if self.uses_candidates { shared.candidates.as_ref() } else { None };
            shared.selection = Some(ix.select(&score, &lens, cand)?);
        }
        let sel = shared.selection.clone().unwrap_or(Selection::Visible);
        Ok(Some(match sel {
            Selection::Visible => Keys { rows: comp.rows, mask: host_mask(n, total, |i, j| j < lens[i])?, block: 0 },
            Selection::Mask(mask) => Keys { rows: comp.rows, mask, block: 0 },
            Selection::Gather { idx, mask } => {
                let k = idx.shape()[1];
                let rows = comp.rows.gather(&idx.reshape(&[n * k])?)?.reshape(&[n, k, comp.rows.shape()[1]])?;
                Keys { rows, mask, block: 0 }
            }
        }))
    }

    /// Rotated K = V rows `[m, head_dim]` of `x [m, D]` at `positions` (a DSpark stage's
    /// window rows, from the target features).
    pub fn main_kv(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (m, hd) = (x.shape()[0], self.head_dim);
        let (cos, sin) = self.rope.tables(positions)?;
        let kv = self.kv_norm.forward(&self.wkv.forward(x)?)?.cast(DType::F32)?.reshape(&[m, 1, hd])?;
        self.rope.apply(&kv, &cos, &sin, false)?.reshape(&[m, hd])
    }

    /// A DSpark stage (upstream `DSparkAttention`): the block `x [b, D]` (normed) at
    /// positions `pos..pos + b` attends to its window `ring [m, head_dim]` and to every
    /// block row (not causal). Returns `[b, D]`.
    pub fn forward_block(&self, x: &Tensor, pos: usize, ring: Option<&Tensor>) -> Result<Tensor> {
        let b = x.shape()[0];
        let (h, hd) = (self.n_head, self.head_dim);
        let positions: Vec<usize> = (pos..pos + b).collect();
        let (cos, sin) = self.rope.tables(&positions)?;
        let qr = self.q_norm.forward(&self.wq_a.forward(x)?)?.cast(DType::F32)?;
        let q = self.wq_b.forward(&qr)?.cast(DType::F32)?.reshape(&[b, h, hd])?;
        let q = self.rope.apply(&q, &cos, &sin, false)?;
        let kv = self.main_kv(x, &positions)?;
        let rows = match ring {
            Some(r) => Tensor::cat(&[r, &kv], 0)?,
            None => kv,
        };
        let mask = Tensor::zeros(&[b, rows.shape()[0]], DType::F32);
        let o = sink_attention(&q, &[Keys { rows, mask, block: 0 }], &self.sinks)?;
        let o = self.rope.apply(&o, &cos, &sin, true)?;
        grouped_output(&o, &self.wo_a, self.wo_a_act_quant, &self.wo_b, self.groups)
    }

    /// `x [n, D]` (normed block input), absolute positions `pos..pos + n`. Returns the block
    /// output `[n, D]` and the layer's next state.
    pub fn forward(&self, x: &Tensor, pos: usize, st: &LayerState, shared: &mut Shared) -> Result<(Tensor, LayerState)> {
        let n = x.shape()[0];
        let (h, hd) = (self.n_head, self.head_dim);
        let positions: Vec<usize> = (pos..pos + n).collect();
        let (cos, sin) = self.rope.tables(&positions)?;

        let qr = self.q_norm.forward(&self.wq_a.forward(x)?)?.cast(DType::F32)?;
        let q = self.wq_b.forward(&qr)?.cast(DType::F32)?.reshape(&[n, h, hd])?;
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
        let visible = |i: usize, j: usize| {
            let (qp, kp) = (pos + i, first + j);
            kp <= qp && qp - kp < win
        };
        let mut next = st.clone();
        // A short forward keeps its own rows beyond the window too: it can be rewound.
        let extra = if n <= super::REWIND_MAX { n } else { 0 };
        let keep = (win - 1 + extra).min(s_raw);
        next.raw = if keep > 0 { Some(raw.slice(0, s_raw - keep, s_raw)?) } else { None };

        let window = if n >= 2 * win {
            // A prompt chunk: each block of `win` queries reads only the 2 win - 1 raw rows
            // its windows cover (not all m + n rows: ~1/16 of the scores of a 2048 chunk).
            let (b, kb) = (win, 2 * win - 1);
            let nb = n.div_ceil(b);
            // raw row of slot t of block c: m + c b - (win - 1) + t (clamped, masked outside)
            let row = |c: usize, t: usize| (m + c * b + t).checked_sub(win - 1).filter(|&j| j < s_raw);
            let idx: Vec<i32> = (0..nb).flat_map(|c| (0..kb).map(move |t| (c, t))).map(|(c, t)| row(c, t).unwrap_or(0) as i32).collect();
            let rows = raw.gather(&Tensor::from_i32(&idx, &[nb * kb])?)?.reshape(&[nb, kb, hd])?;
            let mask = host_mask(n, kb, |i, t| row(i / b, t).is_some_and(|j| visible(i, j)))?;
            Keys { rows, mask, block: b }
        } else {
            Keys { rows: raw, mask: host_mask(n, s_raw, visible)?, block: 0 }
        };
        let mut parts = vec![window];
        if self.ratio > 0
            && let Some(k) = self.compressed_keys(x, &qr, pos, st, &mut next, shared)?
        {
            parts.push(k);
        }
        let o = sink_attention(&q, &parts, &self.sinks)?;
        let o = self.rope.apply(&o, &cos, &sin, true)?;
        let out = grouped_output(&o, &self.wo_a, self.wo_a_act_quant, &self.wo_b, self.groups)?;
        Ok((out, next))
    }
}
