//! KV compressor of the kv-source layers (upstream `Compressor` + the indexer's key owner;
//! llama.cpp `build_attention41` compressor part) as a streaming lapid graph.
//!
//! Every `ratio` (1 or 2) tokens close one group and emit one compressed row; no positional
//! bias, no overlap:
//!
//! ```text
//! latent = rmsnorm(sum_j softmax_j(x_j Wgate^T) * x_j Wkv^T)    (ratio 1: rmsnorm(x Wkv^T))
//! key    = rope(rmsnorm(latent Wk^T))                            index key, from the RoPE-free latent
//! row    = rope(latent)                                          K = V of the compressed entry
//! ```
//! both rotated at the group's first position. State ([`CompState`]): the projected rows of
//! the open group (ratio 2) and the emitted rows / keys ([`RowBuffer`]s, written in place).

use lapid::{DType, Result, Tensor};
use lapid_nn::attention::RowBuffer;
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Linear, RmsNorm};

#[derive(Clone, Default)]
pub(crate) struct CompState {
    /// Open group rows `[pos % ratio, head_dim]` (kv, gate score); ratio 2 only.
    pub pend_kv: Option<Tensor>,
    pub pend_score: Option<Tensor>,
    /// Emitted entries `[pos / ratio, head_dim]` (rotated) and index keys `[.., index_dim]`.
    pub rows: RowBuffer,
    pub keys: RowBuffer,
    /// Ratio 2, after a short forward (rewindable): the projected rows of its tokens and of
    /// the group they continued, from position `tail_start` (rebuild the open group).
    pub tail_kv: Option<Tensor>,
    pub tail_score: Option<Tensor>,
    pub tail_start: usize,
}

impl CompState {
    pub fn slots(&self) -> [Option<&Tensor>; 6] {
        [self.pend_kv.as_ref(), self.pend_score.as_ref(), self.rows.buffer(), self.keys.buffer(), self.tail_kv.as_ref(), self.tail_score.as_ref()]
    }

    pub fn slots_mut(&mut self) -> [&mut Option<Tensor>; 6] {
        [&mut self.pend_kv, &mut self.pend_score, self.rows.buffer_mut(), self.keys.buffer_mut(), &mut self.tail_kv, &mut self.tail_score]
    }

    /// Back to the first `len` tokens (within the last short forward, see `tail_*`); false
    /// when the open group cannot be rebuilt.
    pub fn truncate(&mut self, len: usize, ratio: usize) -> Result<bool> {
        if self.slots().iter().all(Option::is_none) {
            // A layer reading another layer's compressed rows keeps none of its own.
            return Ok(true);
        }
        self.rows.truncate(len / ratio);
        self.keys.truncate(len / ratio);
        if ratio < 2 {
            return Ok(true);
        }
        if len.is_multiple_of(ratio) {
            (self.pend_kv, self.pend_score) = (None, None);
            return Ok(true);
        }
        let (Some(kv), Some(score)) = (&self.tail_kv, &self.tail_score) else { return Ok(false) };
        let (a, b) = ((len - len % ratio).checked_sub(self.tail_start), len.checked_sub(self.tail_start));
        let (Some(a), Some(b)) = (a, b) else { return Ok(false) };
        if b > kv.shape()[0] {
            return Ok(false);
        }
        (self.pend_kv, self.pend_score) = (Some(kv.slice(0, a, b)?), Some(score.slice(0, a, b)?));
        Ok(true)
    }
}

/// Compressed rows visible to the layers of one forward: entries `[T, head_dim]` and index
/// keys `[T, index_dim]`.
#[derive(Clone)]
pub(crate) struct Compressed {
    pub rows: Tensor,
    pub keys: Tensor,
}

/// Latents `[m, head_dim]` of the closed groups and their first positions.
type Latents = (Tensor, Vec<usize>);

pub(crate) struct Compressor {
    pub wkv: Linear,
    /// Gate projection (ratio 2).
    pub wgate: Option<Linear>,
    pub norm: RmsNorm,
    /// Index key projection of the latent and its norm.
    pub wk: Linear,
    pub k_norm: RmsNorm,
    pub ratio: usize,
}

/// Rotate `rows [m, d]` (tail RoPE) at `positions`.
fn rope_rows(rope: &TailRope, rows: &Tensor, positions: &[usize]) -> Result<Tensor> {
    let (m, d) = (rows.shape()[0], rows.shape()[1]);
    let (cos, sin) = rope.tables(positions)?;
    rope.apply(&rows.reshape(&[m, 1, d])?, &cos, &sin, false)?.reshape(&[m, d])
}

impl Compressor {
    /// The latents of the groups closed by `x [n, D]` at positions `pos..pos + n` (and their
    /// first positions), plus the next open-group state.
    fn latents(&self, x: &Tensor, pos: usize, st: &CompState, rewindable: bool) -> Result<(Option<Latents>, CompState)> {
        let n = x.shape()[0];
        let r = self.ratio;
        let mut next = st.clone();
        (next.tail_kv, next.tail_score) = (None, None);
        let kv = self.wkv.forward(x)?.cast(DType::F32)?;
        let Some(wgate) = &self.wgate else {
            let latent = self.norm.forward(&kv)?.cast(DType::F32)?;
            return Ok((Some((latent, (pos..pos + n).collect())), next));
        };
        let score = wgate.forward(x)?.cast(DType::F32)?;
        let (all_kv, all_score) = match (&st.pend_kv, &st.pend_score) {
            (Some(pk), Some(ps)) => (Tensor::cat(&[pk, &kv], 0)?, Tensor::cat(&[ps, &score], 0)?),
            _ => (kv, score),
        };
        if rewindable {
            (next.tail_kv, next.tail_score, next.tail_start) = (Some(all_kv.clone()), Some(all_score.clone()), pos - pos % r);
        }
        let total = all_kv.shape()[0];
        let nw = total / r;
        let rest = |t: &Tensor| -> Result<Option<Tensor>> { if total > nw * r { Ok(Some(t.slice(0, nw * r, total)?)) } else { Ok(None) } };
        next.pend_kv = rest(&all_kv)?;
        next.pend_score = rest(&all_score)?;
        if nw == 0 {
            return Ok((None, next));
        }
        let hd = all_kv.shape()[1];
        let values = all_kv.slice(0, 0, nw * r)?.reshape(&[nw, r, hd])?;
        let w = all_score.slice(0, 0, nw * r)?.reshape(&[nw, r, hd])?.permute(&[0, 2, 1])?.softmax()?; // [nw, hd, r]
        let pooled = values.permute(&[0, 2, 1])?.mul(&w)?.sum(2, false)?; // [nw, hd]
        let latent = self.norm.forward(&pooled)?.cast(DType::F32)?;
        let start = pos - pos % r;
        Ok((Some((latent, (0..nw).map(|j| start + j * r).collect())), next))
    }

    /// Feed `x [n, D]` at positions `pos..pos + n`. Returns every row emitted so far (`None`
    /// while there are none) and the next state (`rewindable`: keeps what
    /// [`CompState::truncate`] needs).
    pub fn step(&self, x: &Tensor, pos: usize, rope: &TailRope, st: &CompState, rewindable: bool) -> Result<(Option<Compressed>, CompState)> {
        let (new, mut next) = self.latents(x, pos, st, rewindable)?;
        let Some((latent, positions)) = new else {
            let rows = st.rows.rows()?.zip(st.keys.rows()?).map(|(rows, keys)| Compressed { rows, keys });
            return Ok((rows, next));
        };
        let keys = self.k_norm.forward(&self.wk.forward(&latent)?)?.cast(DType::F32)?;
        let (kb, keys) = st.keys.append(&rope_rows(rope, &keys, &positions)?)?;
        let (rb, rows) = st.rows.append(&rope_rows(rope, &latent, &positions)?)?;
        next.keys = kb;
        next.rows = rb;
        Ok((Some(Compressed { rows, keys }), next))
    }
}
