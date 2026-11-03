//! Indexer of the index-source layers (upstream `Indexer` + `select_candidate_blocks`;
//! llama.cpp `build_attention41` top-k part): which compressed rows each query attends to.
//!
//! ```text
//! q        = rope(qr Wq_b^T)                        [n, H_idx, d_idx]   (no Hadamard rotation)
//! w        = x Wproj^T / sqrt(d_idx H_idx)          [n, H_idx]
//! score    = sum_h w_h relu(q_h . key_t)            [n, T], -inf where not yet visible
//! cand     = top blocks of max(score) per block     (candidate source; the block of the newest
//!                                                    visible row pinned)
//! score    = -inf outside cand                      (index sources after the candidate source)
//! selected = topk(score, min(index_topk, T))
//! ```
//! A query at position p sees rows `0 .. (p + 1) / ratio`. When `T <= index_topk` every
//! visible row is selected; otherwise the selection is a per-query gather (few queries:
//! decode) or a threshold mask over all rows (prefill), equal up to ties.

use lapid::{BinaryOp, DType, Result, Tensor};
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Linear, ops};

use crate::deepseek4::model::compress::NEG;

/// Queries up to which the selection gathers rows per query.
const GATHER_MAX_QUERIES: usize = 32;

/// Compressed rows a group of queries attends to.
#[derive(Clone)]
pub(crate) enum Selection {
    /// Every causally visible row.
    Visible,
    /// Additive mask over all rows `[n, T]`.
    Mask(Tensor),
    /// Rows `idx [n, k]` (i32) per query, additive `mask [n, k]` (drops rows not yet visible).
    Gather { idx: Tensor, mask: Tensor },
}

/// Rows visible to each query at `pos..pos + n`.
pub(crate) fn visible_lens(pos: usize, n: usize, ratio: usize) -> Vec<usize> {
    (pos..pos + n).map(|p| (p + 1) / ratio).collect()
}

/// `[n, len]` 1 / 0 host tensor of `j < lens[i]`.
fn below(lens: &[usize], len: usize) -> Result<Tensor> {
    let m: Vec<f32> = lens.iter().flat_map(|&l| (0..len).map(move |j| if j < l { 1.0 } else { 0.0 })).collect();
    Tensor::from_f32(&m, &[lens.len(), len])
}

/// `keep ? x : NEG`.
fn keep_or_neg(keep: &Tensor, x: &Tensor) -> Result<Tensor> {
    Tensor::where_(keep, x, &Tensor::full(f64::from(NEG), &[1, 1], DType::F32))
}

/// `1 - (a < b)`: a >= b.
fn ge(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.binary(BinaryOp::CmpLt, b)?.neg()?.add_scalar(1.0)
}

/// The k-th largest value per row `[n, 1]` (the row's selection threshold).
fn kth(x: &Tensor, k: usize) -> Result<Tensor> {
    x.topk(k, 1)?.0.neg()?.max(1, true)?.neg()
}

pub(crate) struct Indexer {
    pub wq_b: Linear,
    pub weights_proj: Linear,
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
}

/// The candidate-block step of the candidate source layer.
pub(crate) struct Candidates {
    pub topk_blocks: usize,
    pub block_size: usize,
}

impl Candidates {
    /// Keep mask `[n, T]` (1 / 0) of the best `topk_blocks` blocks per query (by their best
    /// `score`), always with the block of the newest visible row.
    pub fn select(&self, score: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let (n, t, bs) = (score.shape()[0], score.shape()[1], self.block_size);
        let nb = t.div_ceil(bs);
        let bmax = score.pad(1, 0, nb * bs - t, f64::from(NEG))?.reshape(&[n, nb, bs])?.max(2, false)?; // [n, nb]
        let pin: Vec<f32> = lens.iter().flat_map(|&l| (0..nb).map(move |b| if l > 0 && b == (l - 1) / bs { 1.0 } else { 0.0 })).collect();
        let bmax = Tensor::where_(&Tensor::from_f32(&pin, &[n, nb])?, &Tensor::full(-f64::from(NEG), &[1, 1], DType::F32), &bmax)?;
        let live = bmax.binary_scalar(BinaryOp::CmpLt, f64::from(NEG) / 2.0)?.neg()?.add_scalar(1.0)?; // > -inf
        let k = self.topk_blocks.min(nb);
        let keep = if k == nb { live } else { ge(&bmax, &kth(&bmax, k)?)?.mul(&live)? };
        keep.unsqueeze(2)?.expand(&[n, nb, bs])?.reshape(&[n, nb * bs])?.slice(1, 0, t)
    }
}

impl Indexer {
    /// Scores `[n, T]` of the queries at `pos..` against the index `keys [T, d_idx]`, NEG where
    /// a row is not visible yet (`lens`).
    pub fn scores(&self, x: &Tensor, qr: &Tensor, keys: &Tensor, pos: usize, lens: &[usize], rope: &TailRope) -> Result<Tensor> {
        let n = x.shape()[0];
        let t = keys.shape()[0];
        let (h, d) = (self.heads, self.dim);
        let positions: Vec<usize> = (pos..pos + n).collect();
        let (cos, sin) = rope.tables(&positions)?;
        let q = self.wq_b.forward(qr)?.cast(DType::F32)?.reshape(&[n, h, d])?;
        let q = rope.apply(&q, &cos, &sin, false)?.reshape(&[n * h, d])?;
        let w = self.weights_proj.forward(x)?.cast(DType::F32)?.mul_scalar(1.0 / ((d * h) as f64).sqrt())?; // [n, h]
        let s = ops::relu(&q.matmul_t(keys)?)?.reshape(&[n, h, t])?;
        let s = s.mul(&w.unsqueeze(2)?)?.sum(1, false)?; // [n, T]
        if lens.iter().all(|&l| l >= t) {
            return Ok(s);
        }
        keep_or_neg(&below(lens, t)?, &s)
    }

    /// The top-k selection from `score [n, T]`, restricted to `candidates` (keep mask) if given.
    pub fn select(&self, score: &Tensor, lens: &[usize], candidates: Option<&Tensor>) -> Result<Selection> {
        let (n, t) = (score.shape()[0], score.shape()[1]);
        let k = self.topk.min(t);
        if k == t {
            return Ok(Selection::Visible);
        }
        let score = match candidates {
            Some(c) => keep_or_neg(c, score)?,
            None => score.clone(),
        };
        if n <= GATHER_MAX_QUERIES {
            let idx = score.topk_indices(k, 1)?; // [n, k]
            let len: Vec<f32> = lens.iter().map(|&l| l as f32).collect();
            let vis = idx.cast(DType::F32)?.binary(BinaryOp::CmpLt, &Tensor::from_f32(&len, &[n, 1])?)?;
            let mask = keep_or_neg(&vis, &Tensor::full(0.0, &[1, 1], DType::F32))?;
            return Ok(Selection::Gather { idx, mask });
        }
        let keep = ge(&score, &kth(&score, k)?)?.mul(&below(lens, t)?)?;
        Ok(Selection::Mask(keep_or_neg(&keep, &Tensor::full(0.0, &[1, 1], DType::F32))?))
    }
}
