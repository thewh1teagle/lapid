//! Multi-head attention: MHA / GQA / MQA, causal, sliding-window and custom masks, cross
//! attention, optional per-head q/k RMSNorm, RoPE and a KV cache.
//!
//! Heads are strided views ([B, S, H*Dh] -> [B, H, S, Dh] is a reshape + permute, no copy).
//! GQA / MQA repeat K/V heads with an expand view; the compiler materializes the repeat
//! where a kernel needs contiguous heads.

mod kv_cache;
mod row_buffer;

pub use kv_cache::KvCache;
pub use row_buffer::RowBuffer;

use lapid::{Result, Tensor, bail};

use crate::layers::{Linear, RmsNorm};
use crate::rope::Rope;

/// Which keys each query may see. Masks align the *last* query with the *last* key (the
/// usual layout when keys = cache + new tokens).
#[derive(Clone)]
pub enum Mask {
    /// Every key (encoders, cross attention).
    None,
    /// Key j is visible to query i iff j <= i + (Sk - Sq).
    Causal,
    /// Causal and at most `window` positions back: visible iff 0 <= pos(i) - pos(j) < window.
    Sliding(usize),
    /// An additive mask broadcastable to [B, H, Sq, Sk] (0 = visible, -inf = hidden),
    /// optionally combined with the causal rule.
    Custom { additive: Tensor, causal: bool },
}

/// Additive [Sq, Sk] sliding-window mask (keys at least `window` positions behind the query
/// get -inf). Combine with the causal flag for future keys.
pub fn sliding_window_mask(sq: usize, sk: usize, window: usize) -> Result<Tensor> {
    let offset = sk.saturating_sub(sq);
    let mut data = vec![0.0f32; sq * sk];
    for i in 0..sq {
        let qpos = i + offset;
        for j in 0..sk {
            if j + window <= qpos {
                data[i * sk + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&data, &[sq, sk])
}

/// Additive key-padding mask [B, 1, 1, Sk]: keys at or beyond `lengths[b]` are hidden.
pub fn key_padding_mask(lengths: &[usize], sk: usize) -> Result<Tensor> {
    let mut data = vec![0.0f32; lengths.len() * sk];
    for (b, &len) in lengths.iter().enumerate() {
        for j in len.min(sk)..sk {
            data[b * sk + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_f32(&data, &[lengths.len(), 1, 1, sk])
}

/// Repeat K/V heads for GQA: [B, Hkv, S, D] -> [B, Hkv * n_rep, S, D] (head h reads kv head
/// h / n_rep, as HF `repeat_kv`).
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let s = x.shape();
    if s.len() != 4 {
        bail!("shape_mismatch", "repeat_kv expects [B, H, S, D], got {s:?}");
    }
    let (b, h, n, d) = (s[0], s[1], s[2], s[3]);
    x.unsqueeze(2)?.expand(&[b, h, n_rep, n, d])?.reshape(&[b, h * n_rep, n, d])
}

/// softmax(q k^T * scale + mask) v with GQA head repetition.
/// q [B, H, Sq, D], k / v [B, Hkv, Sk, D] with H % Hkv == 0.
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, mask: &Mask) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 {
        bail!("shape_mismatch", "attention expects [B, H, S, D] tensors, got q {:?} k {:?}", q.shape(), k.shape());
    }
    let (h, hkv) = (q.shape()[1], k.shape()[1]);
    if hkv == 0 || h % hkv != 0 {
        bail!("shape_mismatch", "attention: {h} query heads are not a multiple of {hkv} kv heads");
    }
    let n_rep = h / hkv;
    let (b, sq, d, sk0) = (q.shape()[0], q.shape()[2], q.shape()[3], k.shape()[2]);
    let all_visible = match mask {
        Mask::None | Mask::Causal => true,
        Mask::Sliding(w) => *w >= 1 && sk0 <= *w,
        Mask::Custom { .. } => false,
    };
    if n_rep > 1 && sq == 1 && all_visible && k.shape() == [b, hkv, sk0, d] && v.shape() == [b, hkv, sk0, d] {
        // GQA decode: the n_rep query heads sharing a kv head become n_rep query rows of that
        // head (every key is visible to the single query), so K / V are never repeated.
        let qf = q.reshape(&[b, hkv, n_rep, d])?;
        return Tensor::attention(&qf, k, v, None, scale, false)?.reshape(&[b, h, 1, d]);
    }
    let (k, v) = (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?);
    let (sq, sk) = (q.shape()[2], k.shape()[2]);
    match mask {
        Mask::None => Tensor::attention(q, &k, &v, None, scale, false),
        Mask::Causal => Tensor::attention(q, &k, &v, None, scale, true),
        Mask::Sliding(window) => {
            if *window == 0 {
                bail!("invalid_argument", "sliding window must be >= 1");
            }
            if sk <= *window {
                // Nothing falls out of the window: plain causal (no mask leaf).
                return Tensor::attention(q, &k, &v, None, scale, true);
            }
            let m = sliding_window_mask(sq, sk, *window)?;
            Tensor::attention(q, &k, &v, Some(&m), scale, true)
        }
        Mask::Custom { additive, causal } => Tensor::attention(q, &k, &v, Some(additive), scale, *causal),
    }
}

/// [B, S, H*Dh] -> [B, H, S, Dh] (strided view).
pub fn split_heads(x: &Tensor, n_head: usize) -> Result<Tensor> {
    let s = x.shape();
    if s.len() != 3 || n_head == 0 || !s[2].is_multiple_of(n_head) {
        bail!("shape_mismatch", "split_heads: {s:?} is not [B, S, {n_head} * Dh]");
    }
    let (b, n, d) = (s[0], s[1], s[2]);
    x.reshape(&[b, n, n_head, d / n_head])?.permute(&[0, 2, 1, 3])
}

/// [B, H, S, Dh] -> [B, S, H*Dh].
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let s = x.shape();
    if s.len() != 4 {
        bail!("shape_mismatch", "merge_heads expects [B, H, S, Dh], got {s:?}");
    }
    let (b, h, n, dh) = (s[0], s[1], s[2], s[3]);
    x.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, n, h * dh])
}

/// Attention block with q / k / v / o projections.
#[derive(Clone)]
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    /// Per-head RMSNorm of queries / keys (LFM, Gemma 3), applied before RoPE.
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl Attention {
    /// `n_kv_head == n_head` is MHA, `1` is MQA, anything dividing `n_head` is GQA.
    /// head_dim = q.out_features / n_head, scale = 1 / sqrt(head_dim).
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, n_head: usize, n_kv_head: usize) -> Result<Attention> {
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            bail!("invalid_argument", "attention: n_head {n_head} must be a multiple of n_kv_head {n_kv_head}");
        }
        let qo = q.out_features();
        if !qo.is_multiple_of(n_head) {
            bail!("shape_mismatch", "attention: q_proj out {qo} not divisible by {n_head} heads");
        }
        let head_dim = qo / n_head;
        for (name, l) in [("k_proj", &k), ("v_proj", &v)] {
            if l.out_features() != n_kv_head * head_dim {
                bail!(
                    "shape_mismatch",
                    "attention: {name} out {} != {n_kv_head} kv heads x {head_dim}",
                    l.out_features()
                );
            }
        }
        Ok(Attention {
            q,
            k,
            v,
            o,
            q_norm: None,
            k_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn with_qk_norm(mut self, q_norm: RmsNorm, k_norm: RmsNorm) -> Attention {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Attention {
        self.scale = scale;
        self
    }

    /// Query heads of x [B, S, D]: [B, H, S, Dh] (q-norm applied).
    pub fn project_q(&self, x: &Tensor) -> Result<Tensor> {
        let q = split_heads(&self.q.forward(x)?, self.n_head)?;
        match &self.q_norm {
            Some(n) => n.forward(&q),
            None => Ok(q),
        }
    }

    /// Key / value heads of x [B, S, D]: ([B, Hkv, S, Dh], [B, Hkv, S, Dh]) (k-norm applied).
    pub fn project_kv(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut k = split_heads(&self.k.forward(x)?, self.n_kv_head)?;
        if let Some(n) = &self.k_norm {
            k = n.forward(&k)?;
        }
        let v = split_heads(&self.v.forward(x)?, self.n_kv_head)?;
        Ok((k, v))
    }

    /// o_proj(attention(q heads, k, v)): q [B, H, Sq, Dh] -> [B, Sq, D].
    pub fn attend_heads(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Mask) -> Result<Tensor> {
        let a = scaled_dot_product_attention(q, k, v, self.scale, mask)?;
        self.o.forward(&merge_heads(&a)?)
    }

    /// o_proj(attention(q_proj(x), k, v)) — cross attention, or self attention with K/V
    /// computed by the caller.
    pub fn attend(&self, x: &Tensor, k: &Tensor, v: &Tensor, mask: &Mask) -> Result<Tensor> {
        self.attend_heads(&self.project_q(x)?, k, v, mask)
    }

    /// Self attention of x [B, S, D] at positions `cache.offset()..` (0 without a cache):
    /// projections, q/k norm, RoPE, cache append (layer `layer`), attention, output proj.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: Option<&Rope>,
        cache: Option<(&mut KvCache, usize)>,
        mask: &Mask,
    ) -> Result<Tensor> {
        let mut q = self.project_q(x)?;
        let (mut k, mut v) = self.project_kv(x)?;
        let offset = cache.as_ref().map_or(0, |(c, _)| c.offset());
        if let Some(r) = rope {
            q = r.apply(&q, offset)?;
            k = r.apply(&k, offset)?;
        }
        if let Some((c, layer)) = cache {
            (k, v) = c.append(layer, &k, &v)?;
        }
        self.attend_heads(&q, &k, &v, mask)
    }
}

#[cfg(test)]
mod tests;
