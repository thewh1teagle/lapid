//! Composites: native Softmax / LayerNorm / Attention kernels, or the reference
//! decomposition (`lapid_tensor::decompose`) lowered like any other subgraph.

use std::sync::Arc;

use lapid_tensor::{DType, Op, Result, Tensor};

use super::lexpr::LExpr;
use super::{Lowerer, internal, op_name, src_of, view};
use crate::ir::{
    AttentionDims, AttentionSpec, BufId, Kernel, LayerNormDims, LayerNormSpec, RowDims, SinkhornDims, SoftmaxSpec, View,
};

/// Emit `Kernel::RmsNorm` (every backend has it; `LAPID_RMS_NORM=0` falls back to the
/// decomposition: reduce + one fused elementwise).
fn native_rms_norm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAPID_RMS_NORM").map_or(true, |v| v.trim() != "0"))
}

/// Largest matrix side of a native `Kernel::Sinkhorn` (larger ones are decomposed).
const SINKHORN_MAX_N: usize = 8;

impl Lowerer {
    pub(super) fn lower_composite(&mut self, t: &Tensor) -> Result<LExpr> {
        // Quantized expert stacks have no decomposition (they cannot be sliced in a graph).
        let force = matches!(t.op(), Op::MatMulId) && t.src().get(1).is_some_and(|w| w.dtype().is_quant());
        if self.opts.native_composites || force {
            let native = match t.op() {
                Op::Softmax => self.native_softmax(t)?,
                Op::LayerNorm { eps } => self.native_layer_norm(t, *eps)?,
                Op::RmsNorm { eps } => self.native_rms_norm(t, *eps)?,
                Op::Attention { scale, causal, has_mask } => self.native_attention(t, *scale, *causal, *has_mask)?,
                Op::MatMulId => self.native_matmul_id(t)?,
                Op::GatedDeltaNet { scale } => self.native_gated_delta_net(t, *scale)?,
                Op::Sinkhorn { iters, eps } if self.opts.native_sinkhorn => self.native_sinkhorn(t, *iters, *eps)?,
                _ => None,
            };
            if let Some(v) = native {
                return Ok(v);
            }
        }
        let Some(root) = lapid_tensor::decompose::decompose(t)? else {
            return Err(internal(format!("no decomposition for {} node #{}", op_name(t.op()), t.id())));
        };
        self.lower_subgraph(t, &root)
    }

    /// Lower the decomposition `root` of composite `t`. Its leaves are (ancestors of) the
    /// already-lowered srcs of `t`; `root` inherits `t`'s users and output claim.
    fn lower_subgraph(&mut self, t: &Tensor, root: &Tensor) -> Result<LExpr> {
        if self.vals.contains_key(&root.id()) {
            return self.val(root);
        }
        let order = lapid_tensor::toposort(std::slice::from_ref(root));
        for n in &order {
            if self.vals.contains_key(&n.id()) {
                continue;
            }
            for s in n.src() {
                if !self.vals.contains_key(&s.id()) {
                    *self.users.entry(s.id()).or_insert(0) += 1;
                }
            }
        }
        let u = self.users_of(t);
        self.users.insert(root.id(), u);
        if let Some(&i) = self.out_claim.get(&t.id()) {
            self.out_claim.entry(root.id()).or_insert(i);
        }
        for n in &order {
            self.lower_node(n)?;
        }
        self.val(root)
    }

    fn native_sinkhorn(&mut self, t: &Tensor, iters: usize, eps: f32) -> Result<Option<LExpr>> {
        let x = src_of(t, 0)?;
        let r = x.rank();
        let n = x.shape().get(r.wrapping_sub(1)).copied().unwrap_or(0);
        if r < 2 || x.dtype() != DType::F32 || t.shape() != x.shape() || n > SINKHORN_MAX_N {
            return Ok(None);
        }
        let count = view::numel(&x.shape()[..r - 2]);
        let e = self.val(x)?;
        let (xb, _) = self.float_contig(&e, x.shape(), DType::F32)?;
        let out = self.alloc_node(t)?;
        if count * n > 0 {
            self.push(Kernel::Sinkhorn(SinkhornDims { count, n, iters, eps: eps.to_bits() }), vec![xb], out);
        }
        Ok(Some(LExpr::buf(out, DType::F32, View::contiguous(t.shape()))))
    }

    /// Row softmax over the last axis of a row-major input.
    fn native_softmax(&mut self, t: &Tensor) -> Result<Option<LExpr>> {
        let x = src_of(t, 0)?;
        let r = x.rank();
        if r == 0 || x.dtype().is_quant() || t.shape() != x.shape() {
            return Ok(None);
        }
        let cols = x.shape()[r - 1];
        let rows = view::numel(&x.shape()[..r - 1]);
        let e = self.val(x)?;
        let (xb, xdt) = self.float_contig(&e, x.shape(), x.dtype())?;
        let out = self.alloc_node(t)?;
        if rows * cols > 0 {
            let spec = SoftmaxSpec { in_dtype: xdt, out_dtype: t.dtype() };
            self.push(Kernel::Softmax(Arc::new(spec), RowDims { rows, cols }), vec![xb], out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }

    fn native_rms_norm(&mut self, t: &Tensor, eps: f32) -> Result<Option<LExpr>> {
        if !native_rms_norm_enabled() {
            return Ok(None);
        }
        let x = src_of(t, 0)?;
        let w = src_of(t, 1)?;
        let r = x.rank();
        if r == 0 || t.shape() != x.shape() || w.numel() != x.shape()[r - 1] {
            return Ok(None);
        }
        if x.dtype().is_quant() || w.dtype().is_quant() {
            return Ok(None);
        }
        let cols = x.shape()[r - 1];
        let rows = view::numel(&x.shape()[..r - 1]);
        let xe = self.val(x)?;
        let (xb, xdt) = self.float_contig(&xe, x.shape(), x.dtype())?;
        let we = self.val(w)?;
        let (wb, wdt) = self.float_contig(&we, w.shape(), w.dtype())?;
        let out = self.alloc_node(t)?;
        if rows * cols > 0 {
            let spec = LayerNormSpec { in_dtype: xdt, param_dtype: wdt, out_dtype: t.dtype() };
            let dims = LayerNormDims { rows, cols, eps: eps.to_bits() };
            self.push(Kernel::RmsNorm(Arc::new(spec), dims), vec![xb, wb], out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }

    fn native_layer_norm(&mut self, t: &Tensor, eps: f32) -> Result<Option<LExpr>> {
        let x = src_of(t, 0)?;
        let w = src_of(t, 1)?;
        let b = src_of(t, 2)?;
        let r = x.rank();
        if r == 0 || t.shape() != x.shape() {
            return Ok(None);
        }
        let cols = x.shape()[r - 1];
        if w.numel() != cols || b.numel() != cols || [x, w, b].iter().any(|s| s.dtype().is_quant()) {
            return Ok(None);
        }
        let rows = view::numel(&x.shape()[..r - 1]);
        let xe = self.val(x)?;
        let (xb, xdt) = self.float_contig(&xe, x.shape(), x.dtype())?;
        let we = self.val(w)?;
        let (mut wb, mut wdt) = self.float_contig(&we, w.shape(), w.dtype())?;
        let be = self.val(b)?;
        let (mut bb, mut bdt) = self.float_contig(&be, b.shape(), b.dtype())?;
        if wdt != bdt {
            // One param dtype per kernel: promote both to f32.
            if wdt != DType::F32 {
                wb = self.materialize_contig(&LExpr::buf(wb, wdt, View::contiguous(w.shape())), w.shape(), DType::F32)?;
                wdt = DType::F32;
            }
            if bdt != DType::F32 {
                bb = self.materialize_contig(&LExpr::buf(bb, bdt, View::contiguous(b.shape())), b.shape(), DType::F32)?;
                bdt = DType::F32;
            }
        }
        debug_assert_eq!(wdt, bdt);
        let out = self.alloc_node(t)?;
        if rows * cols > 0 {
            let spec = LayerNormSpec { in_dtype: xdt, param_dtype: wdt, out_dtype: t.dtype() };
            let dims = LayerNormDims { rows, cols, eps: eps.to_bits() };
            self.push(Kernel::LayerNorm(Arc::new(spec), dims), vec![xb, wb, bb], out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }

    /// q/k/v are read through their views (reshape + permute of a projection is free);
    /// the mask is broadcast with zero strides.
    fn native_attention(&mut self, t: &Tensor, scale: f32, causal: bool, has_mask: bool) -> Result<Option<LExpr>> {
        let q = src_of(t, 0)?;
        let k = src_of(t, 1)?;
        let v = src_of(t, 2)?;
        if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
            return Ok(None);
        }
        let (bsz, heads, sq, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let sk = k.shape()[2];
        let kv_shape = [bsz, heads, sk, d];
        if k.shape() != kv_shape || v.shape() != kv_shape || t.shape() != q.shape() {
            return Ok(None);
        }
        if [q, k, v].iter().any(|s| s.dtype().is_quant()) {
            return Ok(None);
        }
        let mask_shape = [bsz, heads, sq, sk];
        let mask_node = if has_mask {
            let m = src_of(t, 3)?;
            let fits = view::broadcast_to(&View::contiguous(m.shape()), m.shape(), &mask_shape).is_some();
            if !fits || m.dtype().is_quant() {
                return Ok(None);
            }
            Some(m)
        } else {
            None
        };

        let mut parts: Vec<(BufId, DType, View)> = Vec::with_capacity(3);
        for s in [q, k, v] {
            let e = self.val(s)?;
            parts.push(self.float_buffer(&e, s.shape(), s.dtype())?);
        }
        if !(parts[0].1 == parts[1].1 && parts[1].1 == parts[2].1) {
            // One q/k/v dtype per kernel: promote the odd ones to f32.
            for (i, s) in [q, k, v].into_iter().enumerate() {
                if parts[i].1 != DType::F32 {
                    let (b, dt, vw) = parts[i].clone();
                    let nb = self.materialize_contig(&LExpr::buf(b, dt, vw), s.shape(), DType::F32)?;
                    parts[i] = (nb, DType::F32, View::contiguous(s.shape()));
                }
            }
        }
        let mask = match mask_node {
            Some(m) => {
                let e = self.val(m)?;
                let (b, dt, vw) = self.float_buffer(&e, m.shape(), m.dtype())?;
                let bv = view::broadcast_to(&vw, m.shape(), &mask_shape)
                    .ok_or_else(|| internal("attention mask view lost its broadcastability"))?;
                Some((b, dt, bv))
            }
            None => None,
        };

        let out = self.alloc_node(t)?;
        if bsz * heads * sq * d > 0 {
            let spec = AttentionSpec {
                dtype: parts[0].1,
                mask_dtype: mask.as_ref().map(|m| m.1),
                causal,
                out_dtype: t.dtype(),
            };
            let mut inputs = vec![parts[0].0, parts[1].0, parts[2].0];
            if let Some(m) = &mask {
                inputs.push(m.0);
            }
            let dims = AttentionDims {
                batch: bsz,
                heads,
                sq,
                sk,
                d,
                scale: scale.to_bits(),
                q: parts[0].2.clone(),
                k: parts[1].2.clone(),
                v: parts[2].2.clone(),
                mask: mask.map(|m| m.2),
            };
            self.push(Kernel::Attention(Arc::new(spec), dims), inputs, out);
        }
        Ok(Some(LExpr::buf(out, t.dtype(), View::contiguous(t.shape()))))
    }
}
