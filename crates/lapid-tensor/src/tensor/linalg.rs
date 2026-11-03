//! Builders for matmul, gather and the composite ops (softmax, layer norm, attention).

use crate::bail;
use crate::dtype::DType;
use crate::error::Result;
use crate::op::Op;
use crate::tensor::{Shape, Srcs, Tensor};
use smallvec::smallvec;
use crate::tensor::check::{self, broadcast_shapes};

impl Tensor {
    /// `self @ b` (or `self @ b^T` when `b_transposed`). Batch dims broadcast numpy-style.
    ///
    /// Normalization done here so the compiler sees simple cases: `a` always carries the
    /// full output batch dims (via Reshape+Expand if needed); `b` is either 2-D (shared by
    /// every batch, batch stride 0) or carries the full output batch dims too.
    pub(super) fn matmul_impl(&self, b: &Tensor, b_transposed: bool) -> Result<Tensor> {
        let a = self;
        if !a.dtype().is_float() {
            bail!("unsupported_dtype", "matmul lhs must be float, got {}", a.dtype().name());
        }
        if b.dtype().is_quant() {
            if !b_transposed || b.rank() != 2 {
                bail!(
                    "unsupported_dtype",
                    "quantized matmul rhs ({}) must be a 2-D transposed weight (use matmul_t)",
                    b.dtype().name()
                );
            }
        } else if !b.dtype().is_float() {
            bail!("unsupported_dtype", "matmul rhs must be float or quantized, got {}", b.dtype().name());
        }
        if a.rank() < 2 || b.rank() < 2 {
            bail!("invalid_argument", "matmul needs rank >= 2 operands, got {:?} @ {:?}", a.shape(), b.shape());
        }
        let (ar, br) = (a.rank(), b.rank());
        let (m, k) = (a.shape()[ar - 2], a.shape()[ar - 1]);
        let (kb, n) = if b_transposed {
            (b.shape()[br - 1], b.shape()[br - 2])
        } else {
            (b.shape()[br - 2], b.shape()[br - 1])
        };
        if k != kb {
            let t = if b_transposed { "^T" } else { "" };
            bail!("shape_mismatch", "matmul {:?} @ {:?}{t}: inner dims {k} != {kb}", a.shape(), b.shape());
        }
        let a_batch = &a.shape()[..ar - 2];
        let b_batch = &b.shape()[..br - 2];
        let batch = broadcast_shapes(a_batch, b_batch)?;

        let with_batch = |tail: &[usize]| -> Shape {
            let mut s = batch.clone();
            s.extend_from_slice(tail);
            s
        };
        let a2 = a.broadcast_to(&with_batch(&a.shape()[ar - 2..]))?;
        let b2 = if br == 2 { b.clone() } else { b.broadcast_to(&with_batch(&b.shape()[br - 2..]))? };
        let shape = with_batch(&[m, n]);
        Ok(Tensor::from_op(Op::MatMul { b_transposed }, smallvec![a2, b2], shape, a.dtype()))
    }

    pub(super) fn gather_impl(&self, indices: &Tensor) -> Result<Tensor> {
        if self.rank() != 2 {
            bail!("shape_mismatch", "gather table must be 2-D [V, D], got {:?}", self.shape());
        }
        if !(self.dtype().is_float() || self.dtype().is_quant()) {
            bail!("unsupported_dtype", "gather table must be float or quantized, got {}", self.dtype().name());
        }
        if indices.dtype() != DType::I32 {
            bail!("unsupported_dtype", "gather indices must be i32, got {}", indices.dtype().name());
        }
        let mut shape = Shape::from_slice(indices.shape());
        shape.push(self.shape()[1]);
        Ok(Tensor::from_op(Op::Gather, smallvec![self.clone(), indices.clone()], shape, DType::F32))
    }

    pub(super) fn softmax_impl(&self) -> Result<Tensor> {
        check::float_dtype(self, "softmax")?;
        if self.rank() == 0 {
            bail!("invalid_argument", "softmax of a rank-0 tensor");
        }
        Ok(Tensor::from_op(Op::Softmax, smallvec![self.clone()], self.shape(), self.dtype()))
    }

    pub(super) fn sinkhorn_impl(&self, iters: usize, eps: f32) -> Result<Tensor> {
        let r = self.rank();
        if self.dtype() != DType::F32 || r < 2 || self.shape()[r - 1] != self.shape()[r - 2] || iters == 0 {
            bail!("invalid_argument", "sinkhorn needs f32 [..., n, n] and iters >= 1, got {} {:?}, iters {iters}", self.dtype().name(), self.shape());
        }
        Ok(Tensor::from_op(Op::Sinkhorn { iters, eps }, smallvec![self.clone()], self.shape(), DType::F32))
    }

    pub(super) fn rms_norm_impl(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        check::float_dtype(self, "rms_norm")?;
        check::float_dtype(weight, "rms_norm weight")?;
        if self.rank() == 0 {
            bail!("invalid_argument", "rms_norm of a rank-0 tensor");
        }
        let d = self.shape()[self.rank() - 1];
        if weight.shape() != [d] {
            bail!("shape_mismatch", "rms_norm over last dim {d}: weight {:?} must be [{d}]", weight.shape());
        }
        if !eps.is_finite() || eps < 0.0 {
            bail!("invalid_argument", "rms_norm eps must be finite and >= 0, got {eps}");
        }
        let srcs: Srcs = smallvec![self.clone(), weight.clone()];
        Ok(Tensor::from_op(Op::RmsNorm { eps }, srcs, self.shape(), self.dtype()))
    }

    pub(super) fn layer_norm_impl(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        check::float_dtype(self, "layer_norm")?;
        check::float_dtype(weight, "layer_norm weight")?;
        check::float_dtype(bias, "layer_norm bias")?;
        if self.rank() == 0 {
            bail!("invalid_argument", "layer_norm of a rank-0 tensor");
        }
        let d = self.shape()[self.rank() - 1];
        if weight.shape() != [d] || bias.shape() != [d] {
            bail!(
                "shape_mismatch",
                "layer_norm over last dim {d}: weight {:?} / bias {:?} must be [{d}]",
                weight.shape(),
                bias.shape()
            );
        }
        if weight.dtype() != bias.dtype() {
            bail!(
                "unsupported_dtype",
                "layer_norm weight ({}) and bias ({}) dtypes differ",
                weight.dtype().name(),
                bias.dtype().name()
            );
        }
        if !eps.is_finite() || eps < 0.0 {
            bail!("invalid_argument", "layer_norm eps must be finite and >= 0, got {eps}");
        }
        let srcs: Srcs = smallvec![self.clone(), weight.clone(), bias.clone()];
        Ok(Tensor::from_op(Op::LayerNorm { eps }, srcs, self.shape(), self.dtype()))
    }
}

/// Validate and build an `Op::Attention` node. The mask (if any) is broadcast to exactly
/// `[B, H, Sq, Sk]` (an `Expand` view, so no memory is spent on it).
pub(super) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        check::float_dtype(t, "attention")?;
        if t.rank() != 4 {
            bail!("shape_mismatch", "attention {name} must be [B, H, S, D], got {:?}", t.shape());
        }
    }
    if k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        bail!(
            "unsupported_dtype",
            "attention q/k/v dtypes differ: {} {} {}",
            q.dtype().name(),
            k.dtype().name(),
            v.dtype().name()
        );
    }
    let (b, h, sq, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let sk = k.shape()[2];
    if k.shape() != [b, h, sk, d] || v.shape() != [b, h, sk, d] {
        bail!(
            "shape_mismatch",
            "attention: q {:?} needs k and v of [{b}, {h}, Sk, {d}], got k {:?} v {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    if causal && sq > sk {
        bail!("invalid_argument", "causal attention needs Sq ({sq}) <= Sk ({sk})");
    }
    if !scale.is_finite() {
        bail!("invalid_argument", "attention scale must be finite, got {scale}");
    }
    let mut srcs: Srcs = smallvec![q.clone(), k.clone(), v.clone()];
    if let Some(m) = mask {
        check::float_dtype(m, "attention mask")?;
        let full = [b, h, sq, sk];
        if broadcast_shapes(m.shape(), &full)?[..] != full[..] {
            bail!("shape_mismatch", "attention mask {:?} is not broadcastable to {full:?}", m.shape());
        }
        srcs.push(m.broadcast_to(&full)?);
    }
    let op = Op::Attention { scale, causal, has_mask: mask.is_some() };
    Ok(Tensor::from_op(op, srcs, q.shape(), q.dtype()))
}
