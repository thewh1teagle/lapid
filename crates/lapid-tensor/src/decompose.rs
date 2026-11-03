//! Reference decompositions of composite ops into primitives. CONTRACT signatures —
//! implementer: agent "tensor". Used by backends without a native kernel and by spec
//! conformance tests.
//!
//! All decompositions compute in f32 (inputs are cast up, the result is cast back to the
//! node's dtype), mirroring the "all arithmetic is f32" rule of the kernel IR.

use crate::bail;
use crate::dtype::DType;
use crate::error::Result;
use crate::op::{Op, UnaryOp};
use crate::tensor::Tensor;

/// Decompose a composite node (Softmax / LayerNorm / RmsNorm / Attention / MatMulId with float experts /
/// GatedDeltaNet / Sinkhorn) into an equivalent
/// subgraph of primitives. Returns `None` for non-composite nodes.
pub fn decompose(node: &Tensor) -> Result<Option<Tensor>> {
    let src = node.src();
    let out = match node.op() {
        Op::Softmax => {
            expect_srcs(node, 1)?;
            softmax(&src[0].cast(DType::F32)?)?
        }
        Op::LayerNorm { eps } => {
            expect_srcs(node, 3)?;
            layer_norm(&src[0], &src[1], &src[2], *eps)?
        }
        Op::RmsNorm { eps } => {
            expect_srcs(node, 2)?;
            rms_norm(&src[0], &src[1], *eps)?
        }
        Op::Attention { scale, causal, has_mask } => {
            expect_srcs(node, if *has_mask { 4 } else { 3 })?;
            let mask = if *has_mask { Some(&src[3]) } else { None };
            attention(&src[0], &src[1], &src[2], mask, *scale, *causal)?
        }
        Op::MatMulId => {
            expect_srcs(node, 3)?;
            if src[1].dtype().is_quant() {
                // Quantized expert stacks cannot be sliced in the graph: native kernels only.
                return Ok(None);
            }
            matmul_id(&src[0], &src[1], &src[2])?
        }
        Op::GatedDeltaNet { scale } => {
            expect_srcs(node, 6)?;
            gated_delta_net(src, *scale)?
        }
        Op::Sinkhorn { iters, eps } => {
            expect_srcs(node, 1)?;
            sinkhorn(&src[0], *iters, *eps)?
        }
        _ => return Ok(None),
    };
    Ok(Some(out.cast(node.dtype())?))
}

fn expect_srcs(node: &Tensor, n: usize) -> Result<()> {
    if node.src().len() != n {
        bail!("invalid_argument", "{:?} node has {} srcs, expected {n}", node.op(), node.src().len());
    }
    Ok(())
}

/// exp(x - max) / sum(exp(x - max)) over the last axis; `x` must be f32.
fn softmax(x: &Tensor) -> Result<Tensor> {
    let m = x.max(-1, true)?;
    let e = x.sub(&m)?.exp()?;
    let s = e.sum(-1, true)?;
    e.div(&s)
}

/// See `Op::Sinkhorn`.
fn sinkhorn(x: &Tensor, iters: usize, eps: f32) -> Result<Tensor> {
    let r = x.rank() as isize;
    let eps = f64::from(eps);
    let norm = |c: &Tensor, axis: isize| -> Result<Tensor> { c.div(&c.sum(axis, true)?.add_scalar(eps)?) };
    let mut c = norm(&softmax(x)?.add_scalar(eps)?, r - 2)?;
    for _ in 1..iters {
        c = norm(&norm(&c, r - 1)?, r - 2)?;
    }
    Ok(c)
}

/// (x - mean) * rsqrt(var + eps) * w + b over the last axis (biased variance).
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.cast(DType::F32)?;
    let mean = x.mean(-1, true)?;
    let xc = x.sub(&mean)?;
    let var = xc.mul(&xc)?.mean(-1, true)?;
    let inv = var.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?;
    xc.mul(&inv)?.mul(&w.cast(DType::F32)?)?.add(&b.cast(DType::F32)?)
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.cast(DType::F32)?;
    let ms = x.mul(&x)?.mean(-1, true)?;
    let inv = ms.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?;
    x.mul(&inv)?.mul(&w.cast(DType::F32)?)
}

/// softmax(q @ k^T * scale + mask [+ causal]) @ v.
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let (q, k, v) = (q.cast(DType::F32)?, k.cast(DType::F32)?, v.cast(DType::F32)?);
    let mut scores = q.matmul_t(&k)?.mul_scalar(f64::from(scale))?;
    if let Some(m) = mask {
        scores = scores.add(&m.cast(DType::F32)?)?;
    }
    if causal {
        let sq = q.shape()[q.rank() - 2];
        let sk = k.shape()[k.rank() - 2];
        scores = scores.add(&causal_mask(sq, sk)?)?;
    }
    softmax(&scores)?.matmul(&v)
}

/// `out[t, r] = w[ids[t, r]] @ x[t]` via a gather of whole expert matrices (float experts).
fn matmul_id(x: &Tensor, w: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (t, k) = (x.shape()[0], x.shape()[1]);
    let (e, n) = (w.shape()[0], w.shape()[1]);
    let r = ids.shape()[1];
    // [E, N*K] table -> [T, R, N*K] -> [T, R, N, K]
    let sel = w.cast(DType::F32)?.reshape(&[e, n * k])?.gather(ids)?.reshape(&[t, r, n, k])?;
    let xa = x.cast(DType::F32)?.reshape(&[t, 1, 1, k])?.expand(&[t, r, 1, k])?;
    xa.matmul_t(&sel)?.reshape(&[t, r, n])
}

/// The gated delta rule unrolled over T with primitives (f32). See `Op::GatedDeltaNet`.
fn gated_delta_net(src: &[Tensor], scale: f32) -> Result<Tensor> {
    let (q, k, v, g, beta) = (&src[0], &src[1], &src[2], &src[3], &src[4]);
    let (b, t, h, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let dv = v.shape()[3];
    let mut s = src[5].clone(); // [B, H, Dk, Dv]
    let mut outs = Vec::with_capacity(t + 1);
    let step = |x: &Tensor, i: usize, tail: &[usize]| -> Result<Tensor> {
        let mut shape = vec![b, h];
        shape.extend_from_slice(tail);
        x.slice(1, i, i + 1)?.reshape(&shape)
    };
    for i in 0..t {
        let qt = step(q, i, &[dk, 1])?.mul_scalar(f64::from(scale))?;
        let kt = step(k, i, &[dk, 1])?;
        let vt = step(v, i, &[1, dv])?;
        let gt = step(g, i, &[1, 1])?.exp()?;
        let bt = step(beta, i, &[1, 1])?;
        s = s.mul(&gt)?;
        let kv = s.mul(&kt)?.sum(2, true)?; // Sᵀk: [B, H, 1, Dv]
        let delta = vt.sub(&kv)?.mul(&bt)?;
        s = s.add(&kt.mul(&delta)?)?;
        outs.push(s.mul(&qt)?.sum(2, true)?); // [B, H, 1, Dv]
    }
    outs.push(s);
    let refs: Vec<&Tensor> = outs.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Additive [sq, sk] mask: 0 where key j is visible to query i (j <= i + sk - sq),
/// -inf elsewhere. Built on the host; it broadcasts over batch and heads.
pub(crate) fn causal_mask(sq: usize, sk: usize) -> Result<Tensor> {
    let offset = sk.saturating_sub(sq);
    let mut data = vec![0.0f32; sq * sk];
    for (i, row) in data.chunks_exact_mut(sk.max(1)).enumerate().take(sq) {
        for (j, v) in row.iter_mut().enumerate() {
            if j > i + offset {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(&data, &[sq, sk])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, DType::F32)
    }

    #[test]
    fn non_composite_is_none() {
        let t = leaf(&[2, 3]).exp().unwrap();
        assert!(decompose(&t).unwrap().is_none());
    }

    #[test]
    fn softmax_keeps_shape_and_dtype() {
        let x = Tensor::zeros(&[2, 5], DType::F16);
        let d = decompose(&x.softmax().unwrap()).unwrap().unwrap();
        assert_eq!(d.shape(), &[2, 5]);
        assert_eq!(d.dtype(), DType::F16);
    }

    #[test]
    fn layer_norm_shape() {
        let x = leaf(&[3, 4, 8]);
        let ln = x.layer_norm(&leaf(&[8]), &leaf(&[8]), 1e-5).unwrap();
        let d = decompose(&ln).unwrap().unwrap();
        assert_eq!(d.shape(), &[3, 4, 8]);
    }

    #[test]
    fn attention_shape_with_mask_and_causal() {
        let q = leaf(&[1, 2, 3, 4]);
        let k = leaf(&[1, 2, 5, 4]);
        let m = leaf(&[3, 5]);
        let a = Tensor::attention(&q, &k, &k, Some(&m), 0.5, true).unwrap();
        let d = decompose(&a).unwrap().unwrap();
        assert_eq!(d.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn causal_mask_values() {
        let m = causal_mask(2, 3).unwrap();
        let Op::Host(h) = m.op() else { panic!("expected host leaf") };
        let v: &[f32] = bytemuck::cast_slice(h.bytes());
        // offset = 1: row 0 sees j <= 1, row 1 sees all.
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], f32::NEG_INFINITY);
        assert!(v[3..].iter().all(|&x| x == 0.0));
    }
}
