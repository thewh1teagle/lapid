//! Lazy tensor graph. CONTRACT FILE — the struct definitions and every `pub fn`
//! signature are owned by the coordinator. Implementers (agent "tensor") replace the
//! `todo!()` bodies; they may add private helpers / new files but must not rename or
//! change signatures of existing pub items.
//!
//! A `Tensor` is an immutable handle to a node of a DAG. Building ops never computes;
//! evaluation is done by the `lapid` crate (compiler + backend). Every builder validates
//! shapes/dtypes and returns `Result` with code "shape_mismatch" / "unsupported_dtype" /
//! "invalid_argument". Binary ops follow numpy broadcasting: the builder inserts
//! `Expand` (and `Reshape` to add leading 1-dims) so the `Op::Binary` node's srcs all
//! have exactly the output shape. Scalars (`*_scalar`) become `Op::Const` nodes of the
//! output shape and dtype.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::bail;
use crate::buffer::{DeviceData, HostData};
use crate::dtype::DType;
use crate::error::Result;
use crate::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use smallvec::{SmallVec, smallvec};

mod check;
mod index;
mod linalg;
mod moe;
mod node;
#[cfg(test)]
mod tests;

use check::broadcast_shapes;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

use node::NodeRef;
pub use node::{Node, Shape, Srcs};

#[derive(Clone, Debug)]
pub struct Tensor(pub(crate) NodeRef);

impl Tensor {
    // ---------- raw construction ----------
    /// Create a node. No validation — builders below validate before calling this.
    pub fn from_op(op: Op, src: impl Into<Srcs>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
        Tensor(NodeRef::new(Node {
            id: next_id(),
            op,
            src: src.into(),
            shape: shape.into(),
            dtype,
            name: None,
            mark: Default::default(),
        }))
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }
    pub fn op(&self) -> &Op {
        &self.0.op
    }
    pub fn src(&self) -> &[Tensor] {
        &self.0.src
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.0.shape.iter().product()
    }
    pub fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }
    /// Returns a tensor that is the same node content with a debug name attached
    /// (a new node with identical op/src; cheap).
    pub fn named(&self, name: impl Into<String>) -> Tensor {
        Tensor(NodeRef::new(Node {
            id: next_id(),
            op: self.op().clone(),
            src: self.src().into(),
            shape: self.shape().into(),
            dtype: self.dtype(),
            name: Some(name.into().into_boxed_str()),
            mark: Default::default(),
        }))
    }
    /// Resolve a possibly-negative axis.
    pub fn axis(&self, axis: isize) -> Result<usize> {
        check::resolve_axis(axis, self.rank())
    }

    // ---------- leaves ----------
    pub fn from_host(data: HostData, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let expected = check::storage_bytes(shape, dtype)?;
        let got = data.bytes().len();
        if got != expected {
            bail!(
                "shape_mismatch",
                "host data has {got} bytes but shape {shape:?} of {} needs {expected}",
                dtype.name()
            );
        }
        Ok(Tensor::from_op(Op::Host(data), Srcs::new(), shape, dtype))
    }
    pub fn from_device(data: DeviceData, shape: &[usize], dtype: DType) -> Tensor {
        Tensor::from_op(Op::Device(data), Srcs::new(), shape, dtype)
    }
    pub fn from_f32(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        check::numel_matches(data.len(), shape)?;
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Tensor::from_host(HostData::from_vec(bytes), shape, DType::F32)
    }
    pub fn from_i32(data: &[i32], shape: &[usize]) -> Result<Tensor> {
        check::numel_matches(data.len(), shape)?;
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Tensor::from_host(HostData::from_vec(bytes), shape, DType::I32)
    }
    pub fn full(value: f64, shape: &[usize], dtype: DType) -> Tensor {
        Tensor::from_op(Op::Const(value), Srcs::new(), shape, dtype)
    }
    pub fn zeros(shape: &[usize], dtype: DType) -> Tensor {
        Tensor::full(0.0, shape, dtype)
    }

    // ---------- elementwise ----------
    pub fn unary(&self, op: UnaryOp) -> Result<Tensor> {
        let dt = self.dtype();
        let int_ok = dt == DType::I32 && matches!(op, UnaryOp::Neg | UnaryOp::Abs);
        if !(dt.is_float() || int_ok) {
            bail!("unsupported_dtype", "unary {op:?} is not defined for {}", dt.name());
        }
        Ok(Tensor::from_op(Op::Unary(op), smallvec![self.clone()], self.shape(), dt))
    }
    pub fn binary(&self, op: BinaryOp, rhs: &Tensor) -> Result<Tensor> {
        check::elementwise_dtype(self, "binary")?;
        index::check_int_only(op, self.dtype())?;
        check::elementwise_dtype(rhs, "binary")?;
        if self.dtype() != rhs.dtype() {
            bail!(
                "unsupported_dtype",
                "binary {op:?}: dtype mismatch {} vs {} (cast one side first)",
                self.dtype().name(),
                rhs.dtype().name()
            );
        }
        let shape = broadcast_shapes(self.shape(), rhs.shape())?;
        let a = self.broadcast_to(&shape)?;
        let b = rhs.broadcast_to(&shape)?;
        Ok(Tensor::from_op(Op::Binary(op), smallvec![a, b], shape, self.dtype()))
    }
    pub fn binary_scalar(&self, op: BinaryOp, rhs: f64) -> Result<Tensor> {
        check::elementwise_dtype(self, "binary")?;
        index::check_int_only(op, self.dtype())?;
        let c = Tensor::full(rhs, self.shape(), self.dtype());
        Ok(Tensor::from_op(Op::Binary(op), smallvec![self.clone(), c], self.shape(), self.dtype()))
    }
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Add, rhs)
    }
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Sub, rhs)
    }
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Mul, rhs)
    }
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Div, rhs)
    }
    pub fn maximum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Max, rhs)
    }
    pub fn minimum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Min, rhs)
    }
    pub fn add_scalar(&self, v: f64) -> Result<Tensor> {
        self.binary_scalar(BinaryOp::Add, v)
    }
    pub fn mul_scalar(&self, v: f64) -> Result<Tensor> {
        self.binary_scalar(BinaryOp::Mul, v)
    }
    pub fn neg(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Neg)
    }
    pub fn exp(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Exp)
    }
    pub fn log(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Log)
    }
    pub fn sqrt(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Sqrt)
    }
    pub fn tanh(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Tanh)
    }
    pub fn erf(&self) -> Result<Tensor> {
        self.unary(UnaryOp::Erf)
    }
    /// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))), built from primitives.
    pub fn gelu(&self) -> Result<Tensor> {
        check::float_dtype(self, "lapid")?;
        let cdf = self
            .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?
            .erf()?
            .add_scalar(1.0)?
            .mul_scalar(0.5)?;
        self.mul(&cdf)
    }
    /// cond != 0 ? a : b, with broadcasting of all three.
    pub fn where_(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        check::elementwise_dtype(cond, "where")?;
        check::elementwise_dtype(a, "where")?;
        check::elementwise_dtype(b, "where")?;
        if a.dtype() != b.dtype() {
            bail!(
                "unsupported_dtype",
                "where: branch dtypes differ ({} vs {})",
                a.dtype().name(),
                b.dtype().name()
            );
        }
        let shape = broadcast_shapes(&broadcast_shapes(cond.shape(), a.shape())?, b.shape())?;
        let srcs: Srcs = smallvec![cond.broadcast_to(&shape)?, a.broadcast_to(&shape)?, b.broadcast_to(&shape)?];
        Ok(Tensor::from_op(Op::Where, srcs, shape, a.dtype()))
    }
    pub fn cast(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype().is_quant() || dtype.is_quant() {
            bail!(
                "unsupported_dtype",
                "cast {} -> {}: quantized casts are not graph ops (use lapid-quant)",
                self.dtype().name(),
                dtype.name()
            );
        }
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        Ok(Tensor::from_op(Op::Cast(dtype), smallvec![self.clone()], self.shape(), dtype))
    }

    // ---------- reductions ----------
    pub fn reduce(&self, op: ReduceOp, axis: isize, keepdim: bool) -> Result<Tensor> {
        check::elementwise_dtype(self, "reduce")?;
        let axis = self.axis(axis)?;
        let mut shape = Shape::from_slice(self.shape());
        if keepdim {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        Ok(Tensor::from_op(Op::Reduce { op, axis, keepdim }, smallvec![self.clone()], shape, self.dtype()))
    }
    pub fn sum(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.reduce(ReduceOp::Sum, axis, keepdim)
    }
    pub fn max(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.reduce(ReduceOp::Max, axis, keepdim)
    }
    /// sum / n
    pub fn mean(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        check::float_dtype(self, "mean")?;
        let n = self.shape()[self.axis(axis)?];
        // n == 0 yields 0 * inf = NaN, matching numpy's mean of an empty axis.
        self.sum(axis, keepdim)?.mul_scalar(1.0 / n as f64)
    }
    /// Index (i32) of the max along the last axis (first on ties); the axis is removed.
    pub fn argmax_last(&self) -> Result<Tensor> {
        self.argmax(-1)
    }

    // ---------- movement ----------
    /// One dim may be usize::MAX meaning "infer".
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let shape = check::infer_shape(self.numel(), self.shape(), shape)?;
        if shape[..] == *self.shape() {
            return Ok(self.clone());
        }
        if self.dtype().is_quant() {
            bail!(
                "unsupported_dtype",
                "cannot reshape quantized {} tensor {:?} to {shape:?}",
                self.dtype().name(),
                self.shape()
            );
        }
        Ok(Tensor::from_op(Op::Reshape, smallvec![self.clone()], shape, self.dtype()))
    }
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor> {
        check::not_quant(self, "permute")?;
        let rank = self.rank();
        let mut seen: SmallVec<[bool; 8]> = SmallVec::from_elem(false, rank);
        if perm.len() != rank || !perm.iter().all(|&p| p < rank && !std::mem::replace(&mut seen[p], true)) {
            bail!("invalid_argument", "permute: {perm:?} is not a permutation of 0..{rank}");
        }
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return Ok(self.clone());
        }
        let shape: Shape = perm.iter().map(|&p| self.shape()[p]).collect();
        Ok(Tensor::from_op(Op::Permute(perm.to_vec()), smallvec![self.clone()], shape, self.dtype()))
    }
    pub fn transpose(&self, a: isize, b: isize) -> Result<Tensor> {
        let (a, b) = (self.axis(a)?, self.axis(b)?);
        let mut perm: Shape = (0..self.rank()).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }
    /// Numpy-style broadcast: missing leading dims are added, size-1 dims are expanded.
    pub fn expand(&self, shape: &[usize]) -> Result<Tensor> {
        self.broadcast_to(shape)
    }
    pub fn unsqueeze(&self, axis: isize) -> Result<Tensor> {
        let axis = check::resolve_axis(axis, self.rank() + 1)?;
        let mut shape = Shape::from_slice(self.shape());
        shape.insert(axis, 1);
        self.reshape(&shape)
    }
    pub fn squeeze(&self, axis: isize) -> Result<Tensor> {
        let axis = self.axis(axis)?;
        if self.shape()[axis] != 1 {
            bail!("shape_mismatch", "squeeze: axis {axis} of {:?} has size {} (must be 1)", self.shape(), self.shape()[axis]);
        }
        let mut shape = Shape::from_slice(self.shape());
        shape.remove(axis);
        self.reshape(&shape)
    }
    pub fn slice(&self, axis: isize, start: usize, end: usize) -> Result<Tensor> {
        check::not_quant(self, "slice")?;
        let axis = self.axis(axis)?;
        let dim = self.shape()[axis];
        if start > end || end > dim {
            bail!("invalid_argument", "slice [{start}, {end}) out of range for axis {axis} of {:?}", self.shape());
        }
        if start == 0 && end == dim {
            return Ok(self.clone());
        }
        let mut shape = Shape::from_slice(self.shape());
        shape[axis] = end - start;
        Ok(Tensor::from_op(Op::Slice { axis, start, end }, smallvec![self.clone()], shape, self.dtype()))
    }
    pub fn pad(&self, axis: isize, before: usize, after: usize, value: f64) -> Result<Tensor> {
        check::not_quant(self, "pad")?;
        let axis = self.axis(axis)?;
        if before == 0 && after == 0 {
            return Ok(self.clone());
        }
        let mut shape = Shape::from_slice(self.shape());
        shape[axis] += before + after;
        Ok(Tensor::from_op(Op::Pad { axis, before, after, value }, smallvec![self.clone()], shape, self.dtype()))
    }
    pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
        let Some(first) = tensors.first() else {
            bail!("invalid_argument", "cat of zero tensors");
        };
        let axis = first.axis(axis)?;
        for t in tensors {
            check::not_quant(t, "cat")?;
            if t.dtype() != first.dtype() {
                bail!("unsupported_dtype", "cat: dtype mismatch {} vs {}", first.dtype().name(), t.dtype().name());
            }
            let same = t.rank() == first.rank()
                && t.shape().iter().zip(first.shape()).enumerate().all(|(d, (a, b))| d == axis || a == b);
            if !same {
                bail!("shape_mismatch", "cat along axis {axis}: {:?} vs {:?}", first.shape(), t.shape());
            }
        }
        // Empty pieces contribute nothing (e.g. an empty KV cache on the first step).
        let parts: Vec<Tensor> = tensors.iter().filter(|t| t.shape()[axis] > 0).map(|t| (*t).clone()).collect();
        match parts.len() {
            0 => Ok((*first).clone()),
            1 => Ok(parts[0].clone()),
            _ => {
                let mut shape = Shape::from_slice(first.shape());
                shape[axis] = parts.iter().map(|t| t.shape()[axis]).sum();
                let dtype = first.dtype();
                Ok(Tensor::from_op(Op::Cat { axis }, parts, shape, dtype))
            }
        }
    }
    pub fn contiguous(&self) -> Result<Tensor> {
        check::not_quant(self, "contiguous")?;
        Ok(Tensor::from_op(Op::Contiguous, smallvec![self.clone()], self.shape(), self.dtype()))
    }

    // ---------- linear algebra / indexing ----------
    /// a @ b with batch broadcasting.
    pub fn matmul(&self, b: &Tensor) -> Result<Tensor> {
        self.matmul_impl(b, false)
    }
    /// a @ b^T where b is [n, k] (or batched [..., n, k]). Used by Linear: x @ W^T.
    /// Accepts quantized 2-D b.
    pub fn matmul_t(&self, b: &Tensor) -> Result<Tensor> {
        self.matmul_impl(b, true)
    }
    /// self is the table [V, D] (any float or quant dtype); indices I32. Output F32.
    pub fn gather(&self, indices: &Tensor) -> Result<Tensor> {
        self.gather_impl(indices)
    }

    // ---------- composites ----------
    pub fn softmax(&self) -> Result<Tensor> {
        self.softmax_impl()
    }
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        self.layer_norm_impl(weight, bias, eps)
    }

    /// RMS norm over the last axis: `x * rsqrt(mean(x^2) + eps) * weight` (f32 math, result in
    /// x's dtype). `weight` is `[D]`, float.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        self.rms_norm_impl(weight, eps)
    }
    /// Sinkhorn normalization of the trailing `[n, n]` matrices of f32 logits (see
    /// `Op::Sinkhorn`).
    pub fn sinkhorn(&self, iters: usize, eps: f32) -> Result<Tensor> {
        self.sinkhorn_impl(iters, eps)
    }
    /// q [B,H,Sq,D], k/v [B,H,Sk,D]; mask optional additive.
    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        linalg::attention(q, k, v, mask, scale, causal)
    }

    // ---------- private helpers ----------
    /// Broadcast to `shape` (numpy rules): prepend 1-dims with `Reshape`, then `Expand`.
    /// Returns `self` unchanged when the shape already matches.
    fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        if self.shape() == shape {
            return Ok(self.clone());
        }
        check::not_quant(self, "expand")?;
        if self.rank() > shape.len() {
            bail!("shape_mismatch", "cannot broadcast {:?} to lower-rank {shape:?}", self.shape());
        }
        let lead = shape.len() - self.rank();
        let compatible = self.shape().iter().zip(&shape[lead..]).all(|(&s, &d)| s == d || s == 1);
        if !compatible {
            bail!("shape_mismatch", "cannot broadcast {:?} to {shape:?}", self.shape());
        }
        let t = if lead > 0 {
            let mut padded = Shape::from_elem(1, lead);
            padded.extend_from_slice(self.shape());
            self.reshape(&padded)?
        } else {
            self.clone()
        };
        if t.shape() == shape {
            return Ok(t);
        }
        let dtype = t.dtype();
        Ok(Tensor::from_op(Op::Expand, smallvec![t], shape, dtype))
    }
}
