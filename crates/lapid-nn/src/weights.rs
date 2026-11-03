//! Loading weights from a `.lapid` file into layers, plus the fake-quant reference
//! precisions every quantized model is verified against.

use lapid::{DType, Device, Error, LapidFile, ReduceOp, Result, Tensor, UnaryOp};

use crate::layers::{Conv1d, Embedding, LayerNorm, Linear, RmsNorm};

/// How a model is built from its file.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Weights as stored (quantized kernels for int8/int4 weights).
    Stored,
    /// Fake-quant reference: quantized weights (int8/int4/k-quants, incl. 3-D expert stacks)
    /// dequantized to f32, f32 matmuls.
    FakeQuantWeights,
    /// As `FakeQuantWeights`, plus every input of a quantized matmul quantized to int8 per
    /// 32-block (`d = max|x| / 127`, round to nearest even) — the arithmetic of kernels that
    /// quantize activations on the fly.
    FakeQuantActivations,
}

/// Fetches weights from a `.lapid` file, uploads them to the device and builds layers.
/// Names follow the checkpoint (`{prefix}.weight`, `{prefix}.bias`).
pub struct Loader<'a> {
    pub file: &'a LapidFile,
    pub device: &'a Device,
    pub precision: Precision,
}

impl<'a> Loader<'a> {
    pub fn new(file: &'a LapidFile, device: &'a Device, precision: Precision) -> Loader<'a> {
        Loader { file, device, precision }
    }

    /// Weight `name` on the device (dequantized to f32 for the fake-quant precisions).
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let t = self.file.get(name)?;
        if self.precision != Precision::Stored && t.dtype().is_quant() {
            return self.device.upload(&dequantized(self.file, name, &t)?);
        }
        self.device.upload(&t)
    }

    /// Weight `name` as a host leaf, never copied to the device: the file's (mmap'd) bytes
    /// as stored, or dequantized to f32 for the fake-quant precisions. For weights a model
    /// moves to the device itself, e.g. MoE expert stacks read through an
    /// [`ExpertCache`](crate::llm::expert_cache::ExpertCache).
    pub fn host_tensor_checked(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.file.get(name)?;
        let t = if self.precision != Precision::Stored && t.dtype().is_quant() { dequantized(self.file, name, &t)? } else { t };
        self.expect_shape(&t, name, shape)?;
        Ok(t)
    }

    /// Whether the file has a tensor `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.file.get(name).is_ok()
    }

    /// Whether matmuls with weight `name` quantize their input (fake-quant activations of a
    /// weight stored quantized).
    pub fn act_quant(&self, name: &str) -> Result<bool> {
        Ok(self.precision == Precision::FakeQuantActivations && self.file.get(name)?.dtype().is_quant())
    }

    /// Weight as a realized f32 tensor (e.g. positional tables stored as f16).
    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let t = self.tensor(name)?;
        if t.dtype() == DType::F32 {
            return Ok(t);
        }
        let out = self.device.eval(&[&t.cast(DType::F32)?])?;
        out.into_iter().next().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))
    }

    /// Check a weight's shape (catches wrong / mismatched files early).
    pub fn expect_shape(&self, t: &Tensor, name: &str, shape: &[usize]) -> Result<()> {
        if t.shape() != shape {
            return Err(Error::new(
                "shape_mismatch",
                format!("{name}: expected shape {shape:?}, file has {:?}", t.shape()),
            ));
        }
        Ok(())
    }

    /// `{prefix}.weight` [out, in] and optionally `{prefix}.bias` [out].
    pub fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        let name = format!("{prefix}.weight");
        let weight = self.tensor(&name)?;
        let bias = if bias { Some(self.tensor(&format!("{prefix}.bias"))?) } else { None };
        Ok(Linear::new(weight, bias).with_act_quant(self.act_quant(&name)?))
    }

    /// [`linear`](Self::linear) checked to be `[out, inp]`.
    pub fn linear_checked(&self, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Linear> {
        let lin = self.linear(prefix, bias)?;
        self.expect_shape(&lin.weight, &format!("{prefix}.weight"), &[out, inp])?;
        Ok(lin)
    }

    /// [`tensor`](Self::tensor) checked to have `shape`.
    pub fn tensor_checked(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.tensor(name)?;
        self.expect_shape(&t, name, shape)?;
        Ok(t)
    }

    /// [`rms_norm`](Self::rms_norm) checked to be `[dim]`.
    pub fn rms_norm_checked(&self, prefix: &str, dim: usize, eps: f32) -> Result<RmsNorm> {
        let n = self.rms_norm(prefix, eps)?;
        self.expect_shape(&n.weight, &format!("{prefix}.weight"), &[dim])?;
        Ok(n)
    }

    /// `{prefix}.weight` and `{prefix}.bias`.
    pub fn layer_norm(&self, prefix: &str, eps: f32) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{prefix}.weight"))?,
            self.tensor(&format!("{prefix}.bias"))?,
            eps,
        ))
    }

    /// `{prefix}.weight`.
    pub fn rms_norm(&self, prefix: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm::new(self.tensor(&format!("{prefix}.weight"))?, eps))
    }

    /// `{prefix}.weight` [vocab, dim].
    pub fn embedding(&self, prefix: &str) -> Result<Embedding> {
        Ok(Embedding::new(self.tensor(&format!("{prefix}.weight"))?))
    }

    /// `{prefix}.weight` [Cout, Cin/groups, K] and optionally `{prefix}.bias`; stride 1, no
    /// padding, one group — configure with the `Conv1d::with_*` builders.
    pub fn conv1d(&self, prefix: &str, bias: bool) -> Result<Conv1d> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        let bias = if bias { Some(self.tensor(&format!("{prefix}.bias"))?) } else { None };
        Ok(Conv1d::new(weight, bias))
    }
}

/// Quantize-dequantize the last axis of `x` in int8 blocks of 32 (the on-the-fly activation
/// quantization of quantized matmul kernels): `d = max|x| / 127`, `q = round(x / d)`, `x' = q d`.
pub fn fake_quant_act(x: &Tensor) -> Result<Tensor> {
    /// Adding and subtracting 1.5 * 2^23 rounds an f32 of magnitude < 2^22 to nearest even.
    const ROUND_MAGIC: f64 = 12_582_912.0;
    let shape = x.shape().to_vec();
    let k = shape.last().copied().unwrap_or(0);
    let group = lapid::quant::GROUP;
    if k % group != 0 {
        return Err(Error::new(
            "shape_mismatch",
            format!("activation quantization needs last dim % {group} == 0, got {shape:?}"),
        ));
    }
    let mut blocks = shape.clone();
    blocks.pop();
    blocks.extend([k / group, group]);
    let xb = x.contiguous()?.reshape(&blocks)?;
    let d = xb.unary(UnaryOp::Abs)?.reduce(ReduceOp::Max, -1, true)?.mul_scalar(1.0 / 127.0)?;
    // Blocks of zeros: d = 0 → inverse 0 (as in the kernels), not inf.
    let safe = d.maximum(&Tensor::full(f32::MIN_POSITIVE as f64, d.shape(), DType::F32))?;
    let q = xb.mul(&safe.unary(UnaryOp::Recip)?)?.add_scalar(ROUND_MAGIC)?.add_scalar(-ROUND_MAGIC)?;
    q.mul(&d)?.reshape(&shape)
}

/// f32 copy of a quantized weight: exactly the values the quantized kernels multiply with.
/// 2-D `[rows, cols]` matrices or 3-D `[E, rows, cols]` expert stacks (E matrices back to back),
/// any quantized format (int8 / int4 / k-quants).
fn dequantized(file: &LapidFile, name: &str, t: &Tensor) -> Result<Tensor> {
    let Some(format) = t.dtype().quant_format() else {
        return Err(Error::new("internal", format!("{name}: {} is not a quantized dtype", t.dtype().name())));
    };
    let (experts, rows, cols) = match *t.shape() {
        [rows, cols] => (1, rows, cols),
        [e, rows, cols] => (e, rows, cols),
        _ => {
            return Err(Error::new(
                "shape_mismatch",
                format!("{name}: quantized weight must be 2-D or a 3-D expert stack, got {:?}", t.shape()),
            ));
        }
    };
    let bytes = file.bytes(name)?;
    let per = lapid::quant::layout(format, rows, cols).total;
    if bytes.len() != per * experts {
        return Err(Error::new("format", format!("{name}: {} bytes, layout needs {}", bytes.len(), per * experts)));
    }
    let mut out = Vec::with_capacity(experts * rows * cols);
    for e in 0..experts {
        out.extend(lapid::quant::dequantize(format, &bytes[e * per..(e + 1) * per], rows, cols));
    }
    Tensor::from_f32(&out, t.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::eval;

    #[test]
    fn fake_quant_act_matches_int8_quantizer() {
        let (rows, cols) = (3, 64);
        let mut x: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 0.013).collect();
        x[5] = 40.0; // outlier block
        x[64..96].fill(0.0); // zero block
        let t = Tensor::from_f32(&x, &[rows, cols]).expect("tensor");
        let got = eval(&fake_quant_act(&t).expect("graph"));
        let q = lapid::quant::quantize(lapid::quant::Format::Int8, &x, rows, cols);
        let want = lapid::quant::dequantize(lapid::quant::Format::Int8, &q, rows, cols);
        // The stored scale is f16 in the weight format; activations keep an f32 scale.
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 2e-3 * w.abs().max(0.05), "{i}: {g} vs {w}");
        }
        assert!(got[64..96].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fake_quant_act_needs_block_multiple() {
        let t = Tensor::zeros(&[2, 33], DType::F32);
        assert_eq!(fake_quant_act(&t).unwrap_err().code, "shape_mismatch");
    }
}
