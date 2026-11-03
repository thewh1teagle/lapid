//! Host-side conversion of raw tensor bytes (little endian, row-major) to f32 / i32.

use lapid_tensor::{DType, Result, bail};
use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Decode `bytes` (exactly `dtype.storage_bytes(shape)`) as f32, dequantizing if needed.
pub(crate) fn to_f32(bytes: &[u8], dtype: DType, shape: &[usize]) -> Result<Vec<f32>> {
    let n = numel(shape);
    Ok(match dtype {
        DType::F32 => match bytemuck::try_cast_slice::<u8, f32>(bytes) {
            Ok(v) => v.to_vec(),
            Err(_) => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        },
        DType::F16 => {
            let mut out = vec![0.0f32; n];
            match bytemuck::try_cast_slice::<u8, f16>(bytes) {
                Ok(h) => h.convert_to_f32_slice(&mut out),
                Err(_) => {
                    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                        *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
                    }
                }
            }
            out
        }
        DType::BF16 => {
            let mut out = vec![0.0f32; n];
            match bytemuck::try_cast_slice::<u8, bf16>(bytes) {
                Ok(h) => h.convert_to_f32_slice(&mut out),
                Err(_) => {
                    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                        *o = bf16::from_le_bytes([c[0], c[1]]).to_f32();
                    }
                }
            }
            out
        }
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => {
            let Some(format) = dtype.quant_format() else {
                bail!("unsupported_dtype", "no quant format for {}", dtype.name());
            };
            // 2-D matrix, or a 3-D expert stack of matrices back to back.
            let (count, rows, cols) = match *shape {
                [rows, cols] => (1, rows, cols),
                [e, rows, cols] => (e, rows, cols),
                _ => bail!("shape_mismatch", "quantized tensors are 2-D or 3-D, got {shape:?}"),
            };
            let one = lapid_quant::layout(format, rows, cols).total;
            if bytes.len() < count * one {
                bail!("shape_mismatch", "quantized {shape:?} needs {} bytes, got {}", count * one, bytes.len());
            }
            let mut out = Vec::with_capacity(count * rows * cols);
            for e in 0..count {
                out.extend(lapid_quant::dequantize(format, &bytes[e * one..(e + 1) * one], rows, cols));
            }
            out
        }
    })
}

/// Decode `bytes` as i32. Float tensors are truncated toward zero (saturating).
pub(crate) fn to_i32(bytes: &[u8], dtype: DType, shape: &[usize]) -> Result<Vec<i32>> {
    match dtype {
        DType::I32 => Ok(match bytemuck::try_cast_slice::<u8, i32>(bytes) {
            Ok(v) => v.to_vec(),
            Err(_) => bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        }),
        DType::F32 | DType::F16 | DType::BF16 => {
            Ok(to_f32(bytes, dtype, shape)?.into_iter().map(|x| x as i32).collect())
        }
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => {
            bail!("unsupported_dtype", "cannot read quantized {} tensor as i32", dtype.name())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let vals = [1.0f32, -2.5, 0.0, 65504.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
        assert_eq!(to_f32(&bytes, DType::F16, &[4]).unwrap(), vals);
        // unaligned path
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&bytes);
        assert_eq!(to_f32(&shifted[1..], DType::F16, &[4]).unwrap(), vals);
    }

    #[test]
    fn i32_and_f32() {
        let bytes: Vec<u8> = [3i32, -7].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(to_i32(&bytes, DType::I32, &[2]).unwrap(), [3, -7]);
        assert_eq!(to_f32(&bytes, DType::I32, &[2]).unwrap(), [3.0, -7.0]);
        let fb: Vec<u8> = [2.9f32, -1.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(to_i32(&fb, DType::F32, &[2]).unwrap(), [2, -1]);
    }
}
