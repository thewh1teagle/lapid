//! Element types. CONTRACT FILE — owned by the coordinator. Do not change public items
//! without updating plans/lapid-v1/CONTRACT.md.

use serde::{Deserialize, Serialize};

/// Element type of a tensor.
///
/// Plain types store one element per slot. Quantized types (`Int8`, `Int4`) store a
/// 2-D `[rows, cols]` matrix (cols % 32 == 0) in the SoA layout defined by `lapid-quant`:
/// a quants plane followed by an f16 scales plane (one scale per 32 elements of a row).
/// Quantized tensors may only be used as: the weight (`rhs`, transposed, 2-D) of `MatMul`
/// or the table of `Gather` (and may be passed through `Reshape` to the same shape).
/// All compute happens in f32; f16/bf16 are storage formats.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    BF16,
    #[serde(rename = "i32")]
    I32,
    /// 8-bit symmetric ints, group of 32 along the row, f16 scale. x = q * d.
    #[serde(rename = "int8")]
    Int8,
    /// 4-bit ints, group of 32 along the row, f16 scale. x = (q - 8) * d.
    #[serde(rename = "int4")]
    Int4,
    /// ggml Q4_K (256-element super-blocks, AoS as in GGUF; see lapid-quant). Same usage
    /// rules as Int8/Int4 (MatMul b_transposed 2-D, MatMulId experts, Gather table).
    #[serde(rename = "q4_k")]
    Q4K,
    #[serde(rename = "q5_k")]
    Q5K,
    #[serde(rename = "q6_k")]
    Q6K,
    /// MXFP4 (32-element blocks, e8m0 scale + e2m1 values; ggml `block_mxfp4` AoS, see
    /// lapid-quant). Same usage rules as the K formats.
    #[serde(rename = "mxfp4")]
    Mxfp4,
}

impl DType {
    pub const QUANT_GROUP: usize = 32;

    pub fn is_quant(self) -> bool {
        matches!(self, DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4)
    }

    /// The lapid-quant format of a quantized dtype.
    pub fn quant_format(self) -> Option<lapid_quant::Format> {
        Some(match self {
            DType::Int8 => lapid_quant::Format::Int8,
            DType::Int4 => lapid_quant::Format::Int4,
            DType::Q4K => lapid_quant::Format::Q4K,
            DType::Q5K => lapid_quant::Format::Q5K,
            DType::Q6K => lapid_quant::Format::Q6K,
            DType::Mxfp4 => lapid_quant::Format::Mxfp4,
            _ => return None,
        })
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Bytes per element for plain types. Panics for quantized types (use `storage_bytes`).
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => panic!("size() on quantized dtype {self:?}"),
        }
    }

    /// Spec id, e.g. "f16", "int4".
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::Int8 => "int8",
            DType::Int4 => "int4",
            DType::Q4K => "q4_k",
            DType::Q5K => "q5_k",
            DType::Q6K => "q6_k",
            DType::Mxfp4 => "mxfp4",
        }
    }

    pub fn from_name(s: &str) -> Option<DType> {
        Some(match s {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "i32" => DType::I32,
            "int8" => DType::Int8,
            "int4" => DType::Int4,
            "q4_k" => DType::Q4K,
            "q5_k" => DType::Q5K,
            "q6_k" => DType::Q6K,
            "mxfp4" => DType::Mxfp4,
            _ => return None,
        })
    }

    /// Total bytes needed to store `shape` in this dtype. For quantized types the shape
    /// must be 2-D `[rows, cols]` and the result is `lapid_quant::layout(..).total`.
    pub fn storage_bytes(self, shape: &[usize]) -> usize {
        if self.is_quant() {
            assert!(shape.len() == 2 || shape.len() == 3, "quantized tensors are 2-D (or 3-D expert stacks)");
            let q = self.quant_format().expect("quantized dtype");
            if shape.len() == 3 {
                // MatMulId expert stack [E, N, K]: E matrices back to back.
                return shape[0] * lapid_quant::layout(q, shape[1], shape[2]).total;
            }
            lapid_quant::layout(q, shape[0], shape[1]).total
        } else {
            shape.iter().product::<usize>() * self.size()
        }
    }
}
