//! Output dtype policy and tensor encoding, independent of any model.
//!
//! Models classify each tensor by its [`Role`]; the policy maps (role, shape, dtype) to a
//! storage [`Target`]:
//! - fp32: everything f32.
//! - fp16: `Param` (norms, biases, any 1-D tensor) f32, everything else f16.
//! - int8 / int4: `Linear` weights quantized int8 / int4, `LinearInt8` weights int8 for
//!   both; `Dense` weights f16; `Param` f32. Weights whose last dim is not a multiple of
//!   the block size (32) fall back to f16.
//! - q4_k / q5_k / q6_k: `Linear` weights in that ggml k-quant; `LinearInt8` (token
//!   embeddings / output projections) q6_k, as llama.cpp does for its output tensor. Last
//!   dims that are not a multiple of 256 fall back to int8 (then f16).
//! - q4_k_m: llama.cpp's Q4_K_M mix — q4_k, but q6_k for `LinearInt8` and for the
//!   attention value and FFN down projections of the layers `use_more_bits` selects
//!   (first and last eighth, every third in between); see [`target_named`].
//!
//! Quantized tensors are 2-D `[rows, cols]` or 3-D expert stacks `[E, rows, cols]` (each
//! expert matrix in its own layout, back to back).

use lapid_quant::Format;
use lapid_tensor::{DType, Result, bail};
use half::f16;

use crate::OutDType;

/// Precision class of a model tensor (assigned by the model's converter).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// LayerNorm parameters, biases, 1-D tensors: always f32.
    Param,
    /// Linear (matmul) weight `[out, in]`: quantized for int8 / int4.
    Linear,
    /// Linear weight that is never quantized below int8 (e.g. a tied token embedding that
    /// is also the output projection).
    LinearInt8,
    /// Other weights (convolutions, positional tables): f16 unless fp32.
    Dense,
}

/// Storage of one tensor in the output file.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Target {
    F32,
    F16,
    Quant(Format),
}

impl Target {
    pub fn dtype(self) -> DType {
        match self {
            Target::F32 => DType::F32,
            Target::F16 => DType::F16,
            Target::Quant(Format::Int8) => DType::Int8,
            Target::Quant(Format::Int4) => DType::Int4,
            Target::Quant(Format::Q4K) => DType::Q4K,
            Target::Quant(Format::Q5K) => DType::Q5K,
            Target::Quant(Format::Q6K) => DType::Q6K,
            Target::Quant(Format::Mxfp4) => DType::Mxfp4,
        }
    }
}

/// Whether `shape` (2-D, or a 3-D expert stack) can be stored block-quantized in `format`.
pub fn quantizable_as(format: Format, shape: &[usize]) -> bool {
    matches!(shape.len(), 2 | 3) && shape[shape.len() - 1] > 0 && shape[shape.len() - 1] % format.block_elems() == 0
}

/// Whether `shape` can be stored in the 32-element block formats (int8 / int4).
pub fn quantizable(shape: &[usize]) -> bool {
    quantizable_as(Format::Int8, shape)
}

/// `format` if `shape` fits it, else int8, else f16.
fn quant_or_fallback(format: Format, shape: &[usize]) -> Target {
    if quantizable_as(format, shape) {
        Target::Quant(format)
    } else if quantizable(shape) {
        Target::Quant(Format::Int8)
    } else {
        Target::F16
    }
}

/// Storage target for a tensor of `role` and `shape` at output precision `dtype`. Name-free:
/// q4_k_m is treated as q4_k with q6_k `LinearInt8` (use [`target_named`] for the full mix).
pub fn target(role: Role, shape: &[usize], dtype: OutDType) -> Target {
    if dtype == OutDType::F32 || role == Role::Param || shape.len() <= 1 {
        return Target::F32;
    }
    let quant = match (role, dtype) {
        (Role::Dense, _) | (_, OutDType::F16 | OutDType::F32) => return Target::F16,
        (Role::LinearInt8, OutDType::Q4K | OutDType::Q5K | OutDType::Q6K | OutDType::Q4KM) => Format::Q6K,
        (Role::LinearInt8, _) | (_, OutDType::Int8) => Format::Int8,
        (_, OutDType::Int4) => Format::Int4,
        (_, OutDType::Q4K | OutDType::Q4KM) => Format::Q4K,
        (_, OutDType::Q5K) => Format::Q5K,
        (_, OutDType::Q6K) => Format::Q6K,
    };
    quant_or_fallback(quant, shape)
}

/// [`target`] with the name-dependent rules of the q4_k_m mix: attention value and FFN
/// down projections (recognized by their module name, see [`mix_kind`]) in layer `i` of
/// `n_layer` get q6_k when llama.cpp's `use_more_bits(i, n_layer)` holds. Without a layer
/// count only the first-eighth / every-third rule cannot be applied, so they stay q4_k.
pub fn target_named(name: &str, role: Role, shape: &[usize], dtype: OutDType, n_layer: Option<usize>) -> Target {
    let t = target(role, shape, dtype);
    if dtype != OutDType::Q4KM || t != Target::Quant(Format::Q4K) || mix_kind(name).is_none() {
        return t;
    }
    match (layer_index(name), n_layer) {
        (Some(i), Some(n)) if i < n && use_more_bits(i, n) => quant_or_fallback(Format::Q6K, shape),
        _ => t,
    }
}

/// Tensors the q4_k_m mix may upgrade.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MixKind {
    AttnV,
    FfnDown,
}

/// Classify a weight by its module name (the segment before `.weight`): attention value
/// (`v_proj`, `attn_v`, `wv`, `value`, `linear_v`) or FFN down projection (`down_proj`,
/// `ffn_down`, `ffn_down_exps`, `w2`, `fc2`, `linear2`).
pub fn mix_kind(name: &str) -> Option<MixKind> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let module = stem.rsplit('.').next().unwrap_or(stem);
    match module {
        "v_proj" | "attn_v" | "wv" | "value" | "linear_v" => Some(MixKind::AttnV),
        "down_proj" | "ffn_down" | "ffn_down_exps" | "w2" | "fc2" | "linear2" => Some(MixKind::FfnDown),
        _ => None,
    }
}

/// Layer index of a name like `model.layers.7.mlp…`, `blk.7.…`, `blocks.7.…`.
pub fn layer_index(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    while let Some(p) = parts.next() {
        if matches!(p, "layers" | "layer" | "blk" | "blocks" | "h")
            && let Some(i) = parts.clone().next().and_then(|n| n.parse().ok())
        {
            return Some(i);
        }
    }
    None
}

/// llama.cpp `use_more_bits`: first and last eighth of the layers, every third in between.
pub fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
}

/// Encode f32 values (row-major, `shape`) into the target dtype's bytes.
pub fn encode(target: Target, data: &[f32], shape: &[usize]) -> Result<(DType, Vec<u8>)> {
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        bail!("shape_mismatch", "encode: {} values for shape {shape:?}", data.len());
    }
    let bytes = match target {
        Target::F32 => {
            let mut b = vec![0u8; numel * 4];
            crate::parallel::convert(data, &mut b, 4, |v, o| o.copy_from_slice(&v.to_le_bytes()));
            b
        }
        Target::F16 => {
            let mut b = vec![0u8; numel * 2];
            crate::parallel::convert(data, &mut b, 2, |v, o| o.copy_from_slice(&f16::from_f32(*v).to_le_bytes()));
            b
        }
        Target::Quant(f) => {
            if !quantizable_as(f, shape) {
                bail!(
                    "shape_mismatch",
                    "{f:?} tensor must be 2-D (or a 3-D expert stack) with last dim % {} == 0, got {shape:?}",
                    f.block_elems()
                );
            }
            let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let mut out = Vec::new();
            for m in data.chunks_exact(rows * cols) {
                out.extend(quantize_par(f, m, rows, cols));
            }
            out
        }
    };
    Ok((target.dtype(), bytes))
}

/// `lapid_quant::quantize` of a `[rows, cols]` matrix on all cores: row ranges are
/// quantized separately and spliced into the full layout (quants and scales are row-major in
/// every format), so the bytes are identical.
pub fn quantize_par(f: Format, data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let l = lapid_quant::layout(f, rows, cols);
    // Small matrices: not worth the threads.
    if rows * cols < 1 << 18 || rows < 2 {
        return lapid_quant::quantize(f, data, rows, cols);
    }
    let parts = crate::parallel::map_ranges(rows, 64, |r| (r.clone(), lapid_quant::quantize(f, &data[r.start * cols..r.end * cols], r.len(), cols)));
    let mut out = vec![0u8; l.total];
    let (qrow, srow) = (l.qs_bytes / rows, l.scales_bytes / rows);
    for (r, b) in parts {
        let pl = lapid_quant::layout(f, r.len(), cols);
        out[l.qs_offset + r.start * qrow..l.qs_offset + r.end * qrow].copy_from_slice(&b[pl.qs_offset..pl.qs_offset + pl.qs_bytes]);
        if srow > 0 {
            out[l.scales_offset + r.start * srow..l.scales_offset + r.end * srow]
                .copy_from_slice(&b[pl.scales_offset..pl.scales_offset + pl.scales_bytes]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_table() {
        let sq = [768, 768];
        assert_eq!(target(Role::Linear, &sq, OutDType::F32), Target::F32);
        assert_eq!(target(Role::Linear, &sq, OutDType::F16), Target::F16);
        assert_eq!(target(Role::Linear, &sq, OutDType::Int8), Target::Quant(Format::Int8));
        assert_eq!(target(Role::Linear, &sq, OutDType::Int4), Target::Quant(Format::Int4));
        assert_eq!(target(Role::LinearInt8, &[51865, 768], OutDType::Int4), Target::Quant(Format::Int8));
        assert_eq!(target(Role::Dense, &[768, 80, 3], OutDType::Int8), Target::F16);
        assert_eq!(target(Role::Dense, &[1500, 768], OutDType::Int4), Target::F16);
        assert_eq!(target(Role::Param, &[768], OutDType::F16), Target::F32);
        assert_eq!(target(Role::Linear, &[3072], OutDType::Int8), Target::F32);
        assert_eq!(target(Role::Linear, &[10, 33], OutDType::Int8), Target::F16);
        assert_eq!(target(Role::Linear, &sq, OutDType::Q4K), Target::Quant(Format::Q4K));
        assert_eq!(target(Role::Linear, &sq, OutDType::Q5K), Target::Quant(Format::Q5K));
        assert_eq!(target(Role::Linear, &[768, 1152], OutDType::Q4K), Target::Quant(Format::Int8));
        assert_eq!(target(Role::LinearInt8, &[262144, 768], OutDType::Q4KM), Target::Quant(Format::Q6K));
        assert_eq!(target(Role::Linear, &[8, 768, 512], OutDType::Q4K), Target::Quant(Format::Q4K));
        assert_eq!(target(Role::Dense, &sq, OutDType::Q6K), Target::F16);
    }

    #[test]
    fn q4_k_m_mix() {
        let t = |n: &str| target_named(n, Role::Linear, &[2048, 2048], OutDType::Q4KM, Some(24));
        let q6 = Target::Quant(Format::Q6K);
        let q4 = Target::Quant(Format::Q4K);
        // n = 24: more bits for i < 3, i >= 21, (i - 3) % 3 == 2.
        assert_eq!(t("model.layers.0.self_attn.v_proj.weight"), q6);
        assert_eq!(t("model.layers.3.self_attn.v_proj.weight"), q4);
        assert_eq!(t("model.layers.5.feed_forward.w2.weight"), q6);
        assert_eq!(t("blk.22.ffn_down.weight"), q6);
        assert_eq!(t("model.layers.5.self_attn.q_proj.weight"), q4);
        assert_eq!(t("model.layers.4.mlp.down_proj.weight"), q4);
        assert_eq!(target_named("model.layers.0.mlp.down_proj.weight", Role::Linear, &[2048, 2048], OutDType::Q4K, Some(24)), q4);
        assert_eq!(layer_index("model.layers.12.x.weight"), Some(12));
        assert_eq!(layer_index("embed.weight"), None);
        assert_eq!((0..24).filter(|&i| use_more_bits(i, 24)).count(), 12);
    }

    #[test]
    fn encode_sizes() {
        let x = vec![0.25f32; 512];
        assert_eq!(encode(Target::F32, &x[..64], &[64]).map(|r| r.1.len()).ok(), Some(256));
        assert_eq!(encode(Target::F16, &x[..64], &[2, 32]).map(|r| r.1.len()).ok(), Some(128));
        let (dt, b) = encode(Target::Quant(Format::Int4), &x[..64], &[2, 32]).expect("quant");
        assert_eq!((dt, b.len()), (DType::Int4, DType::Int4.storage_bytes(&[2, 32])));
        let (dt, b) = encode(Target::Quant(Format::Q4K), &x, &[2, 1, 256]).expect("experts");
        assert_eq!((dt, b.len()), (DType::Q4K, DType::Q4K.storage_bytes(&[2, 1, 256])));
        assert!(encode(Target::F32, &x[..64], &[3]).is_err());
        assert!(encode(Target::Quant(Format::Q6K), &x[..64], &[2, 32]).is_err());
    }

    #[test]
    fn parallel_quantize_identical() {
        let (rows, cols) = (1031, 512);
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1013) as f32 / 97.0 - 5.0).collect();
        for f in [Format::Int8, Format::Int4, Format::Q4K, Format::Q6K, Format::Mxfp4] {
            assert!(quantize_par(f, &x, rows, cols) == lapid_quant::quantize(f, &x, rows, cols), "{f:?}");
        }
    }
}
