//! ggml tensor types and dequantizers (reference: ggml `ggml-common.h` block layouts and
//! `ggml-quants.c` `dequantize_row_*`). Used by the GGUF importer: F32/F16/BF16 are read
//! directly, Q8_0/Q4_0 can be repacked losslessly into lapid int8/int4
//! (`lapid_quant::from_ggml_blocks`), Q4_K/Q5_K/Q6_K are lapid dtypes stored as-is, every
//! other supported type is dequantized to f32 and re-quantized by the dtype policy.

use lapid_quant::kquant;
use lapid_tensor::{Error, Result, bail};
use half::{bf16, f16};

/// k-quant super-block size.
pub const QK_K: usize = kquant::QK_K;

/// A ggml tensor type (`enum ggml_type`). Unknown or unsupported ids are kept as `Other`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            30 => GgmlType::BF16,
            n => GgmlType::Other(n),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q8K => 15,
            GgmlType::BF16 => 30,
            GgmlType::Other(n) => n,
        }
    }

    /// ggml's name for the type (`"q4_K"`, ...); known-but-unsupported ids are named too.
    pub fn name(self) -> String {
        let s = match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::BF16 => "bf16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2K => "q2_K",
            GgmlType::Q3K => "q3_K",
            GgmlType::Q4K => "q4_K",
            GgmlType::Q5K => "q5_K",
            GgmlType::Q6K => "q6_K",
            GgmlType::Q8K => "q8_K",
            GgmlType::Other(n) => {
                const NAMES: [(u32, &str); 21] = [
                    (16, "iq2_xxs"), (17, "iq2_xs"), (18, "iq3_xxs"), (19, "iq1_s"), (20, "iq4_nl"),
                    (21, "iq3_s"), (22, "iq2_s"), (23, "iq4_xs"), (24, "i8"), (25, "i16"), (26, "i32"),
                    (27, "i64"), (28, "f64"), (29, "iq1_m"), (34, "tq1_0"), (35, "tq2_0"), (39, "mxfp4"),
                    (40, "nvfp4"), (41, "q1_0"), (42, "q2_0"), (4, "q4_2"),
                ];
                return NAMES.iter().find(|(i, _)| *i == n).map_or(format!("type {n}"), |(_, s)| s.to_string());
            }
        };
        s.to_string()
    }

    /// (elements per block, bytes per block); `None` for unsupported types.
    pub fn block(self) -> Option<(usize, usize)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q8_1 => (32, 36),
            GgmlType::Q2K => (QK_K, 84),
            GgmlType::Q3K => (QK_K, 110),
            GgmlType::Q4K => (QK_K, 144),
            GgmlType::Q5K => (QK_K, 176),
            GgmlType::Q6K => (QK_K, 210),
            GgmlType::Q8K => (QK_K, 292),
            // Plain integer / f64 tensors and MXFP4 ({e8m0 scale, 16 bytes of e2m1 pairs}).
            GgmlType::Other(24) => (1, 1),
            GgmlType::Other(25) => (1, 2),
            GgmlType::Other(26) => (1, 4),
            GgmlType::Other(27 | 28) => (1, 8),
            GgmlType::Other(MXFP4) => (32, 17),
            GgmlType::Other(IQ2_XXS) => (QK_K, lapid_quant::iq2_xxs::BYTES),
            GgmlType::Other(IQ2_S) => (QK_K, lapid_quant::iq2_s::BYTES),
            GgmlType::Other(IQ2_XS) => (QK_K, lapid_quant::iq2_xs::BYTES),
            GgmlType::Other(IQ3_XXS) => (QK_K, lapid_quant::iq3_xxs::BYTES),
            GgmlType::Other(_) => return None,
        })
    }

    /// Storage bytes of `n` elements. Errors: "unsupported_gguf_type", "format" (n not a
    /// whole number of blocks).
    /// The lapid quant format storing this type without conversion: Q8_0 → Int8 and
    /// Q4_0 → Int4 (lossless repack), Q4_K/Q5_K/Q6_K, MXFP4 and IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS → the
    /// same format (as-is).
    pub fn lapid_format(self) -> Option<lapid_quant::Format> {
        use lapid_quant::Format;
        Some(match self {
            GgmlType::Q8_0 => Format::Int8,
            GgmlType::Q4_0 => Format::Int4,
            GgmlType::Q4K => Format::Q4K,
            GgmlType::Q5K => Format::Q5K,
            GgmlType::Q6K => Format::Q6K,
            GgmlType::Other(MXFP4) => Format::Mxfp4,
            GgmlType::Other(IQ2_XXS) => Format::Iq2Xxs,
            GgmlType::Other(IQ2_S) => Format::Iq2S,
            GgmlType::Other(IQ2_XS) => Format::Iq2Xs,
            GgmlType::Other(IQ3_XXS) => Format::Iq3Xxs,
            _ => return None,
        })
    }

    pub fn nbytes(self, n: u64) -> Result<u64> {
        let (blk, bsz) = self.block().ok_or_else(|| unsupported(self))?;
        if !n.is_multiple_of(blk as u64) {
            bail!("format", "{} elements is not a multiple of the {} block size {blk}", n, self.name());
        }
        (n / blk as u64).checked_mul(bsz as u64).ok_or_else(|| Error::new("format", "tensor size overflows"))
    }
}

fn unsupported(t: GgmlType) -> Error {
    Error::new("unsupported_gguf_type", format!("ggml tensor type {} (id {}) is not supported", t.name(), t.id()))
}

fn f16_at(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

/// Dequantize `n` elements of type `t` to f32. Errors: "unsupported_gguf_type", "format".
pub fn dequantize(t: GgmlType, data: &[u8], n: usize) -> Result<Vec<f32>> {
    let (_, bsz) = t.block().ok_or_else(|| unsupported(t))?;
    if t.nbytes(n as u64)? != data.len() as u64 {
        bail!("format", "ggml {}: {} bytes do not hold {n} elements", t.name(), data.len());
    }
    let mut out = Vec::with_capacity(n);
    let blocks = data.chunks_exact(bsz);
    match t {
        GgmlType::F32 => out.extend(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        GgmlType::F16 => out.extend(data.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())),
        GgmlType::BF16 => out.extend(data.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())),
        GgmlType::Q8_0 | GgmlType::Q8_1 => {
            let qo = if t == GgmlType::Q8_0 { 2 } else { 4 };
            for b in blocks {
                let d = f16_at(b, 0);
                out.extend(b[qo..qo + 32].iter().map(|&q| (q as i8) as f32 * d));
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 => blocks.for_each(|b| q4(t == GgmlType::Q4_1, b, &mut out)),
        GgmlType::Q5_0 | GgmlType::Q5_1 => blocks.for_each(|b| q5(t == GgmlType::Q5_1, b, &mut out)),
        GgmlType::Q2K => blocks.for_each(|b| q2_k(b, &mut out)),
        GgmlType::Q3K => blocks.for_each(|b| q3_k(b, &mut out)),
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
            let f = match t {
                GgmlType::Q4K => kquant::dequantize_q4k,
                GgmlType::Q5K => kquant::dequantize_q5k,
                _ => kquant::dequantize_q6k,
            };
            let mut y = [0f32; QK_K];
            for b in blocks {
                f(b, &mut y);
                out.extend_from_slice(&y);
            }
        }
        GgmlType::Q8K => {
            for b in blocks {
                let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                out.extend(b[4..4 + QK_K].iter().map(|&q| d * (q as i8) as f32));
            }
        }
        GgmlType::Other(24) => out.extend(data.iter().map(|&b| f32::from(b as i8))),
        GgmlType::Other(25) => out.extend(data.chunks_exact(2).map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])))),
        GgmlType::Other(26) => out.extend(data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)),
        GgmlType::Other(27) => out.extend(data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f32)),
        GgmlType::Other(28) => out.extend(data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f32)),
        GgmlType::Other(MXFP4) => blocks.for_each(|b| mxfp4(b, &mut out)),
        GgmlType::Other(IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS) => {
            let f = match t.id() {
                IQ2_XXS => lapid_quant::iq2_xxs::dequantize_block,
                IQ2_XS => lapid_quant::iq2_xs::dequantize_block,
                IQ2_S => lapid_quant::iq2_s::dequantize_block,
                _ => lapid_quant::iq3_xxs::dequantize_block,
            };
            let mut y = [0f32; QK_K];
            for b in blocks {
                f(b, &mut y);
                out.extend_from_slice(&y);
            }
        }
        GgmlType::Other(_) => return Err(unsupported(t)),
    }
    Ok(out)
}

/// ggml type id of MXFP4.
const MXFP4: u32 = 39;
/// ggml type ids of the supported i-quants.
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const IQ3_XXS: u32 = 18;
const IQ2_S: u32 = 22;

/// block_mxfp4 {e (e8m0), qs[16]}: x[j] = kv[qs[j] & 15] * d, x[j + 16] = kv[qs[j] >> 4] * d with
/// kv = 2 * e2m1 and d = 2^(e - 128) (ggml's `GGML_E8M0_TO_FP32_HALF`).
fn mxfp4(b: &[u8], out: &mut Vec<f32>) {
    const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let bits: u32 = if b[0] < 2 { 0x0020_0000 << b[0] } else { u32::from(b[0] - 1) << 23 };
    let d = f32::from_bits(bits);
    let qs = &b[1..17];
    out.extend(qs.iter().map(|&q| f32::from(KV[usize::from(q & 15)]) * d));
    out.extend(qs.iter().map(|&q| f32::from(KV[usize::from(q >> 4)]) * d));
}

/// block_q4_0 {d, qs[16]} / block_q4_1 {d, m, qs[16]}.
fn q4(has_min: bool, b: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(b, 0);
    let (m, off, qo) = if has_min { (f16_at(b, 2), 0, 4) } else { (0.0, 8, 2) };
    let qs = &b[qo..qo + 16];
    let mut v = [0f32; 32];
    for j in 0..16 {
        v[j] = ((qs[j] & 0x0F) as i32 - off) as f32 * d + m;
        v[j + 16] = ((qs[j] >> 4) as i32 - off) as f32 * d + m;
    }
    out.extend_from_slice(&v);
}

/// block_q5_0 {d, qh[4], qs[16]} / block_q5_1 {d, m, qh[4], qs[16]}.
fn q5(has_min: bool, b: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(b, 0);
    let (m, off, ho) = if has_min { (f16_at(b, 2), 0, 4) } else { (0.0, 16, 2) };
    let qh = u32::from_le_bytes([b[ho], b[ho + 1], b[ho + 2], b[ho + 3]]);
    let qs = &b[ho + 4..ho + 20];
    let mut v = [0f32; 32];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as i32;
        let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
        v[j] = (((qs[j] & 0x0F) as i32 | xh0) - off) as f32 * d + m;
        v[j + 16] = (((qs[j] >> 4) as i32 | xh1) - off) as f32 * d + m;
    }
    out.extend_from_slice(&v);
}

/// block_q2_K {scales[16], qs[64], d, dmin}.
fn q2_k(b: &[u8], out: &mut Vec<f32>) {
    let (scales, qs) = (&b[0..16], &b[16..80]);
    let (d, min) = (f16_at(b, 80), f16_at(b, 82));
    let mut is = 0;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                out.extend(q[half * 16..half * 16 + 16].iter().map(|&x| dl * ((x >> shift) & 3) as f32 - ml));
            }
        }
    }
}

/// block_q3_K {hmask[32], qs[64], scales[12], d}.
fn q3_k(b: &[u8], out: &mut Vec<f32>) {
    let (hm, qs, sc_raw) = (&b[0..32], &b[32..96], &b[96..108]);
    let d_all = f16_at(b, 108);
    // Unpack 16 6-bit scales (same bit shuffling as ggml's kmask1/kmask2 code).
    let mut aux = [0u32; 4];
    for (i, a) in aux.iter_mut().take(3).enumerate() {
        *a = u32::from_le_bytes([sc_raw[4 * i], sc_raw[4 * i + 1], sc_raw[4 * i + 2], sc_raw[4 * i + 3]]);
    }
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    let scales: Vec<i8> = aux.iter().flat_map(|a| a.to_le_bytes()).map(|x| x as i8).collect();
    let mut is = 0;
    let mut m = 1u8;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in half * 16..half * 16 + 16 {
                    let hi = if hm[l] & m != 0 { 0 } else { 4 };
                    out.push(dl * (((q[l] >> shift) & 3) as i32 - hi) as f32);
                }
            }
            m <<= 1;
        }
    }
}

#[cfg(test)]
#[path = "ggml_tests.rs"]
pub(crate) mod tests;
