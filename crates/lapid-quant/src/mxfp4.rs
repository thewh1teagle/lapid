//! MXFP4 (OCP microscaling FP4): 32-element blocks of e2m1 values sharing one e8m0 scale,
//! stored exactly as ggml `block_mxfp4` (17 B, AoS, as in GGUF type 39):
//!
//! ```text
//! e (u8, e8m0) | qs[16]: byte j low nibble = element j, high nibble = element j + 16
//! x = KV[nibble] * d,  KV = 2 * e2m1 = {0,1,2,3,4,6,8,12, -0,-1,-2,-3,-4,-6,-8,-12},
//! d = 2^(e - 128)  (ggml GGML_E8M0_TO_FP32_HALF: half the e8m0 value, folding e2m1's 0.5)
//! ```
//! Bit-exact with ggml `quantize_row_mxfp4_ref` / `dequantize_row_mxfp4`. The DeepSeek V4 /
//! V4.1 FP4 expert weights (e2m1 pairs + one e8m0 scale per 32 inputs) are this format after
//! a nibble reorder ([`from_e2m1_pairs`]), so they convert losslessly.

/// Elements per block.
pub const QK: usize = 32;
/// Bytes per block.
pub const BYTES: usize = 17;
/// 2 * e2m1 values by nibble.
pub const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `d = 2^(e - 128)` as f32 bits (denormal for e < 2, as ggml).
#[inline]
pub fn e8m0_half(e: u8) -> f32 {
    f32::from_bits(if e < 2 { 0x0020_0000 << e } else { u32::from(e - 1) << 23 })
}

fn best_index(x: f32, d: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = (f32::from(KV[0]) * d - x).abs();
    for (i, &k) in KV.iter().enumerate().skip(1) {
        let err = (f32::from(k) * d - x).abs();
        if err < best_err {
            best = i as u8;
            best_err = err;
        }
    }
    best
}

/// Quantize 32 values into one block (`quantize_row_mxfp4_ref`).
pub fn quantize_block(x: &[f32], out: &mut [u8]) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let e: u8 = if amax > 0.0 { (amax.log2().floor() - 2.0 + 127.0) as u8 } else { 0 };
    let d = e8m0_half(e);
    out[0] = e;
    for j in 0..QK / 2 {
        out[1 + j] = best_index(x[j], d) | (best_index(x[j + QK / 2], d) << 4);
    }
}

/// Dequantize one block into 32 values (`dequantize_row_mxfp4`).
pub fn dequantize_block(b: &[u8], y: &mut [f32]) {
    let d = e8m0_half(b[0]);
    for j in 0..QK / 2 {
        let q = b[1 + j];
        y[j] = f32::from(KV[usize::from(q & 15)]) * d;
        y[j + QK / 2] = f32::from(KV[usize::from(q >> 4)]) * d;
    }
}

/// Repack a `[rows, cols]` matrix of packed e2m1 pairs (byte `i` = element `2i` in the low
/// nibble, `2i + 1` in the high nibble, as torch `float4_e2m1fn_x2`) with one e8m0 scale per
/// 32 elements (`scales[r * cols / 32 + g]`, the true power of two `2^(e - 127)`) into MXFP4
/// blocks. Lossless: e2m1 nibbles are the same 4-bit codes, and ggml's `d` is half the
/// e8m0 value because `KV` holds doubled e2m1 values.
pub fn from_e2m1_pairs(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert!(cols % QK == 0, "mxfp4: cols {cols} not a multiple of {QK}");
    assert_eq!(packed.len(), rows * cols / 2, "mxfp4: packed size");
    assert_eq!(scales.len(), rows * cols / QK, "mxfp4: scale count");
    let mut out = vec![0u8; rows * cols / QK * BYTES];
    for (bi, (src, dst)) in packed.chunks_exact(QK / 2).zip(out.chunks_exact_mut(BYTES)).enumerate() {
        dst[0] = scales[bi];
        let mut codes = [0u8; QK];
        for (i, &b) in src.iter().enumerate() {
            codes[2 * i] = b & 15;
            codes[2 * i + 1] = b >> 4;
        }
        for j in 0..QK / 2 {
            dst[1 + j] = codes[j] | (codes[j + QK / 2] << 4);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_exact_grid() {
        // Values on the e2m1 grid times a power of two survive exactly.
        let x: Vec<f32> = (0..32).map(|i| [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][i % 8] * if i % 3 == 0 { -0.25 } else { 0.25 }).collect();
        let mut b = [0u8; BYTES];
        quantize_block(&x, &mut b);
        let mut y = [0f32; 32];
        dequantize_block(&b, &mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn e2m1_pairs_match_blocks() {
        // e = 127 -> scale 1.0; codes 0..16 in order.
        let codes: Vec<u8> = (0..32u8).map(|i| i % 16).collect();
        let packed: Vec<u8> = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
        let blk = from_e2m1_pairs(&packed, &[127], 1, 32);
        let mut y = [0f32; 32];
        dequantize_block(&blk, &mut y);
        let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &c) in codes.iter().enumerate() {
            let v = e2m1[usize::from(c & 7)] * if c & 8 != 0 { -1.0 } else { 1.0 };
            assert_eq!(y[i], v, "element {i}");
        }
    }
}
