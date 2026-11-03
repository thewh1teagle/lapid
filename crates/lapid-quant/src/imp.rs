//! Reference implementation of the int8 / int4 SoA formats and the dispatch to the AoS
//! k-quant blocks (`kquant.rs`); see `lib.rs` and `docs/spec/quant.md`. Scalar, deterministic,
//! bit-exact with ggml's q8_0 / q4_0 / q{4,5,6}_K reference quantizers.

use half::f16;

use crate::kquant;
use crate::{Format, GROUP, Layout, layout};

/// ggml block sizes: q8_0 = f16 d + 32 x i8 ; q4_0 = f16 d + 16 nibble bytes.
const Q8_0_BLOCK: usize = 2 + GROUP;
const Q4_0_BLOCK: usize = 2 + GROUP / 2;

fn check_shape(format: Format, rows: usize, cols: usize) -> Layout {
    let be = format.block_elems();
    assert!(cols > 0 && cols % be == 0, "{format:?} cols must be a positive multiple of {be}, got {cols}");
    let _ = rows.checked_mul(cols).expect("quantized matrix size overflows usize");
    layout(format, rows, cols)
}

/// Quantize one group of 32 values to i8. Returns the f32 scale `d` (stored as f16).
/// ggml q8_0: d = amax / 127, q = round(x / d) (half away from zero).
#[inline]
fn quant_group_i8(x: &[f32], q: &mut [u8]) -> f32 {
    let mut amax = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (dst, &v) in q.iter_mut().zip(x) {
        let r = (v * id).round().clamp(-127.0, 127.0);
        // NaN inputs map to 0 (`as` saturates and maps NaN to 0).
        *dst = (r as i32) as i8 as u8;
    }
    d
}

/// Quantize one group of 32 values to 4 bits (ggml q4_0). `q` receives 16 bytes:
/// byte j low nibble = element j, high nibble = element j + 16.
/// d = max / -8 where `max` is the (first) value with the largest |x|, keeping its sign;
/// q = min(15, trunc(x / d + 8.5)), which equals clamp(floor(x/d + 8.5), 0, 15).
#[inline]
fn quant_group_i4(x: &[f32], q: &mut [u8]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    let d = if max == 0.0 { 0.0 } else { max / -8.0 };
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let half = GROUP / 2;
    for j in 0..half {
        let lo = nibble(x[j] * id);
        let hi = nibble(x[j + half] * id);
        q[j] = lo | (hi << 4);
    }
    d
}

#[inline]
fn nibble(v: f32) -> u8 {
    // `as i32` truncates toward zero and saturates; NaN -> 0.
    ((v + 8.5) as i32).clamp(0, 15) as u8
}

pub fn quantize(format: Format, data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let l = check_shape(format, rows, cols);
    assert_eq!(data.len(), rows * cols, "quantize: data has {} elements, expected {rows}x{cols}", data.len());
    let mut out = vec![0u8; l.total];
    if format.is_iquant() {
        let q = match format {
            Format::Iq2Xxs => crate::iq2_xxs::quantize(data, cols, None),
            Format::Iq2S => crate::iq2_s::quantize(data, cols, None),
            _ => crate::iq3_xxs::quantize(data, cols, None),
        };
        out[..q.len()].copy_from_slice(&q);
        return out;
    }
    if let Some((q, _)) = kfns(format) {
        // Rows are contiguous and a whole number of blocks: one block per 256 values.
        let bb = format.block_bytes();
        for (x, b) in data.chunks_exact(format.block_elems()).zip(out.chunks_exact_mut(bb)) {
            q(x, b);
        }
        return out;
    }
    let groups_per_row = cols / GROUP;
    let (qs, rest) = out.split_at_mut(l.scales_offset);
    let scales = &mut rest[..l.scales_bytes];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let gi = r * groups_per_row + g;
            let x = &data[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            let d = match format {
                Format::Int8 => {
                    let off = r * cols + g * GROUP;
                    quant_group_i8(x, &mut qs[off..off + GROUP])
                }
                _ => {
                    let off = (r * cols + g * GROUP) / 2;
                    quant_group_i4(x, &mut qs[off..off + GROUP / 2])
                }
            };
            scales[gi * 2..gi * 2 + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        }
    }
    out
}

#[inline]
fn scale_at(bytes: &[u8], l: &Layout, gi: usize) -> f32 {
    let o = l.scales_offset + gi * 2;
    f16::from_le_bytes([bytes[o], bytes[o + 1]]).to_f32()
}

pub fn dequantize_row(format: Format, bytes: &[u8], rows: usize, cols: usize, row: usize, out: &mut [f32]) {
    let l = check_shape(format, rows, cols);
    assert!(bytes.len() >= l.scales_offset + l.scales_bytes, "dequantize: buffer too small ({} < {})", bytes.len(), l.total);
    assert!(row < rows, "dequantize_row: row {row} out of range ({rows} rows)");
    assert_eq!(out.len(), cols, "dequantize_row: out has {} elements, expected {cols}", out.len());
    if let Some((_, d)) = kfns(format) {
        let bb = format.block_bytes();
        let nb = cols / format.block_elems();
        let row_bytes = &bytes[row * nb * bb..(row + 1) * nb * bb];
        for (b, y) in row_bytes.chunks_exact(bb).zip(out.chunks_exact_mut(format.block_elems())) {
            d(b, y);
        }
        return;
    }
    let groups_per_row = cols / GROUP;
    for g in 0..groups_per_row {
        let d = scale_at(bytes, &l, row * groups_per_row + g);
        let dst = &mut out[g * GROUP..(g + 1) * GROUP];
        match format {
            Format::Int8 => {
                let off = row * cols + g * GROUP;
                for (o, &b) in dst.iter_mut().zip(&bytes[off..off + GROUP]) {
                    *o = (b as i8) as f32 * d;
                }
            }
            _ => {
                let off = (row * cols + g * GROUP) / 2;
                let half = GROUP / 2;
                for j in 0..half {
                    let b = bytes[off + j];
                    dst[j] = ((b & 0x0F) as i32 - 8) as f32 * d;
                    dst[j + half] = ((b >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
    }
}

pub fn dequantize(format: Format, bytes: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let _ = check_shape(format, rows, cols);
    let mut out = vec![0.0f32; rows * cols];
    for (r, chunk) in out.chunks_exact_mut(cols).enumerate() {
        dequantize_row(format, bytes, rows, cols, r, chunk);
    }
    out
}

pub fn from_ggml_blocks(format: Format, blocks: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let l = check_shape(format, rows, cols);
    if format.is_aos() {
        // k-quants are stored exactly as in ggml: copy, pad to the layout's alignment.
        assert_eq!(blocks.len(), l.qs_bytes, "from_ggml_blocks: got {} bytes, expected {}", blocks.len(), l.qs_bytes);
        let mut out = vec![0u8; l.total];
        out[..l.qs_bytes].copy_from_slice(blocks);
        return out;
    }
    let n_groups = rows * (cols / GROUP);
    let (block, payload) = match format {
        Format::Int8 => (Q8_0_BLOCK, GROUP),
        _ => (Q4_0_BLOCK, GROUP / 2),
    };
    assert_eq!(
        blocks.len(),
        n_groups * block,
        "from_ggml_blocks: got {} bytes, expected {} ({n_groups} blocks of {block} B)",
        blocks.len(),
        n_groups * block
    );
    let mut out = vec![0u8; l.total];
    // Groups are consecutive along the row in both layouts, so group index gi maps to
    // quant offset gi * payload in the SoA quants plane.
    for (gi, b) in blocks.chunks_exact(block).enumerate() {
        out[l.scales_offset + gi * 2..l.scales_offset + gi * 2 + 2].copy_from_slice(&b[0..2]);
        out[gi * payload..(gi + 1) * payload].copy_from_slice(&b[2..2 + payload]);
    }
    out
}

type KFns = (fn(&[f32], &mut [u8]), fn(&[u8], &mut [f32]));

/// Block quantizer / dequantizer of a k-quant format.
fn kfns(format: Format) -> Option<KFns> {
    match format {
        Format::Q4K => Some((kquant::quantize_q4k, kquant::dequantize_q4k)),
        Format::Q5K => Some((kquant::quantize_q5k, kquant::dequantize_q5k)),
        Format::Q6K => Some((kquant::quantize_q6k, kquant::dequantize_q6k)),
        Format::Mxfp4 => Some((crate::mxfp4::quantize_block, crate::mxfp4::dequantize_block)),
        Format::Iq2Xxs => Some((|x, o| crate::iq2_xxs::quantize_block(x, None, o), crate::iq2_xxs::dequantize_block)),
        Format::Iq2S => Some((|x, o| crate::iq2_s::quantize_block(x, None, o), crate::iq2_s::dequantize_block)),
        Format::Iq3Xxs => Some((|x, o| crate::iq3_xxs::quantize_block(x, None, o), crate::iq3_xxs::dequantize_block)),
        Format::Int8 | Format::Int4 => None,
    }
}

pub fn pack_int4(q: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let l = check_shape(Format::Int4, rows, cols);
    assert_eq!(q.len(), rows * cols, "pack_int4: {} quants for {rows}x{cols}", q.len());
    assert_eq!(scales.len(), rows * cols / GROUP, "pack_int4: {} scales for {rows}x{cols}", scales.len());
    let mut out = vec![0u8; l.total];
    for (gi, (g, &d)) in q.chunks_exact(GROUP).zip(scales).enumerate() {
        let dst = &mut out[gi * GROUP / 2..(gi + 1) * GROUP / 2];
        for j in 0..GROUP / 2 {
            dst[j] = (g[j] & 0xF) | ((g[j + GROUP / 2] & 0xF) << 4);
        }
        let o = l.scales_offset + gi * 2;
        out[o..o + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Vec<f32> {
        (0..32).map(|j| j as f32 - 16.0).collect()
    }

    /// Deterministic pseudo-random values in [-scale, scale) (xorshift).
    fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn layout_sizes() {
        let l = layout(Format::Int8, 1, 32);
        assert_eq!((l.qs_bytes, l.scales_offset, l.scales_bytes, l.total), (32, 256, 2, 512));
        let l = layout(Format::Int4, 3, 64);
        assert_eq!((l.qs_bytes, l.scales_offset, l.scales_bytes, l.total), (96, 256, 12, 512));
        let l = layout(Format::Int8, 768, 768);
        assert_eq!(l.scales_offset, 768 * 768);
        assert_eq!(l.total % 256, 0);
    }

    #[test]
    fn i8_test_vector() {
        let q = quantize(Format::Int8, &ramp(), 1, 32);
        let expect: [i8; 32] = [
            -127, -119, -111, -103, -95, -87, -79, -71, -64, -56, -48, -40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40, 48,
            56, 64, 71, 79, 87, 95, 103, 111, 119,
        ];
        for j in 0..32 {
            assert_eq!(q[j] as i8, expect[j], "element {j}");
        }
        assert_eq!(&q[256..258], &0x3008u16.to_le_bytes());
        assert!(q[32..256].iter().all(|&b| b == 0));
        assert!(q[258..].iter().all(|&b| b == 0));
    }

    #[test]
    fn i4_test_vector() {
        let q = quantize(Format::Int4, &ramp(), 1, 32);
        let expect: [u8; 16] =
            [0x80, 0x91, 0x91, 0xa2, 0xa2, 0xb3, 0xb3, 0xc4, 0xc4, 0xd5, 0xd5, 0xe6, 0xe6, 0xf7, 0xf7, 0xf8];
        assert_eq!(&q[..16], &expect);
        // d = -16 / -8 = 2.0
        assert_eq!(&q[256..258], &0x4000u16.to_le_bytes());
        let x = dequantize(Format::Int4, &q, 1, 32);
        assert_eq!(x[0], -16.0);
        assert_eq!(x[16], 0.0);
        assert_eq!(x[31], 14.0);
    }

    #[test]
    fn i4_positive_max_maps_to_minus_eight() {
        // The extreme value has positive sign: d is negative, x_max -> q = 0 -> (0-8)*d = x_max.
        let mut x = vec![0.5f32; 32];
        x[3] = 4.0;
        let q = quantize(Format::Int4, &x, 1, 32);
        let y = dequantize(Format::Int4, &q, 1, 32);
        assert_eq!(y[3], 4.0);
    }

    #[test]
    fn zero_groups() {
        for f in [Format::Int8, Format::Int4] {
            let q = quantize(f, &[0.0; 64], 2, 32);
            let l = layout(f, 2, 32);
            assert!(q[l.scales_offset..l.scales_offset + l.scales_bytes].iter().all(|&b| b == 0));
            assert!(dequantize(f, &q, 2, 32).iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn round_trip_error_bounds() {
        let (rows, cols) = (7, 96);
        let x = noise(rows * cols, 42, 3.0);
        for (f, levels) in [(Format::Int8, 127.0f32), (Format::Int4, 8.0f32)] {
            let q = quantize(f, &x, rows, cols);
            assert_eq!(q.len(), layout(f, rows, cols).total);
            let y = dequantize(f, &q, rows, cols);
            for r in 0..rows {
                for g in 0..cols / 32 {
                    let grp = &x[r * cols + g * 32..r * cols + g * 32 + 32];
                    let amax = grp.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    // i8: half a step. i4: values on the side opposite to the extreme can reach
                    // +8 steps but q saturates at 15 (= +7), so up to one step. Plus f16 scale
                    // rounding (relative 2^-11 per level).
                    let step = amax / levels;
                    let k = if f == Format::Int8 { 0.5 } else { 1.0 };
                    let tol = step * k + amax * 1e-3 + 1e-6;
                    for c in 0..32 {
                        let i = r * cols + g * 32 + c;
                        assert!((x[i] - y[i]).abs() <= tol, "{f:?} ({r},{}) {} vs {} tol {tol}", g * 32 + c, x[i], y[i]);
                    }
                }
            }
            let mut row = vec![0.0; cols];
            dequantize_row(f, &q, rows, cols, 5, &mut row);
            assert_eq!(&row[..], &y[5 * cols..6 * cols]);
        }
    }

    #[test]
    fn k_formats() {
        let (rows, cols) = (3, 512);
        let x = noise(rows * cols, 11, 2.0);
        for (f, tol) in [(Format::Q4K, 0.25f32), (Format::Q5K, 0.12), (Format::Q6K, 0.06)] {
            let q = quantize(f, &x, rows, cols);
            let l = layout(f, rows, cols);
            assert_eq!(q.len(), l.total);
            assert_eq!(l.qs_bytes, rows * 2 * f.block_bytes());
            let y = dequantize(f, &q, rows, cols);
            let mut row = vec![0.0; cols];
            dequantize_row(f, &q, rows, cols, 2, &mut row);
            assert_eq!(&row[..], &y[2 * cols..]);
            assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < tol), "{f:?}");
            // GGUF blocks are our bytes minus the padding.
            assert_eq!(from_ggml_blocks(f, &q[..l.qs_bytes], rows, cols), q);
        }
    }

    #[test]
    fn int4_pack_matches_quantize() {
        let x = ramp();
        let q = quantize(Format::Int4, &x, 1, 32);
        let qs: Vec<u8> = (0..32).map(|j| if j < 16 { q[j] & 0xF } else { q[j - 16] >> 4 }).collect();
        assert_eq!(pack_int4(&qs, &[2.0], 1, 32), q);
    }

    #[test]
    fn ggml_blocks_round_trip() {
        let (rows, cols) = (3, 64);
        let x = noise(rows * cols, 7, 1.0);
        for f in [Format::Int8, Format::Int4] {
            let ours = quantize(f, &x, rows, cols);
            let l = layout(f, rows, cols);
            let payload = if f == Format::Int8 { 32 } else { 16 };
            // Build ggml AoS blocks from our layout, then convert back.
            let mut blocks = Vec::new();
            for gi in 0..rows * cols / 32 {
                blocks.extend_from_slice(&ours[l.scales_offset + gi * 2..l.scales_offset + gi * 2 + 2]);
                blocks.extend_from_slice(&ours[gi * payload..(gi + 1) * payload]);
            }
            assert_eq!(from_ggml_blocks(f, &blocks, rows, cols), ours);
        }
    }
}
