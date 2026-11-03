//! Block-scaled float checkpoints (DeepSeek V3 / V4 / V4.1 official releases): a weight
//! `X.weight` with an `F8_E8M0` companion `X.scale`.
//!
//! - FP8: `X.weight` `F8_E4M3` `[out, in]`, `X.scale` `[out / bo, in / bi]` (ceil): one scale
//!   per `bo x bi` block (128 x 128 in V3, 32 x 32 in V4.1); value = e4m3 * 2^(e - 127),
//!   exact in f32.
//! - FP4: `X.weight` `I8` / `U8` `[out, in / 2]` (e2m1 pairs, low nibble = even element),
//!   `X.scale` `[out, in / 32]`: exactly MXFP4 after a nibble reorder
//!   (`lapid_quant::mxfp4::from_e2m1_pairs`), so it is kept losslessly.

use lapid_quant::mxfp4;
use lapid_tensor::{Result, bail};

use crate::parallel::{fill_rows, map_ranges};

/// Which block-scaled format a weight / scale pair is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Micro {
    /// e4m3 values, 2-D block scales.
    Fp8,
    /// packed e2m1 pairs, one scale per 32 elements of a row.
    Fp4,
}

impl Micro {
    /// Pairing of a weight of safetensors dtype `wdt` with an `F8_E8M0` scale.
    pub fn of(wdt: &str) -> Option<Micro> {
        match wdt {
            "F8_E4M3" => Some(Micro::Fp8),
            "I8" | "U8" => Some(Micro::Fp4),
            _ => None,
        }
    }

    /// Stored-form name reported by `TensorSource::stored_dtype`.
    pub fn name(self) -> &'static str {
        match self {
            Micro::Fp8 => "F8_E4M3+E8M0",
            Micro::Fp4 => "F4_E2M1+E8M0",
        }
    }

    /// Logical `[rows, cols]` of weight `w` / scale `s` (stored shapes), validated.
    pub fn logical_shape(self, w: &[usize], s: &[usize]) -> Result<Vec<usize>> {
        let ([r, c], [sr, sc]) = (w, s) else { bail!("shape_mismatch", "block-scaled weight {w:?} / scale {s:?}: expected 2-D") };
        match self {
            Micro::Fp8 => {
                fp8_block(*r, *sr).and(fp8_block(*c, *sc))?;
                Ok(vec![*r, *c])
            }
            Micro::Fp4 => {
                if *sr != *r || c * 2 != sc * mxfp4::QK {
                    bail!("shape_mismatch", "FP4 weight {w:?} needs scale [{r}, {}], got {s:?}", c * 2 / mxfp4::QK);
                }
                Ok(vec![*r, c * 2])
            }
        }
    }
}

/// e4m3 (fn: no infinities, `S.1111.111` = NaN) -> f32, exact.
pub fn e4m3(b: u8) -> f32 {
    E4M3[usize::from(b)]
}

static E4M3: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut i = 0;
    while i < 256 {
        let (e, m) = ((i >> 3) & 15, i & 7);
        // value = m/8 * 2^-6 (e == 0) or (1 + m/8) * 2^(e-7), built from f32 bits.
        let mag = if e == 15 && m == 7 {
            f32::NAN
        } else if e == 0 {
            // m * 2^-9: exponent 2^-9 .. 2^-7 times a short mantissa, exact.
            m as f32 * f32::from_bits((127 - 9) << 23)
        } else {
            f32::from_bits((((e + 120) as u32) << 23) | ((m as u32) << 20))
        };
        t[i] = if i & 0x80 != 0 { -mag } else { mag };
        i += 1;
    }
    t
};

/// e8m0 -> 2^(e - 127) (`255` = NaN).
pub fn e8m0(e: u8) -> f32 {
    match e {
        255 => f32::NAN,
        0 => f32::from_bits(1 << 22),
        e => f32::from_bits(u32::from(e) << 23),
    }
}

/// Block length along one axis of `n` elements with `s` scales (`s = ceil(n / b)`): the
/// power of two that tiles (128 for `n = 4000, s = 32`), else the exact ratio.
fn fp8_block(n: usize, s: usize) -> Result<usize> {
    if s == 0 || n == 0 {
        bail!("shape_mismatch", "FP8: {s} scales for {n} elements");
    }
    let p = n.div_ceil(s).next_power_of_two();
    let b = if n.div_ceil(p) == s { p } else { n / s };
    if b == 0 || n.div_ceil(b) != s {
        bail!("shape_mismatch", "FP8: {s} scales do not tile {n} elements");
    }
    Ok(b)
}

/// Dequantize an FP8 `[rows, cols]` weight with scales `[sr, sc]`, in parallel.
pub fn fp8_to_f32(w: &[u8], s: &[u8], shape: [usize; 2], sshape: [usize; 2]) -> Result<Vec<f32>> {
    let ([r, c], [sr, sc]) = (shape, sshape);
    let (br, bc) = (fp8_block(r, sr)?, fp8_block(c, sc)?);
    if w.len() != r * c || s.len() != sr * sc {
        bail!("shape_mismatch", "FP8: {} / {} bytes for {shape:?} / {sshape:?}", w.len(), s.len());
    }
    let mut out = vec![0f32; r * c];
    fill_rows(&mut out, c, 16, |rows, o| {
        for (i, row) in rows.zip(o.chunks_exact_mut(c)) {
            let srow = &s[i / br * sc..(i / br + 1) * sc];
            for (j, (y, &q)) in row.iter_mut().zip(&w[i * c..(i + 1) * c]).enumerate() {
                *y = e4m3(q) * e8m0(srow[j / bc]);
            }
        }
    });
    Ok(out)
}

/// FP4 `[rows, cols]` (packed `[rows, cols / 2]`, scales `[rows, cols / 32]`) as unpadded
/// MXFP4 blocks (row-major, `cols / 32 * 17` bytes per row), in parallel.
pub fn fp4_to_mxfp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if cols % mxfp4::QK != 0 || packed.len() != rows * cols / 2 || scales.len() != rows * cols / mxfp4::QK {
        bail!("shape_mismatch", "FP4: {} / {} bytes for [{rows}, {cols}]", packed.len(), scales.len());
    }
    let (pb, sb, ob) = (cols / 2, cols / mxfp4::QK, cols / mxfp4::QK * mxfp4::BYTES);
    let mut out = vec![0u8; rows * ob];
    fill_rows(&mut out, ob, 64, |r, o| {
        o.copy_from_slice(&mxfp4::from_e2m1_pairs(&packed[r.start * pb..r.end * pb], &scales[r.start * sb..r.end * sb], r.len(), cols));
    });
    Ok(out)
}

/// Values of unpadded MXFP4 blocks (`n` elements), in parallel.
pub fn mxfp4_to_f32(blocks: &[u8]) -> Vec<f32> {
    let n = blocks.len() / mxfp4::BYTES;
    let parts = map_ranges(n, 4096, |r| {
        let mut v = vec![0f32; r.len() * mxfp4::QK];
        for (b, y) in blocks[r.start * mxfp4::BYTES..r.end * mxfp4::BYTES].chunks_exact(mxfp4::BYTES).zip(v.chunks_exact_mut(mxfp4::QK)) {
            mxfp4::dequantize_block(b, y);
        }
        v
    });
    parts.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x38), 1.0); // e = 7, m = 0
        assert_eq!(e4m3(0x7e), 448.0); // max
        assert_eq!(e4m3(0x01), 2f32.powi(-9)); // min subnormal
        assert_eq!(e4m3(0x3c), 1.5);
        assert_eq!(e4m3(0xc0), -2.0);
        assert!(e4m3(0x7f).is_nan() && e4m3(0xff).is_nan());
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(130), 8.0);
        assert_eq!(e8m0(0), 2f32.powi(-127));
        assert!(e8m0(255).is_nan());
    }

    #[test]
    fn fp8_blocks() {
        assert_eq!(fp8_block(4096, 32).ok(), Some(128));
        assert_eq!(fp8_block(4000, 32).ok(), Some(128));
        assert_eq!(fp8_block(5120, 160).ok(), Some(32));
        assert_eq!(fp8_block(4000, 40).ok(), Some(100));
        assert!(fp8_block(100, 3).is_err());
        // [4, 64] with 2x32 blocks: scales [2, 2].
        let w: Vec<u8> = (0..256).map(|i| if i % 2 == 0 { 0x38 } else { 0xc0 }).collect();
        let s = [127u8, 128, 126, 129];
        let y = fp8_to_f32(&w, &s, [4, 64], [2, 2]).expect("fp8");
        assert_eq!(&y[..2], &[1.0, -2.0]);
        assert_eq!(y[32], 2.0); // row 0, col block 1: scale 2
        assert_eq!(y[2 * 64], 0.5); // row 2: scale 0.5
        assert_eq!(y[3 * 64 + 33], -8.0);
        assert_eq!(Micro::Fp8.logical_shape(&[4, 64], &[2, 2]).ok(), Some(vec![4, 64]));
        assert_eq!(Micro::Fp4.logical_shape(&[4, 16], &[4, 1]).ok(), Some(vec![4, 32]));
        assert!(Micro::Fp4.logical_shape(&[4, 16], &[4, 2]).is_err());
    }

    #[test]
    fn fp4_matches_reference() {
        let rows = 300;
        let packed: Vec<u8> = (0..rows * 32).map(|i| (i * 37 + 11) as u8).collect();
        let scales: Vec<u8> = (0..rows * 2).map(|i| 120 + (i % 9) as u8).collect();
        let got = fp4_to_mxfp4(&packed, &scales, rows, 64).expect("fp4");
        assert_eq!(got, mxfp4::from_e2m1_pairs(&packed, &scales, rows, 64));
        let v = mxfp4_to_f32(&got);
        // element 1 of row 0 = high nibble of byte 0, scale 2^(120-127)
        let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let code = packed[0] >> 4;
        let want = e2m1[usize::from(code & 7)] * if code & 8 != 0 { -1.0 } else { 1.0 } * 2f32.powi(-7);
        assert_eq!(v[1], want);
        assert_eq!(v.len(), rows * 64);
    }
}
