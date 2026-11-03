//! ggml i-quant weights (IQ2_XXS, IQ2_S, IQ3_XXS; `lapid_quant::iq*`, 256-element AoS
//! super-blocks) x int8 activations quantized per 32-group (`KA`, the K-quant activation
//! layout; no min term, so the `bs` sums are unused).
//!
//! Every format is `w = sc[h] * g * sign` with grid bytes `g` (IQ2: 8, 25, 43; IQ3: 4..62,
//! exact in i8) and a scale per 16 elements `h` (IQ2_S; the 32-group formats repeat theirs),
//! so `dot(w, a) = sum_h sc[h] * da[h / 2] * dot_i32(g * sign, qa)`.
//!
//! Kernels share the `kquant::KTile*` contract: aarch64 `sdot` (grid rows loaded by index,
//! signs applied with `vtst` masks) and a portable scalar reference.

use lapid_quant::{Format, IQ2S_GRID, IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS};

use crate::kquant::{KA, KMat, QK};
use crate::typed::f16_bits_to_f32;

#[inline(always)]
fn f16_at(b: &[u8], o: usize) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

#[inline(always)]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Dequantize one block (256 values).
pub(crate) fn dequant_block(fmt: Format, b: &[u8], out: &mut [f32]) {
    match fmt {
        Format::Iq2Xxs => lapid_quant::iq2_xxs::dequantize_block(b, out),
        Format::Iq2S => lapid_quant::iq2_s::dequantize_block(b, out),
        _ => lapid_quant::iq3_xxs::dequantize_block(b, out),
    }
}

/// Grid bytes of 8-group `j` (0..32) with their signs as `(grid u64, sign bits u8)`.
#[inline(always)]
fn group8(fmt: Format, b: &[u8], j: usize) -> (u64, u8) {
    let (ib, l) = (j / 4, j % 4);
    match fmt {
        Format::Iq2Xxs => {
            let a1 = u32_at(b, 6 + 8 * ib);
            (IQ2XXS_GRID[usize::from(b[2 + 8 * ib + l])], KSIGNS[((a1 >> (7 * l)) & 127) as usize])
        }
        Format::Iq2S => {
            let idx = usize::from(b[2 + j]) | ((usize::from(b[66 + ib]) << (8 - 2 * l)) & 0x300);
            (IQ2S_GRID[idx], b[34 + j])
        }
        _ => {
            let aux = u32_at(b, 66 + 4 * ib);
            let g1 = u64::from(IQ3XXS_GRID[usize::from(b[2 + 8 * ib + 2 * l])]);
            let g2 = u64::from(IQ3XXS_GRID[usize::from(b[2 + 8 * ib + 2 * l + 1])]);
            (g1 | (g2 << 32), KSIGNS[((aux >> (7 * l)) & 127) as usize])
        }
    }
}

/// Scales of the 16 sixteen-element halves of a block.
#[inline(always)]
pub(crate) fn half_scales(fmt: Format, b: &[u8]) -> [f32; 16] {
    let d = f16_at(b, 0);
    let mut s = [0f32; 16];
    for ib in 0..8 {
        let v = match fmt {
            Format::Iq2Xxs => {
                let x = d * (0.5 + (u32_at(b, 6 + 8 * ib) >> 28) as f32) * 0.25;
                [x, x]
            }
            Format::Iq2S => {
                let sc = b[74 + ib];
                [d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25]
            }
            _ => {
                let x = d * (0.5 + (u32_at(b, 66 + 4 * ib) >> 28) as f32) * 0.5;
                [x, x]
            }
        };
        s[2 * ib] = v[0];
        s[2 * ib + 1] = v[1];
    }
    s
}

/// Block quants as signed integers (`grid * sign`), element order, plus the half scales.
#[inline(always)]
pub(crate) fn unpack_block(fmt: Format, b: &[u8], q: &mut [i8; QK]) -> [f32; 16] {
    for j in 0..32 {
        let (g, s) = group8(fmt, b, j);
        let g = g.to_le_bytes();
        for i in 0..8 {
            let v = g[i] as i8;
            q[8 * j + i] = if s & (1 << i) != 0 { -v } else { v };
        }
    }
    half_scales(fmt, b)
}

/// Portable kernel (also the reference of the SIMD one).
///
/// # Safety
/// Same contract as `kquant::KTile1`.
pub(crate) unsafe fn tile_scalar<const R: usize>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    let k = a.k;
    let kg = k / 32;
    let mut acc = [[0f32; 4]; R];
    let mut q = [0i8; QK];
    for c in 0..4 {
        for blk in 0..k / QK {
            let sc = unpack_block(b.fmt, b.block(cols[c], blk), &mut q);
            for r in 0..R {
                let qa = &a.q[rows[r] * k + blk * QK..][..QK];
                for h in 0..16 {
                    let dot: i32 = (16 * h..16 * h + 16).map(|t| qa[t] as i32 * q[t] as i32).sum();
                    acc[r][c] += sc[h] * a.s[rows[r] * kg + blk * 8 + h / 2] * dot as f32;
                }
            }
        }
    }
    acc
}

/// aarch64 `sdot` tile.
///
/// # Safety
/// Same contract as `kquant::KTile1`; requires dotprod.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn tile_sdot<const R: usize>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    use std::arch::aarch64::*;
    let k = a.k;
    let nblk = k / QK;
    let kg = k / 32;
    let fmt = b.fmt;
    let mut out = [[0f32; 4]; R];
    // SAFETY: indices validated by the caller (see `KTile1`): blocks come from row `cols[c]`
    // via `KMat::block`, activation loads stay in row `rows[r]` (k int8, k / 32 scales).
    unsafe {
        let zero = vdupq_n_s32(0);
        let bits = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
        // 16 signed weights of 8-groups j, j + 1
        let dec = |bb: &[u8], j: usize| -> int8x16_t {
            let (g0, s0) = group8(fmt, bb, j);
            let (g1, s1) = group8(fmt, bb, j + 1);
            let g = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(g0), vcreate_u64(g1)));
            let s = vcombine_u8(vdup_n_u8(s0), vdup_n_u8(s1));
            let m = vreinterpretq_s8_u8(vtstq_u8(s, bits));
            vsubq_s8(veorq_s8(g, m), m)
        };
        for c in 0..4 {
            if c > 0 && cols[c] == cols[c - 1] {
                for o in out.iter_mut() {
                    o[c] = o[c - 1];
                }
                continue;
            }
            let mut acc = [vdupq_n_f32(0.0); R];
            for blk in 0..nblk {
                let bb = b.block(cols[c], blk);
                let sc = half_scales(fmt, bb);
                for ib in 0..8 {
                    let w = [dec(bb, 4 * ib), dec(bb, 4 * ib + 2)];
                    let sv = vcombine_f32(vdup_n_f32(sc[2 * ib]), vdup_n_f32(sc[2 * ib + 1]));
                    for r in 0..R {
                        let ap = a.q.as_ptr().add(rows[r] * k + blk * QK + 32 * ib);
                        let d0 = vdotq_s32(zero, vld1q_s8(ap), w[0]);
                        let d1 = vdotq_s32(zero, vld1q_s8(ap.add(16)), w[1]);
                        // lanes [sum d0 (pairs), sum d1 (pairs)] -> per half after one more add
                        let p = vpaddq_s32(d0, d1);
                        let da = *a.s.get_unchecked(rows[r] * kg + blk * 8 + ib);
                        acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(p), vmulq_n_f32(sv, da));
                    }
                }
            }
            for r in 0..R {
                out[r][c] = vaddvq_f32(acc[r]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Isa;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    const FMTS: [Format; 3] = [Format::Iq2Xxs, Format::Iq2S, Format::Iq3Xxs];

    #[test]
    fn unpack_matches_reference_dequant() {
        for fmt in FMTS {
            let w = rnd(2 * 512, 3);
            let bytes = lapid_quant::quantize(fmt, &w, 2, 512);
            let want = lapid_quant::dequantize(fmt, &bytes, 2, 512);
            let km = KMat { fmt, bytes: &bytes, cols: 512 };
            let mut q = [0i8; QK];
            for r in 0..2 {
                for blk in 0..2 {
                    let sc = unpack_block(fmt, km.block(r, blk), &mut q);
                    for t in 0..QK {
                        assert_eq!(sc[t / 16] * q[t] as f32, want[r * 512 + blk * QK + t], "{fmt:?} r{r} t{t}");
                    }
                }
            }
        }
    }

    #[test]
    fn tiles_match_dequant_reference() {
        for fmt in FMTS {
            let (rows, n, k) = (5, 6, 768);
            let x: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
            let kg = k / 32;
            let mut q = vec![0i8; rows * k];
            let mut s = vec![0f32; rows * kg];
            for r in 0..rows {
                crate::matmul::quantize_row(&x[r * k..(r + 1) * k], &mut q[r * k..(r + 1) * k], &mut s[r * kg..(r + 1) * kg]);
            }
            let a = KA { q: &q, s: &s, bs: &[], k };
            let wf: Vec<f32> = rnd(n * k, 9).iter().map(|v| v * 0.05).collect();
            let bytes = lapid_quant::quantize(fmt, &wf, n, k);
            let w = lapid_quant::dequantize(fmt, &bytes, n, k);
            let b = KMat { fmt, bytes: &bytes, cols: k };
            let fast = crate::kquant::select(&Isa::detect(), fmt);
            let cols = [0, 2, 5, 5];
            for (ci, &col) in cols.iter().enumerate() {
                for r in 0..rows {
                    let want: f32 = (0..k).map(|t| q[r * k + t] as f32 * s[r * kg + t / 32] * w[col * k + t]).sum();
                    let got = unsafe { (fast.t1)(&a, [r], &b, cols) }[0][ci];
                    let got_s = unsafe { tile_scalar::<1>(&a, [r], &b, cols) }[0][ci];
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!((want - got).abs() <= tol, "{fmt:?} r{r} c{col}: {want} vs {got}");
                    assert!((want - got_s).abs() <= tol, "{fmt:?} scalar r{r} c{col}: {want} vs {got_s}");
                }
                let got4 = unsafe { (fast.t4)(&a, [0, 1, 3, 4], &b, cols) };
                let want4 = unsafe { tile_scalar::<4>(&a, [0, 1, 3, 4], &b, cols) };
                for r in 0..4 {
                    assert!((got4[r][ci] - want4[r][ci]).abs() <= 1e-4 * want4[r][ci].abs().max(1.0));
                }
            }
        }
    }
}
