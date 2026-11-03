//! ggml i-quant weights (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS; `lapid_quant::iq*`, 256-element AoS
//! super-blocks) x int8 activations quantized per 32-group (`KA`, the K-quant activation
//! layout; no min term, so the `bs` sums are unused).
//!
//! Every format is `w = sc[h] * g * sign` with grid bytes `g` (IQ2: 8, 25, 43; IQ3: 4..62,
//! exact in i8) and a scale per 16 elements `h` (IQ2_S; the 32-group formats repeat theirs),
//! so `dot(w, a) = sum_h sc[h] * da[h / 2] * dot_i32(g * sign, qa)` (IQ2_XS too has a scale
//! per 16).
//!
//! Kernels share the `kquant::KTile*` contract: aarch64 `sdot` (grid rows loaded by index,
//! signs applied with `vtst` masks) and a portable scalar reference.

use lapid_quant::{Format, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS};

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
        Format::Iq2Xs => lapid_quant::iq2_xs::dequantize_block(b, out),
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
        Format::Iq2Xs => {
            let q = u16::from_le_bytes([b[2 + 2 * j], b[3 + 2 * j]]);
            (IQ2XS_GRID[usize::from(q & 511)], KSIGNS[usize::from(q >> 9)])
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
            Format::Iq2Xs | Format::Iq2S => {
                let sc = b[if fmt == Format::Iq2Xs { 66 } else { 74 } + ib];
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

/// Byte masks of sign bytes: entry s has byte j = 0xff when bit j of s is set.
#[cfg(target_arch = "aarch64")]
static MASK8: [u64; 256] = mask8();
/// `MASK8[KSIGNS[i]]`: masks of the 7 stored sign bits plus their parity bit.
#[cfg(target_arch = "aarch64")]
static MASK7: [u64; 128] = mask7();

#[cfg(target_arch = "aarch64")]
const fn mask8() -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut s = 0;
    while s < 256 {
        let mut j = 0;
        while j < 8 {
            if (s >> j) & 1 != 0 {
                t[s] |= 0xff << (8 * j);
            }
            j += 1;
        }
        s += 1;
    }
    t
}

#[cfg(target_arch = "aarch64")]
const fn mask7() -> [u64; 128] {
    let m = mask8();
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = m[(i as u32 | ((i as u32).count_ones() & 1) << 7) as usize];
        i += 1;
    }
    t
}

/// aarch64 `sdot` tile, dispatched on the format.
///
/// # Safety
/// Same contract as `kquant::KTile1`; requires dotprod.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn tile_sdot<const R: usize>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    // SAFETY: forwarded contract.
    unsafe {
        match b.fmt {
            Format::Iq2Xxs => tile_sdot_f::<R, 0>(a, rows, b, cols),
            Format::Iq2S => tile_sdot_f::<R, 1>(a, rows, b, cols),
            Format::Iq2Xs => tile_sdot_f::<R, 3>(a, rows, b, cols),
            _ => tile_sdot_f::<R, 2>(a, rows, b, cols),
        }
    }
}

/// `tile_sdot` for format `F` (0 IQ2_XXS, 1 IQ2_S, 2 IQ3_XXS, 3 IQ2_XS): each 32-element sub-block is
/// decoded once into two vectors of signed weights `(g ^ m) - m` (grid rows by index, sign
/// byte masks from `MASK7` / `MASK8`) and two half scales, then dotted with every row.
///
/// # Safety
/// Same contract as `kquant::KTile1`; requires dotprod.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn tile_sdot_f<const R: usize, const F: u8>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    use std::arch::aarch64::*;
    let k = a.k;
    let nblk = k / QK;
    let kg = k / 32;
    let bb = b.fmt.block_bytes();
    let mut out = [[0f32; 4]; R];
    // SAFETY: indices validated by the caller (see `KTile1`): every block read lies in row
    // `cols[c]` (`nblk` blocks of `bb` bytes), table indices are bounded by construction
    // (u8 / 7-bit / 10-bit fields into 256 / 128 / 1024-entry tables), activation loads stay
    // in row `rows[r]` (k int8, k / 32 scales).
    unsafe {
        let zero = vdupq_n_s32(0);
        let u32_at = |p: *const u8| -> u32 { u32::from_le(std::ptr::read_unaligned(p.cast::<u32>())) };
        let signed = |g: [u64; 2], m: [u64; 2]| -> int8x16_t {
            let g = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(g[0]), vcreate_u64(g[1])));
            let m = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(m[0]), vcreate_u64(m[1])));
            vsubq_s8(veorq_s8(g, m), m)
        };
        let g2 = |t: &[u64], i: usize| *t.get_unchecked(i);
        let g3 = |i: u8| u64::from(*IQ3XXS_GRID.get_unchecked(usize::from(i)));
        for c in 0..4 {
            if c > 0 && cols[c] == cols[c - 1] {
                for o in out.iter_mut() {
                    o[c] = o[c - 1];
                }
                continue;
            }
            let row = b.bytes.as_ptr().add(cols[c] * nblk * bb);
            let mut acc = [vdupq_n_f32(0.0); R];
            for blk in 0..nblk {
                let p = row.add(blk * bb);
                let d = f16_bits_to_f32(u16::from_le(std::ptr::read_unaligned(p.cast::<u16>())));
                for ib in 0..8 {
                    // two vectors of 16 signed weights and the two half scales
                    let (w, sc) = match F {
                        0 => {
                            let q = p.add(2 + 8 * ib);
                            let aux = u32_at(q.add(4));
                            let gi = |l: usize| g2(&IQ2XXS_GRID, usize::from(*q.add(l)));
                            let mi = |l: usize| *MASK7.get_unchecked(((aux >> (7 * l)) & 127) as usize);
                            let x = d * (0.5 + (aux >> 28) as f32) * 0.25;
                            ([signed([gi(0), gi(1)], [mi(0), mi(1)]), signed([gi(2), gi(3)], [mi(2), mi(3)])], [x, x])
                        }
                        1 => {
                            let (qs, sg) = (p.add(2 + 4 * ib), p.add(34 + 4 * ib));
                            let qh = u32::from(*p.add(66 + ib));
                            let s = *p.add(74 + ib);
                            let gi = |l: usize| g2(&IQ2S_GRID, usize::from(*qs.add(l)) | ((qh << (8 - 2 * l) as u32) & 0x300) as usize);
                            let mi = |l: usize| *MASK8.get_unchecked(usize::from(*sg.add(l)));
                            let x = [d * (0.5 + (s & 15) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25];
                            ([signed([gi(0), gi(1)], [mi(0), mi(1)]), signed([gi(2), gi(3)], [mi(2), mi(3)])], x)
                        }
                        3 => {
                            let q = p.add(2 + 8 * ib);
                            let s = *p.add(66 + ib);
                            let qv = |l: usize| u16::from_le(std::ptr::read_unaligned(q.add(2 * l).cast::<u16>()));
                            let gi = |l: usize| g2(&IQ2XS_GRID, usize::from(qv(l) & 511));
                            let mi = |l: usize| *MASK7.get_unchecked(usize::from(qv(l) >> 9));
                            let x = [d * (0.5 + (s & 15) as f32) * 0.25, d * (0.5 + (s >> 4) as f32) * 0.25];
                            ([signed([gi(0), gi(1)], [mi(0), mi(1)]), signed([gi(2), gi(3)], [mi(2), mi(3)])], x)
                        }
                        _ => {
                            let q = p.add(2 + 8 * ib);
                            let aux = u32_at(p.add(66 + 4 * ib));
                            let gi = |l: usize| g3(*q.add(2 * l)) | (g3(*q.add(2 * l + 1)) << 32);
                            let mi = |l: usize| *MASK7.get_unchecked(((aux >> (7 * l)) & 127) as usize);
                            let x = d * (0.5 + (aux >> 28) as f32) * 0.5;
                            ([signed([gi(0), gi(1)], [mi(0), mi(1)]), signed([gi(2), gi(3)], [mi(2), mi(3)])], [x, x])
                        }
                    };
                    let sv = vcombine_f32(vdup_n_f32(sc[0]), vdup_n_f32(sc[1]));
                    for r in 0..R {
                        let ap = a.q.as_ptr().add(rows[r] * k + blk * QK + 32 * ib);
                        let d0 = vdotq_s32(zero, vld1q_s8(ap), w[0]);
                        let d1 = vdotq_s32(zero, vld1q_s8(ap.add(16)), w[1]);
                        // lanes [half 0, half 0, half 1, half 1]
                        let pr = vpaddq_s32(d0, d1);
                        let da = *a.s.get_unchecked(rows[r] * kg + blk * 8 + ib);
                        acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(pr), vmulq_n_f32(sv, da));
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

    const FMTS: [Format; 4] = [Format::Iq2Xxs, Format::Iq2Xs, Format::Iq2S, Format::Iq3Xxs];

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
