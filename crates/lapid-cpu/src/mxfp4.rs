//! MXFP4 weights (`lapid_quant::mxfp4`, ggml `block_mxfp4`: 32 elements / 17 bytes, e8m0
//! scale + e2m1 nibbles) x int8 activations quantized per 32-group (`KA`, the K-quant
//! activation layout; the `bs` sums are not needed: MXFP4 has no min term).
//!
//! Per block: `w = KV[q] * d`, `d = 2^(e - 128)`, `KV` the doubled e2m1 values (small
//! integers -12..12, exact in i8), so `dot(w, a) = d * da * dot_i32(KV[q], qa)`.
//!
//! Kernels share the `kquant::KTile*` contract: aarch64 `sdot` (nibbles decoded with one
//! `tbl` lookup per 16 values; four blocks reduced pairwise into one vector and scaled with
//! one FMA) and a portable scalar reference.

use lapid_quant::mxfp4::{BYTES, KV, QK, e8m0_half};

use crate::kquant::{KA, KMat};

/// Dequantize one block (32 values).
#[inline]
pub(crate) fn dequant_block(b: &[u8], out: &mut [f32]) {
    lapid_quant::mxfp4::dequantize_block(b, out);
}

/// Block quants as signed integers (`KV[q]`), element order, plus the block scale `d`.
#[inline(always)]
pub(crate) fn unpack_block(b: &[u8], q: &mut [i8; QK]) -> f32 {
    for j in 0..QK / 2 {
        q[j] = KV[usize::from(b[1 + j] & 15)];
        q[j + QK / 2] = KV[usize::from(b[1 + j] >> 4)];
    }
    e8m0_half(b[0])
}

/// Portable kernel (also the reference of the SIMD one).
///
/// # Safety
/// Same contract as `kquant::KTile1`.
pub(crate) unsafe fn tile_scalar<const R: usize>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    let k = a.k;
    let kg = k / QK;
    let mut acc = [[0f32; 4]; R];
    let mut q = [0i8; QK];
    for c in 0..4 {
        for blk in 0..kg {
            let d = unpack_block(b.block(cols[c], blk), &mut q);
            for r in 0..R {
                let qa = &a.q[rows[r] * k + blk * QK..][..QK];
                let dot: i32 = qa.iter().zip(&q).map(|(&x, &y)| x as i32 * y as i32).sum();
                acc[r][c] += d * a.s[rows[r] * kg + blk] * dot as f32;
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
    let mut out = [[0f32; 4]; R];
    // SAFETY: indices validated by the caller (see `KTile1`); every weight load reads the 16
    // quant bytes of a block of row `cols[c]` (bytes 1..17), every activation load stays in
    // row `rows[r]` (k int8, k / 32 scales).
    unsafe {
        let zero = vdupq_n_s32(0);
        let kv = vld1q_s8(KV.as_ptr());
        let m0f = vdupq_n_u8(0x0f);
        let aq = a.q.as_ptr();
        let asc = a.s.as_ptr();
        let dec = |p: *const u8| -> [int8x16_t; 2] {
            let v = vld1q_u8(p.add(1));
            [vqtbl1q_s8(kv, vandq_u8(v, m0f)), vqtbl1q_s8(kv, vshrq_n_u8::<4>(v))]
        };
        for c in 0..4 {
            if c > 0 && cols[c] == cols[c - 1] {
                for o in out.iter_mut() {
                    o[c] = o[c - 1];
                }
                continue;
            }
            let row = b.bytes.as_ptr().add(cols[c] * nblk * BYTES);
            let mut acc = [[vdupq_n_f32(0.0); 2]; R];
            let mut tail = [0f32; R];
            let mut blk = 0;
            while blk + 4 <= nblk {
                let p = row.add(blk * BYTES);
                let w = [dec(p), dec(p.add(BYTES)), dec(p.add(2 * BYTES)), dec(p.add(3 * BYTES))];
                let ds = [e8m0_half(*p), e8m0_half(*p.add(BYTES)), e8m0_half(*p.add(2 * BYTES)), e8m0_half(*p.add(3 * BYTES))];
                let dv = vld1q_f32(ds.as_ptr());
                let h = (blk / 4) & 1;
                for r in 0..R {
                    let ap = aq.add(rows[r] * k + blk * QK);
                    let dg: [int32x4_t; 4] = std::array::from_fn(|i| {
                        let o = ap.add(i * QK);
                        vdotq_s32(vdotq_s32(zero, vld1q_s8(o), w[i][0]), vld1q_s8(o.add(16)), w[i][1])
                    });
                    let s = vpaddq_s32(vpaddq_s32(dg[0], dg[1]), vpaddq_s32(dg[2], dg[3]));
                    let da = vld1q_f32(asc.add(rows[r] * nblk + blk));
                    acc[r][h] = vfmaq_f32(acc[r][h], vcvtq_f32_s32(s), vmulq_f32(dv, da));
                }
                blk += 4;
            }
            while blk < nblk {
                let p = row.add(blk * BYTES);
                let w = dec(p);
                let d = e8m0_half(*p);
                for r in 0..R {
                    let ap = aq.add(rows[r] * k + blk * QK);
                    let s = vaddvq_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(ap), w[0]), vld1q_s8(ap.add(16)), w[1]));
                    tail[r] += s as f32 * d * *asc.add(rows[r] * nblk + blk);
                }
                blk += 1;
            }
            for r in 0..R {
                out[r][c] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1])) + tail[r];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Isa;
    use lapid_quant::Format;

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

    #[test]
    fn unpack_matches_reference_dequant() {
        let w = rnd(4 * 64, 3);
        let bytes = lapid_quant::quantize(Format::Mxfp4, &w, 4, 64);
        let want = lapid_quant::dequantize(Format::Mxfp4, &bytes, 4, 64);
        let km = KMat { fmt: Format::Mxfp4, bytes: &bytes, cols: 64 };
        let mut q = [0i8; QK];
        for r in 0..4 {
            for blk in 0..2 {
                let d = unpack_block(km.block(r, blk), &mut q);
                for t in 0..QK {
                    assert_eq!(q[t] as f32 * d, want[r * 64 + blk * QK + t]);
                }
            }
        }
    }

    #[test]
    fn tiles_match_dequant_reference() {
        // k = 7 blocks: one 4-block step and a 3-block tail.
        for k in [224usize, 512] {
            let (rows, n) = (5, 6);
            let x: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
            let kg = k / 32;
            let mut q = vec![0i8; rows * k];
            let mut s = vec![0f32; rows * kg];
            for r in 0..rows {
                crate::matmul::quantize_row(&x[r * k..(r + 1) * k], &mut q[r * k..(r + 1) * k], &mut s[r * kg..(r + 1) * kg]);
            }
            let a = KA { q: &q, s: &s, bs: &[], k };
            let wf: Vec<f32> = rnd(n * k, 9).iter().map(|v| v * 0.05).collect();
            let bytes = lapid_quant::quantize(Format::Mxfp4, &wf, n, k);
            let w = lapid_quant::dequantize(Format::Mxfp4, &bytes, n, k);
            let b = KMat { fmt: Format::Mxfp4, bytes: &bytes, cols: k };
            let fast = crate::kquant::select(&Isa::detect(), Format::Mxfp4);
            let cols = [0, 2, 5, 5];
            for (ci, &col) in cols.iter().enumerate() {
                for r in 0..rows {
                    let want: f32 = (0..k).map(|t| q[r * k + t] as f32 * s[r * kg + t / 32] * w[col * k + t]).sum();
                    let got = unsafe { (fast.t1)(&a, [r], &b, cols) }[0][ci];
                    let got_s = unsafe { tile_scalar::<1>(&a, [r], &b, cols) }[0][ci];
                    let tol = 1e-5 * want.abs().max(1.0);
                    assert!((want - got).abs() <= tol, "k{k} r{r} c{col}: {want} vs {got}");
                    assert!((want - got_s).abs() <= tol, "scalar k{k} r{r} c{col}: {want} vs {got_s}");
                }
                let got4 = unsafe { (fast.t4)(&a, [0, 1, 3, 4], &b, cols) };
                let want4 = unsafe { tile_scalar::<4>(&a, [0, 1, 3, 4], &b, cols) };
                let got2 = unsafe { (fast.t2)(&a, [1, 4], &b, cols) };
                for r in 0..4 {
                    assert!((got4[r][ci] - want4[r][ci]).abs() <= 1e-5 * want4[r][ci].abs().max(1.0));
                }
                for (r, &ri) in [1usize, 4].iter().enumerate() {
                    let w1 = unsafe { tile_scalar::<1>(&a, [ri], &b, cols) }[0][ci];
                    assert!((got2[r][ci] - w1).abs() <= 1e-5 * w1.abs().max(1.0));
                }
            }
        }
    }
}
