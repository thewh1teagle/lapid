//! Quantized-B matmul driver: int8 activations x int8/int4 weights, f32 accumulation.
//!
//! 1. Every A row is quantized once per launch to int8 per 32-block (`d = max|x| / 127`).
//! 2. Tasks cover (row block, column block). Inside, 4 output columns at a time share the
//!    B group loads with up to 4 A rows (4x4 register tile); the A row block stays in L2,
//!    the 4 B rows in L1.
//! 3. Per group: `acc += f32(dot_i32(qa, qb)) * (da * db)`; lanes are reduced at the end.

use lapid_compiler::MatMulDims;
use lapid_quant::GROUP;

use super::qkernels::{self, QA};
use super::{AOp, Epilogue, Workspace};
use crate::math::round_fast;
use crate::par::{self, PAR_MIN_WORK, SyncPtr, block_size, grow};
use crate::kquant::{self, KA};
use crate::qmat::QMat;
use crate::Ctx;

/// Rows per row block (A block of 64 x k int8 stays in L2).
const ROW_BLOCK: usize = 64;

/// Quantize one f32 row to int8 per 32-element group. `x.len() == q.len() == 32 * s.len()`.
pub(crate) fn quantize_row(x: &[f32], q: &mut [i8], s: &mut [f32]) {
    quantize_row_sums(x, q, s, None);
}

/// `quantize_row`, plus `bs[g] = s[g] * sum(q[g])` (the K-quant min terms) when given.
/// The NEON path rounds exactly like the scalar one (to nearest, ties to even).
pub(crate) fn quantize_row_sums(x: &[f32], q: &mut [i8], s: &mut [f32], bs: Option<&mut [f32]>) {
    let ng = s.len();
    assert!(x.len() >= ng * GROUP && q.len() >= ng * GROUP && bs.as_ref().is_none_or(|b| b.len() >= ng));
    #[cfg(target_arch = "aarch64")]
    // SAFETY: lengths asserted above; NEON is part of the aarch64 baseline.
    unsafe {
        use std::arch::aarch64::*;
        let bsp = bs.map(|b| b.as_mut_ptr());
        for g in 0..ng {
            let xp = x.as_ptr().add(g * GROUP);
            let v: [float32x4_t; 8] = std::array::from_fn(|i| vld1q_f32(xp.add(4 * i)));
            let m01 = vmaxq_f32(vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1])), vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3])));
            let m23 = vmaxq_f32(vmaxq_f32(vabsq_f32(v[4]), vabsq_f32(v[5])), vmaxq_f32(vabsq_f32(v[6]), vabsq_f32(v[7])));
            let amax = vmaxvq_f32(vmaxq_f32(m01, m23));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            *s.get_unchecked_mut(g) = d;
            let r: [int32x4_t; 8] = std::array::from_fn(|i| vcvtnq_s32_f32(vmulq_n_f32(v[i], id)));
            let h: [int16x8_t; 4] = std::array::from_fn(|i| vcombine_s16(vqmovn_s32(r[2 * i]), vqmovn_s32(r[2 * i + 1])));
            let lim = vdupq_n_s8(127);
            let neg = vdupq_n_s8(-127);
            let b0 = vmaxq_s8(vminq_s8(vcombine_s8(vqmovn_s16(h[0]), vqmovn_s16(h[1])), lim), neg);
            let b1 = vmaxq_s8(vminq_s8(vcombine_s8(vqmovn_s16(h[2]), vqmovn_s16(h[3])), lim), neg);
            let qp = q.as_mut_ptr().add(g * GROUP);
            vst1q_s8(qp, b0);
            vst1q_s8(qp.add(16), b1);
            if let Some(p) = bsp {
                let sum = vaddlvq_s8(b0) as i32 + vaddlvq_s8(b1) as i32;
                *p.add(g) = d * sum as f32;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    quantize_row_scalar(x, q, s, bs);
}

/// Portable `quantize_row_sums` (the reference of the NEON path).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn quantize_row_scalar(x: &[f32], q: &mut [i8], s: &mut [f32], mut bs: Option<&mut [f32]>) {
    for (g, ((xg, qg), sg)) in x.as_chunks::<GROUP>().0.iter().zip(q.as_chunks_mut::<GROUP>().0).zip(s.iter_mut()).enumerate() {
        let mut mx = [0f32; 8];
        for c in xg.as_chunks::<8>().0 {
            for l in 0..8 {
                mx[l] = mx[l].max(c[l].abs());
            }
        }
        let amax = mx.iter().fold(0f32, |m, &v| m.max(v));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        *sg = d;
        let mut sum = 0i32;
        for (o, &v) in qg.iter_mut().zip(xg) {
            *o = (round_fast(v * id) as i32).clamp(-127, 127) as i8;
            sum += *o as i32;
        }
        if let Some(b) = bs.as_deref_mut() {
            b[g] = d * sum as f32;
        }
    }
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    let rows = batch * m;
    let kg = k / GROUP;
    let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;

    // 1. Quantize activations.
    let qa = grow(&mut ws.q, rows * k);
    let sa = grow(&mut ws.s, rows * kg);
    // K formats: per-group `da * sum(q)` for the min terms.
    let has_bs = matches!(b.fmt, lapid_quant::Format::Q4K | lapid_quant::Format::Q5K);
    let bs = grow(&mut ws.pa_f, if has_bs { rows * kg } else { 0 });
    let a_f32 = a.src.as_f32();
    {
        let (sp, bp) = (SyncPtr(sa.as_mut_ptr()), SyncPtr(bs.as_mut_ptr()));
        // K formats with a min term: per-group `da * sum(q)`.
        par::chunks_mut(parallel && rows > 1, qa, k, || vec![0f32; if a_f32.is_some() { 0 } else { k }], |tmp, r, q| {
            let (sp, bp) = (sp, bp);
            // SAFETY: row r owns scales / sums r*kg..(r+1)*kg exclusively.
            let s = unsafe { std::slice::from_raw_parts_mut(sp.0.add(r * kg), kg) };
            let bsr = has_bs.then(|| unsafe { std::slice::from_raw_parts_mut(bp.0.add(r * kg), kg) });
            let (bt, i) = (r / m, r % m);
            let x: &[f32] = match a_f32 {
                Some(f) => &f[a.row(bt, i)..a.row(bt, i) + k],
                None => {
                    a.src.load_contig(a.row(bt, i), &mut tmp[..k]);
                    &tmp[..k]
                }
            };
            quantize_row_sums(x, q, s, bsr);
        });
    }
    let bsum = if has_bs { Some(&*bs) } else { None };
    let qa = QA { q: qa, s: sa, k };

    // 2. Tiles.
    let rb = ROW_BLOCK.min(rows);
    let row_blocks = rows.div_ceil(rb);
    let target = if parallel { ctx.threads * 4 } else { 1 };
    let nb = block_size(n, target.div_ceil(row_blocks), 4, 16, 512);
    let col_blocks = n.div_ceil(nb);
    let kern = qkernels::select(&ctx.isa, b.fmt);
    let kkern = kquant::select(&ctx.isa, b.fmt);
    let (ka, km) = (KA { q: qa.q, s: qa.s, bs: bsum.unwrap_or(&[]), k }, b.kmat());
    let is_k = b.is_k();
    // SAFETY (both): forwarded contract of `Tile4` / `KTile4` (see the call sites).
    let t4 = |r: [usize; 4], c: [usize; 4]| unsafe { if is_k { (kkern.t4)(&ka, r, &km, c) } else { (kern.t4)(&qa, r, b, c) } };
    let t2 = |r: [usize; 2], c: [usize; 4]| unsafe { if is_k { (kkern.t2)(&ka, r, &km, c) } else { (kern.t2)(&qa, r, b, c) } };
    let t1 = |r: [usize; 1], c: [usize; 4]| unsafe { if is_k { (kkern.t1)(&ka, r, &km, c) } else { (kern.t1)(&qa, r, b, c) } };

    par::for_each(parallel, row_blocks * col_blocks, || (vec![0f32; rb * nb], epi.regs()), |(cbuf, regs), t| {
        let (rbi, cbi) = (t / col_blocks, t % col_blocks);
        let (r0, r1) = (rbi * rb, ((rbi + 1) * rb).min(rows));
        let (j0, j1) = (cbi * nb, ((cbi + 1) * nb).min(n));
        let ldc = nb;
        let mut c = j0;
        while c < j1 {
            let valid = (j1 - c).min(4);
            // Tail columns repeat the last valid column; their results are discarded.
            let last = valid - 1;
            let cols = [c, c + last.min(1), c + last.min(2), c + last];
            let mut r = r0;
            while r + 4 <= r1 {
                // rows < `rows`, cols < n, buffers sized/validated by the caller; the
                // kernel's ISA was detected by `select`.
                let out = t4([r, r + 1, r + 2, r + 3], cols);
                for (q, row) in out.iter().enumerate() {
                    let o = (r + q - r0) * ldc + (c - j0);
                    cbuf[o..o + valid].copy_from_slice(&row[..valid]);
                }
                r += 4;
            }
            if r + 2 <= r1 {
                let out = t2([r, r + 1], cols);
                for (q, row) in out.iter().enumerate() {
                    let o = (r + q - r0) * ldc + (c - j0);
                    cbuf[o..o + valid].copy_from_slice(&row[..valid]);
                }
                r += 2;
            }
            while r < r1 {
                let out = t1([r], cols);
                let o = (r - r0) * ldc + (c - j0);
                cbuf[o..o + valid].copy_from_slice(&out[0][..valid]);
                r += 1;
            }
            c += 4;
        }
        for r in r0..r1 {
            let o = (r - r0) * ldc;
            // SAFETY: this task exclusively owns rows r0..r1, columns j0..j1.
            unsafe { epi.apply(regs, r / m, r % m, j0, &cbuf[o..o + (j1 - j0)]) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_row_roundtrip() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37) % 29) as f32 * 0.37 - 5.0).collect();
        let mut q = vec![0i8; 64];
        let mut s = vec![0f32; 2];
        quantize_row(&x, &mut q, &mut s);
        for g in 0..2 {
            for t in 0..32 {
                let v = q[g * 32 + t] as f32 * s[g];
                assert!((v - x[g * 32 + t]).abs() <= s[g] * 0.5 + 1e-6);
            }
        }
        // NEON (if any) and portable paths agree exactly, sums included.
        let x: Vec<f32> = (0..96).map(|i| ((i * 53) % 61) as f32 * 0.21 - 6.0 + if i % 32 == 5 { 0.5 } else { 0.0 }).collect();
        let (mut q1, mut s1, mut b1) = (vec![0i8; 96], vec![0f32; 3], vec![0f32; 3]);
        let (mut q2, mut s2, mut b2) = (vec![0i8; 96], vec![0f32; 3], vec![0f32; 3]);
        quantize_row_sums(&x, &mut q1, &mut s1, Some(&mut b1));
        quantize_row_scalar(&x, &mut q2, &mut s2, Some(&mut b2));
        assert_eq!((q1, s1, b1), (q2, s2, b2));
        let z = vec![0f32; 32];
        let mut qz = vec![1i8; 32];
        let mut sz = vec![1f32; 1];
        quantize_row(&z, &mut qz, &mut sz);
        assert!(qz.iter().all(|&v| v == 0) && sz[0] == 0.0);
    }
}
