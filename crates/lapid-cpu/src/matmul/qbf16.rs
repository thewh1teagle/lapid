//! Quantized-B GEMM for larger m (encoder, prefill) with bf16 `bfmmla` (aarch64 BF16).
//!
//! Exact on the dequantized weights (~1e-5 relative, no int8 activation blocks): every
//! supported weight format is `w = s[g] * q - mn[g]` with small integer quants `q` (int8,
//! int4 - 8, q4_K 0..15, q5_K 0..31, q6_K -32..31, MXFP4 2*e2m1 in -12..12, i-quant grid * sign in -62..62), which are exact bf16 values. So:
//! * B holds the integer quants as bf16, with an f32 scale per scale group (32 elements; 16
//!   for q6_K) applied after the group's products, and for q4_K / q5_K the min term
//!   `mn[g] * sum(a over g)` subtracted with f32 activation group sums;
//! * activations are split into hi + lo bf16 halves (`a ≈ hi + lo` to ~2^-17), so each
//!   product costs two `bfmmla` (`LAPID_CPU_BF16_SPLIT=0`: hi only, plain bf16 rounding).
//!
//! `bfmmla` multiplies a 2x4 bf16 block by a 4x2 block into a 2x2 f32 tile.
//!
//! Layout (both operands, per pair of rows / columns, k in chunks of 4): pair `p`, chunk `c`
//! → 8 bf16 at `(p * kc + c) * 8` = `[row0[4c..4c+4], row1[4c..4c+4]]`, rows padded to a
//! multiple of 4 and columns to a multiple of 8. B scales / mins: column pair `p`, scale
//! group `g` → `[s0, s1, s0, s1]` at `(p * ng + g) * 4`; A group sums: row pair `p`, 32-group
//! `g` → `[a0, a0, a1, a1]` at `(p * kg + g) * 4`.
//!
//! Tasks own column blocks: each unpacks its weight rows into a thread-local bf16 panel (kept
//! in L2) and runs every row block against it, so B is read from memory once per launch.

use lapid_compiler::MatMulDims;

use super::{AOp, Epilogue, Workspace};
use crate::Ctx;
use crate::qmat::QMat;

/// Rows per row block (multiple of 4).
const ROW_BLOCK: usize = 64;
/// k chunks (of 4) per k block (multiple of 8).
const KC_BLOCK: usize = 128;
/// m from which this path is used.
pub(crate) const MIN_M: usize = 1;

pub(crate) fn available(ctx: &Ctx) -> bool {
    cfg!(target_arch = "aarch64") && ctx.isa.bf16 && ctx.isa.neon
}

/// `LAPID_CPU_BF16_SPLIT=0`: activations rounded to bf16 (no lo half). Any other value (the
/// old `a` included) keeps the exact hi + lo split.
fn split_a() -> bool {
    static MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("LAPID_CPU_BF16_SPLIT").map(|v| v.trim() != "0").unwrap_or(true))
}

pub(crate) fn matmul(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue) {
    #[cfg(target_arch = "aarch64")]
    imp::run(ctx, ws, a, b, d, epi, split_a());
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (ctx, ws, a, b, d, epi, split_a());
        unreachable!("qbf16 requires aarch64 bf16");
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;
    use std::cell::RefCell;

    use lapid_compiler::MatMulDims;
    use lapid_quant::{Format, GROUP};

    use super::{AOp, Ctx, Epilogue, KC_BLOCK, QMat, ROW_BLOCK, Workspace};
    use crate::kquant::{self, QK};
    use crate::matmul::kdot;
    use crate::par::{self, PAR_MIN_WORK, SyncPtr, block_size, grow};

    /// f32 → bf16 bits, round to nearest even (NaN stays NaN, overflow → inf), 4 lanes.
    #[inline(always)]
    fn bf16x4(x: float32x4_t) -> uint16x4_t {
        // SAFETY: register-only NEON ops (baseline on aarch64).
        unsafe {
            let u = vreinterpretq_u32_f32(x);
            let odd = vandq_u32(vshrq_n_u32::<16>(u), vdupq_n_u32(1));
            vshrn_n_u32::<16>(vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7fff), odd)))
        }
    }

    /// Pack two f32 rows (`None` = zero row; len k, a multiple of 32) into pair layout `dst`
    /// (`k * 2`); with `lo`, the rows' bf16 residuals go to `lo` the same way. With `sums`,
    /// the 32-group sums of the rows go there as `[s0, s0, s1, s1]` per group.
    fn pack_a_pair(r: [Option<&[f32]>; 2], k: usize, dst: &mut [u16], mut lo: Option<&mut [u16]>, sums: Option<&mut [f32]>) {
        assert!(dst.len() >= 2 * k && lo.as_ref().is_none_or(|l| l.len() >= 2 * k) && sums.as_ref().is_none_or(|s| s.len() >= k / 8));
        assert!(r.iter().all(|x| x.is_none_or(|x| x.len() >= k)));
        // SAFETY: bounds asserted above.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let ld = |row: Option<&[f32]>, i: usize| row.map_or(zero, |x| vld1q_f32(x.as_ptr().add(i)));
            let sp = sums.map(|s| s.as_mut_ptr());
            for g in 0..k / GROUP {
                let mut acc = [zero; 2];
                for c in g * 8..g * 8 + 8 {
                    let x = [ld(r[0], c * 4), ld(r[1], c * 4)];
                    let h = [bf16x4(x[0]), bf16x4(x[1])];
                    vst1q_u16(dst.as_mut_ptr().add(c * 8), vcombine_u16(h[0], h[1]));
                    if let Some(l) = lo.as_deref_mut() {
                        let back = |h: uint16x4_t| vreinterpretq_f32_u32(vshll_n_u16::<16>(h));
                        let rl = [bf16x4(vsubq_f32(x[0], back(h[0]))), bf16x4(vsubq_f32(x[1], back(h[1])))];
                        vst1q_u16(l.as_mut_ptr().add(c * 8), vcombine_u16(rl[0], rl[1]));
                    }
                    acc[0] = vaddq_f32(acc[0], x[0]);
                    acc[1] = vaddq_f32(acc[1], x[1]);
                }
                if let Some(p) = sp {
                    let (s0, s1) = (vaddvq_f32(acc[0]), vaddvq_f32(acc[1]));
                    std::ptr::copy_nonoverlapping([s0, s0, s1, s1].as_ptr(), p.add(g * 4), 4);
                }
            }
        }
    }

    /// Integer quants of weight row `row` into `q` (len k) and its per-scale-group scales /
    /// mins (`mn` only for q4_K / q5_K).
    fn row_ints(b: &QMat, row: usize, q: &mut [i8], s: &mut [f32], mn: &mut [f32]) {
        if !b.is_k() {
            let mut g8 = [0i8; GROUP];
            for g in 0..b.groups() {
                b.group_i8(row, g, &mut g8);
                q[g * GROUP..(g + 1) * GROUP].copy_from_slice(&g8);
                s[g] = b.scale(row, g);
            }
            return;
        }
        let km = b.kmat();
        if b.fmt == Format::Mxfp4 {
            // KV[q] in -12..12, scale d per 32-group, no min.
            let mut g8 = [0i8; GROUP];
            for g in 0..b.groups() {
                s[g] = crate::mxfp4::unpack_block(km.block(row, g), &mut g8);
                q[g * GROUP..(g + 1) * GROUP].copy_from_slice(&g8);
            }
            return;
        }
        if b.fmt.is_iquant() {
            // grid * sign in -62..62, a scale per 16 elements (IQ2_S) or 32, no min.
            let mut g = [0i8; QK];
            let per = if b.fmt == Format::Iq2S { 16 } else { 8 };
            for blk in 0..b.cols / QK {
                let sc = crate::iquant::unpack_block(b.fmt, km.block(row, blk), &mut g);
                q[blk * QK..(blk + 1) * QK].copy_from_slice(&g);
                for i in 0..per {
                    s[blk * per + i] = sc[i * 16 / per];
                }
            }
            return;
        }
        for blk in 0..b.cols / QK {
            let bb = km.block(row, blk);
            // SAFETY: `bb` is a whole block of `b.fmt`; stores stay inside `q[blk*256..+256]`.
            unsafe {
                let w = match b.fmt {
                    Format::Q4K => kdot::unpack::<4>(bb.as_ptr()),
                    Format::Q5K => kdot::unpack::<5>(bb.as_ptr()),
                    _ => kdot::unpack::<6>(bb.as_ptr()),
                };
                let qp = q.as_mut_ptr().add(blk * QK);
                for (g, v) in w.iter().enumerate() {
                    vst1q_s8(qp.add(g * 32), v[0]);
                    vst1q_s8(qp.add(g * 32 + 16), v[1]);
                }
            }
            if b.fmt == Format::Q6K {
                let d = half::f16::from_le_bytes([bb[208], bb[209]]).to_f32();
                for h in 0..16 {
                    s[blk * 16 + h] = d * (bb[192 + h] as i8) as f32;
                }
            } else {
                let (sc, m) = kquant::group_scales_k4(bb);
                s[blk * 8..blk * 8 + 8].copy_from_slice(&sc);
                mn[blk * 8..blk * 8 + 8].copy_from_slice(&m);
            }
        }
    }

    /// 16 small integers → two vectors of 8 bf16 bits (exact).
    #[inline(always)]
    fn ints_bf16(v: int8x16_t) -> (uint16x8_t, uint16x8_t) {
        // SAFETY: register-only NEON ops (baseline on aarch64).
        unsafe {
            let cv = |x: int16x8_t| {
                let lo = vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(x))));
                let hi = vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_high_s16(x)));
                vcombine_u16(vshrn_n_u32::<16>(lo), vshrn_n_u32::<16>(hi))
            };
            (cv(vmovl_s8(vget_low_s8(v))), cv(vmovl_high_s8(v)))
        }
    }

    /// Interleave two integer rows (len k, multiple of 16) into bf16 pair layout.
    fn pack_b_pair(r0: &[i8], r1: &[i8], k: usize, dst: &mut [u16]) {
        assert!(r0.len() >= k && r1.len() >= k && dst.len() >= k * 2);
        // SAFETY: bounds asserted; 16 elements of each row → 4 chunks of 8 bf16.
        unsafe {
            let d = dst.as_mut_ptr();
            for i in (0..k).step_by(16) {
                let (a0, a1) = ints_bf16(vld1q_s8(r0.as_ptr().add(i)));
                let (b0, b1) = ints_bf16(vld1q_s8(r1.as_ptr().add(i)));
                let o = d.add(i * 2);
                let z = |x: uint16x8_t, y: uint16x8_t, hi: bool| {
                    let (x, y) = (vreinterpretq_u64_u16(x), vreinterpretq_u64_u16(y));
                    vreinterpretq_u16_u64(if hi { vzip2q_u64(x, y) } else { vzip1q_u64(x, y) })
                };
                vst1q_u16(o, z(a0, b0, false));
                vst1q_u16(o.add(8), z(a0, b0, true));
                vst1q_u16(o.add(16), z(a1, b1, false));
                vst1q_u16(o.add(24), z(a1, b1, true));
            }
        }
    }

    /// Per-thread scratch: B panel, its scales / mins, integer rows, accumulator rows.
    #[derive(Default)]
    struct Scratch {
        hb: Vec<u16>,
        sb: Vec<f32>,
        mb: Vec<f32>,
        q: Vec<i8>,
        s: Vec<f32>,
        mn: Vec<f32>,
        cbuf: Vec<f32>,
    }

    thread_local! {
        static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
    }

    pub(super) fn run(ctx: &Ctx, ws: &mut Workspace, a: &AOp, b: &QMat, d: &MatMulDims, epi: &Epilogue, split: bool) {
        let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
        let rows = batch * m;
        let kc = k / 4;
        let parallel = rows * n * k >= PAR_MIN_WORK && ctx.threads > 1;
        let rpairs = rows.div_ceil(4) * 2;
        let pair = kc * 8;
        let kg = k / GROUP;
        // Scale groups of 16 for q6_K, else 32; mins for q4_K / q5_K.
        let sg = if matches!(b.fmt, Format::Q6K | Format::Iq2S) { 16 } else { GROUP };
        let ng = k / sg;
        let has_min = matches!(b.fmt, Format::Q4K | Format::Q5K);
        let apair = if split { 2 * pair } else { pair };
        let Workspace { ha, pa_f, .. } = ws;
        let ha = grow(ha, rpairs * apair);
        let asum = grow(pa_f, if has_min { rpairs * kg * 4 } else { 0 });

        // 1. A rows → bf16 pairs (+ residual pairs, + 32-group sums).
        {
            let (hp, sp) = (SyncPtr(ha.as_mut_ptr()), SyncPtr(asum.as_mut_ptr()));
            par::for_each(parallel, rpairs, || (vec![0f32; k], vec![0f32; k]), |(t0, t1), rp| {
                let (hp, sp) = (hp, sp);
                let load = |row: usize, t: &mut Vec<f32>| -> bool {
                    if row >= rows {
                        return false;
                    }
                    a.src.load_contig(a.row(row / m, row % m), t);
                    true
                };
                let (h0, h1) = (load(rp * 2, t0), load(rp * 2 + 1, t1));
                // SAFETY: row pair rp owns its `apair` elements and `kg * 4` sums.
                let dst = unsafe { std::slice::from_raw_parts_mut(hp.0.add(rp * apair), apair) };
                let (hi, lo) = dst.split_at_mut(pair);
                let r = [h0.then_some(&t0[..]), h1.then_some(&t1[..])];
                // SAFETY: as above.
                let sums = has_min.then(|| unsafe { std::slice::from_raw_parts_mut(sp.0.add(rp * kg * 4), kg * 4) });
                pack_a_pair(r, k, hi, split.then_some(lo), sums);
            });
        }
        let (ha, asum): (&[u16], &[f32]) = (ha, asum);

        // 2. Tasks: column blocks (x row splits when there are few column blocks).
        let target = if parallel { ctx.threads * 8 } else { 1 };
        let nb = block_size(n, target, 8, 8, 256);
        let col_blocks = n.div_ceil(nb);
        let row_blocks = rows.div_ceil(ROW_BLOCK);
        let splits = if parallel { (ctx.threads * 2).div_ceil(col_blocks).clamp(1, row_blocks) } else { 1 };
        let rows_per_split = row_blocks.div_ceil(splits) * ROW_BLOCK;
        let splits = rows.div_ceil(rows_per_split);
        let bpairs = nb / 2;
        par::for_each(parallel, col_blocks * splits, || epi.regs(), |regs, t| {
            let (cbi, si) = (t % col_blocks, t / col_blocks);
            let (j0, j1) = (cbi * nb, ((cbi + 1) * nb).min(n));
            let (rs0, rs1) = (si * rows_per_split, ((si + 1) * rows_per_split).min(rows));
            SCRATCH.with(|sc| {
                let mut sc = sc.borrow_mut();
                let Scratch { hb, sb, mb, q, s, mn, cbuf } = &mut *sc;
                let hb = grow(hb, bpairs * pair);
                let sb = grow(sb, bpairs * ng * 4);
                let mb = grow(mb, if has_min { bpairs * ng * 4 } else { 0 });
                let q = grow(q, 2 * k);
                let s = grow(s, 2 * ng);
                let mn = grow(mn, 2 * ng);
                let cbuf = grow(cbuf, ROW_BLOCK * nb);
                // Unpack this block's weight rows (zero rows past n).
                for p in 0..bpairs {
                    for h in 0..2 {
                        let col = j0 + 2 * p + h;
                        let (qh, sh, mh) = (&mut q[h * k..(h + 1) * k], &mut s[h * ng..(h + 1) * ng], &mut mn[h * ng..(h + 1) * ng]);
                        if col < j1 {
                            row_ints(b, col, qh, sh, mh);
                        } else {
                            qh.fill(0);
                            sh.fill(0.0);
                            mh.fill(0.0);
                        }
                    }
                    pack_b_pair(&q[..k], &q[k..], k, &mut hb[p * pair..(p + 1) * pair]);
                    for g in 0..ng {
                        sb[(p * ng + g) * 4..][..4].copy_from_slice(&[s[g], s[ng + g], s[g], s[ng + g]]);
                        if has_min {
                            mb[(p * ng + g) * 4..][..4].copy_from_slice(&[mn[g], mn[ng + g], mn[g], mn[ng + g]]);
                        }
                    }
                }
                let w = j1 - j0;
                let mut r0 = rs0;
                while r0 < rs1 {
                    let r1 = (r0 + ROW_BLOCK).min(rs1);
                    cbuf.fill(0.0);
                    let mut c0 = 0;
                    while c0 < kc {
                        let cn = KC_BLOCK.min(kc - c0);
                        let mut c = 0;
                        while c < w {
                            let mut r = r0;
                            while r < r1 {
                                let args = Args {
                                    pa: &ha[(r / 2) * apair + c0 * 8..],
                                    pb: &hb[(c / 2) * pair + c0 * 8..],
                                    sb: &sb[((c / 2) * ng + c0 * 4 / sg) * 4..],
                                    mins: has_min.then(|| (&mb[((c / 2) * ng + c0 / 8) * 4..], &asum[((r / 2) * kg + c0 / 8) * 4..])),
                                    chunks: cn,
                                    strides: (apair, pair, ng * 4, kg * 4),
                                    lo: split.then_some(pair),
                                };
                                // SAFETY: packed buffers are padded to whole 4x8 tiles and
                                // hold `kc` chunks per pair; bf16 checked by `available`.
                                let tile = unsafe {
                                    if sg == 16 { kernel_4x8::<4>(&args) } else { kernel_4x8::<8>(&args) }
                                };
                                let valid_c = (w - c).min(8);
                                for (qr, row) in tile.iter().enumerate() {
                                    if r + qr < r1 {
                                        let o = (r + qr - r0) * nb + c;
                                        for (x, &v) in cbuf[o..o + valid_c].iter_mut().zip(row) {
                                            *x += v;
                                        }
                                    }
                                }
                                r += 4;
                            }
                            c += 8;
                        }
                        c0 += cn;
                    }
                    for r in r0..r1 {
                        let o = (r - r0) * nb;
                        // SAFETY: this task exclusively owns rows rs0..rs1, columns j0..j1.
                        unsafe { epi.apply(regs, r / m, r % m, j0, &cbuf[o..o + w]) };
                    }
                    r0 = r1;
                }
            });
        });
    }

    struct Args<'a> {
        /// 2 A pairs (`strides.0` apart), 4 B pairs (`strides.1` apart).
        pa: &'a [u16],
        pb: &'a [u16],
        /// B scales: 4 pairs `strides.2` apart, 4 floats per scale group.
        sb: &'a [f32],
        /// B mins (as `sb`) and A 32-group sums (2 pairs `strides.3` apart).
        mins: Option<(&'a [f32], &'a [f32])>,
        /// k chunks (multiple of 8).
        chunks: usize,
        strides: (usize, usize, usize, usize),
        /// Offset of each A pair's low (residual) half.
        lo: Option<usize>,
    }

    #[inline(always)]
    unsafe fn bfmmla(acc: float32x4_t, a: uint16x8_t, b: uint16x8_t) -> float32x4_t {
        let mut acc = acc;
        // SAFETY: register-only instruction; bf16 enabled on the calling function.
        unsafe {
            std::arch::asm!(
                "bfmmla {0:v}.4s, {1:v}.8h, {2:v}.8h",
                inout(vreg) acc,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        acc
    }

    /// 4 rows x 8 columns over `chunks` k chunks; scale groups of `G` chunks.
    ///
    /// # Safety
    /// The slices hold 2 A / 4 B pairs (+ scales, mins, sums) of `chunks` chunks; the CPU
    /// supports BF16.
    #[target_feature(enable = "neon,bf16")]
    unsafe fn kernel_4x8<const G: usize>(x: &Args) -> [[f32; 8]; 4] {
        let (pas, ps, ss, sas) = x.strides;
        // SAFETY: bounds per the function contract.
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
            let (a, b, s) = (x.pa.as_ptr(), x.pb.as_ptr(), x.sb.as_ptr());
            for g in 0..x.chunks / G {
                // Separate hi / lo accumulators: 16 independent bfmmla chains.
                let mut t = [[vdupq_n_f32(0.0); 4]; 2];
                let mut tl = [[vdupq_n_f32(0.0); 4]; 2];
                for c in g * G..g * G + G {
                    let av = [vld1q_u16(a.add(c * 8)), vld1q_u16(a.add(pas + c * 8))];
                    let bv = [vld1q_u16(b.add(c * 8)), vld1q_u16(b.add(ps + c * 8)), vld1q_u16(b.add(2 * ps + c * 8)), vld1q_u16(b.add(3 * ps + c * 8))];
                    for pr in 0..2 {
                        for pc in 0..4 {
                            t[pr][pc] = bfmmla(t[pr][pc], av[pr], bv[pc]);
                        }
                    }
                    if let Some(lo) = x.lo {
                        let al = [vld1q_u16(a.add(lo + c * 8)), vld1q_u16(a.add(pas + lo + c * 8))];
                        for pr in 0..2 {
                            for pc in 0..4 {
                                tl[pr][pc] = bfmmla(tl[pr][pc], al[pr], bv[pc]);
                            }
                        }
                    }
                }
                for pc in 0..4 {
                    let sv = vld1q_f32(s.add(pc * ss + g * 4));
                    for pr in 0..2 {
                        acc[pr][pc] = vfmaq_f32(acc[pr][pc], vaddq_f32(t[pr][pc], tl[pr][pc]), sv);
                    }
                }
                if let Some((mb, asum)) = x.mins {
                    // Mins only come with 32-element groups (G == 8): group g of the sums.
                    let av = [vld1q_f32(asum.as_ptr().add(g * 4)), vld1q_f32(asum.as_ptr().add(sas + g * 4))];
                    for pc in 0..4 {
                        let mv = vld1q_f32(mb.as_ptr().add(pc * ss + g * 4));
                        for pr in 0..2 {
                            acc[pr][pc] = vfmsq_f32(acc[pr][pc], av[pr], mv);
                        }
                    }
                }
            }
            let mut out = [[0f32; 8]; 4];
            for pr in 0..2 {
                for pc in 0..4 {
                    let mut v = [0f32; 4];
                    vst1q_f32(v.as_mut_ptr(), acc[pr][pc]);
                    out[2 * pr][2 * pc] = v[0];
                    out[2 * pr][2 * pc + 1] = v[1];
                    out[2 * pr + 1][2 * pc] = v[2];
                    out[2 * pr + 1][2 * pc + 1] = v[3];
                }
            }
            out
        }
    }
}
