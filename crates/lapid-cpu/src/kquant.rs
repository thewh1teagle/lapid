//! ggml K-quant super-blocks (`q4_K`, `q5_K`, `q6_K`; 256 elements, AoS as in GGUF):
//! dequantization and int8-activation dot products.
//!
//! Activations are quantized per 32-element group (`quant::quantize_row`: scale `da`, plus
//! `bs = da * sum(q)` for the min terms of q4_K / q5_K), which is finer than ggml's
//! per-256 `q8_K` and lets one activation layout serve every weight format. Per 32-group:
//! * q4_K / q5_K: `w = d*sc*q - dmin*m` → `d*sc*da*dot(q, a) - dmin*m*bs`;
//! * q6_K: two 16-element halves with their own i8 scale: `d*da*(sc0*dot0 + sc1*dot1)`.
//!
//! Kernels: aarch64 `sdot` (dotprod, `matmul/kdot.rs`) and portable scalar, both over R rows
//! x 4 columns like the int8 / int4 tiles in `matmul/qkernels.rs`.

use lapid_quant::Format;

use crate::typed::f16_bits_to_f32;

pub(crate) const QK: usize = 256;

#[inline(always)]
fn f16_at(b: &[u8], o: usize) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

/// ggml `get_scale_min_k4`: 6-bit scale and min of sub-block `j` (0..8).
#[inline(always)]
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Per 32-group multipliers of a q4_K / q5_K block: `(d*sc[g], dmin*m[g])`.
#[inline(always)]
pub(crate) fn group_scales_k4(b: &[u8]) -> ([f32; 8], [f32; 8]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let mut s = [0f32; 8];
    let mut m = [0f32; 8];
    for j in 0..8 {
        let (sc, mn) = scale_min_k4(j, &b[4..16]);
        s[j] = d * sc as f32;
        m[j] = dmin * mn as f32;
    }
    (s, m)
}

/// Unpack the quants of one block into 256 signed values (q4_K 0..15, q5_K 0..31,
/// q6_K -32..31), in element order.
#[inline(always)]
pub(crate) fn unpack_block(fmt: Format, b: &[u8], out: &mut [i8; QK]) {
    match fmt {
        Format::Q4K => {
            let qs = &b[16..144];
            for j in 0..4 {
                for l in 0..32 {
                    let x = qs[j * 32 + l];
                    out[j * 64 + l] = (x & 0xF) as i8;
                    out[j * 64 + 32 + l] = (x >> 4) as i8;
                }
            }
        }
        Format::Q5K => {
            let (qh, qs) = (&b[16..48], &b[48..176]);
            for j in 0..4 {
                let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
                for l in 0..32 {
                    let x = qs[j * 32 + l];
                    let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    out[j * 64 + l] = ((x & 0xF) + h1) as i8;
                    out[j * 64 + 32 + l] = ((x >> 4) + h2) as i8;
                }
            }
        }
        Format::Q6K => {
            for n in 0..2 {
                let ql = &b[n * 64..n * 64 + 64];
                let qh = &b[128 + n * 32..128 + n * 32 + 32];
                let o = &mut out[n * 128..n * 128 + 128];
                for l in 0..32 {
                    o[l] = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8) - 32;
                    o[l + 32] = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                    o[l + 64] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                    o[l + 96] = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;
                }
            }
        }
        Format::Int8 | Format::Int4 | Format::Mxfp4 => unreachable!("not a K format"),
    }
}

/// Dequantize one block (256 elements; 32 for MXFP4).
pub(crate) fn dequant_block(fmt: Format, b: &[u8], out: &mut [f32]) {
    if fmt == Format::Mxfp4 {
        return crate::mxfp4::dequant_block(b, out);
    }
    let mut q = [0i8; QK];
    unpack_block(fmt, b, &mut q);
    match fmt {
        Format::Q4K | Format::Q5K => {
            let (s, m) = group_scales_k4(b);
            for g in 0..8 {
                for l in 0..32 {
                    out[g * 32 + l] = s[g] * q[g * 32 + l] as f32 - m[g];
                }
            }
        }
        Format::Q6K => {
            let d = f16_at(b, 208);
            for h in 0..16 {
                // 16-element sub-block h: half n = h / 8, scale index within the half.
                let sc = d * (b[192 + h] as i8) as f32;
                let (n, w, part) = (h / 8, (h % 8) / 2, h % 2);
                let base = n * 128 + w * 32 + part * 16;
                for l in 0..16 {
                    out[base + l] = sc * q[base + l] as f32;
                }
            }
        }
        Format::Int8 | Format::Int4 | Format::Mxfp4 => unreachable!("not a K format"),
    }
}

/// Quantized activations for K formats: row r, group g at `q[r*k + g*32..]`, scale
/// `s[r*k/32 + g]`, `bs[r*k/32 + g] = s * sum(q)`.
pub(crate) struct KA<'a> {
    pub q: &'a [i8],
    pub s: &'a [f32],
    pub bs: &'a [f32],
    pub k: usize,
}

/// Borrowed AoS block matrix `[rows, cols]` (rows of `cols / block_elems` blocks: K-quants
/// 256, MXFP4 32).
#[derive(Clone, Copy)]
pub(crate) struct KMat<'a> {
    pub fmt: Format,
    pub bytes: &'a [u8],
    pub cols: usize,
}

impl<'a> KMat<'a> {
    #[inline(always)]
    pub fn block(&self, row: usize, blk: usize) -> &'a [u8] {
        let bb = self.fmt.block_bytes();
        let o = (row * (self.cols / self.fmt.block_elems()) + blk) * bb;
        &self.bytes[o..o + bb]
    }
}

/// # Safety
/// Row indices < rows of `KA`, column indices < rows of `KMat`, `KA.k == KMat.cols`, and the
/// kernel's ISA must be available (guaranteed by `select`).
pub(crate) type KTile1 = unsafe fn(&KA, [usize; 1], &KMat, [usize; 4]) -> [[f32; 4]; 1];
pub(crate) type KTile2 = unsafe fn(&KA, [usize; 2], &KMat, [usize; 4]) -> [[f32; 4]; 2];
pub(crate) type KTile4 = unsafe fn(&KA, [usize; 4], &KMat, [usize; 4]) -> [[f32; 4]; 4];

#[derive(Clone, Copy)]
pub(crate) struct KKernels {
    pub t1: KTile1,
    pub t2: KTile2,
    pub t4: KTile4,
}

pub(crate) fn select(isa: &crate::Isa, fmt: Format) -> KKernels {
    if fmt == Format::Mxfp4 {
        use crate::mxfp4::tile_scalar as ts;
        #[cfg(target_arch = "aarch64")]
        if isa.dotprod {
            use crate::mxfp4::tile_sdot as t;
            return KKernels { t1: t::<1>, t2: t::<2>, t4: t::<4> };
        }
        return KKernels { t1: ts::<1>, t2: ts::<2>, t4: ts::<4> };
    }
    #[cfg(target_arch = "aarch64")]
    if isa.dotprod {
        use crate::matmul::kdot::tile;
        return match fmt {
            Format::Q4K => KKernels { t1: tile::<1, 4>, t2: tile::<2, 4>, t4: tile::<4, 4> },
            Format::Q5K => KKernels { t1: tile::<1, 5>, t2: tile::<2, 5>, t4: tile::<4, 5> },
            _ => KKernels { t1: tile::<1, 6>, t2: tile::<2, 6>, t4: tile::<4, 6> },
        };
    }
    let _ = (isa, fmt);
    KKernels { t1: tile_scalar::<1>, t2: tile_scalar::<2>, t4: tile_scalar::<4> }
}

/// Portable kernel (also the reference of the SIMD ones).
pub(crate) unsafe fn tile_scalar<const R: usize>(a: &KA, rows: [usize; R], b: &KMat, cols: [usize; 4]) -> [[f32; 4]; R] {
    let k = a.k;
    let kg = k / 32;
    let mut acc = [[0f32; 4]; R];
    let mut q = [0i8; QK];
    for c in 0..4 {
        for blk in 0..k / QK {
            let bb = b.block(cols[c], blk);
            unpack_block(b.fmt, bb, &mut q);
            for r in 0..R {
                let ao = rows[r] * k + blk * QK;
                let go = rows[r] * kg + blk * 8;
                let qa = &a.q[ao..ao + QK];
                let dot = |lo: usize, n: usize| -> f32 {
                    let mut s = 0i32;
                    for t in lo..lo + n {
                        s += qa[t] as i32 * q[t] as i32;
                    }
                    s as f32
                };
                match b.fmt {
                    Format::Q6K => {
                        let d = f16_at(bb, 208);
                        for g in 0..8 {
                            let (n, w) = (g / 4, g % 4);
                            let s0 = (bb[192 + n * 8 + 2 * w] as i8) as f32;
                            let s1 = (bb[192 + n * 8 + 2 * w + 1] as i8) as f32;
                            let v = s0 * dot(g * 32, 16) + s1 * dot(g * 32 + 16, 16);
                            acc[r][c] += d * a.s[go + g] * v;
                        }
                    }
                    _ => {
                        let (s, m) = group_scales_k4(bb);
                        for g in 0..8 {
                            acc[r][c] += s[g] * a.s[go + g] * dot(g * 32, 32) - m[g] * a.bs[go + g];
                        }
                    }
                }
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Isa;

    /// Random-ish blocks with sane scales (finite f16 d / dmin).
    pub(crate) fn make_blocks(fmt: Format, n_blocks: usize, seed: u32) -> Vec<u8> {
        let bb = fmt.block_bytes();
        let mut v = vec![0u8; n_blocks * bb];
        let mut s = seed.max(1);
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *x = (s >> 8) as u8;
        }
        for b in v.chunks_exact_mut(bb) {
            let d = half::f16::from_f32(0.01).to_le_bytes();
            let dm = half::f16::from_f32(0.004).to_le_bytes();
            match fmt {
                Format::Q6K => b[208..210].copy_from_slice(&d),
                _ => {
                    b[0..2].copy_from_slice(&d);
                    b[2..4].copy_from_slice(&dm);
                }
            }
        }
        v
    }

    /// Reference dequantization straight from the ggml C code (dequantize_row_q*_K).
    fn ggml_ref(fmt: Format, b: &[u8]) -> Vec<f32> {
        let mut y = vec![0f32; QK];
        match fmt {
            Format::Q4K | Format::Q5K => {
                let (d, min) = (f16_at(b, 0), f16_at(b, 2));
                let (qh, ql) = if fmt == Format::Q5K { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
                let mut is = 0;
                let (mut u1, mut u2) = (1u8, 2u8);
                let mut yi = 0;
                for j in 0..4 {
                    let q = &ql[j * 32..];
                    let (sc, m) = scale_min_k4(is, &b[4..16]);
                    let (d1, m1) = (d * sc as f32, min * m as f32);
                    let (sc, m) = scale_min_k4(is + 1, &b[4..16]);
                    let (d2, m2) = (d * sc as f32, min * m as f32);
                    for l in 0..32 {
                        let h = if fmt == Format::Q5K && qh[l] & u1 != 0 { 16 } else { 0 };
                        y[yi] = d1 * ((q[l] & 0xF) + h) as f32 - m1;
                        yi += 1;
                    }
                    for l in 0..32 {
                        let h = if fmt == Format::Q5K && qh[l] & u2 != 0 { 16 } else { 0 };
                        y[yi] = d2 * ((q[l] >> 4) + h) as f32 - m2;
                        yi += 1;
                    }
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            _ => {
                let d = f16_at(b, 208);
                for n in 0..2 {
                    let ql = &b[n * 64..];
                    let qh = &b[128 + n * 32..];
                    let sc = &b[192 + n * 8..];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[n * 128 + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[n * 128 + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[n * 128 + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[n * 128 + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        y
    }

    #[test]
    fn dequant_matches_ggml() {
        for fmt in [Format::Q4K, Format::Q5K, Format::Q6K] {
            let blocks = make_blocks(fmt, 3, 7);
            for b in blocks.chunks_exact(fmt.block_bytes()) {
                let mut got = vec![0f32; QK];
                dequant_block(fmt, b, &mut got);
                assert_eq!(got, ggml_ref(fmt, b), "{fmt:?}");
            }
        }
    }

    #[test]
    fn tiles_match_dequant_reference() {
        let (rows, n, k) = (5, 6, 512);
        let x: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let kg = k / 32;
        let mut q = vec![0i8; rows * k];
        let mut s = vec![0f32; rows * kg];
        for r in 0..rows {
            crate::matmul::quantize_row(&x[r * k..(r + 1) * k], &mut q[r * k..(r + 1) * k], &mut s[r * kg..(r + 1) * kg]);
        }
        let bs: Vec<f32> = (0..rows * kg).map(|g| s[g] * q[g * 32..g * 32 + 32].iter().map(|&v| v as i32).sum::<i32>() as f32).collect();
        let a = KA { q: &q, s: &s, bs: &bs, k };
        let isa = Isa::detect();
        for fmt in [Format::Q4K, Format::Q5K, Format::Q6K] {
            let bytes = make_blocks(fmt, n * k / QK, 11);
            let b = KMat { fmt, bytes: &bytes, cols: k };
            let fast = select(&isa, fmt);
            let cols = [0, 2, 5, 5];
            // Reference: dequantized weights x dequantized activations.
            let mut w = vec![0f32; k];
            for (ci, &col) in cols.iter().enumerate() {
                for blk in 0..k / QK {
                    dequant_block(fmt, b.block(col, blk), &mut w[blk * QK..(blk + 1) * QK]);
                }
                for r in 0..rows {
                    let want: f32 = (0..k).map(|t| q[r * k + t] as f32 * s[r * kg + t / 32] * w[t]).sum();
                    let got = unsafe { (fast.t1)(&a, [r], &b, cols) }[0][ci];
                    let got_s = unsafe { tile_scalar::<1>(&a, [r], &b, cols) }[0][ci];
                    let tol = 1e-3 * want.abs().max(1.0);
                    assert!((want - got).abs() <= tol, "{fmt:?} r{r} c{col}: {want} vs {got}");
                    assert!((want - got_s).abs() <= tol, "{fmt:?} scalar r{r} c{col}: {want} vs {got_s}");
                }
                let got2 = unsafe { (fast.t2)(&a, [1, 4], &b, cols) };
                let want2 = unsafe { tile_scalar::<2>(&a, [1, 4], &b, cols) };
                for r in 0..2 {
                    assert!((got2[r][ci] - want2[r][ci]).abs() <= 1e-3 * want2[r][ci].abs().max(1.0));
                }
                let got4 = unsafe { (fast.t4)(&a, [0, 1, 3, 4], &b, cols) };
                let want4 = unsafe { tile_scalar::<4>(&a, [0, 1, 3, 4], &b, cols) };
                for r in 0..4 {
                    assert!((got4[r][ci] - want4[r][ci]).abs() <= 1e-3 * want4[r][ci].abs().max(1.0));
                }
            }
        }
    }
}
