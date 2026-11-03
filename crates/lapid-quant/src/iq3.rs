//! ggml IQ3_XXS (3.0625 bpw): 256-element super-blocks stored exactly as in GGUF (AoS,
//! little endian). Bit-exact port of `quantize_row_iq3_xxs_impl(256, ..)` and
//! `dequantize_row_iq3_xxs`.
//!
//! ```text
//! IQ3_XXS (98 B): f16 d | qs[64] grid indices (one per 4 elements) |
//!                 8 x u32 (4 x 7 sign bits + 4-bit scale) per 32-group
//!   x = d * (0.5 + s) * 0.5 * iq3xxs_grid[idx][j] * sign
//! ```

use half::f16;

use crate::iqgrid::{IQ3XXS_GRID, KMASK, KSIGNS, QK_K, fmax, lattice_iq3xxs, nearest_int, quantize_rows};

pub const IQ3XXS_BYTES: usize = 98;

/// `dequantize_row_iq3_xxs` for one block.
pub fn dequantize_iq3xxs(b: &[u8], y: &mut [f32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let qs = &b[2..66];
    for ib in 0..8 {
        let o = 66 + 4 * ib;
        let aux = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l])].to_le_bytes();
            let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l + 1])].to_le_bytes();
            let y8 = &mut y[32 * ib + 8 * l..32 * ib + 8 * l + 8];
            for j in 0..4 {
                let s1 = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                let s2 = if signs & KMASK[j + 4] != 0 { -1.0 } else { 1.0 };
                y8[j] = db * f32::from(g1[j]) * s1;
                y8[j + 4] = db * f32::from(g2[j]) * s2;
            }
        }
    }
}

/// Packed lattice point of 4 levels (3 bits each).
#[inline]
fn pack4(l: &[i8]) -> usize {
    l.iter().enumerate().fold(0, |u, (i, &v)| u | ((v as usize) << (3 * i)))
}

/// `quantize_row_iq3_xxs_impl` for one block (`qw = None`: ggml's default weighting).
pub fn quantize_iq3xxs(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const KMAX_Q: i32 = 8;
    let lat = lattice_iq3xxs();
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut waux = [0f32; 32];
    let mut is_on_grid_aux = [true; 8];
    let mut block_signs = [0u8; 8];
    // q3: 64 grid indices, then 8 scales_and_signs words
    let mut q3 = [0u8; 64];
    let mut sas = [0u32; 8];
    out[..IQ3XXS_BYTES].fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &xbl[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for ib in 0..8 {
        let xb = &xbl[32 * ib..32 * ib + 32];
        match qw {
            Some(w) => {
                for i in 0..32 {
                    weight[i] = w[32 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt();
                }
            }
            None => {
                for i in 0..32 {
                    weight[i] = xb[i] * xb[i];
                }
            }
        }
        for i in 0..32 {
            waux[i] = weight[i].sqrt();
        }
        for k in 0..4 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            if nflip % 2 != 0 {
                let mut imin = 0;
                let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for i in 1..8 {
                    let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if ax < min {
                        min = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= 1 << imin;
            }
            block_signs[k] = s & 127;
        }
        let mut max = xval[0];
        for &v in &xval[1..] {
            max = fmax(max, v);
        }
        l.fill(0);
        if max < 1e-8 {
            scales[ib] = 0.0;
            continue;
        }
        let mut best = 0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;
        let mut is_on_grid = [true; 8];
        for is in -15..=15 {
            let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.2) / max;
            let this_scale = 1.0 / id;
            for k in 0..8 {
                for i in 0..4 {
                    let q = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                    laux[4 * k + i] = q.clamp(0, KMAX_Q - 1) as i8;
                }
                let u = pack4(&laux[4 * k..4 * k + 4]);
                is_on_grid_aux[k] = lat.map[u] >= 0;
                if !is_on_grid_aux[k] {
                    lat.best_neighbour(u, &xval[4 * k..], &waux[4 * k..], this_scale, &mut laux[4 * k..4 * k + 4]);
                }
            }
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for i in 0..32 {
                let w = weight[i];
                let q = (2 * laux[i] + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
                is_on_grid = is_on_grid_aux;
            }
        }
        if is_on_grid.contains(&false) && scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..8 {
                if is_on_grid[k] {
                    continue;
                }
                let mut u = 0usize;
                for i in 0..4 {
                    let q = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)).clamp(0, KMAX_Q - 1);
                    u |= (q as usize) << (3 * i);
                }
                let gi = if lat.map[u] < 0 {
                    lat.best_neighbour(u, &xval[4 * k..], &waux[4 * k..], scale, &mut l[4 * k..4 * k + 4])
                } else {
                    lat.map[u] as usize
                };
                let pg = &lat.grid[gi];
                for i in 0..4 {
                    l[4 * k + i] = (pg[i] - 1) / 2;
                }
            }
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for i in 0..32 {
                let w = weight[i];
                let q = (2 * l[i] + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            // as ggml: only the first four sign groups are flipped
            scale = -scale;
            for s in &mut block_signs[..4] {
                *s = !*s & 127;
            }
        }
        for k in 0..8 {
            let u = pack4(&l[4 * k..4 * k + 4]);
            let gi = lat.map[u];
            assert!(gi >= 0, "iq3_xxs: point {u} not on grid");
            q3[8 * ib + k] = gi as u8;
        }
        sas[ib] = u32::from(block_signs[0])
            | (u32::from(block_signs[1]) << 7)
            | (u32::from(block_signs[2]) << 14)
            | (u32::from(block_signs[3]) << 21);
        scales[ib] = scale;
        max_scale = fmax(max_scale, scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    out[..2].copy_from_slice(&f16::from_f32(d * 1.0125).to_le_bytes());
    let id = 1.0 / d;
    for ib in 0..8 {
        let q = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        sas[ib] |= (q as u32) << 28;
    }
    out[2..66].copy_from_slice(&q3);
    for (i, w) in sas.iter().enumerate() {
        out[66 + 4 * i..70 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
}

/// Quantize `rows` (row-major, `n_per_row` % 256 == 0) to IQ3_XXS blocks with optional
/// importance weights (`imatrix`, one per column). Multithreaded; deterministic.
pub fn quantize(rows: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    quantize_rows(rows, n_per_row, imatrix, IQ3XXS_BYTES, quantize_iq3xxs)
}
