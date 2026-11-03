//! ggml IQ2_XXS (2.0625 bpw) and IQ2_S (2.5625 bpw): 256-element super-blocks stored exactly
//! as in GGUF (AoS, little endian). Bit-exact ports of `quantize_row_iq2_{xxs,s}_impl` and
//! `dequantize_row_iq2_{xxs,s}` (see `iqgrid` for the shared lattice machinery).
//!
//! ```text
//! IQ2_XXS (66 B): f16 d | qs: 8 x (u32 grid indices (4 x u8) | u32 signs (4 x 7 bits) + 4-bit scale)
//!   32-group ib: x = d * (0.5 + s) * 0.25 * iq2xxs_grid[idx][j] * sign
//! IQ2_S (82 B):   f16 d | qs[32] grid index low bytes | signs[32] | qh[8] (2 high index bits
//!   per 8-group) | scales[8] (4 bits per 16-group)
//!   16-group: x = d * (0.5 + s) * 0.25 * iq2s_grid[idx][j] * sign
//! ```

use half::f16;

use crate::iqgrid::{IQ2S_GRID, IQ2XXS_GRID, KMASK, KSIGNS, QK_K, fmax, lattice_iq2s, lattice_iq2xxs, make_qp_quants, nearest_int, quantize_rows};

pub const IQ2XXS_BYTES: usize = 66;
pub const IQ2S_BYTES: usize = 82;

#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// `dequantize_row_iq2_xxs` for one block.
pub fn dequantize_iq2xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for ib in 0..8 {
        let a0 = u32_at(b, 2 + 8 * ib);
        let a1 = u32_at(b, 6 + 8 * ib);
        let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize].to_le_bytes();
            let signs = KSIGNS[((a1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let s = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                y[32 * ib + 8 * l + j] = db * f32::from(grid[j]) * s;
            }
        }
    }
}

/// `dequantize_row_iq2_s` for one block.
pub fn dequantize_iq2s(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib in 0..8 {
        let db = [d * (0.5 + (scales[ib] & 15) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx = usize::from(qs[4 * ib + l]) | ((usize::from(qh[ib]) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                let s = if signs[4 * ib + l] & KMASK[j] != 0 { -1.0 } else { 1.0 };
                y[32 * ib + 8 * l + j] = dl * f32::from(grid[j]) * s;
            }
        }
    }
}

/// Packed lattice point of 8 levels (2 bits each).
#[inline]
fn pack8(l: &[i8]) -> usize {
    l.iter().enumerate().fold(0, |u, (i, &v)| u | ((v as usize) << (2 * i)))
}

/// `quantize_row_iq2_xxs_impl` for one block. ggml requires importance weights for IQ2_XXS;
/// without them (`qw = None`) this uses unit weights (`weight = sqrt(sigma2 + x^2)`).
pub fn quantize_iq2xxs(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const KMAX_Q: i32 = 3;
    let lat = lattice_iq2xxs();
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut waux = [0f32; 32];
    let mut block_signs = [0u8; 4];
    let mut q2 = [0u32; 16];
    out[..IQ2XXS_BYTES].fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &xbl[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = sumx2 / QK_K as f32;
    for ib in 0..8 {
        let xb = &xbl[32 * ib..32 * ib + 32];
        for i in 0..32 {
            let q = qw.map_or(1.0, |w| w[32 * ib + i]);
            weight[i] = q * (sigma2 + xb[i] * xb[i]).sqrt();
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
        if max < 1e-15 {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut lu = [0u8; 32];
        let mut scale = make_qp_quants(32, KMAX_Q + 1, &xval, &mut lu, &weight);
        for i in 0..32 {
            l[i] = lu[i] as i8;
        }
        let eff_max = scale * KMAX_Q as f32;
        if eff_max <= 0.0 {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut best = 0f32;
        for is in -6..=6 {
            let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / eff_max;
            let this_scale = 1.0 / id;
            for k in 0..4 {
                for i in 0..8 {
                    let q = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                    laux[8 * k + i] = q.clamp(0, KMAX_Q - 1) as i8;
                }
                let u = pack8(&laux[8 * k..8 * k + 8]);
                if lat.map[u] < 0 {
                    lat.best_neighbour(u, &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..8 * k + 8]);
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
            }
        }
        if scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..4 {
                let mut u = 0usize;
                for i in 0..8 {
                    let q = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, KMAX_Q - 1);
                    u |= (q as usize) << (2 * i);
                }
                let gi = if lat.map[u] < 0 {
                    lat.best_neighbour(u, &xval[8 * k..], &waux[8 * k..], scale, &mut l[8 * k..8 * k + 8])
                } else {
                    lat.map[u] as usize
                };
                let pg = &lat.grid[gi];
                for i in 0..8 {
                    l[8 * k + i] = (pg[i] - 1) / 2;
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
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s & 127;
            }
        }
        for k in 0..4 {
            let u = pack8(&l[8 * k..8 * k + 8]);
            let gi = lat.map[u];
            assert!(gi >= 0, "iq2_xxs: point {u} not on grid");
            q2[2 * ib] |= (gi as u32) << (8 * k);
            q2[2 * ib + 1] |= u32::from(block_signs[k]) << (7 * k);
        }
        scales[ib] = scale;
        max_scale = fmax(max_scale, scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    out[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    let id = 1.0 / d;
    for ib in 0..8 {
        let q = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        q2[2 * ib + 1] |= (q as u32) << 28;
    }
    for (i, w) in q2.iter().enumerate() {
        out[2 + 4 * i..6 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
}

/// `quantize_row_iq2_s_impl` for one block (`qw = None`: ggml's default weighting).
pub fn quantize_iq2s(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const KMAX_Q: i32 = 3;
    let lat = lattice_iq2s();
    let mut scales = [0f32; 16];
    let mut weight = [0f32; 16];
    let mut xval = [0f32; 16];
    let mut l = [0i8; 16];
    let mut laux = [0i8; 16];
    let mut waux = [0f32; 16];
    let mut is_on_grid_aux = [true; 2];
    let mut block_signs = [0u8; 2];
    let y = &mut out[..IQ2S_BYTES];
    y.fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &xbl[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for ib in 0..16 {
        let xb = &xbl[16 * ib..16 * ib + 16];
        match qw {
            Some(w) => {
                for i in 0..16 {
                    weight[i] = w[16 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt();
                }
            }
            None => {
                for i in 0..16 {
                    weight[i] = 0.25 * sigma2 + xb[i] * xb[i];
                }
            }
        }
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        for k in 0..2 {
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    s |= 1 << i;
                }
            }
            block_signs[k] = s;
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
        let mut is_on_grid = [true; 2];
        for is in -9..=9 {
            let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
            let this_scale = 1.0 / id;
            for k in 0..2 {
                for i in 0..8 {
                    let q = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                    laux[8 * k + i] = q.clamp(0, KMAX_Q - 1) as i8;
                }
                let u = pack8(&laux[8 * k..8 * k + 8]);
                is_on_grid_aux[k] = lat.map[u] >= 0;
                if !is_on_grid_aux[k] {
                    lat.best_neighbour(u, &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..8 * k + 8]);
                }
            }
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for i in 0..16 {
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
            for k in 0..2 {
                if is_on_grid[k] {
                    continue;
                }
                let mut u = 0usize;
                for i in 0..8 {
                    let q = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, KMAX_Q - 1);
                    u |= (q as usize) << (2 * i);
                    l[8 * k + i] = q as i8;
                }
                if lat.map[u] < 0 {
                    lat.best_neighbour(u, &xval[8 * k..], &waux[8 * k..], scale, &mut l[8 * k..8 * k + 8]);
                }
            }
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            for i in 0..16 {
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
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s;
            }
        }
        for k in 0..2 {
            let u = pack8(&l[8 * k..8 * k + 8]);
            let gi = lat.map[u];
            assert!(gi >= 0, "iq2_s: point {u} not on grid");
            let i8 = 2 * ib + k;
            y[2 + i8] = (gi & 255) as u8;
            y[66 + i8 / 4] |= ((gi >> 8) << (2 * (i8 % 4))) as u8;
            y[34 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = fmax(max_scale, scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16::from_f32(d * 0.9875).to_le_bytes());
    let id = 1.0 / d;
    for ib in 0..16 {
        let q = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            y[74 + ib / 2] = q;
        } else {
            y[74 + ib / 2] |= q << 4;
        }
    }
}

/// Quantize `rows` (row-major, `n_per_row` % 256 == 0) to IQ2_XXS blocks with optional
/// importance weights (`imatrix`, one per column). Multithreaded; deterministic.
pub fn quantize_xxs(rows: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    quantize_rows(rows, n_per_row, imatrix, IQ2XXS_BYTES, quantize_iq2xxs)
}

/// Quantize to IQ2_S blocks (see [`quantize_xxs`]).
pub fn quantize_s(rows: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    quantize_rows(rows, n_per_row, imatrix, IQ2S_BYTES, quantize_iq2s)
}
