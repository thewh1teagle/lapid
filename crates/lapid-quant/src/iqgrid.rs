//! Shared machinery of the ggml i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S):
//! dequantization grids and sign tables (`ggml-common.h`), the quantizer lattices with their
//! point maps and neighbour lists (`iq2xs_init_impl` / `iq3xs_init_impl`), and the helpers
//! the row quantizers share (`make_qp_quants`, `iq{2,3}_find_best_neighbour`).
//!
//! Everything is a bit-exact port of ggml `ggml-quants.c` (compiled without floating-point
//! contraction); f32 expressions keep the C evaluation order.

use std::sync::OnceLock;

use crate::iqtables::*;

/// Elements per super-block.
pub const QK_K: usize = 256;

/// ggml `kmask_iq2xs`.
pub const KMASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// ggml `ksigns_iq2xs`: 7 sign bits plus an 8th bit making the count of set bits even.
pub static KSIGNS: [u8; 128] = ksigns();

const fn ksigns() -> [u8; 128] {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = i as u8 | (((i as u32).count_ones() as u8 & 1) << 7);
        i += 1;
    }
    t
}

/// 2-bit grid point -> dequantization grid (8 bytes, little endian u64): l -> {8, 25, 43}.
const fn grid2<const N: usize>(k: &[u16; N]) -> [u64; N] {
    const V: [u64; 3] = [0x08, 0x19, 0x2b];
    let mut t = [0u64; N];
    let mut j = 0;
    while j < N {
        let mut i = 0;
        while i < 8 {
            t[j] |= V[((k[j] >> (2 * i)) & 3) as usize] << (8 * i);
            i += 1;
        }
        j += 1;
    }
    t
}

/// 3-bit grid point -> dequantization grid (4 bytes, little endian u32).
const fn grid3<const N: usize>(k: &[u16; N], v: [u32; 8]) -> [u32; N] {
    let mut t = [0u32; N];
    let mut j = 0;
    while j < N {
        let mut i = 0;
        while i < 4 {
            t[j] |= v[((k[j] >> (3 * i)) & 7) as usize] << (8 * i);
            i += 1;
        }
        j += 1;
    }
    t
}

/// ggml `iq2xxs_grid`.
pub static IQ2XXS_GRID: [u64; 256] = grid2(&KGRID_2BIT_256);
/// ggml `iq2xs_grid`.
pub static IQ2XS_GRID: [u64; 512] = grid2(&KGRID_2BIT_512);
/// ggml `iq2s_grid`.
pub static IQ2S_GRID: [u64; 1024] = grid2(&KGRID_2BIT_1024);
/// ggml `iq3xxs_grid`.
pub static IQ3XXS_GRID: [u32; 256] = grid3(&KGRID_3BIT_256, [4, 12, 20, 28, 36, 44, 52, 62]);
/// ggml `iq3s_grid`.
pub static IQ3S_GRID: [u32; 512] = grid3(&KGRID_3BIT_512, [1, 3, 5, 7, 9, 11, 13, 15]);

/// ggml `nearest_int`: round half to even via the 1.5 * 2^23 trick (|x| <= 4194303).
#[inline]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// C `MAX(a, b)` on floats.
#[inline]
pub(crate) fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// A quantizer lattice: `D`-dimensional points with odd coordinates `2l + 1`, the map from
/// a packed point (`D` fields of `bits` bits) to its grid index, and for points off the grid
/// a neighbour list (ggml `kmap_*` / `kneighbors_*`: `map[u] = -(offset + 1)`, at
/// `neighbours[offset]` the count n followed by n grid indices ordered by (distance, index)).
pub(crate) struct Lattice<const D: usize> {
    pub grid: Vec<[i8; D]>,
    pub map: Vec<i32>,
    pub neighbours: Vec<u16>,
}

impl<const D: usize> Lattice<D> {
    fn new(kgrid: &[u16], bits: u32, nwant: usize) -> Self {
        let mask = (1u16 << bits) - 1;
        let pos = |v: u32| -> [i8; D] { std::array::from_fn(|i| (2 * ((v >> (bits * i as u32)) as u16 & mask) + 1) as i8) };
        let grid: Vec<[i8; D]> = kgrid.iter().map(|&k| pos(u32::from(k))).collect();
        let kmap_size = if bits == 2 { 43692 } else { 4096 };
        let mut map = vec![-1i32; kmap_size];
        for (i, g) in grid.iter().enumerate() {
            let mut index = 0usize;
            for (k, &c) in g.iter().enumerate() {
                index |= (((c - 1) / 2) as usize) << (bits as usize * k);
            }
            map[index] = i as i32;
        }
        // neighbour lists of the off-grid points, computed in parallel, concatenated in order
        let off: Vec<usize> = (0..kmap_size).filter(|&i| map[i] < 0).collect();
        let lists = par_map(&off, |&i| {
            let p = pos(i as u32);
            let mut dist: Vec<(i32, i32)> = grid
                .iter()
                .enumerate()
                .map(|(j, g)| (g.iter().zip(&p).map(|(&a, &b)| (i32::from(a) - i32::from(b)).pow(2)).sum(), j as i32))
                .collect();
            dist.sort_unstable();
            let mut out = vec![0u16];
            let (mut d2, mut nhave) = (dist[0].0, 1);
            for &(d, j) in &dist {
                if d > d2 {
                    if nhave == nwant {
                        break;
                    }
                    d2 = d;
                    nhave += 1;
                }
                out.push(j as u16);
            }
            out[0] = (out.len() - 1) as u16;
            out
        });
        let mut neighbours = Vec::new();
        for (&i, l) in off.iter().zip(lists) {
            map[i] = -(neighbours.len() as i32 + 1);
            neighbours.extend(l);
        }
        Lattice { grid, map, neighbours }
    }

    /// ggml `iq{2,3}_find_best_neighbour` for the off-grid packed point `u`: the neighbour
    /// minimizing the weighted squared error at `scale`; writes its levels to `l`.
    pub fn best_neighbour(&self, u: usize, xval: &[f32], weight: &[f32], scale: f32, l: &mut [i8]) -> usize {
        let start = (-self.map[u] - 1) as usize;
        let n = usize::from(self.neighbours[start]);
        assert!(n > 0, "i-quant lattice point without neighbours");
        let mut best_d2 = f32::MAX;
        let mut best = usize::MAX;
        for &j in &self.neighbours[start + 1..=start + n] {
            let pg = &self.grid[usize::from(j)];
            let mut d2 = 0f32;
            for i in 0..D {
                let q = f32::from(pg[i]);
                let diff = scale * q - xval[i];
                d2 += weight[i] * diff * diff;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best = usize::from(j);
            }
        }
        let pg = &self.grid[best];
        for i in 0..D {
            l[i] = (pg[i] - 1) / 2;
        }
        best
    }
}

/// IQ2_XXS lattice (256 points, 2 nearest distances).
pub(crate) fn lattice_iq2xxs() -> &'static Lattice<8> {
    static L: OnceLock<Lattice<8>> = OnceLock::new();
    L.get_or_init(|| Lattice::new(&KGRID_2BIT_256, 2, 2))
}

/// IQ2_XS lattice (512 points, 2 nearest distances).
pub(crate) fn lattice_iq2xs() -> &'static Lattice<8> {
    static L: OnceLock<Lattice<8>> = OnceLock::new();
    L.get_or_init(|| Lattice::new(&KGRID_2BIT_512, 2, 2))
}

/// IQ2_S lattice (1024 points, nearest distance only).
pub(crate) fn lattice_iq2s() -> &'static Lattice<8> {
    static L: OnceLock<Lattice<8>> = OnceLock::new();
    L.get_or_init(|| Lattice::new(&KGRID_2BIT_1024, 2, 1))
}

/// IQ3_XXS lattice (256 points, 2 nearest distances).
pub(crate) fn lattice_iq3xxs() -> &'static Lattice<4> {
    static L: OnceLock<Lattice<4>> = OnceLock::new();
    L.get_or_init(|| Lattice::new(&KGRID_3BIT_256, 3, 2))
}

/// ggml `make_qp_quants`: non-negative quants `L` in 0..=nmax for non-negative `x`, weighted.
pub(crate) fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], w: &[f32]) -> f32 {
    let mut max = 0f32;
    for &v in &x[..n] {
        max = fmax(max, v);
    }
    if max < 1e-15 {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let diff = x[i] - scale * f32::from(l[i]);
        best_mse += w[i] * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let q = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * q as f32;
            mse += w[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).min(nmax);
        l[i] = q as u8;
        sumlx += w[i] * x[i] * q as f32;
        suml2 += w[i] * q as f32 * q as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let li = f32::from(l[i]);
            let mut slx = sumlx - w[i] * x[i] * li;
            let mut sl2 = suml2 - w[i] * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != i32::from(l[i]) {
                    slx += w[i] * x[i] * new_l as f32;
                    sl2 += w[i] * new_l as f32 * new_l as f32;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// `f` over `items` on all cores (contiguous chunks), results in order.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len().max(1));
    let chunk = items.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items.chunks(chunk).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().expect("i-quant worker panicked")).collect()
    })
}

/// Quantize `rows` rows of `n_per_row` values block by block on all cores: `block(x, qw,
/// out)` gets 256 values, their importance weights (`None` without an imatrix) and
/// `bytes` output bytes.
pub(crate) fn quantize_rows(
    rows: &[f32],
    n_per_row: usize,
    imatrix: Option<&[f32]>,
    bytes: usize,
    block: impl Fn(&[f32], Option<&[f32]>, &mut [u8]) + Sync,
) -> Vec<u8> {
    assert!(n_per_row > 0 && n_per_row % QK_K == 0, "i-quant rows must be a positive multiple of {QK_K}, got {n_per_row}");
    assert_eq!(rows.len() % n_per_row, 0, "i-quant input is not a whole number of rows");
    if let Some(w) = imatrix {
        assert_eq!(w.len(), n_per_row, "imatrix length must equal n_per_row");
    }
    let nb = rows.len() / QK_K;
    let mut out = vec![0u8; nb * bytes];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // whole blocks per worker; block i of the input is column block i % (n_per_row / 256)
    let per = nb.div_ceil(threads).max(1);
    let bpr = n_per_row / QK_K;
    std::thread::scope(|s| {
        for (ci, (dst, src)) in out.chunks_mut(per * bytes).zip(rows.chunks(per * QK_K)).enumerate() {
            let block = &block;
            s.spawn(move || {
                for (bi, (o, x)) in dst.chunks_exact_mut(bytes).zip(src.chunks_exact(QK_K)).enumerate() {
                    let cb = (ci * per + bi) % bpr;
                    block(x, imatrix.map(|w| &w[cb * QK_K..(cb + 1) * QK_K]), o);
                }
            });
        }
    });
    out
}
