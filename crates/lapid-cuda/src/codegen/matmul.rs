//! MatMul planning (GEMV / tensor-core GEMM / scalar GEMM), the scalar tiled GEMM
//! (`kernels/gemm.ptx`) and the fused epilogue shared by all three.
//!
//! Params layout (all variants):
//! `[batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_off_bytes, P11, P12,
//!   out.off, out.bs, out.rs, extra_e(off, s_batch, s_m, s_n)...]` with P11 = column blocks
//! (GEMV), else unused. The output is written at `out.off + bt*out.bs + i*out.rs + j`
//! (strided outputs write straight into a slice of a larger buffer, e.g. a KV cache).

use std::sync::Arc;

use lapid_compiler::{MatMulDims, MatMulSpec};
use lapid_tensor::{DType, Result};

use super::expr::{check_root, render, uses, uses_as};
use super::ptx::{self, Gen};
use super::{Key, Params, Plan, bad, check_plain, gemv, mma, scales_offset, template};
use crate::context::Caps;
use lapid_tensor::Error;

const GEMM: &str = include_str!("../kernels/gemm.ptx");

/// Params word of the output view (offset, batch stride, row stride).
pub(super) const OUT_BASE: usize = 13;
pub(super) const EXTRA_BASE: usize = 16;
/// Largest m served by the GEMV kernel.
pub(super) const GEMV_MAX_M: usize = 4;
/// Outputs (batch * m * n) up to which a transposed-B product whose GEMM grid would not
/// fill the device runs as GEMV (a warp per output, k split across warps) instead: e.g.
/// decode attention scores (64 heads x a few dozen keys x 512), a single 64 x 64 tile
/// walking k alone otherwise.
const GEMV_SMALL_OUT: usize = 4096;
const TILE: usize = 64;

pub(super) fn plan(spec: &Arc<MatMulSpec>, d: &MatMulDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.a_dtype, "matmul A")?;
    check_plain(spec.out_dtype, "matmul output")?;
    for &t in &spec.extra_dtypes {
        check_plain(t, "matmul epilogue input")?;
    }
    if d.extra_views.len() != spec.extra_dtypes.len() {
        return Err(bad("matmul: extra_views / extra_dtypes length mismatch"));
    }
    if let Some(e) = &spec.epilogue
        && e.num_loads() > 1 + spec.extra_dtypes.len()
    {
        return Err(bad("matmul: epilogue loads more inputs than bound"));
    }
    if let Some(e) = &spec.epilogue {
        check_root(e, false, "matmul epilogue")?;
    }
    let quant = spec.b_dtype.is_quant();
    if quant && !spec.b_transposed {
        return Err(bad("matmul: quantized B must be transposed"));
    }
    let qb = super::quant_block(spec.b_dtype);
    if quant && !d.k.is_multiple_of(qb) {
        return Err(bad(format!(
            "matmul: quantized B ({}) needs k % {qb} == 0 (k = {})",
            spec.b_dtype.name(),
            d.k
        )));
    }
    let tiles = d.batch * d.m.div_ceil(TILE) * d.n.div_ceil(TILE);
    let small = d.batch * d.m * d.n <= GEMV_SMALL_OUT && tiles < caps.multiprocessors as usize;
    let gemv = spec.b_transposed && (d.m <= GEMV_MAX_M || small);
    let so = if quant {
        scales_offset(spec.b_dtype, d.n, d.k)
    } else {
        0
    };
    let mut p = Params::default();
    p.u(d.batch)?.u(d.m)?.u(d.n)?.u(d.k)?;
    p.u(d.a.offset)?.u(d.a.batch_stride)?.u(d.a.row_stride)?;
    p.u(d.b.offset)?.u(d.b.batch_stride)?.u(d.b.row_stride)?;
    p.u(so)?;
    let empty = d.batch == 0 || d.m == 0 || d.n == 0;
    if gemv {
        let w = gemv::split(d.batch * d.m * d.n, d.k);
        let cb = d.n.div_ceil(gemv::COLS / w as usize);
        p.u(cb)?.u(0)?;
        out_view(&mut p, d)?;
        for v in &d.extra_views {
            p.view(v, 3)?;
        }
        let (va, vb) = gemv::vec_variants(spec, d);
        let blocks = (d.batch * d.m * cb) as u64;
        let grid = if empty {
            [0, 0, 0]
        } else {
            super::grid1(blocks, caps)?
        };
        return Ok(Plan::new(Key::Gemv(spec.clone(), va, vb, w), p.0, grid, 256));
    }
    let use_mma = mma_ok(spec, caps);
    // B larger than A (e.g. a logits projection): column-tile-major grid, B read once
    let b_bytes = match spec.b_dtype.quant_format() {
        Some(f) => lapid_quant::layout(f, d.n, d.k).total,
        None => d.n * d.k * spec.b_dtype.size(),
    };
    let swap = use_mma && d.batch == 1 && d.m > mma::BM && b_bytes > d.m * d.k * spec.a_dtype.size();
    p.u(swap as usize)?.u(0)?;
    out_view(&mut p, d)?;
    for v in &d.extra_views {
        p.view(v, 3)?;
    }
    let (bm, bn) = if use_mma {
        (mma::BM, mma::BN)
    } else {
        (TILE, TILE)
    };
    let (tm, tn) = (d.m.div_ceil(bm), d.n.div_ceil(bn));
    if tm > caps.max_grid[1] as usize
        || d.batch > caps.max_grid[2] as usize
        || tn > caps.max_grid[0] as usize
    {
        return Err(Error::new(
            "too_large",
            format!("matmul grid {tn}x{tm}x{} exceeds device limits", d.batch),
        ));
    }
    if swap && (tn > caps.max_grid[1] as usize || tm > caps.max_grid[0] as usize) {
        return Err(Error::new("too_large", format!("matmul grid {tm}x{tn} exceeds device limits")));
    }
    let grid = if empty {
        [0, 0, 0]
    } else if swap {
        [tm as u32, tn as u32, 1]
    } else {
        [tn as u32, tm as u32, d.batch as u32]
    };
    let key = if use_mma {
        let (va, vb) = mma::vec_variants(spec, d);
        Key::GemmMma(spec.clone(), va, vb)
    } else {
        Key::Gemm(spec.clone())
    };
    Ok(Plan::new(key, p.0, grid, 256))
}

fn out_view(p: &mut Params, d: &MatMulDims) -> Result<()> {
    if d.batch > 0 && d.m > 0 && d.n > 0 {
        // the whole output must be addressable with 32-bit element indices
        let last = d.out.offset as u128
            + (d.batch as u128 - 1) * d.out.batch_stride as u128
            + (d.m as u128 - 1) * d.out.row_stride as u128
            + d.n as u128;
        if last > u32::MAX as u128 {
            return Err(Error::new("too_large", "matmul output exceeds 32-bit indexing"));
        }
    }
    p.u(d.out.offset)?.u(d.out.batch_stride)?.u(d.out.row_stride)?;
    Ok(())
}

/// Whether the tensor-core GEMM serves `spec`: f16 or quantized weights (f32 weights keep
/// the exact f32 path), f32 / f16 activations.
pub(super) fn mma_ok(spec: &MatMulSpec, caps: &Caps) -> bool {
    caps.mma()
        && (spec.b_dtype == DType::F16 || spec.b_dtype.is_quant())
        && matches!(spec.a_dtype, DType::F32 | DType::F16)
}

/// Buffer names of a matmul kernel: a, b, x0.., out.
pub(super) fn names(spec: &MatMulSpec) -> Vec<String> {
    let mut v = vec!["a".to_string(), "b".to_string()];
    v.extend((0..spec.extra_dtypes.len()).map(|e| format!("x{e}")));
    v.push("out".to_string());
    v
}

pub(super) fn args(spec: &MatMulSpec) -> (String, String) {
    let n = names(spec);
    ptx::args(&n.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Store of one output element. Expects `%acc` (f32), `%bt`, `%gi`, `%gj`, `%M`, `%N`
/// (u32) in scope; loads the epilogue extras at (bt, gi, gj).
pub(super) fn epilogue(spec: &MatMulSpec) -> String {
    let mut g = Gen::new();
    let value = match &spec.epilogue {
        None => "%acc".to_string(),
        Some(e) => {
            let mut xs = vec![(String::new(), String::new()); spec.extra_dtypes.len()];
            for (x, &t) in spec.extra_dtypes.iter().enumerate() {
                if !uses(e, 1 + x) {
                    continue;
                }
                let b = EXTRA_BASE + 4 * x;
                let idx = g.param(b);
                let s = g.param(b + 1);
                g.op(format!("mad.lo.u32 {idx}, %bt, {s}, {idx}"));
                let s = g.param(b + 2);
                g.op(format!("mad.lo.u32 {idx}, %gi, {s}, {idx}"));
                let s = g.param(b + 3);
                g.op(format!("mad.lo.u32 {idx}, %gj, {s}, {idx}"));
                let (wf, wb) = (uses_as(e, 1 + x, false), uses_as(e, 1 + x, true));
                xs[x] = ptx::load_input(&mut g, t, &format!("%g_x{x}"), &idx, wf, wb, false);
            }
            let acc_bits = if uses_as(e, 0, true) {
                let r = g.reg("b32");
                g.op(format!("mov.b32 {r}, %acc"));
                r
            } else {
                String::new()
            };
            render(&mut g, e, &|l, bits| match (l, bits) {
                (0, false) => "%acc".to_string(),
                (0, true) => acc_bits.clone(),
                (l, false) => xs[l - 1].0.clone(),
                (l, true) => xs[l - 1].1.clone(),
            })
        }
    };
    let o = g.param(OUT_BASE);
    let s = g.param(OUT_BASE + 1);
    g.op(format!("mad.lo.u32 {o}, %bt, {s}, {o}"));
    let s = g.param(OUT_BASE + 2);
    g.op(format!("mad.lo.u32 {o}, %gi, {s}, {o}"));
    g.op(format!("add.u32 {o}, {o}, %gj"));
    ptx::store(&mut g, spec.out_dtype, "%g_out", &o, &value);
    g.finish()
}

/// Load one element of B at (row j = `gj`, k = `gk`) of the logical [n, k] (transposed) or
/// [k, n] matrix into `dst` (zero when `valid` is false). Expects `%bbase`, `%brs`, `%so`.
fn load_b(g: &mut Gen, spec: &MatMulSpec, gj: &str, gk: &str, valid: &str, dst: &str) {
    let idx = g.reg("u32");
    if spec.b_dtype.is_quant() {
        g.op(format!("mad.lo.u32 {idx}, {gj}, %K, {gk}"));
    } else if spec.b_transposed {
        g.op(format!("mad.lo.u32 {idx}, {gj}, %brs, %bbase"));
        g.op(format!("add.u32 {idx}, {idx}, {gk}"));
    } else {
        g.op(format!("mad.lo.u32 {idx}, {gk}, %brs, %bbase"));
        g.op(format!("add.u32 {idx}, {idx}, {gj}"));
    }
    g.op(format!("selp.b32 {idx}, {idx}, 0, {valid}"));
    if spec.b_dtype.is_quant() {
        ptx::load_quant(g, spec.b_dtype, "%g_b", "%so", &idx, dst);
    } else {
        ptx::load(g, spec.b_dtype, "%g_b", &idx, dst, true);
    }
    g.op(format!("selp.f32 {dst}, {dst}, 0f00000000, {valid}"));
}

/// Scalar GEMM tile loads: 4 elements of A and of B per thread into As[kk][m] / Bs[kk][n].
fn tile_loads(spec: &MatMulSpec) -> (String, String) {
    let mut la = String::new();
    let mut lb = String::new();
    for u in 0..4 {
        // A: row = e / 16, kk = e % 16
        let mut g = Gen::new();
        let e = g.reg("u32");
        let (r, kk, gi, gk, idx) = (
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
        );
        let valid = g.reg("pred");
        let v = g.reg("f32");
        g.op(format!("add.u32 {e}, %tz, {}", u * 256));
        g.op(format!("shr.u32 {r}, {e}, 4"));
        g.op(format!("and.b32 {kk}, {e}, 15"));
        g.op(format!("add.u32 {gi}, %row0, {r}"));
        g.op(format!("add.u32 {gk}, %k0, {kk}"));
        g.op(format!("setp.lt.u32 {valid}, {gi}, %M"));
        g.op(format!("setp.lt.and.u32 {valid}, {gk}, %K, {valid}"));
        g.op(format!("mad.lo.u32 {idx}, {gi}, %ars, %abase"));
        g.op(format!("add.u32 {idx}, {idx}, {gk}"));
        g.op(format!("selp.b32 {idx}, {idx}, 0, {valid}"));
        ptx::load(&mut g, spec.a_dtype, "%g_a", &idx, &v, true);
        g.op(format!("selp.f32 {v}, {v}, 0f00000000, {valid}"));
        g.op(format!("mad.lo.u32 {idx}, {kk}, 64, {r}"));
        g.op(format!("mad.lo.u32 {idx}, {idx}, 4, %sm"));
        g.op(format!("st.shared.f32 [{idx}], {v}"));
        la.push_str(&g.finish());

        // B: transposed (or quantized): n = e / 16, kk = e % 16; else kk = e / 64, n = e % 64
        let mut g = Gen::new();
        let e = g.reg("u32");
        let (n, kk, gj, gk, a) = (
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
        );
        let valid = g.reg("pred");
        let v = g.reg("f32");
        g.op(format!("add.u32 {e}, %tz, {}", u * 256));
        if spec.b_transposed {
            g.op(format!("shr.u32 {n}, {e}, 4"));
            g.op(format!("and.b32 {kk}, {e}, 15"));
        } else {
            g.op(format!("shr.u32 {kk}, {e}, 6"));
            g.op(format!("and.b32 {n}, {e}, 63"));
        }
        g.op(format!("add.u32 {gj}, %col0, {n}"));
        g.op(format!("add.u32 {gk}, %k0, {kk}"));
        g.op(format!("setp.lt.u32 {valid}, {gj}, %N"));
        g.op(format!("setp.lt.and.u32 {valid}, {gk}, %K, {valid}"));
        load_b(&mut g, spec, &gj, &gk, &valid, &v);
        g.op(format!("mad.lo.u32 {a}, {kk}, 64, {n}"));
        g.op(format!("mad.lo.u32 {a}, {a}, 4, %sm"));
        g.op(format!("st.shared.f32 [{a}+4096], {v}"));
        lb.push_str(&g.finish());
    }
    (la, lb)
}

/// Scalar tiled GEMM (any dtypes / devices).
pub(super) fn source(spec: &MatMulSpec) -> String {
    let (args, ptrs) = args(spec);
    let (la, lb) = tile_loads(spec);
    template::fill(
        GEMM,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD_A", &la),
            ("LOAD_B", &lb),
            ("EPILOGUE", &epilogue(spec)),
        ],
        &[],
    )
}
