//! MatMulId (mixture-of-experts matmul, ggml MUL_MAT_ID semantics).
//!
//! Few tokens (decode): the GEMV kernel with one "batch" per (token, slot) row whose B
//! pointer is moved to the row's expert (`Key::GemvId`). Many tokens (prefill): a routing
//! stage groups the rows by expert into scratch (`kernels/moe_route.ptx`), then the
//! tensor-core GEMM runs over (expert, 64-row tile) pairs with gathered A rows and scattered
//! output rows (`Key::GemmMmaId`), so each expert's weights are read once per column tile.
//!
//! Both variants share the matmul params layout with: word 12 = r, 16 = E, 17 = expert
//! stride in bytes (a quantized expert is one lapid-quant matrix of `layout(n, k).total`).

use std::sync::Arc;

use lapid_compiler::{MatMulDims, MatMulIdDims, MatMulIdSpec, MatMulSpec, MatView};
use lapid_tensor::{Error, Result};

use super::ptx;
use super::{Key, Params, Plan, bad, check_plain, gemv, grid1, matmul, mma, quant_block, scales_offset, template};
use crate::context::Caps;

const ROUTE: &str = include_str!("../kernels/moe_route.ptx");
/// Most experts the routed GEMM path handles: two shared `u32` counters per expert in the
/// routing block, 48 KiB (the static shared memory limit). Expert caches index a pool of
/// slots as the experts (e.g. ~4500 slots of DeepSeek V4.1 experts on a 128 GB GB10).
const MAX_E: usize = 6144;
/// Most (token, slot) rows served by the GEMV variant regardless of t.
const GEMV_MAX_ROWS: usize = 32;

pub(super) fn plan(spec: &Arc<MatMulIdSpec>, d: &MatMulIdDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.x_dtype, "matmul_id x")?;
    check_plain(spec.out_dtype, "matmul_id output")?;
    let quant = spec.w_dtype.is_quant();
    let qb = quant_block(spec.w_dtype);
    if quant && !d.k.is_multiple_of(qb) {
        return Err(bad(format!(
            "matmul_id: quantized experts ({}) need k % {qb} == 0 (k = {})",
            spec.w_dtype.name(),
            d.k
        )));
    }
    let ms = Arc::new(MatMulSpec {
        a_dtype: spec.x_dtype,
        b_dtype: spec.w_dtype,
        b_transposed: true,
        extra_dtypes: vec![],
        out_dtype: spec.out_dtype,
        epilogue: None,
    });
    let stride = match spec.w_dtype.quant_format() {
        Some(f) => lapid_quant::layout(f, d.n, d.k).total,
        None => d.n * d.k * spec.w_dtype.size(),
    };
    let rows = d.t.checked_mul(d.r).ok_or_else(|| bad("matmul_id: size overflow"))?;
    let so = if quant { scales_offset(spec.w_dtype, d.n, d.k) } else { 0 };
    let empty = rows == 0 || d.n == 0;
    // GEMV rows also when (token, slot) rows are few overall (e.g. a down projection over
    // the t * r routed rows of one decode token, r = 1): the routed GEMM would run mostly
    // empty 64-row tiles plus a routing launch.
    let gemv_path = d.t <= matmul::GEMV_MAX_M
        || rows <= GEMV_MAX_ROWS
        || !matmul::mma_ok(&ms, caps)
        || d.e > MAX_E;
    let tail = |p: &mut Params| -> Result<()> {
        // out view (contiguous [rows, n]), extras: E, stride
        p.u(0)?.u(d.n)?.u(0)?;
        p.u(d.e)?.u(stride)?;
        Ok(())
    };
    if gemv_path {
        let w = gemv::split(rows * d.n, d.k);
        let cb = d.n.div_ceil(gemv::COLS / w as usize);
        let mut p = Params::default();
        p.u(rows)?.u(1)?.u(d.n)?.u(d.k)?;
        p.u(0)?.u(d.k)?.u(0)?;
        p.u(0)?.u(0)?.u(d.k)?;
        p.u(so)?.u(cb)?.u(d.r)?;
        tail(&mut p)?;
        let md = dims(rows, 1, d.n, d.k);
        let (va, vb) = gemv::vec_variants(&ms, &md);
        let grid = if empty {
            [0, 0, 0]
        } else {
            grid1((rows * cb) as u64, caps)?
        };
        return Ok(Plan::new(Key::GemvId(ms, va, vb, w), p.0, grid, 256));
    }
    let scratch = 4 * (2 * (d.e + 1) + rows);
    // few rows per used expert (many experts): 32-row tiles waste less tensor-core work
    let small = rows < 48 * d.e.min(rows);
    let bm = if small { mma::SMALL_BM } else { mma::BM };
    let mut route = Params::default();
    route.u(rows)?.u(d.e)?.u(bm.trailing_zeros() as usize)?;
    let mut p = Params::default();
    p.u(1)?.u(rows)?.u(d.n)?.u(d.k)?;
    p.u(0)?.u(0)?.u(d.k)?;
    p.u(0)?.u(0)?.u(d.k)?;
    p.u(so)?.u(0)?.u(d.r)?;
    tail(&mut p)?;
    let md = dims(1, rows, d.n, d.k);
    let (va, vb) = mma::vec_variants(&ms, &md);
    let tiles_n = d.n.div_ceil(mma::BN);
    // every expert's last tile may be partial: at most rows / bm + min(E, rows) tiles
    let tiles = rows.div_ceil(bm) + d.e.min(rows);
    if tiles > caps.max_grid[1] as usize || tiles_n > caps.max_grid[0] as usize {
        return Err(Error::new("too_large", format!("matmul_id grid {tiles_n}x{tiles} exceeds device limits")));
    }
    let grid = if empty {
        [0, 0, 0]
    } else {
        [tiles_n as u32, tiles as u32, 1]
    };
    let mut gemm = Plan::new(Key::GemmMmaId(ms, va, vb, small), p.0, grid, 256);
    gemm.scratch = scratch;
    let mut first = Plan::new(Key::MoeRoute, route.0, if empty { [0, 0, 0] } else { [1, 1, 1] }, 1024);
    first.scratch = scratch;
    first.next = Some(Box::new(gemm));
    Ok(first)
}

fn dims(batch: usize, m: usize, n: usize, k: usize) -> MatMulDims {
    MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView {
            offset: 0,
            batch_stride: k,
            row_stride: k,
        },
        b: MatView {
            offset: 0,
            batch_stride: 0,
            row_stride: k,
        },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    }
}

pub(super) fn source_route() -> String {
    let (args, ptrs) = ptx::args(&["x", "w", "ids", "out", "scratch", "flag"]);
    let max_e = MAX_E.to_string();
    template::fill(ROUTE, &[("ARGS", &args), ("PTRS", &ptrs), ("MAX_E", &max_e)], &[])
}
