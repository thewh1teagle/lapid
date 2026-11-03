//! Index kernels: `GatherAxis` (take_along_axis, exact element copy) and `TopK` (k = 1:
//! packed-u64 max reduction; k > 1: radix select + ordered collection + rank sort, or for
//! rows of up to `TOPK_SMALL_COLS` a rank per column in one scan, e.g. a MoE router).
//! Out-of-range gather indices set the backend error flag (bound after the output, see
//! `Key::uses_flag`); `execute` turns it into `Error("index_out_of_range")`.

use std::sync::Arc;

use lapid_compiler::{GatherAxisDims, GatherAxisSpec, TopKDims, TopKSpec};
use lapid_tensor::{Error, Result};

use super::ptx::{self, Gen};
use super::{Key, Params, Plan, bad, check_plain, grid1, grid_threads, template};
use crate::context::Caps;

const GATHER_AXIS: &str = include_str!("../kernels/gather_axis.ptx");
const TOPK1: &str = include_str!("../kernels/topk1.ptx");
const TOPK: &str = include_str!("../kernels/topk.ptx");
const TOPK_MERGE: &str = include_str!("../kernels/topk_merge.ptx");
const TOPK_SMALL: &str = include_str!("../kernels/topk_small.ptx");

/// Longest row of the small top-k kernel (`Key::TopKSmall`: keys in shared memory, every
/// column ranked by 4 threads scanning a quarter of the row each).
const TOPK_SMALL_COLS: usize = 1024;

/// Columns per block of the split argmax (rows longer than this use several blocks).
const ARGMAX_CHUNK: usize = 8192;
/// Most blocks per row of the split argmax.
const ARGMAX_MAX_CHUNKS: usize = 64;

/// Largest k of the general top-k kernel (candidates live in shared memory).
pub(crate) const TOPK_MAX: usize = 2048;

pub(super) fn plan_gather_axis(spec: &Arc<GatherAxisSpec>, d: &GatherAxisDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.dtype, "gather_axis")?;
    let total = d
        .outer
        .checked_mul(d.n)
        .and_then(|v| v.checked_mul(d.inner))
        .ok_or_else(|| bad("gather_axis: size overflow"))?;
    let mut p = Params::default();
    p.u(total)?.u(d.n)?.u(d.inner)?.u(d.axis_len)?;
    p.view(&d.x, 3)?;
    p.view(&d.idx, 3)?;
    let grid = grid_threads(total as u64, caps)?;
    Ok(Plan::new(Key::GatherAxis(spec.clone()), p.0, grid, 256))
}

pub(super) fn plan_topk(spec: &Arc<TopKSpec>, d: &TopKDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "topk input")?;
    if d.rows == 0 {
        return Ok(Plan::new(Key::TopK(spec.clone(), true), vec![0, 0, 0], [0, 0, 0], 256));
    }
    if d.k == 0 || d.k > d.cols {
        return Err(bad(format!("topk: k = {} with {} columns", d.k, d.cols)));
    }
    if d.k > TOPK_MAX {
        return Err(Error::new(
            "unsupported",
            format!("topk: k = {} > {TOPK_MAX} is not supported on this backend", d.k),
        ));
    }
    let numel = d.rows.checked_mul(d.cols).ok_or_else(|| bad("topk: size overflow"))?;
    let grid = grid1(d.rows as u64, caps)?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.k)?.u(numel)?;
    if d.k > 1 && d.cols <= TOPK_SMALL_COLS {
        return Ok(Plan::new(Key::TopKSmall(spec.clone()), p.0, [grid[0], d.cols.div_ceil(64) as u32, 1], 256));
    }
    if d.k > 1 {
        return Ok(Plan::new(Key::TopK(spec.clone(), false), p.0, grid, 256));
    }
    // argmax: long rows split over grid.y blocks, partials merged by a second stage
    let nch = d.cols.div_ceil(ARGMAX_CHUNK).clamp(1, ARGMAX_MAX_CHUNKS);
    let per = d.cols.div_ceil(nch);
    p.u(per)?.u(nch)?;
    let mut plan = Plan::new(Key::TopK(spec.clone(), true), p.0.clone(), [grid[0], nch as u32, 1], 256);
    if nch > 1 {
        plan.scratch = d.rows * nch * 8;
        plan.next = Some(Box::new(Plan::new(Key::TopKMerge, p.0, grid, 32)));
    }
    Ok(plan)
}

pub(super) fn source_gather_axis(spec: &GatherAxisSpec) -> String {
    let (args, ptrs) = ptx::args(&["x", "idx", "out", "flag"]);
    let mut g = Gen::new();
    ptx::copy_raw(&mut g, spec.dtype, "%g_x", "%src", "%g_out", "%e", true);
    let mut z = Gen::new();
    let (size, t) = if spec.dtype.size() == 2 { (2, "b16") } else { (4, "b32") };
    let a = z.addr("%g_out", "%e", size);
    let r = z.reg(t);
    z.op(format!("mov.{t} {r}, 0"));
    z.op(format!("st.global.{t} [{a}], {r}"));
    template::fill(
        GATHER_AXIS,
        &[("ARGS", &args), ("PTRS", &ptrs), ("COPY", &g.finish()), ("ZERO", &z.finish())],
        &[],
    )
}

/// Order-preserving u32 key of the f32 register `x` (-0 folded into +0, so equal values
/// tie): non-negative → bits | 2^31, negative → !bits.
fn key(g: &mut Gen, x: &str, key: &str) {
    let b = g.reg("b32");
    let z = g.reg("pred");
    let m = g.reg("b32");
    g.op(format!("mov.b32 {b}, {x}"));
    g.op(format!("setp.eq.b32 {z}, {b}, 0x80000000"));
    g.op(format!("selp.b32 {b}, 0, {b}, {z}"));
    g.op(format!("shr.s32 {m}, {b}, 31"));
    g.op(format!("or.b32 {m}, {m}, 0x80000000"));
    g.op(format!("xor.b32 {key}, {b}, {m}"));
}

pub(super) fn source_topk_merge() -> String {
    let (args, ptrs) = ptx::args(&["x", "out", "scratch"]);
    let mut g = Gen::new();
    let (lo, hi, lo2, hi2, t) = (g.reg("b32"), g.reg("b32"), g.reg("b32"), g.reg("b32"), g.reg("b64"));
    for s in [16, 8, 4, 2, 1] {
        g.op(format!("mov.b64 {{{lo}, {hi}}}, %best"));
        g.op(format!("shfl.sync.bfly.b32 {lo2}, {lo}, {s}, 31, -1"));
        g.op(format!("shfl.sync.bfly.b32 {hi2}, {hi}, {s}, 31, -1"));
        g.op(format!("mov.b64 {t}, {{{lo2}, {hi2}}}"));
        g.op(format!("max.u64 %best, %best, {t}"));
    }
    template::fill(TOPK_MERGE, &[("ARGS", &args), ("PTRS", &ptrs), ("WARP_MAX", &g.finish())], &[])
}

/// One step of the small top-k rank scan: keys `%d[0..4]` (columns `%c2 + i`).
fn scan4() -> String {
    let mut g = Gen::new();
    let k: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
    let (c, gt, eq, lt) = (g.reg("b32"), g.reg("pred"), g.reg("pred"), g.reg("pred"));
    g.op(format!("ld.shared.v4.u32 {{{}, {}, {}, {}}}, [%d]", k[0], k[1], k[2], k[3]));
    for (i, ki) in k.iter().enumerate() {
        g.op(format!("add.u32 {c}, %c2, {i}"));
        g.op(format!("setp.gt.u32 {gt}, {ki}, %key"));
        g.op(format!("setp.eq.u32 {eq}, {ki}, %key"));
        g.op(format!("setp.lt.u32 {lt}, {c}, %j"));
        g.op(format!("and.pred {eq}, {eq}, {lt}"));
        g.op(format!("or.pred {gt}, {gt}, {eq}"));
        g.op(format!("@{gt} add.u32 %rank, %rank, 1"));
    }
    g.finish()
}

pub(super) fn source_topk_small(spec: &TopKSpec) -> String {
    let (args, ptrs) = ptx::args(&["x", "out"]);
    let mut l = Gen::new();
    ptx::load(&mut l, spec.in_dtype, "%g_x", "%e", "%x", true);
    let mut kg = Gen::new();
    key(&mut kg, "%x", "%key");
    template::fill(
        TOPK_SMALL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("MAXC", &TOPK_SMALL_COLS.to_string()),
            ("LOAD", &l.finish()),
            ("KEY", &kg.finish()),
            ("SCAN4", &scan4()),
        ],
        &[],
    )
}

pub(super) fn source_topk(spec: &TopKSpec, k1: bool) -> String {
    let names: &[&str] = if k1 { &["x", "out", "scratch"] } else { &["x", "out"] };
    let (args, ptrs) = ptx::args(names);
    if k1 {
        // 4 independent loads per iteration: columns j, j+256, j+512, j+768
        let mut g = Gen::new();
        for u in 0..4 {
            let jj = g.reg("u32");
            let v = g.reg("pred");
            let e = g.reg("u32");
            let x = g.reg("f32");
            let kk = g.reg("b32");
            let nj = g.reg("b32");
            let pk = g.reg("b64");
            g.op(format!("add.u32 {jj}, %j, {}", 256 * u));
            g.op(format!("setp.lt.u32 {v}, {jj}, %end"));
            g.op(format!("add.u32 {e}, %base, {jj}"));
            g.op(format!("selp.b32 {e}, {e}, %base, {v}"));
            ptx::load(&mut g, spec.in_dtype, "%g_x", &e, &x, true);
            key(&mut g, &x, &kk);
            g.op(format!("not.b32 {nj}, {jj}"));
            g.op(format!("mov.b64 {pk}, {{{nj}, {kk}}}"));
            g.op(format!("selp.b64 {pk}, {pk}, 0, {v}"));
            g.op(format!("max.u64 %best, %best, {pk}"));
        }
        return template::fill(TOPK1, &[("ARGS", &args), ("PTRS", &ptrs), ("BODY", &g.finish())], &[]);
    }
    let mut l = Gen::new();
    ptx::load(&mut l, spec.in_dtype, "%g_x", "%e", "%x", true);
    let mut kg = Gen::new();
    key(&mut kg, "%x", "%key");
    // block prefix of the per-warp ballot counts (wt[0..8] = keys > T, wt[8..16] = == T)
    let mut pg = Gen::new();
    let w: Vec<String> = (0..16).map(|_| pg.reg("b32")).collect();
    for q in 0..4 {
        pg.op(format!(
            "ld.shared.v4.u32 {{{}, {}, {}, {}}}, [wt+{}]",
            w[4 * q],
            w[4 * q + 1],
            w[4 * q + 2],
            w[4 * q + 3],
            16 * q
        ));
    }
    pg.op("mov.u32 %totg, 0");
    pg.op("mov.u32 %tote, 0");
    let lower = pg.reg("pred");
    let t = pg.reg("b32");
    for i in 0..8 {
        pg.op(format!("setp.lt.u32 {lower}, {i}, %warp"));
        pg.op(format!("selp.b32 {t}, {}, 0, {lower}", w[i]));
        pg.op(format!("add.u32 %preg, %preg, {t}"));
        pg.op(format!("selp.b32 {t}, {}, 0, {lower}", w[8 + i]));
        pg.op(format!("add.u32 %pree, %pree, {t}"));
        pg.op(format!("add.u32 %totg, %totg, {}", w[i]));
        pg.op(format!("add.u32 %tote, %tote, {}", w[8 + i]));
    }
    template::fill(
        TOPK,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("KMAX", &TOPK_MAX.to_string()),
            ("LOAD", &l.finish()),
            ("KEY", &kg.finish()),
            ("PREFIX", &pg.finish()),
        ],
        &[],
    )
}
