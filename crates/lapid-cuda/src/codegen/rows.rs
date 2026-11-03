//! Row kernels: softmax and layer norm (one block per row), and gather.

use std::sync::Arc;

use lapid_compiler::{GatherDims, GatherSpec, LayerNormDims, LayerNormSpec, RowDims, SoftmaxSpec};
use lapid_tensor::Result;

use super::ptx::{self, Gen, Red, block_reduce};
use super::{Key, Params, Plan, bad, check_plain, grid_threads, grid1, scales_offset, template};
use crate::context::Caps;

const SOFTMAX: &str = include_str!("../kernels/softmax.ptx");
const LAYER_NORM: &str = include_str!("../kernels/layer_norm.ptx");
const GATHER: &str = include_str!("../kernels/gather.ptx");

pub(super) fn plan_softmax(spec: &Arc<SoftmaxSpec>, d: &RowDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.in_dtype, "softmax input")?;
    check_plain(spec.out_dtype, "softmax output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?;
    let grid = if d.cols == 0 {
        [0, 0, 0]
    } else {
        grid1(d.rows as u64, caps)?
    };
    Ok(Plan::new(Key::Softmax(spec.clone()), p.0, grid, 256))
}

/// Rows up to which a layer norm of long rows runs 1024-thread blocks (few blocks: each
/// must cover its row with more loads in flight).
const WIDE_ROWS: usize = 64;
/// Shortest row for the 1024-thread variant.
const WIDE_COLS: usize = 4096;

pub(super) fn plan_layer_norm(
    spec: &Arc<LayerNormSpec>,
    d: &LayerNormDims,
    caps: &Caps,
    rms: bool,
) -> Result<Plan> {
    check_plain(spec.in_dtype, "layer_norm input")?;
    check_plain(spec.param_dtype, "layer_norm params")?;
    check_plain(spec.out_dtype, "layer_norm output")?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.bits(d.eps);
    let grid = if d.cols == 0 {
        [0, 0, 0]
    } else {
        grid1(d.rows as u64, caps)?
    };
    let wide = d.rows <= WIDE_ROWS && d.cols >= WIDE_COLS;
    Ok(Plan::new(Key::LayerNorm(spec.clone(), rms, wide), p.0, grid, if wide { 1024 } else { 256 }))
}

pub(super) fn plan_gather(spec: &Arc<GatherSpec>, d: &GatherDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.out_dtype, "gather output")?;
    let qb = super::quant_block(spec.table_dtype);
    if spec.table_dtype.is_quant() && !d.cols.is_multiple_of(qb) {
        return Err(bad(format!("gather: quantized table needs cols % {qb} == 0")));
    }
    let so = if spec.table_dtype.is_quant() {
        scales_offset(spec.table_dtype, d.rows, d.cols)
    } else {
        0
    };
    let total = d
        .count
        .checked_mul(d.cols)
        .ok_or_else(|| bad("gather: size overflow"))?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.count)?.u(so)?.u(total)?;
    let grid = if d.rows == 0 {
        [0, 0, 0]
    } else {
        grid_threads(total as u64, caps)?
    };
    Ok(Plan::new(Key::Gather(spec.clone()), p.0, grid, 256))
}

fn snippet(f: impl FnOnce(&mut Gen)) -> String {
    let mut g = Gen::new();
    f(&mut g);
    g.finish()
}

pub(super) fn source_softmax(spec: &SoftmaxSpec) -> String {
    let (args, ptrs) = ptx::args(&["x", "out"]);
    let load = snippet(|g| ptx::load(g, spec.in_dtype, "%g_x", "%idx", "%x", true));
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%idx", "%e"));
    template::fill(
        SOFTMAX,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD_X", &load),
            ("REDUCE_MAX", &block_reduce(Red::Max, "%m")),
            ("REDUCE_SUM", &block_reduce(Red::Sum, "%sum")),
            ("STORE", &store),
        ],
        &[],
    )
}

/// Sum of `v` over a block of `warps` warps through the module-scope `red` array (every
/// thread gets the result; block-uniform control flow).
fn block_sum(v: &str, warps: u32) -> String {
    let mut g = Gen::new();
    ptx::warp_reduce(&mut g, Red::Sum, v);
    let (p, a) = (g.reg("pred"), g.reg("b32"));
    g.op(format!("setp.eq.u32 {p}, %lane, 0"));
    g.op(format!("mov.u32 {a}, red"));
    g.op(format!("mad.lo.u32 {a}, %warp, 4, {a}"));
    g.op(format!("@{p} st.shared.f32 [{a}], {v}"));
    g.op("bar.sync 0");
    g.op(format!("setp.lt.u32 {p}, %lane, {warps}"));
    g.op(format!("mov.f32 {v}, 0f00000000"));
    g.op(format!("mov.u32 {a}, red"));
    g.op(format!("mad.lo.u32 {a}, %lane, 4, {a}"));
    g.op(format!("@{p} ld.shared.f32 {v}, [{a}]"));
    ptx::warp_reduce(&mut g, Red::Sum, v);
    g.op("bar.sync 0");
    g.finish()
}

/// Loads of the 4 row elements `j + t * nt` (t < 4) of x, clamped to the row start beyond
/// `cols`: (value registers, in-range predicates, element indices).
fn load4(g: &mut Gen, spec: &LayerNormSpec, nt: u32) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (mut xs, mut ps, mut idxs) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..4 {
        let (jt, p, idx, x) = (g.reg("u32"), g.reg("pred"), g.reg("u32"), g.reg("f32"));
        g.op(format!("add.u32 {jt}, %j, {}", t * nt));
        g.op(format!("setp.lt.u32 {p}, {jt}, %cols"));
        g.op(format!("selp.b32 {jt}, {jt}, 0, {p}"));
        g.op(format!("add.u32 {idx}, %base, {jt}"));
        ptx::load(g, spec.in_dtype, "%g_x", &idx, &x, true);
        xs.push(x);
        ps.push(p);
        idxs.push(jt);
    }
    (xs, ps, idxs)
}

pub(super) fn source_layer_norm(spec: &LayerNormSpec, rms: bool, wide: bool) -> String {
    let nt: u32 = if wide { 1024 } else { 256 };
    let names: &[&str] = if rms { &["x", "w", "out"] } else { &["x", "w", "b", "out"] };
    let (args, ptrs) = ptx::args(names);
    let sum4 = snippet(|g| {
        let (xs, ps, _) = load4(g, spec, nt);
        for (x, p) in xs.iter().zip(&ps) {
            g.op(format!("selp.f32 {x}, {x}, 0f00000000, {p}"));
            g.op(format!("add.f32 %sum, %sum, {x}"));
        }
    });
    let sq4 = snippet(|g| {
        let (xs, ps, _) = load4(g, spec, nt);
        for (x, p) in xs.iter().zip(&ps) {
            g.op(format!("sub.f32 {x}, {x}, %mean"));
            g.op(format!("selp.f32 {x}, {x}, 0f00000000, {p}"));
            g.op(format!("fma.rn.f32 %sq, {x}, {x}, %sq"));
        }
    });
    let out4 = snippet(|g| {
        let (xs, ps, js) = load4(g, spec, nt);
        let mut ws = Vec::new();
        for j in &js {
            let (w, b) = (g.reg("f32"), g.reg("f32"));
            ptx::load(g, spec.param_dtype, "%g_w", j, &w, true);
            if !rms {
                ptx::load(g, spec.param_dtype, "%g_b", j, &b, true);
            }
            ws.push((w, b));
        }
        for t in 0..4 {
            let (x, (w, b)) = (&xs[t], &ws[t]);
            let (y, idx) = (g.reg("f32"), g.reg("u32"));
            g.op(format!("sub.f32 {x}, {x}, %mean"));
            g.op(format!("mul.f32 {x}, {x}, %rstd"));
            if rms {
                g.op(format!("mul.f32 {y}, {x}, {w}"));
            } else {
                g.op(format!("fma.rn.f32 {y}, {x}, {w}, {b}"));
            }
            g.op(format!("add.u32 {idx}, %base, {}", js[t]));
            ptx::store_if(g, spec.out_dtype, "%g_out", &idx, &y, Some(&ps[t]));
        }
    });
    template::fill(
        LAYER_NORM,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("NT", &nt.to_string()),
            ("STEP", &(4 * nt).to_string()),
            ("SUM4", &sum4),
            ("SQ4", &sq4),
            ("OUT4", &out4),
            ("REDUCE_SUM", &block_sum("%sum", nt / 32)),
            ("REDUCE_SQ", &block_sum("%sq", nt / 32)),
        ],
        if rms { &["RMS"] } else { &[] },
    )
}

pub(super) fn source_gather(spec: &GatherSpec) -> String {
    let (args, ptrs) = ptx::args(&["tab", "idx", "out", "flag"]);
    let load = snippet(|g| {
        if spec.table_dtype.is_quant() {
            ptx::load_quant(g, spec.table_dtype, "%g_tab", "%so", "%src", "%v");
        } else {
            ptx::load(g, spec.table_dtype, "%g_tab", "%src", "%v", true);
        }
    });
    let store = snippet(|g| ptx::store(g, spec.out_dtype, "%g_out", "%e", "%v"));
    template::fill(
        GATHER,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("LOAD", &load),
            ("STORE", &store),
        ],
        &[],
    )
}
