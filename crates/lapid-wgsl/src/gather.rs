//! Gather rows of a (possibly quantized) table by i32 indices.

use std::sync::Arc;

use lapid_compiler::{GatherAxisDims, GatherAxisSpec, GatherDims, GatherSpec};
use lapid_tensor::{DType, Error, Result};

use super::{common, template};
use super::{Key, Params, Plan, bad, check_plain, check_quant_supported, grid_threads};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/gather.wgsl");

/// Scales-plane offset of a quantized `[rows, cols]` matrix, in u32 words.
pub fn scales_words(d: DType, rows: usize, cols: usize) -> usize {
    match d.quant_format() {
        Some(f) if !f.is_aos() => lapid_quant::layout(f, rows, cols).scales_offset / 4,
        _ => 0,
    }
}

pub(super) fn plan(spec: &Arc<GatherSpec>, d: &GatherDims, caps: &Caps) -> Result<Plan> {
    plan_part(spec, d, 0, d.rows, caps)
}

/// Gather from rows `[base, base + d.rows)` of a table with `all_rows` rows (a split launch,
/// see `split`): indices outside the part are skipped, outside the table flagged.
pub(super) fn plan_part(spec: &Arc<GatherSpec>, d: &GatherDims, base: usize, all_rows: usize, caps: &Caps) -> Result<Plan> {
    check_plain(spec.out_dtype, "gather output")?;
    check_quant_supported(spec.table_dtype, "gather table")?;
    if let Some(f) = spec.table_dtype.quant_format()
        && !d.cols.is_multiple_of(f.block_elems())
    {
        return Err(bad(format!("gather: {} table needs cols % {} == 0", spec.table_dtype.name(), f.block_elems())));
    }
    let so = if spec.table_dtype.is_quant() { scales_words(spec.table_dtype, d.rows, d.cols) } else { 0 };
    let total = d.count.checked_mul(d.cols).ok_or_else(|| bad("gather: size overflow"))?;
    let mut p = Params::default();
    p.u(d.rows)?.u(d.cols)?.u(d.count)?.u(so)?.u(total)?.u(base)?.u(all_rows)?;
    if all_rows == 0 && d.count > 0 {
        return Err(Error::new("index_out_of_range", "gather: index into an empty table"));
    }
    let groups = grid_threads(total as u64, caps);
    Ok(Plan { scratch: 0, next: None, key: Key::Gather(spec.clone()), params: p.0, groups })
}

pub(super) fn source(spec: &GatherSpec, caps: &Caps) -> String {
    let mut bindings = common::input(1, "tab", spec.table_dtype, caps);
    bindings.push_str(&common::output(3, spec.out_dtype, caps));
    let load = if spec.table_dtype.is_quant() { "tab_deq(so, src)" } else { "ld_tab(src)" };
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("LOAD", load)], &[], 256, false)
}

const GATHER_AXIS: &str = include_str!("kernels/gather_axis.wgsl");

pub(super) fn plan_axis(spec: &Arc<GatherAxisSpec>, d: &GatherAxisDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.dtype, "gather_axis")?;
    let total = d.outer.checked_mul(d.n).and_then(|x| x.checked_mul(d.inner)).ok_or_else(|| bad("gather_axis: size overflow"))?;
    if total > 0 && d.axis_len == 0 {
        return Err(Error::new("index_out_of_range", "gather_axis: index into an empty axis"));
    }
    let mut p = Params::default();
    p.u(d.outer)?.u(d.n)?.u(d.inner)?.u(d.axis_len)?.u(total)?;
    p.view(&d.x, 3)?.view(&d.idx, 3)?;
    let items = if spec.dtype.size() == 2 { total.div_ceil(2) } else { total };
    Ok(Plan { scratch: 0, next: None, key: Key::GatherAxis(spec.clone()), params: p.0, groups: grid_threads(items as u64, caps) })
}

pub(super) fn source_axis(spec: &GatherAxisSpec, caps: &Caps) -> String {
    let flags: &[&str] = if spec.dtype.size() == 2 { &["HALF"] } else { &[] };
    template::module(caps, GATHER_AXIS, &[], flags, 256, false)
}
