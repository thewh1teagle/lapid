//! Activation pre-pass of the i-quant GEMV (template `kernels/q8_rows.ptx`): rows of A
//! quantized to int8 with an f32 scale per 32 elements into the launch scratch, read by the
//! `dp4a` lane dot products (`iquant::gemv_dot`).

use lapid_tensor::{DType, Result};

use super::{Key, Params, Plan, grid1, ptx, template};
use crate::context::Caps;

const KERNEL: &str = include_str!("../kernels/q8_rows.ptx");

/// Scratch layout for `rows` quantized rows of length `k`: (scales offset, total bytes).
pub(super) fn layout(rows: usize, k: usize) -> (usize, usize) {
    let so = lapid_quant::align_up(rows * k, 16);
    (so, so + rows * k / 32 * 4)
}

/// The pre-pass stage for `rows` rows of A (row r at `a_off + (r / m) a_bs + (r % m) a_rs`)
/// of a launch with `inputs` bound inputs (A first); `next` runs after it.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(a_dtype: DType, inputs: usize, rows: usize, m: usize, k: usize, a: [usize; 3], next: Plan, caps: &Caps) -> Result<Plan> {
    let (so, total) = layout(rows, k);
    let mut p = Params::default();
    p.u(rows)?.u(m.max(1))?.u(k)?.u(a[0])?.u(a[1])?.u(a[2])?.u(so)?;
    let grid = if next.is_empty() { [0, 0, 0] } else { grid1((rows * k / 32).div_ceil(8) as u64, caps)? };
    let mut first = Plan::new(Key::Q8Rows(a_dtype, inputs as u32), p.0, grid, 256);
    let mut next = next;
    next.scratch = next.scratch.max(total);
    first.scratch = next.scratch;
    first.next = Some(Box::new(next));
    Ok(first)
}

pub(super) fn source(a_dtype: DType, inputs: u32) -> String {
    let mut names: Vec<String> = (0..inputs).map(|i| format!("in{i}")).collect();
    names.push("out".into());
    names.push("scratch".into());
    let (args, ptrs) = ptx::args(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let mut g = ptx::Gen::new();
    ptx::load(&mut g, a_dtype, "%g_in0", "%src", "%x", true);
    template::fill(KERNEL, &[("ARGS", &args), ("PTRS", &ptrs), ("LOAD_X", &g.finish())], &[])
}
