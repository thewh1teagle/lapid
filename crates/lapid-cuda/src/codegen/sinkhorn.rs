//! `Kernel::Sinkhorn` (template `kernels/sinkhorn.ptx`): one thread per matrix, the matrix
//! in registers, every step unrolled for the baked-in n.

use lapid_compiler::SinkhornDims;
use lapid_tensor::Result;

use super::{Key, Params, Plan, bad, grid1, ptx, template};
use crate::context::Caps;

const KERNEL: &str = include_str!("../kernels/sinkhorn.ptx");
/// Largest n (n * n registers per thread).
pub(crate) const MAX_N: usize = 8;
const BLOCK: u32 = 64;

pub(super) fn plan(d: &SinkhornDims, caps: &Caps) -> Result<Plan> {
    if d.n == 0 || d.n > MAX_N {
        return Err(bad(format!("sinkhorn: n = {} (1..={MAX_N})", d.n)));
    }
    let mut p = Params::default();
    p.u(d.count)?.u(d.iters)?.bits(d.eps);
    let grid = grid1(d.count.div_ceil(BLOCK as usize) as u64, caps)?;
    Ok(Plan::new(Key::Sinkhorn(d.n as u32), p.0, grid, BLOCK))
}

fn c(n: usize, i: usize, j: usize) -> String {
    format!("%c{}", i * n + j)
}

/// `c[i][j] *= 1 / (sum + eps)` over every row (`rows`) or column (within ~2 ulp of the
/// decomposition's division).
fn normalize(n: usize, rows: bool) -> String {
    let mut s = String::new();
    for a in 0..n {
        let cell = |b: usize| if rows { c(n, a, b) } else { c(n, b, a) };
        s.push_str(&format!("    mov.f32 %s, {};\n", cell(0)));
        for b in 1..n {
            s.push_str(&format!("    add.f32 %s, %s, {};\n", cell(b)));
        }
        s.push_str("    add.f32 %s, %s, %eps;\n    rcp.approx.f32 %s, %s;\n");
        for b in 0..n {
            s.push_str(&format!("    mul.f32 {0}, {0}, %s;\n", cell(b)));
        }
    }
    s
}

fn softmax(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!("    mov.f32 %mx, {};\n", c(n, i, 0)));
        for j in 1..n {
            s.push_str(&format!("    max.f32 %mx, %mx, {};\n", c(n, i, j)));
        }
        s.push_str("    mov.f32 %s, 0f00000000;\n");
        for j in 0..n {
            let x = c(n, i, j);
            s.push_str(&format!("    sub.f32 {x}, {x}, %mx;\n    mul.f32 {x}, {x}, 0f3FB8AA3B;\n    ex2.approx.f32 {x}, {x};\n    add.f32 %s, %s, {x};\n"));
        }
        s.push_str("    rcp.rn.f32 %s, %s;\n");
        for j in 0..n {
            s.push_str(&format!("    mul.f32 {0}, {0}, %s;\n    add.f32 {0}, {0}, %eps;\n", c(n, i, j)));
        }
    }
    s
}

pub(super) fn source(n: u32) -> String {
    let n = n as usize;
    let (args, ptrs) = ptx::args(&["x", "out"]);
    let load: String = (0..n * n).map(|k| format!("    ld.global.nc.f32 %c{k}, [%a+{}];\n", 4 * k)).collect();
    let store: String = (0..n * n).map(|k| format!("    st.global.f32 [%a+{}], %c{k};\n", 4 * k)).collect();
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("N", &n.to_string()),
            ("NN", &(n * n).to_string()),
            ("BYTES", &(4 * n * n).to_string()),
            ("LOAD", &load),
            ("SOFTMAX", &softmax(n)),
            ("ROWS", &normalize(n, true)),
            ("COLS", &normalize(n, false)),
            ("STORE", &store),
        ],
        &[],
    )
}
