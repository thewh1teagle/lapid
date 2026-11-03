//! Matrix-vector kernel for small m with transposed B (template `kernels/gemv.ptx`): warp
//! per column. Rust fills the 8-element dot product for the B dtype (int8 / int4 words,
//! 16-byte f16 / f32 vectors when aligned, guarded scalar loads otherwise).

use lapid_compiler::{MatMulDims, MatMulSpec};
use lapid_tensor::DType;

use super::kquant;
use super::matmul::{args, epilogue};
use super::ptx::{self, Gen};
use super::template;

const KERNEL: &str = include_str!("../kernels/gemv.ptx");

/// Warps per block; without a k split, output columns per block (one warp each).
pub(crate) const COLS: usize = 8;
/// Target warps per launch: fewer columns than this split k over 2, 4 or 8 warps per column.
const TARGET_WARPS: usize = 8192;

/// Warps per output column for `cols` output columns (all rows / batches) of length `k`:
/// split while the launch has fewer than `TARGET_WARPS` warps and every warp keeps at
/// least one full 512-element iteration.
pub(crate) fn split(cols: usize, k: usize) -> u32 {
    let mut w = 1;
    while w < COLS && cols * w < TARGET_WARPS && k >= 1024 * w {
        w *= 2;
    }
    w as u32
}

/// Alignment variants: (A is f32 and readable as float4, B (f16 / f32) readable as
/// 16-byte vectors). Both need k % 8 == 0 so a lane's 8-element chunk never crosses k.
pub(crate) fn vec_variants(spec: &MatMulSpec, d: &MatMulDims) -> (bool, bool) {
    let a = &d.a;
    let vec_a = spec.a_dtype == DType::F32
        && d.k.is_multiple_of(8)
        && a.offset.is_multiple_of(4)
        && (a.row_stride.is_multiple_of(4) || d.m <= 1)
        && (a.batch_stride.is_multiple_of(4) || d.batch <= 1);
    let b = &d.b;
    let align = match spec.b_dtype {
        DType::F16 => 8,
        DType::F32 => 4,
        _ => 0,
    };
    let vec_b = align > 0
        && d.k.is_multiple_of(8)
        && b.offset.is_multiple_of(align)
        && b.row_stride.is_multiple_of(align)
        && (b.batch_stride.is_multiple_of(align) || d.batch <= 1);
    (vec_a, vec_b)
}

/// Load A[arow + k + t] for t in 0..n into `dst[t]` (`bounded`: zero beyond K).
fn load_a(
    g: &mut Gen,
    spec: &MatMulSpec,
    vec_a: bool,
    k: &str,
    n: usize,
    bounded: bool,
) -> Vec<String> {
    let v: Vec<String> = (0..n).map(|_| g.reg("f32")).collect();
    if vec_a {
        let idx = g.reg("u32");
        g.op(format!("add.u32 {idx}, %arow, {k}"));
        let a = g.addr("%g_a", &idx, 4);
        for c in 0..n / 4 {
            g.op(format!(
                "ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{a}+{}]",
                v[4 * c],
                v[4 * c + 1],
                v[4 * c + 2],
                v[4 * c + 3],
                16 * c
            ));
        }
        return v;
    }
    for (t, dst) in v.iter().enumerate() {
        let idx = g.reg("u32");
        g.op(format!("add.u32 {idx}, %arow, {k}"));
        g.op(format!("add.u32 {idx}, {idx}, {t}"));
        if bounded {
            let p = g.reg("pred");
            let kt = g.reg("u32");
            g.op(format!("add.u32 {kt}, {k}, {t}"));
            g.op(format!("setp.lt.u32 {p}, {kt}, %K"));
            g.op(format!("selp.b32 {idx}, {idx}, %arow, {p}"));
            ptx::load(g, spec.a_dtype, "%g_a", &idx, dst, true);
            g.op(format!("selp.f32 {dst}, {dst}, 0f00000000, {p}"));
        } else {
            ptx::load(g, spec.a_dtype, "%g_a", &idx, dst, true);
        }
    }
    v
}

/// `out = sum_t A[k + t] * B[gj, k + t]` over the lane's 8 elements at k-offset `k`.
fn dot8(spec: &MatMulSpec, vec_a: bool, vec_b: bool, k: &str, out: &str) -> String {
    let mut g = Gen::new();
    match spec.b_dtype {
        DType::Q4K | DType::Q5K | DType::Q6K => kquant::gemv_dot(&mut g, spec, vec_a, k, out),
        DType::Mxfp4 => super::mxfp4::gemv_dot(&mut g, spec, vec_a, k, out),
        DType::Int8 | DType::Int4 => {
            let int8 = spec.b_dtype == DType::Int8;
            let e = g.reg("u32");
            g.op(format!("mad.lo.u32 {e}, %gj, %K, {k}"));
            // scale of the 32-group
            let gi = g.reg("u32");
            g.op(format!("shr.u32 {gi}, {e}, 5"));
            g.op(format!("mad.lo.u32 {gi}, {gi}, 2, %so"));
            let sa = g.addr("%g_b", &gi, 1);
            let h = g.reg("b16");
            let sc = g.reg("f32");
            g.op(format!("ld.global.nc.b16 {h}, [{sa}]"));
            let (w, a) = if int8 {
                let w = [g.reg("b32"), g.reg("b32")];
                let ba = g.addr("%g_b", &e, 1);
                g.op(format!(
                    "ld.global.nc.v2.u32 {{{}, {}}}, [{ba}]",
                    w[0], w[1]
                ));
                let a = load_a(&mut g, spec, vec_a, k, 8, false);
                (w.to_vec(), a)
            } else {
                // word e / 8 holds group bytes 4*wo.. (wo = (k / 8) % 4): low nibbles are
                // elements kg + 4wo + 0..4, high nibbles kg + 16 + 4wo + 0..4
                let w = g.reg("b32");
                let wb = g.reg("u32");
                g.op(format!("shr.u32 {wb}, {e}, 3"));
                g.op(format!("shl.b32 {wb}, {wb}, 2"));
                let ba = g.addr("%g_b", &wb, 1);
                g.op(format!("ld.global.nc.u32 {w}, [{ba}]"));
                let (klo, khi) = (g.reg("u32"), g.reg("u32"));
                g.op(format!("and.b32 {klo}, {k}, -32"));
                g.op(format!("bfe.u32 {khi}, {k}, 3, 2"));
                g.op(format!("mad.lo.u32 {klo}, {khi}, 4, {klo}"));
                g.op(format!("add.u32 {khi}, {klo}, 16"));
                let mut a = load_a(&mut g, spec, vec_a, &klo, 4, false);
                a.extend(load_a(&mut g, spec, vec_a, &khi, 4, false));
                (vec![w], a)
            };
            g.op(format!("cvt.f32.f16 {sc}, {h}"));
            let q = g.reg("s32");
            let f = g.reg("f32");
            g.op(format!("mov.f32 {out}, 0f00000000"));
            for t in 0..8 {
                if int8 {
                    g.op(format!("bfe.s32 {q}, {}, {}, 8", w[t / 4], 8 * (t % 4)));
                    g.op(format!("cvt.rn.f32.s32 {f}, {q}"));
                } else {
                    // t < 4: low nibble of byte t; t >= 4: high nibble of byte t - 4
                    let bit = if t < 4 { 8 * t } else { 8 * (t - 4) + 4 };
                    g.op(format!("bfe.u32 {q}, {}, {bit}, 4", w[0]));
                    g.op(format!("cvt.rn.f32.s32 {f}, {q}"));
                    g.op(format!("sub.f32 {f}, {f}, 0f41000000"));
                }
                g.op(format!("fma.rn.f32 {out}, {f}, {}, {out}", a[t]));
            }
            g.op(format!("mul.f32 {out}, {out}, {sc}"));
        }
        DType::F16 if vec_b => {
            let idx = g.reg("u32");
            g.op(format!("mad.lo.u32 {idx}, %gj, %brs, %bbase"));
            g.op(format!("add.u32 {idx}, {idx}, {k}"));
            let ba = g.addr("%g_b", &idx, 2);
            let w: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
            g.op(format!(
                "ld.global.nc.v4.b32 {{{}, {}, {}, {}}}, [{ba}]",
                w[0], w[1], w[2], w[3]
            ));
            let a = load_a(&mut g, spec, vec_a, k, 8, false);
            let (h0, h1) = (g.reg("b16"), g.reg("b16"));
            let (f0, f1) = (g.reg("f32"), g.reg("f32"));
            g.op(format!("mov.f32 {out}, 0f00000000"));
            for (c, wc) in w.iter().enumerate() {
                g.op(format!("mov.b32 {{{h0}, {h1}}}, {wc}"));
                g.op(format!("cvt.f32.f16 {f0}, {h0}"));
                g.op(format!("cvt.f32.f16 {f1}, {h1}"));
                g.op(format!("fma.rn.f32 {out}, {f0}, {}, {out}", a[2 * c]));
                g.op(format!("fma.rn.f32 {out}, {f1}, {}, {out}", a[2 * c + 1]));
            }
        }
        DType::F32 if vec_b => {
            let idx = g.reg("u32");
            g.op(format!("mad.lo.u32 {idx}, %gj, %brs, %bbase"));
            g.op(format!("add.u32 {idx}, {idx}, {k}"));
            let ba = g.addr("%g_b", &idx, 4);
            let b: Vec<String> = (0..8).map(|_| g.reg("f32")).collect();
            g.op(format!(
                "ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{ba}]",
                b[0], b[1], b[2], b[3]
            ));
            g.op(format!(
                "ld.global.nc.v4.f32 {{{}, {}, {}, {}}}, [{ba}+16]",
                b[4], b[5], b[6], b[7]
            ));
            let a = load_a(&mut g, spec, vec_a, k, 8, false);
            g.op(format!("mov.f32 {out}, 0f00000000"));
            for t in 0..8 {
                g.op(format!("fma.rn.f32 {out}, {}, {}, {out}", b[t], a[t]));
            }
        }
        d => {
            // guarded scalar path (any plain dtype, any alignment, k not a multiple of 8)
            let a = load_a(&mut g, spec, false, k, 8, true);
            let brow = g.reg("u32");
            g.op(format!("mad.lo.u32 {brow}, %gj, %brs, %bbase"));
            let b = g.reg("f32");
            let p = g.reg("pred");
            let (idx, kt) = (g.reg("u32"), g.reg("u32"));
            g.op(format!("mov.f32 {out}, 0f00000000"));
            for (t, at) in a.iter().enumerate() {
                g.op(format!("add.u32 {kt}, {k}, {t}"));
                g.op(format!("setp.lt.u32 {p}, {kt}, %K"));
                g.op(format!("add.u32 {idx}, {brow}, {kt}"));
                g.op(format!("selp.b32 {idx}, {idx}, {brow}, {p}"));
                ptx::load(&mut g, d, "%g_b", &idx, &b, true);
                g.op(format!("selp.f32 {b}, {b}, 0f00000000, {p}"));
                g.op(format!("fma.rn.f32 {out}, {b}, {at}, {out}"));
            }
        }
    }
    g.finish()
}

/// `id`: MatMulId variant (inputs a, b, ids; binds the error flag; see `moe`).
pub(super) fn source(spec: &MatMulSpec, vec_a: bool, vec_b: bool, id: bool, w: u32) -> String {
    let (args, ptrs) = if id {
        ptx::args(&["a", "b", "ids", "out", "flag"])
    } else {
        args(spec)
    };
    let da = dot8(spec, vec_a, vec_b, "%ka", "%sa");
    let db = dot8(spec, vec_a, vec_b, "%kb", "%sb");
    let mut flags = Vec::new();
    if id {
        flags.push("ID");
    }
    if w > 1 {
        flags.push("SPLIT");
    }
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("DOT_A", &da),
            ("DOT_B", &db),
            ("EPILOGUE", &epilogue(spec)),
            ("W", &w.to_string()),
            ("WLOG", &w.trailing_zeros().to_string()),
            ("WMASK", &(w - 1).to_string()),
            ("CPB", &(COLS as u32 / w).to_string()),
            ("KSTEP", &(512 * w).to_string()),
            ("SPLIT_SUM", &split_sum(w)),
        ],
        &flags,
    )
}

/// Sum of the W partials of a column (lane 0 of its first warp, `%ra` = its shared slot).
fn split_sum(w: u32) -> String {
    let mut s = String::new();
    for i in 1..w {
        s.push_str(&format!("    ld.shared.f32 %sh, [%ra+{}];\n    add.f32 %part, %part, %sh;\n", 4 * i));
    }
    s
}
