//! Tensor-core GEMM (template `kernels/gemm_mma.ptx`). Rust fills the per-dtype tile
//! loaders (global → registers for the next k-tile, registers → f16 shared memory after
//! the current tile's math), the unrolled mma block and the accumulator staging.

use lapid_compiler::{MatMulDims, MatMulSpec};
use lapid_tensor::DType;

use super::kquant;
use super::matmul::{args, epilogue};
use super::ptx::{self, Gen};
use super::template;

const KERNEL: &str = include_str!("../kernels/gemm_mma.ptx");

/// Output tile per block.
pub(crate) const BM: usize = 64;
pub(crate) const BN: usize = 128;
/// Tile rows of the small MatMulId variant.
pub(crate) const SMALL_BM: usize = 32;
/// Shared row stride in halves (32 + 8 padding).
const LDS: usize = 40;

/// (A f32 readable as float4, transposed f16 B readable as 16-byte vectors).
pub(crate) fn vec_variants(spec: &MatMulSpec, d: &MatMulDims) -> (bool, bool) {
    let a = &d.a;
    let vec_a = spec.a_dtype == DType::F32
        && d.k.is_multiple_of(4)
        && a.offset.is_multiple_of(4)
        && (a.row_stride.is_multiple_of(4) || d.m <= 1)
        && (a.batch_stride.is_multiple_of(4) || d.batch <= 1);
    let b = &d.b;
    let vec_b = spec.b_dtype == DType::F16
        && spec.b_transposed
        && d.k.is_multiple_of(8)
        && b.offset.is_multiple_of(8)
        && b.row_stride.is_multiple_of(8)
        && (b.batch_stride.is_multiple_of(8) || d.batch <= 1);
    (vec_a, vec_b)
}

/// Row start (element index of A) of the thread's A-load rows `u` in the MatMulId variant:
/// tile row r -> list entry `lbase + row0 + r` -> source row id -> token = id / r.
/// `shift`: log2 of the threads per tile row of the A loader.
fn id_rows(n: usize, shift: u32) -> String {
    let mut g = Gen::new();
    let (e, gi, li, rid) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    let v = g.reg("pred");
    for u in 0..n {
        g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
        g.op(format!("shr.u32 {e}, {e}, {shift}"));
        g.op(format!("add.u32 {gi}, %row0, {e}"));
        g.op(format!("setp.lt.u32 {v}, {gi}, %M"));
        g.op(format!("selp.b32 {gi}, {gi}, 0, {v}"));
        g.op(format!("add.u32 {li}, %lbase, {gi}"));
        let a = g.addr("%g_list", &li, 4);
        g.op(format!("ld.global.u32 {rid}, [{a}]"));
        g.op(format!("div.u32 {rid}, {rid}, %rr"));
        g.op(format!("mad.lo.u32 %arow{u}, {rid}, %ars, %abase"));
    }
    g.finish()
}

/// (REGS, A_LOAD, A_STORE, A_ROWS); `rows`: tile rows (BM, or 32 for the small variant).
fn a_parts(spec: &MatMulSpec, vec_a: bool, id: bool, rows: usize) -> (String, String, String, String) {
    let (regs, load, store) = a_parts_inner(spec, vec_a, id, rows);
    if !id {
        return (regs, load, store, String::new());
    }
    let (n, shift) = if vec_a { (rows / 32, 3) } else { (rows / 8, 5) };
    (format!("{regs}    .reg .b32 %arow<{n}>;\n"), load, store, id_rows(n, shift))
}

fn a_row_index(g: &mut Gen, id: bool, u: usize, gi: &str, idx: &str) {
    if id {
        g.op(format!("mov.u32 {idx}, %arow{u}"));
    } else {
        g.op(format!("mad.lo.u32 {idx}, {gi}, %ars, %abase"));
    }
}

fn a_parts_inner(spec: &MatMulSpec, vec_a: bool, id: bool, rows: usize) -> (String, String, String) {
    let mut load = String::new();
    let mut store = String::new();
    if vec_a {
        let regs = "    .reg .f32 %ax<8>;\n    .reg .pred %av<2>;\n".to_string();
        for u in 0..rows / 32 {
            // e = tid + 256u: row e / 8, k chunk (e % 8) * 4
            let mut g = Gen::new();
            let (e, r, gk, gi, idx) = (
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
            );
            g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
            g.op(format!("shr.u32 {r}, {e}, 3"));
            g.op(format!("and.b32 {gk}, {e}, 7"));
            g.op(format!("shl.b32 {gk}, {gk}, 2"));
            g.op(format!("add.u32 {gk}, {gk}, %k0"));
            g.op(format!("add.u32 {gi}, %row0, {r}"));
            g.op(format!("setp.lt.u32 %av{u}, {gi}, %M"));
            g.op(format!("setp.lt.and.u32 %av{u}, {gk}, %K, %av{u}"));
            a_row_index(&mut g, id, u, &gi, &idx);
            g.op(format!("add.u32 {idx}, {idx}, {gk}"));
            g.op(format!("selp.b32 {idx}, {idx}, 0, %av{u}"));
            let a = g.addr("%g_a", &idx, 4);
            g.op(format!(
                "ld.global.nc.v4.f32 {{%ax{}, %ax{}, %ax{}, %ax{}}}, [{a}]",
                4 * u,
                4 * u + 1,
                4 * u + 2,
                4 * u + 3
            ));
            load.push_str(&g.finish());

            let mut g = Gen::new();
            let (e, r, c, h0, h1) = (
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
                g.reg("b32"),
                g.reg("b32"),
            );
            let x: Vec<String> = (0..4).map(|_| g.reg("f32")).collect();
            for (t, xt) in x.iter().enumerate() {
                g.op(format!(
                    "selp.f32 {xt}, %ax{}, 0f00000000, %av{u}",
                    4 * u + t
                ));
            }
            g.op(format!("cvt.rn.f16x2.f32 {h0}, {}, {}", x[1], x[0]));
            g.op(format!("cvt.rn.f16x2.f32 {h1}, {}, {}", x[3], x[2]));
            g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
            g.op(format!("shr.u32 {r}, {e}, 3"));
            g.op(format!("and.b32 {c}, {e}, 7"));
            g.op(format!("mul.lo.u32 {r}, {r}, {}", 2 * LDS));
            g.op(format!("mad.lo.u32 {r}, {c}, 8, {r}"));
            g.op(format!("add.u32 {r}, {r}, %sa_st"));
            g.op(format!("st.shared.v2.b32 [{r}], {{{h0}, {h1}}}"));
            store.push_str(&g.finish());
        }
        return (regs, load, store);
    }
    let f16 = spec.a_dtype == DType::F16;
    let regs = if f16 {
        "    .reg .b16 %ah<8>;\n    .reg .pred %av<8>;\n"
    } else {
        "    .reg .f32 %ax<8>;\n    .reg .pred %av<8>;\n"
    };
    for u in 0..rows / 8 {
        // e = tid + 256u: row e / 32, k e % 32
        let mut g = Gen::new();
        let (e, r, gk, gi, idx) = (
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
        );
        g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
        g.op(format!("shr.u32 {r}, {e}, 5"));
        g.op(format!("and.b32 {gk}, {e}, 31"));
        g.op(format!("add.u32 {gk}, {gk}, %k0"));
        g.op(format!("add.u32 {gi}, %row0, {r}"));
        g.op(format!("setp.lt.u32 %av{u}, {gi}, %M"));
        g.op(format!("setp.lt.and.u32 %av{u}, {gk}, %K, %av{u}"));
        a_row_index(&mut g, id, u, &gi, &idx);
        g.op(format!("add.u32 {idx}, {idx}, {gk}"));
        g.op(format!("selp.b32 {idx}, {idx}, 0, %av{u}"));
        if f16 {
            let a = g.addr("%g_a", &idx, 2);
            g.op(format!("ld.global.nc.b16 %ah{u}, [{a}]"));
        } else {
            let a = g.addr("%g_a", &idx, 4);
            g.op(format!("ld.global.nc.f32 %ax{u}, [{a}]"));
        }
        load.push_str(&g.finish());

        let mut g = Gen::new();
        let (e, r, c, h) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("b16"));
        if f16 {
            g.op(format!("selp.b16 {h}, %ah{u}, 0, %av{u}"));
        } else {
            let x = g.reg("f32");
            g.op(format!("selp.f32 {x}, %ax{u}, 0f00000000, %av{u}"));
            g.op(format!("cvt.rn.f16.f32 {h}, {x}"));
        }
        g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
        g.op(format!("shr.u32 {r}, {e}, 5"));
        g.op(format!("and.b32 {c}, {e}, 31"));
        g.op(format!("mad.lo.u32 {r}, {r}, {LDS}, {c}"));
        g.op(format!("mad.lo.u32 {r}, {r}, 2, %sa_st"));
        g.op(format!("st.shared.b16 [{r}], {h}"));
        store.push_str(&g.finish());
    }
    (regs.to_string(), load, store)
}

/// Pack f32 registers pairwise (low half = first) into f16x2 words.
fn pack(g: &mut Gen, f: &[String]) -> Vec<String> {
    f.chunks(2)
        .map(|p| {
            let h = g.reg("b32");
            g.op(format!("cvt.rn.f16x2.f32 {h}, {}, {}", p[1], p[0]));
            h
        })
        .collect()
}

/// (REGS, B_LOAD, B_STORE)
fn b_parts(spec: &MatMulSpec, vec_b: bool) -> (String, String, String) {
    let mut load = String::new();
    let mut store = String::new();
    match spec.b_dtype {
        DType::Q4K | DType::Q5K | DType::Q6K => kquant::mma_b_parts(spec.b_dtype, LDS),
        DType::Mxfp4 => super::mxfp4::mma_b_parts(LDS),
        d if d.is_iquant() => super::iquant::mma_b_parts(d, LDS),
        DType::Int8 | DType::Int4 => {
            let int8 = spec.b_dtype == DType::Int8;
            let words = if int8 { 4 } else { 2 };
            let regs =
                format!("    .reg .b32 %bw<{words}>;\n    .reg .b16 %bh;\n    .reg .pred %bv;\n");
            // thread t: tile row t / 2 (output column), half t % 2 of the 32-element group
            let mut g = Gen::new();
            let (row, half, gj, e, off) = (
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
                g.reg("u32"),
            );
            g.op(format!("shr.u32 {row}, %tz, 1"));
            g.op(format!("and.b32 {half}, %tz, 1"));
            g.op(format!("add.u32 {gj}, %col0, {row}"));
            g.op(format!("setp.lt.u32 %bv, {gj}, %N"));
            g.op(format!("mad.lo.u32 {e}, {gj}, %K, %k0"));
            g.op(format!("selp.b32 {e}, {e}, 0, %bv"));
            if int8 {
                g.op(format!("mad.lo.u32 {off}, {half}, 16, {e}"));
                let a = g.addr("%g_b", &off, 1);
                g.op(format!(
                    "ld.global.nc.v4.u32 {{%bw0, %bw1, %bw2, %bw3}}, [{a}]"
                ));
            } else {
                g.op(format!("shr.u32 {off}, {e}, 1"));
                g.op(format!("mad.lo.u32 {off}, {half}, 8, {off}"));
                let a = g.addr("%g_b", &off, 1);
                g.op(format!("ld.global.nc.v2.u32 {{%bw0, %bw1}}, [{a}]"));
            }
            g.op(format!("shr.u32 {off}, {e}, 5"));
            g.op(format!("mad.lo.u32 {off}, {off}, 2, %so"));
            let a = g.addr("%g_b", &off, 1);
            g.op(format!("ld.global.nc.b16 %bh, [{a}]"));
            load.push_str(&g.finish());

            let mut g = Gen::new();
            let (s, q) = (g.reg("f32"), g.reg("s32"));
            g.op(format!("cvt.f32.f16 {s}, %bh"));
            g.op(format!("selp.f32 {s}, {s}, 0f00000000, %bv"));
            let (row, half, dst) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
            g.op(format!("shr.u32 {row}, %tz, 1"));
            g.op(format!("and.b32 {half}, %tz, 1"));
            g.op(format!("mad.lo.u32 {dst}, {row}, {}, %sb_st", 2 * LDS));
            let deq = |g: &mut Gen, w: usize, bit: usize, signed: bool| -> String {
                let f = g.reg("f32");
                if signed {
                    g.op(format!("bfe.s32 {q}, %bw{w}, {bit}, 8"));
                    g.op(format!("cvt.rn.f32.s32 {f}, {q}"));
                } else {
                    g.op(format!("bfe.u32 {q}, %bw{w}, {bit}, 4"));
                    g.op(format!("cvt.rn.f32.s32 {f}, {q}"));
                    g.op(format!("sub.f32 {f}, {f}, 0f41000000"));
                }
                g.op(format!("mul.f32 {f}, {f}, {s}"));
                f
            };
            if int8 {
                // 16 bytes = elements half*16 .. +16
                let f: Vec<String> = (0..16)
                    .map(|t| deq(&mut g, t / 4, 8 * (t % 4), true))
                    .collect();
                let h = pack(&mut g, &f);
                g.op(format!("mad.lo.u32 {dst}, {half}, 32, {dst}"));
                g.op(format!(
                    "st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}}",
                    h[0], h[1], h[2], h[3]
                ));
                g.op(format!(
                    "st.shared.v4.b32 [{dst}+16], {{{}, {}, {}, {}}}",
                    h[4], h[5], h[6], h[7]
                ));
            } else {
                // 8 bytes = group bytes half*8 .. +8: low nibbles → elements half*8 + t,
                // high nibbles → 16 + half*8 + t
                let lo: Vec<String> = (0..8)
                    .map(|t| deq(&mut g, t / 4, 8 * (t % 4), false))
                    .collect();
                let hi: Vec<String> = (0..8)
                    .map(|t| deq(&mut g, t / 4, 8 * (t % 4) + 4, false))
                    .collect();
                let hl = pack(&mut g, &lo);
                let hh = pack(&mut g, &hi);
                g.op(format!("mad.lo.u32 {dst}, {half}, 16, {dst}"));
                g.op(format!(
                    "st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}}",
                    hl[0], hl[1], hl[2], hl[3]
                ));
                g.op(format!(
                    "st.shared.v4.b32 [{dst}+32], {{{}, {}, {}, {}}}",
                    hh[0], hh[1], hh[2], hh[3]
                ));
            }
            store.push_str(&g.finish());
            (regs, load, store)
        }
        _ if vec_b => {
            let regs = "    .reg .b32 %bw<8>;\n    .reg .pred %bv<2>;\n".to_string();
            for u in 0..2 {
                // e = tid + 256u: row e / 4, 16-byte chunk e % 4
                let mut g = Gen::new();
                let (e, row, gj, gk, idx) = (
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                );
                g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
                g.op(format!("shr.u32 {row}, {e}, 2"));
                g.op(format!("and.b32 {gk}, {e}, 3"));
                g.op(format!("shl.b32 {gk}, {gk}, 3"));
                g.op(format!("add.u32 {gk}, {gk}, %k0"));
                g.op(format!("add.u32 {gj}, %col0, {row}"));
                g.op(format!("setp.lt.u32 %bv{u}, {gj}, %N"));
                g.op(format!("setp.lt.and.u32 %bv{u}, {gk}, %K, %bv{u}"));
                g.op(format!("mad.lo.u32 {idx}, {gj}, %brs, %bbase"));
                g.op(format!("add.u32 {idx}, {idx}, {gk}"));
                g.op(format!("selp.b32 {idx}, {idx}, 0, %bv{u}"));
                let a = g.addr("%g_b", &idx, 2);
                g.op(format!(
                    "ld.global.nc.v4.b32 {{%bw{}, %bw{}, %bw{}, %bw{}}}, [{a}]",
                    4 * u,
                    4 * u + 1,
                    4 * u + 2,
                    4 * u + 3
                ));
                load.push_str(&g.finish());

                let mut g = Gen::new();
                let (e, row, c) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
                let w: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
                for (t, wt) in w.iter().enumerate() {
                    g.op(format!("selp.b32 {wt}, %bw{}, 0, %bv{u}", 4 * u + t));
                }
                g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
                g.op(format!("shr.u32 {row}, {e}, 2"));
                g.op(format!("and.b32 {c}, {e}, 3"));
                g.op(format!("mul.lo.u32 {row}, {row}, {}", 2 * LDS));
                g.op(format!("mad.lo.u32 {row}, {c}, 16, {row}"));
                g.op(format!("add.u32 {row}, {row}, %sb_st"));
                g.op(format!(
                    "st.shared.v4.b32 [{row}], {{{}, {}, {}, {}}}",
                    w[0], w[1], w[2], w[3]
                ));
                store.push_str(&g.finish());
            }
            (regs, load, store)
        }
        _ => {
            // generic f16 B (unaligned or not transposed): 16 scalar elements per thread
            let regs = "    .reg .b16 %bq<16>;\n    .reg .pred %bv<16>;\n".to_string();
            for u in 0..16 {
                let mut g = Gen::new();
                let (e, row, kk, gj, gk, idx) = (
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                    g.reg("u32"),
                );
                g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
                if spec.b_transposed {
                    g.op(format!("shr.u32 {row}, {e}, 5"));
                    g.op(format!("and.b32 {kk}, {e}, 31"));
                } else {
                    g.op(format!("and.b32 {row}, {e}, 127"));
                    g.op(format!("shr.u32 {kk}, {e}, 7"));
                }
                g.op(format!("add.u32 {gj}, %col0, {row}"));
                g.op(format!("add.u32 {gk}, %k0, {kk}"));
                g.op(format!("setp.lt.u32 %bv{u}, {gj}, %N"));
                g.op(format!("setp.lt.and.u32 %bv{u}, {gk}, %K, %bv{u}"));
                if spec.b_transposed {
                    g.op(format!("mad.lo.u32 {idx}, {gj}, %brs, %bbase"));
                    g.op(format!("add.u32 {idx}, {idx}, {gk}"));
                } else {
                    g.op(format!("mad.lo.u32 {idx}, {gk}, %brs, %bbase"));
                    g.op(format!("add.u32 {idx}, {idx}, {gj}"));
                }
                g.op(format!("selp.b32 {idx}, {idx}, 0, %bv{u}"));
                let a = g.addr("%g_b", &idx, 2);
                g.op(format!("ld.global.nc.b16 %bq{u}, [{a}]"));
                load.push_str(&g.finish());

                let mut g = Gen::new();
                let (e, row, kk, h) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("b16"));
                g.op(format!("selp.b16 {h}, %bq{u}, 0, %bv{u}"));
                g.op(format!("add.u32 {e}, %tz, {}", 256 * u));
                if spec.b_transposed {
                    g.op(format!("shr.u32 {row}, {e}, 5"));
                    g.op(format!("and.b32 {kk}, {e}, 31"));
                } else {
                    g.op(format!("and.b32 {row}, {e}, 127"));
                    g.op(format!("shr.u32 {kk}, {e}, 7"));
                }
                g.op(format!("mad.lo.u32 {row}, {row}, {LDS}, {kk}"));
                g.op(format!("mad.lo.u32 {row}, {row}, 2, %sb_st"));
                g.op(format!("st.shared.b16 [{row}], {h}"));
                store.push_str(&g.finish());
            }
            (regs, load, store)
        }
    }
}

/// Two k16 steps over the current buffers: ldmatrix A (2 m16 tiles) and B (4 n8 tiles),
/// then 8 mma per step.
fn mma_block(mt: usize) -> String {
    let mut s = String::new();
    for kk in 0..2 {
        let ko = 32 * kk;
        for mi in 0..mt {
            let r = 4 * mi;
            s.push_str(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%fa{}, %fa{}, %fa{}, %fa{}}}, [%pa+{}];\n",
                r,
                r + 1,
                r + 2,
                r + 3,
                mi * 16 * 2 * LDS + ko
            ));
        }
        for p in 0..2 {
            let r = 4 * p;
            s.push_str(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%fb{}, %fb{}, %fb{}, %fb{}}}, [%pb+{}];\n",
                r,
                r + 1,
                r + 2,
                r + 3,
                p * 16 * 2 * LDS + ko
            ));
        }
        for mi in 0..mt {
            for ni in 0..4 {
                let c = mi * 16 + ni * 4;
                let a = mi * 4;
                let b = ni * 2;
                s.push_str(&format!(
                    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%c{c}, %c{}, %c{}, %c{}}}, {{%fa{a}, %fa{}, %fa{}, %fa{}}}, {{%fb{b}, %fb{}}}, {{%c{c}, %c{}, %c{}, %c{}}};\n",
                    c + 1, c + 2, c + 3, a + 1, a + 2, a + 3, b + 1, c + 1, c + 2, c + 3
                ));
            }
        }
    }
    s
}

/// Accumulators → Cs (f32, row stride 132): c[mi][ni][0..2] at (row, col), [2..4] at row + 8.
fn c_store(mt: usize) -> String {
    let mut s = String::new();
    for mi in 0..mt {
        for ni in 0..4 {
            let c = mi * 16 + ni * 4;
            let off = mi * 16 * 132 * 4 + ni * 8 * 4;
            s.push_str(&format!(
                "    st.shared.v2.f32 [%c_lane+{off}], {{%c{c}, %c{}}};\n",
                c + 1
            ));
            s.push_str(&format!(
                "    st.shared.v2.f32 [%c_lane+{}], {{%c{}, %c{}}};\n",
                off + 8 * 132 * 4,
                c + 2,
                c + 3
            ));
        }
    }
    s
}

/// `id`: MatMulId variant (rows grouped by expert, see `moe`); binds a, b, ids, out,
/// scratch, flag. `small`: 32-row tiles (MatMulId with few rows per expert).
pub(super) fn source(spec: &MatMulSpec, vec_a: bool, vec_b: bool, id: bool, small: bool) -> String {
    let rows = if small { SMALL_BM } else { BM };
    let mt = rows / 32;
    let (args, ptrs) = if id {
        ptx::args(&["a", "b", "ids", "out", "scratch", "flag"])
    } else {
        args(spec)
    };
    let (ra, la, sa, arows) = a_parts(spec, vec_a, id, rows);
    let (rb, lb, sb) = b_parts(spec, vec_b);
    let regs = format!("{ra}{rb}");
    template::fill(
        KERNEL,
        &[
            ("ARGS", &args),
            ("PTRS", &ptrs),
            ("REGS", &regs),
            ("A_LOAD", &la),
            ("B_LOAD", &lb),
            ("A_STORE", &sa),
            ("B_STORE", &sb),
            ("A_ROWS", &arows),
            ("MMA", &mma_block(mt)),
            ("C_STORE", &c_store(mt)),
            ("WM", &(16 * mt).to_string()),
            ("A_BUF", &(rows * 2 * LDS).to_string()),
            ("B_OFF", &(2 * rows * 2 * LDS).to_string()),
            ("SMEM", &(2 * rows * 2 * LDS + 2 * BN * 2 * LDS).max(rows * 132 * 4).to_string()),
            ("TSH", &rows.trailing_zeros().to_string()),
            ("EPI_N", &(rows * BN).to_string()),
            ("EPILOGUE", &if id { id_epilogue(spec) } else { epilogue(spec) }),
        ],
        if id { &["ID"] } else { &[] },
    )
}

/// MatMulId epilogue: output row = source row id `list[lbase + gi]` (contiguous [t*r, n]).
fn id_epilogue(spec: &MatMulSpec) -> String {
    let mut g = Gen::new();
    let (li, o) = (g.reg("u32"), g.reg("u32"));
    g.op(format!("add.u32 {li}, %lbase, %gi"));
    let a = g.addr("%g_list", &li, 4);
    g.op(format!("ld.global.u32 {o}, [{a}]"));
    g.op(format!("mad.lo.u32 {o}, {o}, %N, %gj"));
    ptx::store(&mut g, spec.out_dtype, "%g_out", &o, "%acc");
    g.finish()
}
