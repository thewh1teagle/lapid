//! ggml i-quants (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS, AoS exactly as in GGUF; see lapid-quant):
//! element dequantization (gather, scalar GEMM), the GEMV lane dot product over int8
//! activations (`dp4a`, after the `q8_rows` pre-pass) and the tensor-core GEMM B-tile loader.
//!
//! Block layouts (bytes, 256 elements = 8 sub-blocks of 32 = 32 groups of 8):
//! - IQ2_XXS (66): `d f16 | 8 x (idx u8[4] | aux u32)`; group l of sub-block ib: grid
//!   `iq2xxs_grid[idx[l]]`, signs `ksigns[(aux >> 7l) & 127]`, x = d (0.5 + aux>>28) / 4 * g.
//! - IQ2_XS (74): `d | qs u16[32] | scales[8]`; group j = 4ib + l: grid
//!   `iq2xs_grid[qs[j] & 511]`, signs `ksigns[qs[j] >> 9]`, 4-bit scale per 16.
//! - IQ2_S (82): `d | qs[32] | signs[32] | qh[8] | scales[8]`; group j = 4ib + l: grid
//!   `iq2s_grid[qs[j] | ((qh[ib] >> 2l) & 3) << 8]`, signs `signs[j]`, 4-bit scale per 16.
//! - IQ3_XXS (98): `d | idx u8[64] | aux u32[8]`; group l of ib: grids `iq3xxs_grid` of
//!   `idx[8ib + 2l]`, `idx[8ib + 2l + 1]` (4 bytes each), signs as IQ2_XXS, x = d (0.5 + s) / 2.
//!
//! Grids and sign masks are module-scope `.global` tables (read through the L1 / texture
//! path). Signs enter the integer dot products as byte masks `m` (0xff per negative value):
//! `dot(a, ±g) = dot(a, g) - 2 dot(a, g & m)`. Blocks are 2-byte aligned (66, 82, 98 bytes),
//! so every multi-byte field is read as u16 halves.

use std::fmt::Write;

use lapid_quant::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS};
use lapid_tensor::DType;

use super::ptx::Gen;

fn block_bytes(d: DType) -> usize {
    d.quant_format().map(|f| f.block_bytes()).unwrap_or(0)
}

/// Byte mask of the 8 sign bits `s`: byte j = 0xff when bit j is set.
fn mask8(s: u8) -> u64 {
    (0..8).fold(0u64, |m, j| if s >> j & 1 != 0 { m | 0xff << (8 * j) } else { m })
}

/// Module-scope tables for format `d`: the grid `iq_grid` and the sign masks `iq_sign`
/// (indexed by the 7 stored sign bits for IQ2_XXS / IQ3_XXS, by the sign byte for IQ2_S).
pub(crate) fn tables(d: DType) -> String {
    fn table<T: std::fmt::Display>(s: &mut String, name: &str, ty: &str, align: usize, v: impl Iterator<Item = T>) {
        let vals: Vec<String> = v.map(|x| x.to_string()).collect();
        let _ = write!(s, ".global .align {align} .{ty} {name}[{}] = {{", vals.len());
        for (i, x) in vals.iter().enumerate() {
            if i % 8 == 0 {
                s.push_str("\n    ");
            }
            s.push_str(x);
            if i + 1 < vals.len() {
                s.push_str(", ");
            }
        }
        s.push_str("\n};\n");
    }
    let mut s = String::new();
    match d {
        DType::Iq2Xxs => table(&mut s, "iq_grid", "b64", 8, IQ2XXS_GRID.iter()),
        DType::Iq2Xs => table(&mut s, "iq_grid", "b64", 8, IQ2XS_GRID.iter()),
        DType::Iq2S => table(&mut s, "iq_grid", "b64", 8, IQ2S_GRID.iter()),
        _ => table(&mut s, "iq_grid", "b32", 4, IQ3XXS_GRID.iter()),
    }
    if d == DType::Iq2S {
        table(&mut s, "iq_sign", "b64", 8, (0..=255u8).map(mask8));
    } else {
        table(&mut s, "iq_sign", "b64", 8, KSIGNS.iter().map(|&k| mask8(k)));
    }
    s
}

/// `ptr + blk * block_bytes` (64-bit).
fn block_addr(g: &mut Gen, d: DType, ptr: &str, blk: &str) -> String {
    let a = g.reg("b64");
    g.op(format!("mul.wide.u32 {a}, {blk}, {}", block_bytes(d)));
    g.op(format!("add.s64 {a}, {a}, {ptr}"));
    a
}

/// `base + off` where `off` is a u32 register.
fn offset(g: &mut Gen, base: &str, off: &str) -> String {
    let a = g.reg("b64");
    g.op(format!("cvt.u64.u32 {a}, {off}"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    a
}

/// u32 little endian at byte address `a + imm` (2-byte aligned) from two u16 loads.
fn ld32(g: &mut Gen, a: &str, imm: usize) -> String {
    let (lo, hi, w) = (g.reg("b16"), g.reg("b16"), g.reg("b32"));
    g.op(format!("ld.global.nc.b16 {lo}, [{a}+{imm}]"));
    g.op(format!("ld.global.nc.b16 {hi}, [{a}+{}]", imm + 2));
    g.op(format!("mov.b32 {w}, {{{lo}, {hi}}}"));
    w
}

/// Where the lookup tables are read: the module's global tables, or the shared copies made
/// by [`smem_prologue`].
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Tab {
    Global,
    Shared,
}

/// Load `dst` (one or two b32 registers) from entry `idx` (u32 register) of `size` bytes of
/// table `name` ("grid" / "sign").
fn lookup(g: &mut Gen, tab: Tab, name: &str, idx: &str, size: usize, dst: &[&str]) {
    let regs = if dst.len() == 1 { dst[0].to_string() } else { format!("{{{}}}", dst.join(", ")) };
    let v = if dst.len() == 1 { "" } else { ".v2" };
    if tab == Tab::Shared {
        let a = g.reg("u32");
        g.op(format!("mov.u32 {a}, iq_s{name}"));
        g.op(format!("mad.lo.u32 {a}, {idx}, {size}, {a}"));
        g.op(format!("ld.shared{v}.u32 {regs}, [{a}]"));
        return;
    }
    let (t, a) = (g.reg("b64"), g.reg("b64"));
    g.op(format!("mov.u64 {t}, iq_{name}"));
    g.op(format!("mul.wide.u32 {a}, {idx}, {size}"));
    g.op(format!("add.s64 {a}, {a}, {t}"));
    g.op(format!("ld.global.nc{v}.u32 {regs}, [{a}]"));
}

/// Bytes of the (grid, sign) tables of format `d`.
fn table_bytes(d: DType) -> (usize, usize) {
    match d {
        DType::Iq2Xxs => (256 * 8, 128 * 8),
        DType::Iq2Xs => (512 * 8, 128 * 8),
        DType::Iq2S => (1024 * 8, 256 * 8),
        _ => (256 * 4, 128 * 8),
    }
}

/// Function-scope code copying the tables into shared memory (`iq_sgrid`, `iq_ssign`) with
/// every thread of the block (`threads`), then a block barrier: must run before any thread
/// leaves the kernel.
pub(crate) fn smem_prologue(d: DType, threads: usize) -> String {
    let (gb, sb) = table_bytes(d);
    let mut s = format!("    .shared .align 16 .b8 iq_sgrid[{gb}];\n    .shared .align 16 .b8 iq_ssign[{sb}];\n");
    for (name, bytes) in [("grid", gb), ("sign", sb)] {
        let mut g = Gen::new();
        let (i, sa, w) = (g.reg("u32"), g.reg("u32"), [g.reg("b32"), g.reg("b32"), g.reg("b32"), g.reg("b32")]);
        let (t, a) = (g.reg("b64"), g.reg("b64"));
        let p = g.reg("pred");
        g.op(format!("mov.u64 {t}, iq_{name}"));
        for c in 0..bytes.div_ceil(16 * threads) {
            g.op(format!("mov.u32 {i}, %tid.x"));
            g.op(format!("add.u32 {i}, {i}, {}", c * threads));
            g.op(format!("setp.lt.u32 {p}, {i}, {}", bytes / 16));
            g.op(format!("shl.b32 {i}, {i}, 4"));
            g.op(format!("cvt.u64.u32 {a}, {i}"));
            g.op(format!("add.s64 {a}, {a}, {t}"));
            g.op(format!("@{p} ld.global.nc.v4.u32 {{{}, {}, {}, {}}}, [{a}]", w[0], w[1], w[2], w[3]));
            g.op(format!("mov.u32 {sa}, iq_s{name}"));
            g.op(format!("add.u32 {sa}, {sa}, {i}"));
            g.op(format!("@{p} st.shared.v4.u32 [{sa}], {{{}, {}, {}, {}}}", w[0], w[1], w[2], w[3]));
        }
        s.push_str(&g.finish());
    }
    s.push_str("    bar.sync 0;\n");
    s
}

/// The block's f16 `d` as f32.
fn d_f32(g: &mut Gen, base: &str) -> String {
    let (h, f) = (g.reg("b16"), g.reg("f32"));
    g.op(format!("ld.global.nc.b16 {h}, [{base}]"));
    g.op(format!("cvt.f32.f16 {f}, {h}"));
    f
}

/// `d * (0.5 + s) * mul` for the 4-bit scale `s` (u32 register).
fn sub_scale(g: &mut Gen, d: &str, s: &str, mul: f32) -> String {
    let f = g.reg("f32");
    g.op(format!("cvt.rn.f32.u32 {f}, {s}"));
    g.op(format!("add.f32 {f}, {f}, 0f3F000000"));
    g.op(format!("mul.f32 {f}, {d}, {f}"));
    g.op(format!("mul.f32 {f}, {f}, {}", super::ptx::fimm(mul)));
    f
}

/// The 8 grid bytes (two words) and the sign mask (two words) of group `l` (0..4) of a
/// sub-block, given its loaded fields. `f`: (idx word(s), sign word, qh byte) per format.
struct Group {
    g: [String; 2],
    m: [String; 2],
}

/// Raw fields of one sub-block `ib` of the block at `base`.
struct Sub {
    /// IQ2_XXS: idx bytes; IQ2_S: idx low bytes; IQ3_XXS: idx bytes 0..4, 4..8; IQ2_XS:
    /// qs u16 pairs 0..2, 2..4.
    idx: Vec<String>,
    /// IQ2_XXS / IQ3_XXS: aux (signs + scale); IQ2_S: sign bytes.
    aux: String,
    /// IQ2_S: high index bits byte; IQ2_S / IQ2_XS: the scales byte.
    qh: String,
    sc: String,
}

/// Load the fields of sub-block `ib` (u32 register) of the block at `base`.
fn load_sub(g: &mut Gen, d: DType, base: &str, ib: &str) -> Sub {
    let o = g.reg("u32");
    match d {
        DType::Iq2Xxs => {
            g.op(format!("shl.b32 {o}, {ib}, 3"));
            let a = offset(g, base, &o);
            let idx = ld32(g, &a, 2);
            let aux = ld32(g, &a, 6);
            Sub { idx: vec![idx], aux, qh: String::new(), sc: String::new() }
        }
        DType::Iq2Xs => {
            g.op(format!("shl.b32 {o}, {ib}, 3"));
            let a = offset(g, base, &o);
            let i0 = ld32(g, &a, 2);
            let i1 = ld32(g, &a, 6);
            let a = offset(g, base, ib);
            let sc = g.reg("u32");
            g.op(format!("ld.global.nc.u8 {sc}, [{a}+66]"));
            Sub { idx: vec![i0.clone(), i1], aux: i0, qh: String::new(), sc }
        }
        DType::Iq2S => {
            g.op(format!("shl.b32 {o}, {ib}, 2"));
            let a = offset(g, base, &o);
            let idx = ld32(g, &a, 2);
            let aux = ld32(g, &a, 34);
            let a = offset(g, base, ib);
            let (qh, sc) = (g.reg("u32"), g.reg("u32"));
            g.op(format!("ld.global.nc.u8 {qh}, [{a}+66]"));
            g.op(format!("ld.global.nc.u8 {sc}, [{a}+74]"));
            Sub { idx: vec![idx], aux, qh, sc }
        }
        _ => {
            g.op(format!("shl.b32 {o}, {ib}, 3"));
            let a = offset(g, base, &o);
            let i0 = ld32(g, &a, 2);
            let i1 = ld32(g, &a, 6);
            g.op(format!("shl.b32 {o}, {ib}, 2"));
            let a = offset(g, base, &o);
            let aux = ld32(g, &a, 66);
            Sub { idx: vec![i0, i1], aux, qh: String::new(), sc: String::new() }
        }
    }
}

/// Group selector: an immediate 0..4 or a u32 register.
enum L<'a> {
    Imm(usize),
    Reg(&'a str),
}

impl L<'_> {
    /// `l * mul + add` as an operand (immediate or a fresh register).
    fn lin(&self, g: &mut Gen, mul: usize, add: usize) -> String {
        match self {
            L::Imm(l) => (l * mul + add).to_string(),
            L::Reg(r) => {
                let v = g.reg("u32");
                g.op(format!("mad.lo.u32 {v}, {r}, {mul}, {add}"));
                v
            }
        }
    }
}

/// Index word `l / 2` of a sub-block with two (IQ3_XXS, IQ2_XS).
fn word(g: &mut Gen, s: &Sub, l: &L) -> String {
    match l {
        L::Imm(i) => s.idx[i / 2].clone(),
        L::Reg(r) => {
            let (w, p) = (g.reg("b32"), g.reg("pred"));
            g.op(format!("setp.ge.u32 {p}, {r}, 2"));
            g.op(format!("selp.b32 {w}, {}, {}, {p}", s.idx[1], s.idx[0]));
            w
        }
    }
}

/// `16 (l % 2)`: the bit offset of group `l` in its index word (IQ3_XXS, IQ2_XS).
fn odd16(g: &mut Gen, l: &L) -> String {
    match l {
        L::Imm(i) => (16 * (i % 2)).to_string(),
        L::Reg(r) => {
            let v = g.reg("u32");
            g.op(format!("and.b32 {v}, {r}, 1"));
            g.op(format!("shl.b32 {v}, {v}, 4"));
            v
        }
    }
}

/// Formats with a scale per 16 elements (else per 32).
fn scale16(d: DType) -> bool {
    matches!(d, DType::Iq2S | DType::Iq2Xs)
}

/// Grid bytes and sign mask of group `l` (0..4) of a loaded sub-block.
fn group(g: &mut Gen, d: DType, tab: Tab, s: &Sub, l: L) -> Group {
    let (gw, mw) = ([g.reg("b32"), g.reg("b32")], [g.reg("b32"), g.reg("b32")]);
    let (idx, si) = (g.reg("u32"), g.reg("u32"));
    match d {
        DType::Iq2Xs => {
            // qs[l]: u16 l % 2 of word l / 2: 9-bit index, 7 sign bits
            let w = word(g, s, &l);
            let lo = odd16(g, &l);
            g.op(format!("bfe.u32 {idx}, {w}, {lo}, 9"));
            let p = g.reg("u32");
            g.op(format!("add.u32 {p}, {lo}, 9"));
            g.op(format!("bfe.u32 {si}, {w}, {p}, 7"));
            lookup(g, tab, "grid", &idx, 8, &[&gw[0], &gw[1]]);
        }
        DType::Iq2Xxs | DType::Iq2S => {
            let p = l.lin(g, 8, 0);
            g.op(format!("bfe.u32 {idx}, {}, {p}, 8", s.idx[0]));
            if d == DType::Iq2S {
                let h = g.reg("u32");
                let p = l.lin(g, 2, 0);
                g.op(format!("bfe.u32 {h}, {}, {p}, 2", s.qh));
                g.op(format!("bfi.b32 {idx}, {h}, {idx}, 8, 2"));
                let p = l.lin(g, 8, 0);
                g.op(format!("bfe.u32 {si}, {}, {p}, 8", s.aux));
            } else {
                let p = l.lin(g, 7, 0);
                g.op(format!("bfe.u32 {si}, {}, {p}, 7", s.aux));
            }
            lookup(g, tab, "grid", &idx, 8, &[&gw[0], &gw[1]]);
        }
        _ => {
            // idx bytes 2l, 2l + 1: word l / 2, bits 16 (l % 2) + 8h
            let w = word(g, s, &l);
            let lo = odd16(g, &l);
            for (h, gh) in gw.iter().enumerate() {
                let pos = g.reg("u32");
                g.op(format!("add.u32 {pos}, {lo}, {}", 8 * h));
                g.op(format!("bfe.u32 {idx}, {w}, {pos}, 8"));
                lookup(g, tab, "grid", &idx, 4, &[gh]);
            }
            let p = l.lin(g, 7, 0);
            g.op(format!("bfe.u32 {si}, {}, {p}, 7", s.aux));
        }
    }
    lookup(g, tab, "sign", &si, 8, &[&mw[0], &mw[1]]);
    Group { g: gw, m: mw }
}

/// The f32 scale of the 16-element half `h` (0 / 1) of a loaded sub-block (without `da`).
fn half_scale(g: &mut Gen, dt: DType, s: &Sub, d: &str, h: usize) -> String {
    let v = g.reg("u32");
    match dt {
        DType::Iq2S | DType::Iq2Xs => g.op(format!("bfe.u32 {v}, {}, {}, 4", s.sc, 4 * h)),
        _ => g.op(format!("shr.u32 {v}, {}, 28", s.aux)),
    }
    sub_scale(g, d, &v, if dt == DType::Iq3Xxs { 0.5 } else { 0.25 })
}

/// GEMV lane dot product over one 32-element sub-block with int8 activations:
/// `out = sum A * B[gj, k..k+32]`, A from the `q8_rows` scratch (int8 at `%arow + k`, f32
/// scale per 32 at byte `%so + 4 (%arow + k) / 32`). Expects `%gj`, `%K`, `%arow`, `%so`,
/// `%g_b`, `%g_scratch` and the shared tables of [`smem_prologue`].
pub(crate) fn gemv_dot(g: &mut Gen, d: DType, k: &str, out: &str) {
    let (t, blk, ib, ia) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {t}, %K, 8"));
    g.op(format!("shr.u32 {blk}, {k}, 8"));
    g.op(format!("mad.lo.u32 {blk}, %gj, {t}, {blk}"));
    g.op(format!("bfe.u32 {ib}, {k}, 5, 3"));
    let base = block_addr(g, d, "%g_b", &blk);
    // activations: 32 int8 + scale
    g.op(format!("add.u32 {ia}, %arow, {k}"));
    let aa = offset(g, "%g_scratch", &ia);
    let a: Vec<String> = (0..8).map(|_| g.reg("b32")).collect();
    g.op(format!("ld.global.nc.v4.u32 {{{}, {}, {}, {}}}, [{aa}]", a[0], a[1], a[2], a[3]));
    g.op(format!("ld.global.nc.v4.u32 {{{}, {}, {}, {}}}, [{aa}+16]", a[4], a[5], a[6], a[7]));
    g.op(format!("shr.u32 {ia}, {ia}, 5"));
    g.op(format!("shl.b32 {ia}, {ia}, 2"));
    g.op(format!("add.u32 {ia}, {ia}, %so"));
    let sa = offset(g, "%g_scratch", &ia);
    let da = g.reg("f32");
    g.op(format!("ld.global.nc.f32 {da}, [{sa}]"));
    let s = load_sub(g, d, &base, &ib);
    let df = d_f32(g, &base);
    // per 16-element half: positive and masked dot products
    let mut halves = Vec::new();
    for h in 0..2 {
        let (p, n) = (g.reg("s32"), g.reg("s32"));
        g.op(format!("mov.s32 {p}, 0"));
        g.op(format!("mov.s32 {n}, 0"));
        for l in 2 * h..2 * h + 2 {
            let gr = group(g, d, Tab::Shared, &s, L::Imm(l));
            for w in 0..2 {
                let m = g.reg("b32");
                g.op(format!("and.b32 {m}, {}, {}", gr.g[w], gr.m[w]));
                g.op(format!("dp4a.s32.s32 {p}, {}, {}, {p}", a[2 * l + w], gr.g[w]));
                g.op(format!("dp4a.s32.s32 {n}, {}, {m}, {n}", a[2 * l + w]));
            }
        }
        g.op(format!("mad.lo.s32 {p}, {n}, -2, {p}"));
        halves.push(p);
    }
    let f = g.reg("f32");
    if scale16(d) {
        let (s0, s1) = (half_scale(g, d, &s, &df, 0), half_scale(g, d, &s, &df, 1));
        let f1 = g.reg("f32");
        g.op(format!("cvt.rn.f32.s32 {f}, {}", halves[0]));
        g.op(format!("cvt.rn.f32.s32 {f1}, {}", halves[1]));
        g.op(format!("mul.f32 {f}, {f}, {s0}"));
        g.op(format!("fma.rn.f32 {f}, {f1}, {s1}, {f}"));
    } else {
        let sc = half_scale(g, d, &s, &df, 0);
        g.op(format!("add.s32 {}, {}, {}", halves[0], halves[0], halves[1]));
        g.op(format!("cvt.rn.f32.s32 {f}, {}", halves[0]));
        g.op(format!("mul.f32 {f}, {f}, {sc}"));
    }
    g.op(format!("mul.f32 {out}, {f}, {da}"));
}

/// Dequantize element `e` (row-major linear index, u32 register) into `dst`.
pub(crate) fn load_elem(g: &mut Gen, d: DType, ptr: &str, e: &str, dst: &str) {
    let (blk, ib, l, j) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {blk}, {e}, 8"));
    g.op(format!("bfe.u32 {ib}, {e}, 5, 3"));
    g.op(format!("bfe.u32 {l}, {e}, 3, 2"));
    g.op(format!("and.b32 {j}, {e}, 7"));
    let base = block_addr(g, d, ptr, &blk);
    let s = load_sub(g, d, &base, &ib);
    let df = d_f32(g, &base);
    // the group's grid bytes and mask word holding element j
    let gr = group(g, d, Tab::Global, &s, L::Reg(&l));
    let (gb, mb, hi) = (g.reg("b32"), g.reg("b32"), g.reg("pred"));
    let p = g.reg("pred");
    g.op(format!("setp.ge.u32 {hi}, {j}, 4"));
    g.op(format!("selp.b32 {gb}, {}, {}, {hi}", gr.g[1], gr.g[0]));
    g.op(format!("selp.b32 {mb}, {}, {}, {hi}", gr.m[1], gr.m[0]));
    let (sh, q, m) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("and.b32 {sh}, {j}, 3"));
    g.op(format!("shl.b32 {sh}, {sh}, 3"));
    g.op(format!("bfe.u32 {q}, {gb}, {sh}, 8"));
    g.op(format!("bfe.u32 {m}, {mb}, {sh}, 1"));
    // scale of the element's half (l / 2)
    let sc = g.reg("f32");
    if scale16(d) {
        let (s0, s1) = (half_scale(g, d, &s, &df, 0), half_scale(g, d, &s, &df, 1));
        g.op(format!("setp.ge.u32 {p}, {l}, 2"));
        g.op(format!("selp.f32 {sc}, {s1}, {s0}, {p}"));
    } else {
        let s0 = half_scale(g, d, &s, &df, 0);
        g.op(format!("mov.f32 {sc}, {s0}"));
    }
    let f = g.reg("f32");
    g.op(format!("cvt.rn.f32.u32 {f}, {q}"));
    g.op(format!("mul.f32 {f}, {sc}, {f}"));
    g.op(format!("setp.ne.u32 {p}, {m}, 0"));
    g.op(format!("@{p} neg.f32 {f}, {f}"));
    g.op(format!("mov.f32 {dst}, {f}"));
}

/// Tensor-core GEMM B tile (128 columns x 32 k = one sub-block per column): thread t loads
/// the fields of column t / 2, sub-block (k0 / 32) % 8, and dequantizes its half t % 2
/// (16 elements, groups 2h, 2h + 1) into f16 shared memory (row stride `lds` halves).
/// Returns (REGS, B_LOAD, B_STORE).
pub(crate) fn mma_b_parts(d: DType, lds: usize) -> (String, String, String) {
    let regs = "    .reg .pred %bv;\n    .reg .b32 %iqi<2>;\n    .reg .b32 %iqa, %iqh, %iqs;\n    .reg .f32 %iqd;\n".to_string();
    // ---- load: raw fields into %iqi*, %iqa, %iqh, %iqs, %iqd
    let mut g = Gen::new();
    let (row, gj, blk, t, ib) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("add.u32 {gj}, %col0, {row}"));
    g.op(format!("setp.lt.u32 %bv, {gj}, %N"));
    g.op(format!("selp.b32 {gj}, {gj}, 0, %bv"));
    g.op(format!("shr.u32 {t}, %K, 8"));
    g.op(format!("shr.u32 {blk}, %k0, 8"));
    g.op(format!("mad.lo.u32 {blk}, {gj}, {t}, {blk}"));
    g.op(format!("bfe.u32 {ib}, %k0, 5, 3"));
    let base = block_addr(&mut g, d, "%g_b", &blk);
    let s = load_sub(&mut g, d, &base, &ib);
    for (i, w) in s.idx.iter().enumerate() {
        g.op(format!("mov.b32 %iqi{i}, {w}"));
    }
    g.op(format!("mov.b32 %iqa, {}", s.aux));
    if d == DType::Iq2S {
        g.op(format!("mov.b32 %iqh, {}", s.qh));
    }
    if scale16(d) {
        g.op(format!("mov.b32 %iqs, {}", s.sc));
    }
    let df = d_f32(&mut g, &base);
    g.op(format!("mov.f32 %iqd, {df}"));
    let load = g.finish();

    // ---- store
    let mut g = Gen::new();
    let s = Sub {
        idx: (0..if matches!(d, DType::Iq3Xxs | DType::Iq2Xs) { 2 } else { 1 }).map(|i| format!("%iqi{i}")).collect(),
        aux: "%iqa".into(),
        qh: "%iqh".into(),
        sc: "%iqs".into(),
    };
    let (half, row, dst) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("and.b32 {half}, %tz, 1"));
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("mad.lo.u32 {dst}, {row}, {}, %sb_st", 2 * lds));
    g.op(format!("mad.lo.u32 {dst}, {half}, 32, {dst}"));
    let p = g.reg("pred");
    g.op(format!("setp.ne.u32 {p}, {half}, 0"));
    // scale of this half (zero for columns past N)
    let (s0, s1) = (half_scale(&mut g, d, &s, "%iqd", 0), half_scale(&mut g, d, &s, "%iqd", 1));
    let sc = g.reg("f32");
    g.op(format!("selp.f32 {sc}, {s1}, {s0}, {p}"));
    g.op(format!("selp.f32 {sc}, {sc}, 0f00000000, %bv"));
    // grid / mask words of this half's two groups (l = 2h, 2h + 1)
    let mut words = Vec::new();
    for li in 0..2 {
        let l = g.reg("u32");
        g.op(format!("mad.lo.u32 {l}, {half}, 2, {li}"));
        let gr = group(&mut g, d, Tab::Global, &s, L::Reg(&l));
        for w in 0..2 {
            words.push((gr.g[w].clone(), gr.m[w].clone()));
        }
    }
    // 16 values as f16 pairs: grid bytes spliced into the mantissa of 1024 (exact integers
    // after subtracting 1024), mask bytes to the sign bits, times the scale (f16)
    let (k64, zero, sh, s2, h1024) = (g.reg("b32"), g.reg("b32"), g.reg("b16"), g.reg("b32"), g.reg("b32"));
    g.op(format!("mov.b32 {k64}, 0x64646464"));
    g.op(format!("mov.b32 {h1024}, 0x64006400"));
    g.op(format!("mov.b32 {zero}, 0"));
    g.op(format!("cvt.rn.f16.f32 {sh}, {sc}"));
    g.op(format!("mov.b32 {s2}, {{{sh}, {sh}}}"));
    let mut packed = Vec::new();
    for (gw, mw) in &words {
        for (vs, ms) in [("0x4140", "0x1404"), ("0x4342", "0x3424")] {
            let (h, m) = (g.reg("b32"), g.reg("b32"));
            g.op(format!("prmt.b32 {h}, {gw}, {k64}, {vs}"));
            g.op(format!("sub.f16x2 {h}, {h}, {h1024}"));
            g.op(format!("prmt.b32 {m}, {mw}, {zero}, {ms}"));
            g.op(format!("and.b32 {m}, {m}, 0x80008000"));
            g.op(format!("xor.b32 {h}, {h}, {m}"));
            g.op(format!("mul.rn.f16x2 {h}, {h}, {s2}"));
            packed.push(h);
        }
    }
    g.op(format!("st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}}", packed[0], packed[1], packed[2], packed[3]));
    g.op(format!("st.shared.v4.b32 [{dst}+16], {{{}, {}, {}, {}}}", packed[4], packed[5], packed[6], packed[7]));
    let store = g.finish();
    (regs, load, store)
}
