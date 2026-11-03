//! Slot fillers shared by all kernels: a tiny register/instruction builder (`Gen`), typed
//! global loads and stores, dequantization, params access, index decomposition, warp and
//! block reductions, and the kernel argument list.
//!
//! Generated snippets only use registers they declare (`%_N`, scoped to the snippet's
//! `{ }` block) plus the registers the template names explicitly; they never contain
//! labels, so they can be pasted anywhere in a template.

use std::fmt::Write;

use lapid_tensor::DType;

/// PTX f32 immediate with the exact bits of `v`.
pub(crate) fn fimm(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

/// Register + instruction builder for one snippet.
pub(crate) struct Gen {
    decls: String,
    body: String,
    n: usize,
    prefix: &'static str,
}

impl Default for Gen {
    fn default() -> Gen {
        Gen::new()
    }
}

impl Gen {
    /// Registers `%_N` (for scoped snippets).
    pub fn new() -> Gen {
        Gen::named("_")
    }

    /// Registers `%<prefix>N` (for function-scope code that scoped snippets must not shadow).
    pub fn named(prefix: &'static str) -> Gen {
        Gen {
            decls: String::new(),
            body: String::new(),
            n: 0,
            prefix,
        }
    }

    /// Fresh register of PTX type `ty` ("f32", "b32", "u32", "s32", "b64", "pred", "b16").
    pub fn reg(&mut self, ty: &str) -> String {
        let r = format!("%{}{}", self.prefix, self.n);
        self.n += 1;
        let _ = writeln!(self.decls, "    .reg .{ty} {r};");
        r
    }

    /// One instruction (no trailing `;`).
    pub fn op(&mut self, s: impl AsRef<str>) {
        self.body.push_str("    ");
        self.body.push_str(s.as_ref());
        self.body.push_str(";\n");
    }

    /// `ld.param.u32` of params word `w` into a fresh register.
    pub fn param(&mut self, w: usize) -> String {
        let r = self.reg("u32");
        self.op(format!("ld.param.u32 {r}, [P+{}]", 4 * w));
        r
    }

    /// f32 constant in a fresh register.
    pub fn konst(&mut self, v: f32) -> String {
        let r = self.reg("f32");
        self.op(format!("mov.f32 {r}, {}", fimm(v)));
        r
    }

    /// `ptr + idx * size` as a 64-bit address.
    pub fn addr(&mut self, ptr: &str, idx: &str, size: usize) -> String {
        let a = self.reg("b64");
        if size == 1 {
            self.op(format!("cvt.u64.u32 {a}, {idx}"));
        } else {
            self.op(format!("mul.wide.u32 {a}, {idx}, {size}"));
        }
        self.op(format!("add.s64 {a}, {a}, {ptr}"));
        a
    }

    /// The snippet as a scoped block.
    pub fn finish(self) -> String {
        format!("{{\n{}{}}}\n", self.decls, self.body)
    }

    /// Declarations + code at the enclosing scope (registers stay visible afterwards).
    pub fn finish_flat(self) -> String {
        format!("{}{}", self.decls, self.body)
    }
}

/// Load element `idx` (u32 register, in elements) of a plain-dtype buffer at `ptr` into the
/// f32 register `dst`. `nc` uses the read-only (non-coherent) cache path.
pub(crate) fn load(g: &mut Gen, d: DType, ptr: &str, idx: &str, dst: &str, nc: bool) {
    let ld = if nc { "ld.global.nc" } else { "ld.global" };
    match d {
        DType::F32 => {
            let a = g.addr(ptr, idx, 4);
            g.op(format!("{ld}.f32 {dst}, [{a}]"));
        }
        DType::F16 => {
            let a = g.addr(ptr, idx, 2);
            let h = g.reg("b16");
            g.op(format!("{ld}.b16 {h}, [{a}]"));
            g.op(format!("cvt.f32.f16 {dst}, {h}"));
        }
        DType::BF16 => {
            let a = g.addr(ptr, idx, 2);
            let h = g.reg("b16");
            let u = g.reg("b32");
            g.op(format!("{ld}.b16 {h}, [{a}]"));
            g.op(format!("cvt.u32.u16 {u}, {h}"));
            g.op(format!("shl.b32 {u}, {u}, 16"));
            g.op(format!("mov.b32 {dst}, {u}"));
        }
        DType::I32 => {
            let a = g.addr(ptr, idx, 4);
            let s = g.reg("s32");
            g.op(format!("{ld}.s32 {s}, [{a}]"));
            g.op(format!("cvt.rn.f32.s32 {dst}, {s}"));
        }
        // quantized buffers go through `load_quant`
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 | DType::Iq2Xxs | DType::Iq2S | DType::Iq3Xxs => {
            g.op(format!("mov.f32 {dst}, 0f7FC00000"))
        }
    }
}

/// Load element `idx` of an expression input as the registers the expression needs:
/// (`f32` value when `want_f`, raw 32-bit word when `want_bits`). I32 / F32 storage read the
/// word once; narrower dtypes expose the bits of their f32 value.
pub(crate) fn load_input(
    g: &mut Gen,
    d: DType,
    ptr: &str,
    idx: &str,
    want_f: bool,
    want_bits: bool,
    nc: bool,
) -> (String, String) {
    let (mut f, mut w) = (String::new(), String::new());
    if want_bits && matches!(d, DType::F32 | DType::I32) {
        let ld = if nc { "ld.global.nc" } else { "ld.global" };
        let a = g.addr(ptr, idx, 4);
        w = g.reg("b32");
        g.op(format!("{ld}.b32 {w}, [{a}]"));
        if want_f {
            f = g.reg("f32");
            if d == DType::I32 {
                g.op(format!("cvt.rn.f32.s32 {f}, {w}"));
            } else {
                g.op(format!("mov.b32 {f}, {w}"));
            }
        }
        return (f, w);
    }
    if want_f || want_bits {
        f = g.reg("f32");
        load(g, d, ptr, idx, &f, nc);
        if want_bits {
            w = g.reg("b32");
            g.op(format!("mov.b32 {w}, {f}"));
        }
    }
    (f, w)
}

/// Store the raw 32-bit word `w` as element `idx` of an I32 (or F32) buffer.
pub(crate) fn store_bits(g: &mut Gen, ptr: &str, idx: &str, w: &str) {
    let a = g.addr(ptr, idx, 4);
    g.op(format!("st.global.b32 [{a}], {w}"));
}

/// Copy element `src` of `sp` to element `dst` of `dp` without conversion (plain dtypes).
pub(crate) fn copy_raw(g: &mut Gen, d: DType, sp: &str, src: &str, dp: &str, dst: &str, nc: bool) {
    let ld = if nc { "ld.global.nc" } else { "ld.global" };
    let (size, t) = match d.size() {
        2 => (2, "b16"),
        _ => (4, "b32"),
    };
    let a = g.addr(sp, src, size);
    let r = g.reg(t);
    g.op(format!("{ld}.{t} {r}, [{a}]"));
    let b = g.addr(dp, dst, size);
    g.op(format!("st.global.{t} [{b}], {r}"));
}

/// Store the f32 register `v` as element `idx` of a plain-dtype buffer at `ptr`.
pub(crate) fn store(g: &mut Gen, d: DType, ptr: &str, idx: &str, v: &str) {
    store_if(g, d, ptr, idx, v, None)
}

/// `store`, executed only where the predicate register `pred` is true (when given).
pub(crate) fn store_if(g: &mut Gen, d: DType, ptr: &str, idx: &str, v: &str, pred: Option<&str>) {
    let at = pred.map(|p| format!("@{p} ")).unwrap_or_default();
    match d {
        DType::F32 => {
            let a = g.addr(ptr, idx, 4);
            g.op(format!("{at}st.global.f32 [{a}], {v}"));
        }
        DType::F16 => {
            let a = g.addr(ptr, idx, 2);
            let h = g.reg("b16");
            g.op(format!("cvt.rn.f16.f32 {h}, {v}"));
            g.op(format!("{at}st.global.b16 [{a}], {h}"));
        }
        DType::BF16 => {
            // round to nearest even; NaN → canonical quiet NaN
            let a = g.addr(ptr, idx, 2);
            let u = g.reg("b32");
            let t = g.reg("b32");
            let p = g.reg("pred");
            let h = g.reg("b16");
            g.op(format!("mov.b32 {u}, {v}"));
            g.op(format!("shr.u32 {t}, {u}, 16"));
            g.op(format!("and.b32 {t}, {t}, 1"));
            g.op(format!("add.u32 {t}, {t}, 32767"));
            g.op(format!("add.u32 {t}, {t}, {u}"));
            g.op(format!("shr.u32 {t}, {t}, 16"));
            g.op(format!("setp.nan.f32 {p}, {v}, {v}"));
            g.op(format!("selp.b32 {t}, 32704, {t}, {p}"));
            g.op(format!("cvt.u16.u32 {h}, {t}"));
            g.op(format!("{at}st.global.b16 [{a}], {h}"));
        }
        DType::I32 => {
            let a = g.addr(ptr, idx, 4);
            let s = g.reg("s32");
            g.op(format!("cvt.rzi.s32.f32 {s}, {v}"));
            g.op(format!("{at}st.global.s32 [{a}], {s}"));
        }
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 | DType::Iq2Xxs | DType::Iq2S | DType::Iq3Xxs => {}
    }
}

/// Dequantize element `e` (row-major index into the `[rows, cols]` matrix) of a lapid-quant
/// SoA buffer at `ptr` whose scales plane starts at byte `so` (u32 register) into `dst`.
/// K-quant (AoS) buffers ignore `so`.
pub(crate) fn load_quant(g: &mut Gen, d: DType, ptr: &str, so: &str, e: &str, dst: &str) {
    if super::kquant::is_kquant(d) {
        return super::kquant::load_elem(g, d, ptr, e, dst);
    }
    if d == DType::Mxfp4 {
        return super::mxfp4::load_elem(g, ptr, e, dst);
    }
    if d.is_iquant() {
        return super::iquant::load_elem(g, d, ptr, e, dst);
    }
    let q = g.reg("f32");
    if d == DType::Int8 {
        let a = g.addr(ptr, e, 1);
        let s = g.reg("s32");
        g.op(format!("ld.global.s8 {s}, [{a}]"));
        g.op(format!("cvt.rn.f32.s32 {q}, {s}"));
    } else {
        // byte (e / 32) * 16 + (e % 16), nibble (e % 32) / 16
        let r = g.reg("b32");
        let byte = g.reg("b32");
        let sh = g.reg("b32");
        let w = g.reg("b32");
        g.op(format!("and.b32 {r}, {e}, 31"));
        g.op(format!("shr.u32 {byte}, {e}, 5"));
        g.op(format!("shl.b32 {byte}, {byte}, 4"));
        g.op(format!("and.b32 {sh}, {r}, 15"));
        g.op(format!("add.u32 {byte}, {byte}, {sh}"));
        g.op(format!("shr.u32 {sh}, {r}, 4"));
        g.op(format!("shl.b32 {sh}, {sh}, 2"));
        let a = g.addr(ptr, &byte, 1);
        g.op(format!("ld.global.u8 {w}, [{a}]"));
        g.op(format!("shr.u32 {w}, {w}, {sh}"));
        g.op(format!("and.b32 {w}, {w}, 15"));
        g.op(format!("cvt.rn.f32.u32 {q}, {w}"));
        g.op(format!("sub.f32 {q}, {q}, 0f41000000"));
    }
    let gi = g.reg("b32");
    g.op(format!("shr.u32 {gi}, {e}, 5"));
    g.op(format!("mad.lo.u32 {gi}, {gi}, 2, {so}"));
    let a = g.addr(ptr, &gi, 1);
    let h = g.reg("b16");
    let sc = g.reg("f32");
    g.op(format!("ld.global.b16 {h}, [{a}]"));
    g.op(format!("cvt.f32.f16 {sc}, {h}"));
    g.op(format!("mul.f32 {dst}, {q}, {sc}"));
}

/// Statements decomposing `lin` (u32 register) into coordinates over the dims stored at
/// params words `shape_base..shape_base + n` (row-major). Returns the coordinate registers.
pub(crate) fn decompose(g: &mut Gen, lin: &str, shape_base: usize, n: usize) -> Vec<String> {
    let mut coords = vec![String::new(); n];
    if n == 0 {
        return coords;
    }
    let rem = g.reg("u32");
    g.op(format!("mov.u32 {rem}, {lin}"));
    for d in (1..n).rev() {
        let s = g.param(shape_base + d);
        let c = g.reg("u32");
        g.op(format!("rem.u32 {c}, {rem}, {s}"));
        g.op(format!("div.u32 {rem}, {rem}, {s}"));
        coords[d] = c;
    }
    coords[0] = rem;
    coords
}

/// `P[base] + sum(coords[d] * P[base + 1 + d])` into a fresh register (u32, wrapping).
pub(crate) fn strided_index(g: &mut Gen, base: usize, coords: &[String]) -> String {
    let r = g.param(base);
    for (d, c) in coords.iter().enumerate() {
        let s = g.param(base + 1 + d);
        g.op(format!("mad.lo.u32 {r}, {c}, {s}, {r}"));
    }
    r
}

/// Reduction combiner.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Red {
    Sum,
    Max,
}

impl Red {
    pub fn instr(self) -> &'static str {
        match self {
            Red::Sum => "add.f32",
            Red::Max => "max.f32",
        }
    }
}

/// Butterfly reduction of the f32 register `v` across the 32 lanes of a warp (every lane
/// gets the result).
pub(crate) fn warp_reduce(g: &mut Gen, op: Red, v: &str) {
    let t = g.reg("f32");
    for s in [16, 8, 4, 2, 1] {
        g.op(format!("shfl.sync.bfly.b32 {t}, {v}, {s}, 31, -1"));
        g.op(format!("{} {v}, {v}, {t}", op.instr()));
    }
}

/// Reduction of `v` over a 256-thread block (8 warps) through the module-scope shared
/// array `red` (`.shared .align 16 .f32 red[8]`); every thread gets the result. Needs the
/// template registers `%lane` and `%warp`; must run in block-uniform control flow.
pub(crate) fn block_reduce(op: Red, v: &str) -> String {
    let mut g = Gen::new();
    warp_reduce(&mut g, op, v);
    let p = g.reg("pred");
    let a = g.reg("b32");
    g.op(format!("setp.eq.u32 {p}, %lane, 0"));
    g.op(format!("mov.u32 {a}, red"));
    g.op(format!("mad.lo.u32 {a}, %warp, 4, {a}"));
    g.op(format!("@{p} st.shared.f32 [{a}], {v}"));
    g.op("bar.sync 0");
    let r: Vec<String> = (0..8).map(|_| g.reg("f32")).collect();
    g.op(format!(
        "ld.shared.v4.f32 {{{}, {}, {}, {}}}, [red]",
        r[0], r[1], r[2], r[3]
    ));
    g.op(format!(
        "ld.shared.v4.f32 {{{}, {}, {}, {}}}, [red+16]",
        r[4], r[5], r[6], r[7]
    ));
    let i = op.instr();
    g.op(format!("{i} {}, {}, {}", r[0], r[0], r[1]));
    g.op(format!("{i} {}, {}, {}", r[2], r[2], r[3]));
    g.op(format!("{i} {}, {}, {}", r[4], r[4], r[5]));
    g.op(format!("{i} {}, {}, {}", r[6], r[6], r[7]));
    g.op(format!("{i} {}, {}, {}", r[0], r[0], r[2]));
    g.op(format!("{i} {}, {}, {}", r[4], r[4], r[6]));
    g.op(format!("{i} {v}, {}, {}", r[0], r[4]));
    g.op("bar.sync 0");
    g.finish()
}

/// Kernel argument list and pointer setup for buffers named `names` (inputs, output,
/// scratch, ... in launch order): (`.param` lines, statements loading `%g_<name>` as
/// global addresses). The params block `P` (`PARAM_WORDS` u32 words) always comes last.
pub(crate) fn args(names: &[&str]) -> (String, String) {
    let mut sig = String::new();
    let mut ld = String::new();
    for n in names {
        let _ = writeln!(sig, "    .param .u64 A_{n},");
        let _ = writeln!(ld, "    .reg .b64 %g_{n};");
        let _ = writeln!(ld, "    ld.param.u64 %g_{n}, [A_{n}];");
        let _ = writeln!(ld, "    cvta.to.global.u64 %g_{n}, %g_{n};");
    }
    let _ = write!(sig, "    .param .align 4 .b8 P[{}]", 4 * super::PARAM_WORDS);
    (sig, ld)
}

/// Names of the expression inputs `in0..in{n-1}` plus "out".
pub(crate) fn io_names(n: usize) -> Vec<String> {
    let mut v: Vec<String> = (0..n).map(|i| format!("in{i}")).collect();
    v.push("out".to_string());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediates() {
        assert_eq!(fimm(1.0), "0f3F800000");
        assert_eq!(fimm(f32::NEG_INFINITY), "0fFF800000");
    }

    #[test]
    fn gen_scopes() {
        let mut g = Gen::new();
        let r = g.reg("f32");
        g.op(format!("mov.f32 {r}, {}", fimm(2.0)));
        let s = g.finish();
        assert!(s.starts_with("{\n    .reg .f32 %_0;\n"));
        assert!(s.contains("mov.f32 %_0, 0f40000000;"));
    }
}
