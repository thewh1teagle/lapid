//! Slot fillers shared by all kernels: typed buffer bindings with load/store helpers,
//! float literals and strided index expressions. The fixed WGSL (prelude, entry point,
//! workgroup reductions) lives in `src/kernels/*.wgsl`.

use std::fmt::Write;

use lapid_tensor::DType;

use crate::Caps;

/// WGSL float literal that round-trips the exact f32 value.
pub fn flit(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v:?}");
        if v.is_sign_negative() { format!("({s}f)") } else { format!("{s}f") }
    } else {
        format!("bitcast<f32>({:#010x}u)", v.to_bits())
    }
}

pub const NEG_INF: &str = "bitcast<f32>(0xff800000u)";

/// Storage element type used for a plain dtype.
fn elem_ty(d: DType, caps: &Caps) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::F16 if caps.f16 => "f16",
        _ => "u32",
    }
}

/// Read-only input binding `name` with `fn ld_{name}(e: u32) -> f32` (e in elements).
/// Quantized dtypes get `{name}_q`, `{name}_scale`, `{name}_deq` instead (see `quant_fns`).
pub fn input(binding: usize, name: &str, d: DType, caps: &Caps) -> String {
    let mut s = String::new();
    let ty = elem_ty(d, caps);
    let _ = writeln!(s, "@group(0) @binding({binding}) var<storage, read> {name}: array<{ty}>;");
    let body = match d {
        DType::F32 => format!("return {name}[e];"),
        DType::I32 => format!("return f32({name}[e]);"),
        DType::F16 if caps.f16 => format!("return f32({name}[e]);"),
        DType::F16 => format!("return unpack2x16float({name}[e >> 1u])[e & 1u];"),
        DType::BF16 => format!(
            "let w = {name}[e >> 1u];\n    return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (e & 1u) == 1u));"
        ),
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => return s + &quant_fns(name, d),
    };
    let _ = writeln!(s, "fn ld_{name}(e: u32) -> f32 {{\n    {body}\n}}");
    s
}

/// `fn ldb_{name}(e: u32) -> u32`: the raw 32-bit word of element `e` of a 32-bit input
/// declared by `input` (`Expr::LoadBits`).
pub fn input_bits(name: &str, d: DType) -> String {
    match d {
        DType::F32 | DType::I32 => format!("fn ldb_{name}(e: u32) -> u32 {{\n    return bitcast<u32>({name}[e]);\n}}\n"),
        _ => format!("fn ldb_{name}(e: u32) -> u32 {{\n    return bitcast<u32>(ld_{name}(e));\n}}\n"),
    }
}

/// Dequantization helpers over a `array<u32>` binding named `name` holding a lapid-quant
/// SoA matrix. `e` is the row-major element index (r * cols + c); group index = e / 32;
/// `so` is the scales-plane offset in u32 words.
/// K-quant dtypes get the AoS block helpers of `kquant::fns` (`{name}_deq`, `ld_{name}`).
pub fn quant_fns(name: &str, d: DType) -> String {
    if crate::kquant::is_k(d) {
        return crate::kquant::fns(name, d);
    }
    let q = match d {
        DType::Int8 => format!(
            "fn {name}_q(e: u32) -> f32 {{\n    let w = {name}[e >> 2u];\n    return f32(extractBits(bitcast<i32>(w), (e & 3u) * 8u, 8u));\n}}\n"
        ),
        _ => format!(
            "fn {name}_q(e: u32) -> f32 {{\n    let r = e & 31u;\n    let byte = (e >> 5u) * 16u + (r & 15u);\n    let w = {name}[byte >> 2u];\n    let sh = (byte & 3u) * 8u + (r >> 4u) * 4u;\n    return f32((w >> sh) & 15u) - 8.0;\n}}\n"
        ),
    };
    format!(
        "{q}fn {name}_scale(so: u32, gi: u32) -> f32 {{\n    return unpack2x16float({name}[so + (gi >> 1u)])[gi & 1u];\n}}\n\
         fn {name}_deq(so: u32, e: u32) -> f32 {{\n    return {name}_q(e) * {name}_scale(so, e >> 5u);\n}}\n"
    )
}

/// Output binding `out` with `fn st(e: u32, v: f32)`. 16-bit outputs without native
/// 16-bit storage (and bf16 always) are written with atomic and/or on the containing word,
/// which is correct for any (also strided / odd-length) output layout.
pub fn output(binding: usize, d: DType, caps: &Caps) -> String {
    let mut s = String::new();
    match d {
        DType::F32 => {
            let _ = writeln!(s, "@group(0) @binding({binding}) var<storage, read_write> out: array<f32>;");
            s.push_str("fn st(e: u32, v: f32) {\n    out[e] = v;\n}\n");
        }
        DType::I32 => {
            // f32 values convert like Rust `as` (saturating, NaN -> 0); `st_bits` stores a raw word
            let _ = writeln!(s, "@group(0) @binding({binding}) var<storage, read_write> out: array<i32>;");
            s.push_str("fn st(e: u32, v: f32) {\n    out[e] = bitcast<i32>(gi_f2i(v));\n}\n");
            s.push_str("fn st_bits(e: u32, v: u32) {\n    out[e] = bitcast<i32>(v);\n}\n");
        }
        DType::F16 if caps.f16 => {
            let _ = writeln!(s, "@group(0) @binding({binding}) var<storage, read_write> out: array<f16>;");
            s.push_str("fn st(e: u32, v: f32) {\n    out[e] = f16(v);\n}\n");
        }
        _ => {
            let _ = writeln!(s, "@group(0) @binding({binding}) var<storage, read_write> out: array<atomic<u32>>;");
            s.push_str(
                "fn st16(e: u32, h: u32) {\n    let sh = (e & 1u) * 16u;\n    let w = e >> 1u;\n    atomicAnd(&out[w], ~(0xffffu << sh));\n    atomicOr(&out[w], (h & 0xffffu) << sh);\n}\n",
            );
            if d == DType::BF16 {
                s.push_str(
                    "fn st(e: u32, v: f32) {\n    let bits = bitcast<u32>(v);\n    var h = (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;\n    if (v != v) {\n        h = 0x7fc0u;\n    }\n    st16(e, h);\n}\n",
                );
            } else {
                s.push_str("fn st(e: u32, v: f32) {\n    st16(e, pack2x16float(vec2<f32>(v, 0.0)));\n}\n");
            }
        }
    }
    s
}

/// Expression for `P[base] + c0 * P[base + 1] + ...` (strided index over `coords`).
pub fn strided_index(base: usize, coords: &[String]) -> String {
    let mut s = format!("P[{base}u]");
    for (d, c) in coords.iter().enumerate() {
        let _ = write!(s, " + {c} * P[{}u]", base + 1 + d);
    }
    s
}

/// Statements decomposing `lin` into coordinates `c0..c{n-1}` over dims at
/// `P[shape_base..shape_base + n]` (row-major). Returns (code, coord names).
pub fn decompose(lin: &str, shape_base: usize, n: usize, indent: &str) -> (String, Vec<String>) {
    let mut s = String::new();
    let names: Vec<String> = (0..n).map(|d| format!("c{d}")).collect();
    if n == 0 {
        return (s, names);
    }
    let _ = writeln!(s, "{indent}var rem = {lin};");
    for d in (1..n).rev() {
        let _ = writeln!(s, "{indent}let c{d} = rem % P[{}u];", shape_base + d);
        let _ = writeln!(s, "{indent}rem = rem / P[{}u];", shape_base + d);
    }
    let _ = writeln!(s, "{indent}let c0 = rem;");
    (s, names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_literals() {
        assert_eq!(flit(1.0), "1.0f");
        assert_eq!(flit(-0.5), "(-0.5f)");
        assert_eq!(flit(1e-5), "1e-5f");
        assert_eq!(flit(f32::NEG_INFINITY), "bitcast<f32>(0xff800000u)");
    }
}
