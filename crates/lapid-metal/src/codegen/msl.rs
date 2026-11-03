//! Slot fillers shared by all kernels: buffer bindings, element types, float literals,
//! strided index expressions and `Expr` → MSL rendering.

use std::fmt::Write;

use lapid_compiler::{Conv, Expr, ExprType, IntOp, IntUnaryOp};
use lapid_tensor::{BinaryOp, DType, UnaryOp};

/// MSL element type of a buffer holding `d` (see the prelude for the conventions).
pub(crate) fn elem_ty(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "ushort",
        DType::I32 => "int",
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 => "uchar",
    }
}

/// Kernel parameter lines for the bound buffers: `inputs` (name, dtype) at buffer(1..),
/// then the output `out` (when `out` is `Some`), then `extra` raw declarations
/// (e.g. the attention scratch buffer). Every line ends with ",\n" (the builtins follow).
pub(crate) fn bindings(inputs: &[(String, DType)], out: Option<DType>, extra: &[&str]) -> String {
    let mut s = String::new();
    let mut slot = 1;
    for (name, d) in inputs {
        let _ = writeln!(s, "    device const {}* {name} [[buffer({slot})]],", elem_ty(*d));
        slot += 1;
    }
    if let Some(d) = out {
        let _ = writeln!(s, "    device {}* out [[buffer({slot})]],", elem_ty(d));
        slot += 1;
    }
    for decl in extra {
        let _ = writeln!(s, "    {decl} [[buffer({slot})]],");
        slot += 1;
    }
    s
}

/// MSL float literal that round-trips the exact f32 value.
pub(crate) fn flit(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v:?}");
        if v.is_sign_negative() { format!("({s}f)") } else { format!("{s}f") }
    } else {
        format!("as_type<float>({:#010x}u)", v.to_bits())
    }
}

/// `P[base] + c0 * P[base + 1] + ...` (offset plus strided coordinates).
pub(crate) fn strided_index(base: usize, coords: &[String]) -> String {
    let mut s = format!("P[{base}]");
    for (d, c) in coords.iter().enumerate() {
        let _ = write!(s, " + {c} * P[{}]", base + 1 + d);
    }
    s
}

/// Statements decomposing `lin` into coordinates `c0..c{n-1}` over the dims at
/// `P[shape_base..shape_base + n]` (row-major). Returns (code, coordinate names).
pub(crate) fn decompose(lin: &str, shape_base: usize, n: usize, indent: &str) -> (String, Vec<String>) {
    let mut s = String::new();
    let names: Vec<String> = (0..n).map(|d| format!("c{d}")).collect();
    if n == 0 {
        return (s, names);
    }
    if n == 1 {
        let _ = writeln!(s, "{indent}const uint c0 = {lin};");
        return (s, names);
    }
    let _ = writeln!(s, "{indent}uint rem = {lin};");
    for d in (1..n).rev() {
        let _ = writeln!(s, "{indent}const uint c{d} = rem % P[{}];", shape_base + d);
        let _ = writeln!(s, "{indent}rem = rem / P[{}];", shape_base + d);
    }
    let _ = writeln!(s, "{indent}const uint c0 = rem;");
    (s, names)
}

/// Hoisted loads of expression input `i` (buffer `name`, element index `index`): a
/// `const float x{i}` when the expression reads `Load(i)` and a `const uint w{i}` (raw word)
/// when it reads `LoadBits(i)`. Each line is prefixed by `indent`.
pub(crate) fn load_lines(e: &Expr, i: usize, name: &str, index: &str, indent: &str) -> String {
    let mut s = String::new();
    if uses(e, i) {
        let _ = writeln!(s, "{indent}const float x{i} = ldv({name}, {index});");
    }
    if uses_bits(e, i) {
        let _ = writeln!(s, "{indent}const uint w{i} = ldw({name}, {index});");
    }
    s
}

/// Render `e` as an MSL expression: f32 values are `float`, integer words (`ExprType::Int`)
/// are `uint` (signed operations reinterpret with `as_type<int>`). `Load(i)` is the local
/// `x{i}` and `LoadBits(i)` the local `w{i}` (see `load_lines`).
pub(crate) fn render(e: &Expr) -> String {
    match e {
        Expr::Load(i) => format!("x{i}"),
        Expr::LoadBits(i) => format!("w{i}"),
        Expr::Const(bits) => flit(f32::from_bits(*bits)),
        Expr::ConstBits(v) => format!("{v:#x}u"),
        Expr::Unary(op, a) => {
            let a = render(a);
            match op {
                UnaryOp::Neg => format!("(-{a})"),
                UnaryOp::Exp => format!("precise::exp({a})"),
                UnaryOp::Log => format!("precise::log({a})"),
                UnaryOp::Sqrt => format!("precise::sqrt({a})"),
                UnaryOp::Rsqrt => format!("precise::rsqrt({a})"),
                UnaryOp::Recip => format!("(1.0f / {a})"),
                UnaryOp::Sin => format!("precise::sin({a})"),
                UnaryOp::Cos => format!("precise::cos({a})"),
                UnaryOp::Tanh => format!("tanh_({a})"),
                UnaryOp::Erf => format!("erf_({a})"),
                UnaryOp::Abs => format!("fabs({a})"),
            }
        }
        Expr::Binary(op, a, b) => {
            if *op == BinaryOp::Pow
                && let Expr::Const(bits) = **b
            {
                let c = f32::from_bits(bits);
                let ra = render(a);
                if c == 1.0 {
                    return ra;
                } else if c == 2.0 {
                    return format!("({ra} * {ra})");
                } else if c == 3.0 {
                    return format!("({ra} * {ra} * {ra})");
                } else if c == 0.5 {
                    return format!("precise::sqrt({ra})");
                } else if c == -0.5 {
                    return format!("precise::rsqrt({ra})");
                } else if c == -1.0 {
                    return format!("(1.0f / {ra})");
                }
            }
            let ra = render(a);
            let rb = render(b);
            match op {
                BinaryOp::Add => format!("({ra} + {rb})"),
                BinaryOp::Sub => format!("({ra} - {rb})"),
                BinaryOp::Mul => format!("({ra} * {rb})"),
                BinaryOp::Div => format!("({ra} / {rb})"),
                BinaryOp::Max => format!("fmax({ra}, {rb})"),
                BinaryOp::Min => format!("fmin({ra}, {rb})"),
                BinaryOp::Pow => format!("pow_({ra}, {rb})"),
                BinaryOp::CmpLt => format!("select(0.0f, 1.0f, {ra} < {rb})"),
                BinaryOp::CmpEq => format!("select(0.0f, 1.0f, {ra} == {rb})"),
                // Integer-only ops never reach f32 expressions (the compiler lowers them to
                // `IntBinary`); NaN like the CPU VM.
                BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                    flit(f32::NAN)
                }
            }
        }
        Expr::Where(c, a, b) => {
            let rc = render(c);
            let ra = render(a);
            let rb = render(b);
            let zero = if c.ty() == Some(ExprType::Int) { "0u" } else { "0.0f" };
            format!("select({rb}, {ra}, {rc} != {zero})")
        }
        Expr::IntUnary(op, a) => {
            let a = render(a);
            match op {
                IntUnaryOp::Neg => format!("(0u - {a})"),
                IntUnaryOp::Abs => format!("iabs_({a})"),
                IntUnaryOp::Not => format!("(~{a})"),
            }
        }
        Expr::IntBinary(op, a, b) => {
            let ra = render(a);
            let rb = render(b);
            let s = |x: &str| format!("as_type<int>({x})");
            match op {
                IntOp::Add => format!("({ra} + {rb})"),
                IntOp::Sub => format!("({ra} - {rb})"),
                IntOp::Mul => format!("({ra} * {rb})"),
                IntOp::DivS => format!("divs_({ra}, {rb})"),
                IntOp::DivU => format!("divu_({ra}, {rb})"),
                IntOp::RemS => format!("rems_({ra}, {rb})"),
                IntOp::RemU => format!("remu_({ra}, {rb})"),
                IntOp::MinS => format!("as_type<uint>(min({}, {}))", s(&ra), s(&rb)),
                IntOp::MinU => format!("min({ra}, {rb})"),
                IntOp::MaxS => format!("as_type<uint>(max({}, {}))", s(&ra), s(&rb)),
                IntOp::MaxU => format!("max({ra}, {rb})"),
                IntOp::And => format!("({ra} & {rb})"),
                IntOp::Or => format!("({ra} | {rb})"),
                IntOp::Xor => format!("({ra} ^ {rb})"),
                IntOp::Shl => format!("({ra} << ({rb} & 31u))"),
                IntOp::ShrS => format!("as_type<uint>({} >> ({rb} & 31u))", s(&ra)),
                IntOp::ShrU => format!("({ra} >> ({rb} & 31u))"),
                IntOp::CmpLtS => format!("uint({} < {})", s(&ra), s(&rb)),
                IntOp::CmpLtU => format!("uint({ra} < {rb})"),
                IntOp::CmpEq => format!("uint({ra} == {rb})"),
            }
        }
        Expr::Convert(c, a) => {
            let a = render(a);
            match c {
                Conv::F32ToI32 => format!("f2i_({a})"),
                Conv::F32ToU32 => format!("f2u_({a})"),
                Conv::I32ToF32 => format!("float(as_type<int>({a}))"),
                Conv::U32ToF32 => format!("float({a})"),
                Conv::F32Bits => format!("as_type<uint>({a})"),
                Conv::BitsF32 => format!("as_type<float>({a})"),
            }
        }
    }
}

/// Whether `e` reads `Load(i)` (as f32).
pub(crate) fn uses(e: &Expr, i: usize) -> bool {
    reads(e, &|x| matches!(x, Expr::Load(j) if *j == i))
}

/// Whether `e` reads `LoadBits(i)` (raw word).
pub(crate) fn uses_bits(e: &Expr, i: usize) -> bool {
    reads(e, &|x| matches!(x, Expr::LoadBits(j) if *j == i))
}

/// Whether `e` reads input `i` in any form.
pub(crate) fn uses_any(e: &Expr, i: usize) -> bool {
    uses(e, i) || uses_bits(e, i)
}

fn reads(e: &Expr, leaf: &dyn Fn(&Expr) -> bool) -> bool {
    match e {
        Expr::Load(_) | Expr::LoadBits(_) => leaf(e),
        Expr::Const(_) | Expr::ConstBits(_) => false,
        Expr::Unary(_, a) | Expr::IntUnary(_, a) | Expr::Convert(_, a) => reads(a, leaf),
        Expr::Binary(_, a, b) | Expr::IntBinary(_, a, b) => reads(a, leaf) || reads(b, leaf),
        Expr::Where(c, a, b) => reads(c, leaf) || reads(a, leaf) || reads(b, leaf),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_literals() {
        assert_eq!(flit(1.0), "1.0f");
        assert_eq!(flit(-0.5), "(-0.5f)");
        assert_eq!(flit(1e-5), "1e-5f");
        assert_eq!(flit(f32::NEG_INFINITY), "as_type<float>(0xff800000u)");
        for v in [0.1f32, 0.70710677, 3.4028235e38, 1.1754944e-38, 123456.79] {
            let s = flit(v);
            let parsed: f32 = s.trim_end_matches('f').parse().unwrap();
            assert_eq!(parsed.to_bits(), v.to_bits(), "{s}");
        }
    }

    #[test]
    fn renders_gelu() {
        let x = || Box::new(Expr::Load(0));
        let e = Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(0.5)))),
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::konst(1.0)),
                Box::new(Expr::Unary(
                    UnaryOp::Erf,
                    Box::new(Expr::Binary(BinaryOp::Mul, x(), Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)))),
                )),
            )),
        );
        let s = render(&e);
        assert_eq!(s, "((x0 * 0.5f) * (1.0f + erf_((x0 * 0.70710677f))))");
        assert!(uses(&e, 0));
        assert!(!uses(&e, 1));
    }

    #[test]
    fn decompose_rank3() {
        let (code, names) = decompose("i", 1, 3, "");
        assert_eq!(names, ["c0", "c1", "c2"]);
        assert_eq!(
            code,
            "uint rem = i;\nconst uint c2 = rem % P[3];\nrem = rem / P[3];\nconst uint c1 = rem % P[2];\nrem = rem / P[2];\nconst uint c0 = rem;\n"
        );
        assert_eq!(strided_index(4, &names), "P[4] + c0 * P[5] + c1 * P[6] + c2 * P[7]");
    }

    #[test]
    fn binding_lines() {
        let s = bindings(&[("a".into(), DType::F16), ("b".into(), DType::Int4)], Some(DType::BF16), &["device float* scratch"]);
        assert_eq!(
            s,
            "    device const half* a [[buffer(1)]],\n    device const uchar* b [[buffer(2)]],\n    device ushort* out [[buffer(3)]],\n    device float* scratch [[buffer(4)]],\n"
        );
    }
}
