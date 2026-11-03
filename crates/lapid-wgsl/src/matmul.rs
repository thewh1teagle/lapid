//! MatMul planning (GEMV vs tiled GEMM), the tiled GEMM kernel and the shared epilogue.
//!
//! Params layout (both variants):
//! `[batch, m, n, k, a.off, a.bs, a.rs, b.off, b.bs, b.rs, scales_words, P11, P12,
//!   out.off, out.bs, out.rs, extra_e(off, s_batch, s_m, s_n)...]` with P11/P12 =
//! tiles_n/tiles_m (GEMM) or column blocks (GEMV). The output element (bt, i, j) is at
//! `out.off + bt * out.bs + i * out.rs + j` (contiguous or a slice of a larger buffer).

use std::fmt::Write;
use std::sync::Arc;

use lapid_compiler::{ExprType, MatMulDims, MatMulSpec};
use lapid_tensor::{DType, Result};

use super::{common, template};
use super::expr::{self, render, uses_bits, uses_load};
use super::gather::scales_words;
use super::gemv::{gemv_cols, gemv_nc, vec_variants};
use super::{Key, Params, Plan, bad, check_plain, check_quant_supported, grid};
use crate::Caps;

const GEMM: &str = include_str!("kernels/gemm.wgsl");
const GEMM_COOP: &str = include_str!("kernels/gemm_coop.wgsl");

pub(super) const EXTRA_BASE: usize = 16;
const TILE: usize = 64;
/// Largest m served by the GEMV kernel.
const GEMV_MAX_M: usize = 4;

pub(super) fn plan(spec: &Arc<MatMulSpec>, d: &MatMulDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.a_dtype, "matmul A")?;
    check_plain(spec.out_dtype, "matmul output")?;
    check_quant_supported(spec.b_dtype, "matmul B")?;
    for &t in &spec.extra_dtypes {
        check_plain(t, "matmul epilogue input")?;
    }
    if d.extra_views.len() != spec.extra_dtypes.len() {
        return Err(bad("matmul: extra_views / extra_dtypes length mismatch"));
    }
    if let Some(e) = &spec.epilogue {
        // Load(0) = accumulator (f32), Load(1 + e) = extras
        let mut dts = vec![DType::F32];
        dts.extend_from_slice(&spec.extra_dtypes);
        expr::check(e, &dts, Some(ExprType::F32), "matmul epilogue")?;
    }
    let quant = spec.b_dtype.is_quant();
    if quant && !spec.b_transposed {
        return Err(bad("matmul: quantized B must be transposed"));
    }
    let block = spec.b_dtype.quant_format().map(|f| f.block_elems()).unwrap_or(1);
    if quant && !d.k.is_multiple_of(block) {
        return Err(bad(format!("matmul: {} B needs k % {block} == 0 (k = {})", spec.b_dtype.name(), d.k)));
    }
    let gemv = spec.b_transposed && d.m <= GEMV_MAX_M;
    let so = if quant { scales_words(spec.b_dtype, d.n, d.k) } else { 0 };
    let (p11, p12, items) = if gemv {
        let cb = d.n.div_ceil(gemv_cols(caps) * gemv_nc(spec));
        (cb, 0, (d.batch * d.m * cb) as u64)
    } else if coop_ok(spec, caps) {
        let tn = d.n.div_ceil(COOP_BN);
        let tm = d.m.div_ceil(COOP_BM);
        (tn, tm, (d.batch * tm * tn) as u64)
    } else {
        let tn = d.n.div_ceil(TILE);
        let tm = d.m.div_ceil(TILE);
        (tn, tm, (d.batch * tm * tn) as u64)
    };
    let mut p = Params::default();
    p.u(d.batch)?.u(d.m)?.u(d.n)?.u(d.k)?;
    p.u(d.a.offset)?.u(d.a.batch_stride)?.u(d.a.row_stride)?;
    p.u(d.b.offset)?.u(d.b.batch_stride)?.u(d.b.row_stride)?;
    p.u(so)?.u(p11)?.u(p12)?;
    p.u(d.out.offset)?.u(d.out.batch_stride)?.u(d.out.row_stride)?;
    debug_assert_eq!(p.0.len(), EXTRA_BASE);
    for v in &d.extra_views {
        p.view(v, 3)?;
    }
    let key = if gemv {
        let (va, vb) = vec_variants(spec, d);
        Key::Gemv(spec.clone(), va, vb)
    } else if coop_ok(spec, caps) {
        Key::GemmCoop(spec.clone())
    } else {
        Key::Gemm(spec.clone())
    };
    Ok(Plan { scratch: 0, next: None, key, params: p.0, groups: grid(items, caps) })
}

/// Bindings for epilogue extras (`ld_x{e}`) and the output.
pub(super) fn epilogue_bindings(spec: &MatMulSpec, caps: &Caps) -> String {
    let mut s = String::new();
    for (e, &t) in spec.extra_dtypes.iter().enumerate() {
        s.push_str(&common::input(3 + e, &format!("x{e}"), t, caps));
        if spec.epilogue.as_ref().is_some_and(|x| uses_bits(x, 1 + e)) {
            s.push_str(&common::input_bits(&format!("x{e}"), t));
        }
    }
    s.push_str(&common::output(3 + spec.extra_dtypes.len(), spec.out_dtype, caps));
    s
}

/// Store of one output element. Expects `acc`, `bt`, `i`, `j` in scope.
pub(super) fn epilogue(spec: &MatMulSpec, indent: &str) -> String {
    let mut s = String::new();
    let value = match &spec.epilogue {
        None => "acc".to_string(),
        Some(e) => {
            for x in 0..spec.extra_dtypes.len() {
                let b = EXTRA_BASE + 4 * x;
                let idx = format!("P[{b}u] + bt * P[{}u] + i * P[{}u] + j * P[{}u]", b + 1, b + 2, b + 3);
                if uses_load(e, 1 + x) {
                    let _ = writeln!(s, "{indent}let e{x} = ld_x{x}({idx});");
                }
                if uses_bits(e, 1 + x) {
                    let _ = writeln!(s, "{indent}let eb{x} = ldb_x{x}({idx});");
                }
            }
            render(e, &|l, bits| match (l, bits) {
                (0, false) => "acc".to_string(),
                (0, true) => "bitcast<u32>(acc)".to_string(),
                (l, false) => format!("e{}", l - 1),
                (l, true) => format!("eb{}", l - 1),
            })
        }
    };
    let _ = writeln!(s, "{indent}st(P[13u] + bt * P[14u] + i * P[15u] + j, {value});");
    s
}

fn b_tile(spec: &MatMulSpec) -> String {
    match (spec.b_dtype, spec.b_transposed) {
        (d @ (DType::Q4K | DType::Q5K | DType::Q6K), _) => format!(
            r#"            {{
                // 8 consecutive k of one row (K % 256 == 0: the tile never crosses K)
                let gj = col0 + lr;
                var v0 = vec4<f32>(0.0);
                var v1 = vec4<f32>(0.0);
                if (gj < N && k0 + lk < K) {{
                    let kk = k0 + lk;
                    let bb = (gj * (K / 256u) + kk / 256u) * {bb}u;
                    v0 = b_kdeq4(bb, kk % 256u);
                    v1 = b_kdeq4(bb, kk % 256u + 4u);
                }}
                for (var e = 0u; e < 4u; e += 1u) {{
                    Bs[(lk + e) * 64u + lr] = v0[e];
                    Bs[(lk + 4u + e) * 64u + lr] = v1[e];
                }}
            }}
"#,
            bb = crate::kquant::block_bytes(d)
        ),
        (DType::Int8, _) | (DType::Int4, _) => {
            let i8 = spec.b_dtype == DType::Int8;
            let wi = if i8 { "(gj * K + k0) / 4u + part * 2u" } else { "(gj * K + k0) / 8u + (part % 2u) * 2u" };
            let fill = if i8 {
                r#"                for (var e = 0u; e < 4u; e += 1u) {
                    Bs[(part * 8u + e) * 64u + lr] = f32(extractBits(bitcast<i32>(w0), e * 8u, 8u)) * dsc;
                    Bs[(part * 8u + 4u + e) * 64u + lr] = f32(extractBits(bitcast<i32>(w1), e * 8u, 8u)) * dsc;
                }
"#
            } else {
                r#"                let sh = (part / 2u) * 4u;
                for (var e = 0u; e < 4u; e += 1u) {
                    Bs[(part * 8u + e) * 64u + lr] = (f32((w0 >> (e * 8u + sh)) & 15u) - 8.0) * dsc;
                    Bs[(part * 8u + 4u + e) * 64u + lr] = (f32((w1 >> (e * 8u + sh)) & 15u) - 8.0) * dsc;
                }
"#
            };
            format!(
                r#"            {{
                let gj = col0 + lr;
                let part = g_lid % 4u;
                var dsc: f32 = 0.0;
                var w0: u32 = 0u;
                var w1: u32 = 0u;
                if (gj < N) {{
                    dsc = b_scale(P[10], gj * (K / 32u) + k0 / 32u);
                    let wi = {wi};
                    w0 = b[wi];
                    w1 = b[wi + 1u];
                }}
{fill}            }}
"#
            )
        }
        (_, true) => r#"            {
                let gj = col0 + lr;
                let brow = b_base + gj * P[9];
                for (var e = 0u; e < 8u; e += 1u) {
                    let kk = k0 + lk + e;
                    var v: f32 = 0.0;
                    if (gj < N && kk < K) {
                        v = ld_b(brow + kk);
                    }
                    Bs[(lk + e) * 64u + lr] = v;
                }
            }
"#
        .to_string(),
        (_, false) => r#"            {
                let kr = g_lid / 8u;
                let jc = (g_lid % 8u) * 8u;
                let kk = k0 + kr;
                let brow = b_base + kk * P[9];
                for (var e = 0u; e < 8u; e += 1u) {
                    let gj = col0 + jc + e;
                    var v: f32 = 0.0;
                    if (kk < K && gj < N) {
                        v = ld_b(brow + gj);
                    }
                    Bs[kr * 64u + jc + e] = v;
                }
            }
"#
        .to_string(),
    }
}

/// B binding for the GEMM kernel (`b` + helpers).
fn b_binding(spec: &MatMulSpec, caps: &Caps) -> String {
    if spec.b_dtype.is_quant() {
        // `common::quant_fns` provides b_q / b_scale / b_deq over this binding.
        let mut s = String::from("@group(0) @binding(2) var<storage, read> b: array<u32>;\n");
        s.push_str(&common::quant_fns("b", spec.b_dtype));
        s
    } else {
        common::input(2, "b", spec.b_dtype, caps)
    }
}

/// Scalar tiled GEMM (any dtypes / devices).
pub(super) fn source(spec: &MatMulSpec, caps: &Caps) -> String {
    let mut bindings = common::input(1, "a", spec.a_dtype, caps);
    bindings.push_str(&b_binding(spec, caps));
    bindings.push_str(&epilogue_bindings(spec, caps));
    let epi = epilogue(spec, "                        ");
    template::module(caps, GEMM, &[("BINDINGS", &bindings), ("B_TILE", &b_tile(spec)), ("EPILOGUE", &epi)], &[], 256, false)
}

/// Whether the cooperative-matrix GEMM serves `spec` on this device: f16 or quantized
/// weights (f32 weights keep the exact f32 path), plain A.
pub(super) fn coop_ok(spec: &MatMulSpec, caps: &Caps) -> bool {
    caps.coopmat
        && matches!(spec.b_dtype, DType::F16 | DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K)
        && spec.a_dtype != DType::BF16
}

/// (B_REGS, B_LOAD, B_STORE) slots of the coopmat GEMM: B tile registers, global →
/// registers for k-step `k0`, registers → `Bs[j * LDS + kk]` (f16, dequantized).
fn b_coop_slots(spec: &MatMulSpec) -> (String, String, String) {
    let per = |div: usize| COOP_BN * COOP_BK / div / 256;
    match (spec.b_dtype, spec.b_transposed) {
        (d @ (DType::Q4K | DType::Q5K | DType::Q6K), _) => (
            // thread t: row t / 2 of the tile, 16 consecutive k (half t % 2 of the 32-wide step)
            "var<private> rbk: array<vec4<f32>, 4>;\n".to_string(),
            format!(
                r#"    {{
        let gj = col0 + g_lid / 2u;
        let kk = k0 + (g_lid % 2u) * 16u;
        for (var u = 0u; u < 4u; u += 1u) {{
            rbk[u] = vec4<f32>(0.0);
        }}
        if (gj < N && kk < K) {{
            let bb = (gj * (K / 256u) + kk / 256u) * {bb}u;
            rbk = b_kdeq16(bb, kk % 256u);
        }}
    }}
"#,
                bb = crate::kquant::block_bytes(d)
            ),
            r#"    {
        let o = (g_lid / 2u) * LDS + (g_lid % 2u) * 16u;
        for (var u = 0u; u < 4u; u += 1u) {
            for (var x = 0u; x < 4u; x += 1u) {
                Bs[o + 4u * u + x] = f16(rbk[u][x]);
            }
        }
    }
"#
            .to_string(),
        ),
        (DType::Int8, _) => (
            format!("const B_PER: u32 = {}u;\nvar<private> rbw: array<u32, B_PER>;\nvar<private> rbs: array<f32, B_PER>;\n", per(4)),
            r#"    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let gj = col0 + e / (BK / 4u);
        let gk = k0 + (e % (BK / 4u)) * 4u;
        var dsc: f32 = 0.0;
        var word: u32 = 0u;
        if (gj < N && gk < K) {
            dsc = b_scale(P[10], (gj * K + gk) / 32u);
            word = b[(gj * K + gk) / 4u];
        }
        rbw[t] = word;
        rbs[t] = dsc;
    }
"#
            .to_string(),
            r#"    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let o = (e / (BK / 4u)) * LDS + (e % (BK / 4u)) * 4u;
        let q = vec4<f32>(unpack4xI8(rbw[t])) * rbs[t];
        Bs[o] = f16(q.x);
        Bs[o + 1u] = f16(q.y);
        Bs[o + 2u] = f16(q.z);
        Bs[o + 3u] = f16(q.w);
    }
"#
            .to_string(),
        ),
        (DType::Int4, _) => (
            format!("const B_PER: u32 = {}u;\nvar<private> rbw: array<u32, B_PER>;\nvar<private> rbs: array<f32, B_PER>;\n", per(8)),
            r#"    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let gj = col0 + e / (BK / 8u);
        let w = e % (BK / 8u);
        let gk = k0 + (w / 4u) * 32u;
        var dsc: f32 = 0.0;
        var word: u32 = 0x88888888u;
        if (gj < N && gk < K) {
            dsc = b_scale(P[10], (gj * K + gk) / 32u);
            word = b[(gj * K + k0) / 8u + w];
        }
        rbw[t] = word;
        rbs[t] = dsc;
    }
"#
            .to_string(),
            r#"    for (var t = 0u; t < B_PER; t += 1u) {
        let e = g_lid + t * 256u;
        let w = e % (BK / 8u);
        let o = (e / (BK / 8u)) * LDS + (w / 4u) * 32u + (w % 4u) * 4u;
        let word = rbw[t];
        let dsc = rbs[t];
        for (var i = 0u; i < 4u; i += 1u) {
            Bs[o + i] = f16((f32((word >> (i * 8u)) & 15u) - 8.0) * dsc);
            Bs[o + 16u + i] = f16((f32((word >> (i * 8u + 4u)) & 15u) - 8.0) * dsc);
        }
    }
"#
            .to_string(),
        ),
        (_, bt) => {
            let (c, kk, idx) = if bt {
                ("e / BK", "e % BK", "b_base + gj * P[9] + gk")
            } else {
                ("e % BN", "e / BN", "b_base + gk * P[9] + gj")
            };
            (
                format!("const B_PER: u32 = {}u;\nvar<private> rb: array<f32, B_PER>;\n", per(1)),
                format!(
                    r#"    for (var t = 0u; t < B_PER; t += 1u) {{
        let e = g_lid + t * 256u;
        let gj = col0 + {c};
        let gk = k0 + {kk};
        var v: f32 = 0.0;
        if (gj < N && gk < K) {{
            v = ld_b({idx});
        }}
        rb[t] = v;
    }}
"#
                ),
                format!(
                    r#"    for (var t = 0u; t < B_PER; t += 1u) {{
        let e = g_lid + t * 256u;
        Bs[({c}) * LDS + {kk}] = f16(rb[t]);
    }}
"#
                ),
            )
        }
    }
}

/// Grouped MoE GEMM (`Key::GemmIdCoop`): the coopmat GEMM with the `ID` flag; quantized
/// weights are addressed from the expert's base word `g_eb`.
pub(super) fn source_coop_id(spec: &lapid_compiler::MatMulIdSpec, caps: &Caps) -> String {
    let ms = MatMulSpec { a_dtype: spec.x_dtype, b_dtype: spec.w_dtype, b_transposed: true, extra_dtypes: vec![], out_dtype: spec.out_dtype, epilogue: None };
    let mut bindings = common::input(1, "a", ms.a_dtype, caps);
    bindings.push_str(&b_binding(&ms, caps));
    bindings.push_str(&common::output(4, ms.out_dtype, caps));
    let (bm, bn) = (COOP_BM.to_string(), COOP_BN.to_string());
    let (as_size, bs_size) = ((COOP_BM * (COOP_BK + 8)).to_string(), (COOP_BN * (COOP_BK + 8)).to_string());
    let bk = COOP_BK.to_string();
    let (b_regs, b_load, b_store) = b_coop_slots(&ms);
    let b_load = if ms.b_dtype.is_quant() {
        b_load.replace("let bb = (", "let bb = g_eb * 4u + (").replace("b_scale(P[10]", "b_scale(g_eb + P[10]").replace("b[(gj * K", "b[g_eb + (gj * K")
    } else {
        b_load
    };
    template::module(
        caps,
        GEMM_COOP,
        &[
            ("BINDINGS", &bindings),
            ("BM", &bm),
            ("BN", &bn),
            ("AS_SIZE", &as_size),
            ("BS_SIZE", &bs_size),
            ("BK", &bk),
            ("B_REGS", &b_regs),
            ("B_LOAD", &b_load),
            ("B_STORE", &b_store),
            ("EPILOGUE", ""),
        ],
        &["COOPMAT", "ID"],
        256,
        true,
    )
}

/// Workgroup tile (BM, BN) of the coopmat GEMM; (BM / 32) * (BN / 32) == 8 subgroups.
pub(super) const COOP_BM: usize = 64;
pub(super) const COOP_BN: usize = 128;
const COOP_BK: usize = 32;

pub(super) fn source_coop(spec: &MatMulSpec, caps: &Caps) -> String {
    let mut bindings = common::input(1, "a", spec.a_dtype, caps);
    bindings.push_str(&b_binding(spec, caps));
    bindings.push_str(&epilogue_bindings(spec, caps));
    let epi = epilogue(spec, "                    ");
    let (bm, bn) = (COOP_BM.to_string(), COOP_BN.to_string());
    let (as_size, bs_size) = ((COOP_BM * (COOP_BK + 8)).to_string(), (COOP_BN * (COOP_BK + 8)).to_string());
    let bk = COOP_BK.to_string();
    let (b_regs, b_load, b_store) = b_coop_slots(spec);
    template::module(
        caps,
        GEMM_COOP,
        &[
            ("BINDINGS", &bindings),
            ("BM", &bm),
            ("BN", &bn),
            ("AS_SIZE", &as_size),
            ("BS_SIZE", &bs_size),
            ("BK", &bk),
            ("B_REGS", &b_regs),
            ("B_LOAD", &b_load),
            ("B_STORE", &b_store),
            ("EPILOGUE", &epi),
        ],
        &["COOPMAT"],
        256,
        true,
    )
}
