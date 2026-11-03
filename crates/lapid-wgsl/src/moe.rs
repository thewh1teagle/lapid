//! `Kernel::MatMulId` (mixture-of-experts matmul, ggml MUL_MAT_ID): template
//! `kernels/matmul_id.wgsl`. Expert weights are float `[E, N, K]` or E quantized `[N, K]`
//! matrices back to back (`lapid_quant::layout(..).total` bytes each).

use std::sync::Arc;

use lapid_compiler::{MatMulIdDims, MatMulIdSpec};
use lapid_tensor::{DType, Error, Result};

use super::gather::scales_words;
use super::gemv::gemv_cols;
use super::{Key, Params, Plan, bad, check_plain, check_quant_supported, common, grid, kquant, template};
use crate::Caps;

const KERNEL: &str = include_str!("kernels/matmul_id.wgsl");

pub(super) fn plan(spec: &Arc<MatMulIdSpec>, d: &MatMulIdDims, caps: &Caps) -> Result<Plan> {
    check_plain(spec.x_dtype, "matmul_id x")?;
    check_plain(spec.out_dtype, "matmul_id output")?;
    check_quant_supported(spec.w_dtype, "matmul_id experts")?;
    let (so, stride) = match spec.w_dtype.quant_format() {
        Some(f) => {
            if !d.k.is_multiple_of(f.block_elems()) {
                return Err(bad(format!("matmul_id: {} experts need k % {} == 0", spec.w_dtype.name(), f.block_elems())));
            }
            let total = lapid_quant::layout(f, d.n, d.k).total;
            (scales_words(spec.w_dtype, d.n, d.k), total / 4)
        }
        None => (0, d.n.checked_mul(d.k).ok_or_else(|| bad("matmul_id: size overflow"))?),
    };
    if d.e == 0 && d.t * d.r > 0 {
        return Err(Error::new("index_out_of_range", "matmul_id: no experts"));
    }
    if grouped(spec, d, caps) {
        return plan_grouped(spec, d, so, stride, caps);
    }
    let ncb = d.n.div_ceil(gemv_cols(caps));
    let mut p = Params::default();
    p.u(d.t)?.u(d.r)?.u(d.n)?.u(d.k)?.u(so)?.u(stride)?.u(ncb)?.u(d.e)?;
    // expert offsets (x * stride) must fit 32-bit indexing
    p.u(d.e.checked_mul(stride).ok_or_else(|| bad("matmul_id: size overflow"))?)?;
    let items = (d.t * d.r * ncb) as u64;
    Ok(Plan { scratch: 0, next: None, key: Key::MatMulId(spec.clone()), params: p.0, groups: grid(items, caps) })
}

/// Largest expert count of the grouped (prefill) path (shared counters of `moe_route`).
const ROUTE_EMAX: usize = 4096;

/// Many tokens on a cooperative-matrix device: route pairs by expert, then one grouped GEMM
/// (each expert's weights read once per 64-row tile instead of once per pair).
fn grouped(spec: &MatMulIdSpec, d: &MatMulIdDims, caps: &Caps) -> bool {
    let w = spec.w_dtype;
    // at least ~4 rows per expert on average (decode's down projection, t = 8 pairs with
    // r = 1, stays on the per-pair kernel)
    caps.coopmat
        && d.t > 4
        && d.t * d.r >= 4 * d.e
        && d.e <= ROUTE_EMAX
        && matches!(w, DType::F16 | DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K)
        && spec.x_dtype != DType::BF16
        && (w != DType::F16 || caps.f16 || (d.n * d.k).is_multiple_of(2))
}

fn plan_grouped(spec: &Arc<MatMulIdSpec>, d: &MatMulIdDims, so: usize, stride: usize, caps: &Caps) -> Result<Plan> {
    use super::matmul::{COOP_BM, COOP_BN};
    let tr = d.t * d.r;
    let emax = d.e.next_power_of_two().max(64) as u32;
    let mut rp = Params::default();
    rp.u(tr)?.u(d.e)?.u(COOP_BM)?;
    // b.bs: expert stride in elements of the `b` array (u32 words when quantized or packed f16)
    let b_stride = match spec.w_dtype {
        DType::F16 if !caps.f16 => stride / 2,
        _ => stride,
    };
    let slots = tr.div_ceil(COOP_BM) + d.e.min(tr);
    let tn = d.n.div_ceil(COOP_BN);
    let mut p = Params::default();
    p.u(1)?.u(tr)?.u(d.n)?.u(d.k)?;
    p.u(0)?.u(0)?.u(d.k)?;
    p.u(0)?.u(b_stride)?.u(d.k)?;
    p.u(so)?.u(tn)?.u(slots)?;
    p.u(0)?.u(0)?.u(d.n)?;
    p.u(d.r)?.u(d.e)?;
    let gemm = Plan { key: Key::GemmIdCoop(spec.clone()), params: p.0, groups: grid((slots * tn) as u64, caps), scratch: 0, next: None };
    Ok(Plan {
        key: Key::MoeRoute(emax),
        params: rp.0,
        groups: [1, 1, 1],
        scratch: (2 * d.e + 2 + tr) * 4,
        next: Some(Box::new(gemm)),
    })
}

const ROUTE: &str = include_str!("kernels/moe_route.wgsl");

pub(super) fn source_route(emax: u32, caps: &Caps) -> String {
    template::module(caps, ROUTE, &[("EMAX", &emax.to_string())], &[], 256, false)
}

/// `fn wdeq(x, e)`: element `e` (row-major in the expert's [N, K]) of expert `x`.
fn w_loader(d: DType, caps: &Caps) -> String {
    match d {
        DType::Int8 => "@group(0) @binding(2) var<storage, read> w: array<u32>;\n\
            fn wdeq(x: u32, e: u32) -> f32 {\n    let wb = x * P[5];\n    \
            let q = extractBits(bitcast<i32>(w[wb + (e >> 2u)]), (e & 3u) * 8u, 8u);\n    \
            return f32(q) * unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];\n}\n"
            .to_string(),
        DType::Int4 => "@group(0) @binding(2) var<storage, read> w: array<u32>;\n\
            fn wdeq(x: u32, e: u32) -> f32 {\n    let wb = x * P[5];\n    \
            let r = e & 31u;\n    let byte = (e >> 5u) * 16u + (r & 15u);\n    \
            let q = f32((w[wb + (byte >> 2u)] >> ((byte & 3u) * 8u + (r >> 4u) * 4u)) & 15u) - 8.0;\n    \
            return q * unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];\n}\n"
            .to_string(),
        d if kquant::is_k(d) => {
            let bb = kquant::block_bytes(d);
            format!(
                "@group(0) @binding(2) var<storage, read> w: array<u32>;\n{}\
                 fn wdeq(x: u32, e: u32) -> f32 {{\n    return w_kdeq(x * P[5] * 4u + (e >> 8u) * {bb}u, e & 255u);\n}}\n",
                kquant::fns_dot("w", d)
            )
        }
        d => format!("{}fn wdeq(x: u32, e: u32) -> f32 {{\n    return ld_w(x * P[5] + e);\n}}\n", common::input(2, "w", d, caps)),
    }
}

pub(super) fn source(spec: &MatMulIdSpec, caps: &Caps) -> String {
    // f32 x with K % 4 == 0 (always for K-quant experts): vec4 loads
    let mut bindings = if spec.x_dtype == DType::F32 && (kquant::is_k(spec.w_dtype) || spec.w_dtype == DType::Int8) {
        "@group(0) @binding(1) var<storage, read> a4: array<vec4<f32>>;\n\
         fn lda4(e: u32) -> vec4<f32> {\n    return a4[e >> 2u];\n}\n"
            .to_string()
    } else {
        let mut b = common::input(1, "a", spec.x_dtype, caps);
        b.push_str("fn lda4(e: u32) -> vec4<f32> {\n    return vec4<f32>(ld_a(e), ld_a(e + 1u), ld_a(e + 2u), ld_a(e + 3u));\n}\n");
        b
    };
    bindings.push_str(&w_loader(spec.w_dtype, caps));
    bindings.push_str(&common::output(4, spec.out_dtype, caps));
    let dot = if kquant::is_k(spec.w_dtype) {
        let bb = kquant::block_bytes(spec.w_dtype);
        format!(
            "            let nb = K / 256u;\n            let row_byte = x * P[5] * 4u + j * nb * {bb}u;\n            \
             for (var idx = lane; idx < nb * 8u; idx += lanes) {{\n                let blk = idx >> 3u;\n                \
             part = part + w_kdot(row_byte + blk * {bb}u, idx & 7u, a_row + blk * 256u);\n            }}\n"
        )
    } else if spec.w_dtype == DType::Int8 {
        // one 4-element word per lane step (K % 32 == 0)
        "            let wb = x * P[5];\n            for (var k4 = lane; k4 < K / 4u; k4 += lanes) {\n                \
         let e = j * K + k4 * 4u;\n                let sc = unpack2x16float(w[wb + P[4] + (e >> 6u)])[(e >> 5u) & 1u];\n                \
         part = part + dot(vec4<f32>(unpack4xI8(w[wb + (e >> 2u)])), lda4(a_row + k4 * 4u)) * sc;\n            }\n"
            .to_string()
    } else {
        "            for (var k = lane; k < K; k += lanes) {\n                part = part + ld_a(a_row + k) * wdeq(x, j * K + k);\n            }\n"
            .to_string()
    };
    template::module(caps, KERNEL, &[("BINDINGS", &bindings), ("DOT", &dot)], &[], 256, true)
}
