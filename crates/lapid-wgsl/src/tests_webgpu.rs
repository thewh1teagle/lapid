//! WebGPU portability: every kernel family renders with `Caps { webgpu: true, .. }` (f16 and
//! subgroups on/off) and validates with naga, including control-flow uniformity (browsers'
//! Tint compiler enforces it). Vulkan output is unaffected (`WEBGPU` lines only).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, ReduceOp};

use crate::{Caps, IdxMode, Key, source};

fn caps(f16: bool, sg: bool) -> Caps {
    Caps { f16, subgroup: sg, subgroup_size: 32, full_subgroups: false, max_groups: [65535; 3], max_shared: 16384, coopmat: false, webgpu: true }
}

fn ld(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn mm(a: DType, b: DType, bt: bool, epi: bool) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec {
        a_dtype: a,
        b_dtype: b,
        b_transposed: bt,
        extra_dtypes: if epi { vec![DType::F32] } else { vec![] },
        out_dtype: DType::F32,
        epilogue: epi.then(|| Expr::Binary(BinaryOp::Add, ld(0), ld(1))),
    })
}

fn keys() -> Vec<Key> {
    let mut k = Vec::new();
    for (a, b, out) in [(DType::F32, DType::F32, DType::F32), (DType::F16, DType::BF16, DType::F16), (DType::F32, DType::F16, DType::BF16)] {
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![a, b], out_dtype: out, expr: Expr::Binary(BinaryOp::Mul, ld(0), ld(1)) });
        k.push(Key::Ew(spec.clone(), vec![IdxMode::Flat, IdxMode::Scalar], IdxMode::Flat));
        k.push(Key::Ew(spec, vec![IdxMode::Strided, IdxMode::Strided], IdxMode::Strided));
    }
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        let spec = Arc::new(ReduceSpec { op, rank: 2, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) });
        k.push(Key::Reduce(spec.clone(), true));
        k.push(Key::Reduce(spec, false));
    }
    for b in [DType::F32, DType::F16, DType::Int8, DType::Int4] {
        let s = mm(DType::F32, b, true, true);
        k.push(Key::Gemm(s.clone()));
        k.push(Key::Gemv(s.clone(), false, false));
        k.push(Key::Gemv(s, true, b == DType::F16));
    }
    k.push(Key::Gemm(mm(DType::F32, DType::F16, false, false)));
    for t in [DType::F32, DType::F16, DType::Int8, DType::Int4] {
        k.push(Key::Gather(Arc::new(GatherSpec { table_dtype: t, out_dtype: DType::F32 })));
    }
    k.push(Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: DType::F16 })));
    k.push(Key::Softmax(Arc::new(SoftmaxSpec { in_dtype: DType::F16, out_dtype: DType::F32 })));
    k.push(Key::LayerNorm(Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F16 })));
    k.push(Key::RmsNorm(Arc::new(LayerNormSpec { in_dtype: DType::F16, param_dtype: DType::F32, out_dtype: DType::F16 })));
    k.push(Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), true));
    k.push(Key::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), false));
    // subgroup-only kernels (skipped on caps without subgroups)
    k.push(Key::TopKSmall(Arc::new(TopKSpec { in_dtype: DType::F32 })));
    k.push(Key::GatedDeltaNet8(Arc::new(lapid_compiler::GatedDeltaNetSpec { out_dtype: DType::F32 }), 16, 8));
    for causal in [false, true] {
        for mask in [None, Some(DType::F32)] {
            let s = Arc::new(AttentionSpec { dtype: DType::F16, mask_dtype: mask, causal, out_dtype: DType::F32 });
            k.push(Key::AttnTile(s.clone(), 64));
            k.push(Key::AttnSplit(s.clone(), true, 64));
            k.push(Key::AttnSplit(s.clone(), false, 50));
            k.push(Key::AttnSplit(s.clone(), false, 512));
            k.push(Key::AttnMerge(s));
        }
    }
    k
}

fn validate(key: &Key, c: &Caps) -> Result<(), String> {
    use naga::valid::{Capabilities as VC, ValidationFlags, Validator};
    // naga 30 does not parse `enable subgroups;` yet (browsers require it); the native wgpu
    // backend strips the line the same way
    let src = source(key, c).replace("enable subgroups;", "");
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src))?;
    let mut vcaps = VC::SHADER_FLOAT16_IN_FLOAT32;
    if c.f16 {
        vcaps |= VC::SHADER_FLOAT16;
    }
    if c.subgroup {
        vcaps |= VC::SUBGROUP;
    }
    let mut v = Validator::new(ValidationFlags::all(), vcaps);
    if c.subgroup {
        v.subgroup_stages(naga::valid::ShaderStages::COMPUTE).subgroup_operations(naga::valid::SubgroupOperationSet::all());
    }
    v.validate(&module).map_err(|e| e.emit_to_string(&src))?;
    if src.contains("@builtin(subgroup_id)") || src.contains("@builtin(num_subgroups)") {
        return Err("uses builtins WebGPU does not have".into());
    }
    Ok(())
}

#[test]
fn webgpu_kernels_validate() {
    let mut failures = Vec::new();
    for (f16, sg) in [(false, false), (true, false), (false, true), (true, true)] {
        let c = caps(f16, sg);
        for key in keys() {
            if matches!(key, Key::TopKSmall(_) | Key::GatedDeltaNet8(..)) && !sg {
                continue;
            }
            if let Err(e) = validate(&key, &c) {
                failures.push(format!("{} f16={f16} sg={sg}: {e}", key.name()));
            }
        }
    }
    assert!(failures.is_empty(), "{} kernels failed:\n{}", failures.len(), failures.join("\n"));
}

/// MXFP4 and i-quant weights have no WGSL kernels: planning fails cleanly instead of
/// rendering a shader.
#[test]
fn mxfp4_rejected_at_plan() {
    let c = Caps::baseline();
    for w in [DType::Mxfp4, DType::Iq2Xxs, DType::Iq2Xs, DType::Iq2S, DType::Iq3Xxs] {
        let view = MatView { offset: 0, batch_stride: 0, row_stride: 256 };
        let dims = MatMulDims { batch: 1, m: 1, n: 4, k: 256, a: view.clone(), b: view, extra_views: vec![], out: MatView::contiguous(1, 4) };
        let gather = Kernel::Gather(Arc::new(GatherSpec { table_dtype: w, out_dtype: DType::F32 }), GatherDims { rows: 4, cols: 256, count: 1 });
        let id = Kernel::MatMulId(Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: w, out_dtype: DType::F32 }), MatMulIdDims { t: 1, r: 1, e: 2, n: 4, k: 256 });
        for k in [Kernel::MatMul(mm(DType::F32, w, true, false), dims), gather, id] {
            assert_eq!(crate::plan(&k, &c).err().map(|e| e.code), Some("unsupported_dtype"));
        }
    }
}
