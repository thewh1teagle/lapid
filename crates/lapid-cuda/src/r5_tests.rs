//! Round-5 kernel tests: RMS norm, attention with head dim > 256, f32 precision of the
//! tensor-core attention and of f32 matmuls at model sizes.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::DType;

use crate::matmul_attn_tests::{Att, attention_run, matmul_case};
use crate::tests::{bytes, check, compare, cuda, floats, rnd, run_on};

#[test]
fn rms_norm_matches_cpu() {
    let Some(be) = cuda() else { return };
    for &(rows, cols) in &[(1usize, 1024usize), (7, 2048), (33, 100), (5, 5120), (512, 256), (1, 20481), (64, 4096)] {
        for (xt, wt) in [(DType::F32, DType::F32), (DType::F16, DType::F32), (DType::F32, DType::BF16)] {
            let x = rnd(rows * cols, 21);
            let w = rnd(cols, 22);
            let spec = Arc::new(LayerNormSpec { in_dtype: xt, param_dtype: wt, out_dtype: DType::F32 });
            let dims = LayerNormDims { rows, cols, eps: 1e-6f32.to_bits() };
            let inputs = [bytes(xt, &x), bytes(wt, &w)];
            compare(&be, &format!("rms_norm {rows}x{cols}"), Kernel::RmsNorm(spec.clone(), dims.clone()), &inputs, DType::F32, rows * cols, 1e-5);
            // vs plain math
            let got = floats(DType::F32, &run_on(&be, &Kernel::RmsNorm(spec, dims), &inputs, rows * cols * 4));
            let (xs, ws) = (floats(xt, &inputs[0]), floats(wt, &inputs[1]));
            let mut want = vec![0f32; rows * cols];
            for r in 0..rows {
                let row = &xs[r * cols..(r + 1) * cols];
                let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cols as f64;
                let inv = 1.0 / (ms + 1e-6).sqrt();
                for c in 0..cols {
                    want[r * cols + c] = (row[c] as f64 * inv * ws[c] as f64) as f32;
                }
            }
            check("rms_norm vs f64", &got, &want, 1e-5);
        }
    }
}

/// Native Sinkhorn vs f64 math (`Op::Sinkhorn`).
#[test]
fn sinkhorn_matches_f64() {
    let Some(be) = cuda() else { return };
    let eps = 1e-6f64;
    for &(count, n, iters) in &[(1usize, 4usize, 20usize), (130, 4, 20), (3, 7, 3), (5, 1, 2)] {
        let x: Vec<f32> = rnd(count * n * n, 31).iter().map(|v| v * 6.0).collect();
        let dims = SinkhornDims { count, n, iters, eps: (eps as f32).to_bits() };
        let got = floats(DType::F32, &run_on(&be, &Kernel::Sinkhorn(dims), &[bytes(DType::F32, &x)], count * n * n * 4));
        let mut want = vec![0f32; count * n * n];
        for m in 0..count {
            let mut c: Vec<f64> = x[m * n * n..(m + 1) * n * n].iter().map(|&v| v as f64).collect();
            for i in 0..n {
                let row = &mut c[i * n..(i + 1) * n];
                let mx = row.iter().cloned().fold(f64::MIN, f64::max);
                row.iter_mut().for_each(|v| *v = (*v - mx).exp());
                let s: f64 = row.iter().sum();
                row.iter_mut().for_each(|v| *v = *v / s + eps);
            }
            let cols = |c: &mut Vec<f64>| {
                for j in 0..n {
                    let s: f64 = (0..n).map(|i| c[i * n + j]).sum::<f64>() + eps;
                    (0..n).for_each(|i| c[i * n + j] /= s);
                }
            };
            cols(&mut c);
            for _ in 1..iters {
                for i in 0..n {
                    let s: f64 = c[i * n..(i + 1) * n].iter().sum::<f64>() + eps;
                    c[i * n..(i + 1) * n].iter_mut().for_each(|v| *v /= s);
                }
                cols(&mut c);
            }
            want[m * n * n..(m + 1) * n * n].iter_mut().zip(&c).for_each(|(w, v)| *w = *v as f32);
        }
        check(&format!("sinkhorn {count}x{n}x{n} iters {iters}"), &got, &want, 1e-5);
    }
}

/// Head dims > 256 (Gemma 4 full attention: 512) on the split kernel: decode, prefill
/// (causal, many queries), KV cache strides, f16.
#[test]
fn attention_head_dim_512() {
    let Some(be) = cuda() else { return };
    for &(h, sq, sk, d, cap) in &[
        (8usize, 1usize, 1usize, 512usize, None),
        (8, 1, 700, 512, Some(1024usize)),
        (2, 3, 130, 512, None),
        (2, 40, 40, 512, None),
        (1, 17, 300, 320, None),
        (2, 1, 64, 1024, None),
    ] {
        for dt in [DType::F32, DType::F16] {
            for causal in [false, true] {
                let mask = sq == 3;
                attention_run(&be, Att { dt, mask, causal, b: 1, h, sq, sk, d, cap }, 2e-5);
            }
        }
    }
}

/// f32 q/k/v on the tensor-core flash kernel use hi/lo splits: ~f32 accuracy.
#[test]
fn attention_mma_f32_precise() {
    let Some(be) = cuda() else { return };
    for &(h, sq, sk, d) in &[(4usize, 64usize, 64usize, 32usize), (2, 100, 130, 64), (2, 77, 200, 128), (3, 40, 40, 80)] {
        for causal in [false, true] {
            for mask in [false, true] {
                attention_run(&be, Att { dt: DType::F32, mask, causal, b: 1, h, sq, sk, d, cap: None }, 2e-5);
            }
        }
    }
}

/// f32 x f32 matmuls (fp32 checkpoints) stay exact at model sizes.
#[test]
fn matmul_f32_exact_large_k() {
    let Some(be) = cuda() else { return };
    matmul_case(&be, DType::F32, DType::F32, true, 1, 64, 256, 4096, 1e-5);
    // small outputs over long k: GEMV path (decode attention scores, mHC mixes of a chunk)
    matmul_case(&be, DType::F32, DType::F32, true, 1, 64, 40, 512, 1e-5);
    matmul_case(&be, DType::F32, DType::F32, true, 1, 17, 24, 20480, 1e-5);
    matmul_case(&be, DType::F32, DType::F32, true, 1, 1, 512, 4096, 1e-5);
    matmul_case(&be, DType::F32, DType::F32, false, 1, 20, 96, 4096, 1e-5);
}

/// B larger than A: the tensor-core GEMM runs a column-tile-major grid (P[11] = 1).
#[test]
fn matmul_mma_swapped_grid() {
    let Some(be) = cuda() else { return };
    for b in [DType::F16, DType::Int8, DType::Int4] {
        matmul_case(&be, DType::F32, b, true, 1, 130, 3000, 64, 3e-3);
        matmul_case(&be, DType::F16, b, true, 1, 70, 1000, 96, 3e-3);
    }
}
