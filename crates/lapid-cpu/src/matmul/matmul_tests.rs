//! Quantized matmul with many rows (`qdeq`): f32 activations x dequantized weights must match
//! the f32 product with the dequantized matrix tightly (no activation quantization error).

use std::sync::Arc;

use lapid_compiler::{Backend, Expr, Kernel, Launch, MatMulDims, MatMulSpec, MatView, View};
use lapid_quant::Format;
use lapid_tensor::{BinaryOp, DType, DeviceData};

use crate::{CpuBackend, Isa};

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).powi(2)).sum();
    let den: f64 = want.iter().map(|&w| w * w).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn quantized_large_m_matches_dequantized_f32() {
    let bes = [CpuBackend::with_threads(4), CpuBackend::build(1, Isa::default())];
    // (batch, m, n, k, a row padding, bias epilogue); k = 4096 forces two B column chunks.
    let cases = [(1, 9, 45, 96, 0, false), (2, 35, 70, 128, 5, true), (1, 130, 200, 256, 3, true), (1, 20, 1100, 4096, 0, false)];
    let mut seed = 1;
    for fmt in [Format::Int8, Format::Int4] {
        let dt = if fmt == Format::Int8 { DType::Int8 } else { DType::Int4 };
        for &(batch, m, n, k, pad, epi) in &cases {
            seed += 1;
            let w = rnd(n * k, seed);
            let bytes = lapid_quant::quantize(fmt, &w, n, k);
            let deq = lapid_quant::dequantize(fmt, &bytes, n, k);
            let rs = k + pad;
            let a = rnd(batch * m * rs, seed + 7);
            let bias = rnd(n, seed + 9);
            let mut want = vec![0f64; batch * m * n];
            for r in 0..batch * m {
                for j in 0..n {
                    let dot: f64 = (0..k).map(|p| a[r * rs + p] as f64 * deq[j * k + p] as f64).sum();
                    want[r * n + j] = dot + if epi { bias[j] as f64 } else { 0.0 };
                }
            }
            let spec = Arc::new(MatMulSpec {
                a_dtype: DType::F32,
                b_dtype: dt,
                b_transposed: true,
                extra_dtypes: if epi { vec![DType::F32] } else { vec![] },
                out_dtype: DType::F32,
                epilogue: epi.then(|| Expr::Binary(BinaryOp::Add, Box::new(Expr::Load(0)), Box::new(Expr::Load(1)))),
            });
            let dims = MatMulDims {
                batch,
                m,
                n,
                k,
                a: MatView { offset: 0, batch_stride: m * rs, row_stride: rs },
                b: MatView { offset: 0, batch_stride: 0, row_stride: k },
                extra_views: if epi { vec![View { offset: 0, strides: vec![0, 0, 1] }] } else { vec![] },
                out: MatView::contiguous(m, n),
            };
            for be in &bes {
                let mut bufs: Vec<DeviceData> = vec![be.upload(bytemuck::cast_slice(&a)).unwrap(), be.upload(&bytes).unwrap()];
                if epi {
                    bufs.push(be.upload(bytemuck::cast_slice(&bias)).unwrap());
                }
                let ni = bufs.len();
                bufs.push(be.alloc(batch * m * n * 4).unwrap());
                let l = [Launch { kernel: Kernel::MatMul(spec.clone(), dims.clone()), inputs: (0..ni).collect(), output: ni }];
                be.prepare(&l).unwrap();
                be.execute(&l, &bufs).unwrap();
                let got: Vec<f32> = bytemuck::pod_collect_to_vec(&be.download(&bufs[ni]).unwrap());
                let e = rel_err(&got, &want);
                // bf16 hi + lo activations x exact integer weights (`qbf16`), or f32.
                assert!(e <= 2e-5, "{fmt:?} batch={batch} m={m} n={n} k={k}: rel err {e}");
            }
        }
    }
}

/// K-quant weights with many rows (`qbf16` exact path on BF16 CPUs, `qdeq` elsewhere) and
/// few rows (int8 activations): products against the dequantized f32 matrix.
#[test]
fn kquant_matmul_matches_dequantized_f32() {
    let bes = [CpuBackend::with_threads(4), CpuBackend::build(1, Isa::default())];
    let mut seed = 50;
    for fmt in [Format::Q4K, Format::Q5K, Format::Q6K, Format::Mxfp4, Format::Iq2Xxs, Format::Iq2Xs, Format::Iq2S, Format::Iq3Xxs] {
        let dt = super::kbench_tests::dtype(fmt);
        let mut shapes = vec![(1, 37, 512, 0), (3, 20, 256, 4), (6, 9, 256, 0), (20, 45, 256, 3), (130, 70, 512, 0), (33, 300, 1024, 0)];
        if fmt == Format::Mxfp4 {
            // 32-element blocks: k need not be a multiple of 256.
            shapes.extend([(1, 11, 96, 0), (5, 7, 224, 2), (40, 13, 160, 1)]);
        }
        for &(m, n, k, pad) in &shapes {
            seed += 1;
            let w = rnd(n * k, seed);
            let bytes = lapid_quant::quantize(fmt, &w, n, k);
            let deq = lapid_quant::dequantize(fmt, &bytes, n, k);
            let rs = k + pad;
            let a = rnd(m * rs, seed + 7);
            let mut want = vec![0f64; m * n];
            for r in 0..m {
                for j in 0..n {
                    want[r * n + j] = (0..k).map(|p| a[r * rs + p] as f64 * deq[j * k + p] as f64).sum();
                }
            }
            let spec = Arc::new(MatMulSpec { a_dtype: DType::F32, b_dtype: dt, b_transposed: true, extra_dtypes: vec![], out_dtype: DType::F32, epilogue: None });
            let dims = MatMulDims {
                batch: 1,
                m,
                n,
                k,
                a: MatView { offset: 0, batch_stride: 0, row_stride: rs },
                b: MatView { offset: 0, batch_stride: 0, row_stride: k },
                extra_views: vec![],
                out: MatView::contiguous(m, n),
            };
            for (bi, be) in bes.iter().enumerate() {
                let bufs = [be.upload(bytemuck::cast_slice(&a)).unwrap(), be.upload(&bytes).unwrap(), be.alloc(m * n * 4).unwrap()];
                let l = [Launch { kernel: Kernel::MatMul(spec.clone(), dims.clone()), inputs: vec![0, 1], output: 2 }];
                be.execute(&l, &bufs).unwrap();
                let got: Vec<f32> = bytemuck::pod_collect_to_vec(&be.download(&bufs[2]).unwrap());
                let e = rel_err(&got, &want);
                // m <= 8: int8 activations; else exact (bf16 hi + lo split or f32).
                let tol = if m <= 8 { 2e-2 } else { 2e-5 };
                assert!(e <= tol, "{fmt:?} backend {bi} m={m} n={n} k={k}: rel err {e}");
            }
        }
    }
}

/// Every matmul path honors a strided output view: [batch, m, n] written as [m, batch, n]
/// rows inside a larger buffer at an offset (heads written straight into [S, H, D]).
#[test]
fn matmul_strided_output_view() {
    let bes = [CpuBackend::with_threads(4), CpuBackend::build(1, Isa::default())];
    let (k, n, off) = (64, 40, 7);
    let mut seed = 100;
    for (b_dtype, m) in [(DType::F32, 3), (DType::F32, 30), (DType::F16, 2), (DType::Int8, 2), (DType::Int8, 30), (DType::Int4, 20)] {
        let batch = 3;
        seed += 1;
        let a = rnd(batch * m * k, seed);
        let w = rnd(n * k, seed + 1);
        let (b_bytes, b_vals, b_bs) = match b_dtype {
            DType::Int8 | DType::Int4 => {
                let fmt = if b_dtype == DType::Int8 { Format::Int8 } else { Format::Int4 };
                let q = lapid_quant::quantize(fmt, &w, n, k);
                let deq = lapid_quant::dequantize(fmt, &q, n, k);
                (q, deq, 0)
            }
            DType::F16 => {
                let h: Vec<half::f16> = w.iter().map(|&x| half::f16::from_f32(x)).collect();
                (bytemuck::cast_slice(&h).to_vec(), h.iter().map(|x| x.to_f32()).collect(), 0)
            }
            _ => (bytemuck::cast_slice(&w).to_vec(), w.clone(), 0),
        };
        let out_view = MatView { offset: off, batch_stride: n, row_stride: batch * n };
        let total = off + m * batch * n + 5;
        let spec = Arc::new(MatMulSpec {
            a_dtype: DType::F32,
            b_dtype,
            b_transposed: true,
            extra_dtypes: vec![],
            out_dtype: DType::F32,
            epilogue: None,
        });
        let dims = MatMulDims {
            batch,
            m,
            n,
            k,
            a: MatView::contiguous(m, k),
            b: MatView { offset: 0, batch_stride: b_bs, row_stride: k },
            extra_views: vec![],
            out: out_view.clone(),
        };
        for be in &bes {
            let bufs = [
                be.upload(bytemuck::cast_slice(&a)).unwrap(),
                be.upload(&b_bytes).unwrap(),
                be.upload(bytemuck::cast_slice(&vec![-1f32; total])).unwrap(),
            ];
            let l = [Launch { kernel: Kernel::MatMul(spec.clone(), dims.clone()), inputs: vec![0, 1], output: 2 }];
            be.prepare(&l).unwrap();
            be.execute(&l, &bufs).unwrap();
            let got: Vec<f32> = bytemuck::pod_collect_to_vec(&be.download(&bufs[2]).unwrap());
            let mut want = vec![-1f64; total];
            for bt in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let dot: f64 = (0..k).map(|p| a[(bt * m + i) * k + p] as f64 * b_vals[j * k + p] as f64).sum();
                        want[off + bt * n + i * batch * n + j] = dot;
                    }
                }
            }
            // int8 activations (m <= 8 quantized path) carry ~1e-2 relative error; the bf16
            // path for larger m (BF16 CPUs) ~3e-3.
            let quant = matches!(b_dtype, DType::Int8 | DType::Int4);
            let tol = if quant && m <= 8 { 2e-2 } else if quant { 5e-3 } else { 1e-4 };
            let e = rel_err(&got, &want);
            assert!(e <= tol, "{b_dtype:?} m={m}: rel err {e}");
            for (x, (&g, &w)) in got.iter().zip(&want).enumerate() {
                if w == -1.0 {
                    assert_eq!(g, -1.0, "{b_dtype:?} m={m}: element {x} outside the output view was written");
                }
            }
        }
    }
}
