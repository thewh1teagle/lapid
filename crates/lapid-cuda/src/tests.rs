//! Tests. Source generation tests need no GPU. Kernel tests run on CUDA device 0 against
//! the CPU backend (the reference implementation of every kernel) and are skipped when no
//! CUDA device is available.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_cpu::CpuBackend;
use lapid_tensor::{BinaryOp, DType, DeviceData, ReduceOp, UnaryOp};
use half::{bf16, f16};

use crate::codegen::{self, IdxMode, Key};
use crate::context::Caps;
use crate::{CudaBackend, devices};

// ------------------------------------------------------------------ helpers

pub(crate) fn load(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn gelu_of(x: Box<Expr>) -> Expr {
    Expr::Binary(
        BinaryOp::Mul,
        Box::new(Expr::Binary(
            BinaryOp::Mul,
            x.clone(),
            Box::new(Expr::konst(0.5)),
        )),
        Box::new(Expr::Binary(
            BinaryOp::Add,
            Box::new(Expr::konst(1.0)),
            Box::new(Expr::Unary(
                UnaryOp::Erf,
                Box::new(Expr::Binary(
                    BinaryOp::Mul,
                    x,
                    Box::new(Expr::konst(std::f32::consts::FRAC_1_SQRT_2)),
                )),
            )),
        )),
    )
}

/// gelu(x_a) + 2 * x_b
pub(crate) fn gelu_plus(a: usize, b: usize) -> Expr {
    Expr::Binary(
        BinaryOp::Add,
        Box::new(gelu_of(load(a))),
        Box::new(Expr::Binary(
            BinaryOp::Mul,
            load(b),
            Box::new(Expr::konst(2.0)),
        )),
    )
}

pub(crate) fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub(crate) fn bytes(d: DType, v: &[f32]) -> Vec<u8> {
    match d {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        DType::I32 => v.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect(),
        DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 | DType::Iq2Xxs | DType::Iq2S | DType::Iq3Xxs => panic!("use quant()"),
    }
}

pub(crate) fn floats(d: DType, b: &[u8]) -> Vec<f32> {
    match d {
        DType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::I32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        _ => panic!("quantized output"),
    }
}

pub(crate) fn quant(d: DType, v: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let f = if d == DType::Int8 {
        lapid_quant::Format::Int8
    } else {
        lapid_quant::Format::Int4
    };
    lapid_quant::quantize(f, v, rows, cols)
}

pub(crate) fn cuda() -> Option<CudaBackend> {
    match devices() {
        Ok(d) if !d.is_empty() => CudaBackend::new(0).ok(),
        _ => {
            eprintln!("no CUDA device; skipping");
            None
        }
    }
}

/// Run one kernel on `be` with `inputs` (raw bytes) and an output of `out_bytes`.
pub(crate) fn run_on(
    be: &dyn Backend,
    kernel: &Kernel,
    inputs: &[Vec<u8>],
    out_bytes: usize,
) -> Vec<u8> {
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.upload(&vec![0u8; out_bytes]).unwrap());
    let n = inputs.len();
    let launch = Launch {
        kernel: kernel.clone(),
        inputs: (0..n).collect(),
        output: n,
    };
    be.execute(&[launch], &bufs).unwrap();
    be.download(&bufs[n]).unwrap()
}

/// Relative L2 error of `got` vs `want` (plus max abs diff), asserting `rel <= tol`.
pub(crate) fn check(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max = 0f32;
    for (g, w) in got.iter().zip(want) {
        assert!(
            g.is_finite() == w.is_finite(),
            "{what}: finiteness differs ({g} vs {w})"
        );
        if !w.is_finite() {
            continue;
        }
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
        max = max.max((g - w).abs());
    }
    let rel = (num / den.max(1e-30)).sqrt() as f32;
    assert!(
        rel <= tol,
        "{what}: rel error {rel:.3e} > {tol:.1e} (max abs {max:.3e})"
    );
}

/// Compare CUDA against the CPU backend for one kernel.
pub(crate) fn compare(
    be: &CudaBackend,
    what: &str,
    kernel: Kernel,
    inputs: &[Vec<u8>],
    out: DType,
    numel: usize,
    tol: f32,
) {
    let ob = numel * out.size();
    let got = floats(out, &run_on(be, &kernel, inputs, ob));
    let want = floats(out, &run_on(&CpuBackend::new(), &kernel, inputs, ob));
    check(what, &got, &want, tol);
}

// ------------------------------------------------------------------ no device needed

pub(crate) fn mat_spec(
    a: DType,
    b: DType,
    bt: bool,
    extras: Vec<DType>,
    epi: Option<Expr>,
) -> Arc<MatMulSpec> {
    Arc::new(MatMulSpec {
        a_dtype: a,
        b_dtype: b,
        b_transposed: bt,
        extra_dtypes: extras,
        out_dtype: DType::F32,
        epilogue: epi,
    })
}

/// A representative set of kernel keys (every template and slot variant).
fn all_keys() -> Vec<Key> {
    let mut keys = Vec::new();
    for &(ina, inb, out) in &[
        (DType::F32, DType::F32, DType::F32),
        (DType::F16, DType::BF16, DType::F16),
        (DType::I32, DType::F32, DType::I32),
        (DType::F32, DType::F16, DType::BF16),
    ] {
        let spec = Arc::new(EwSpec {
            rank: 3,
            in_dtypes: vec![ina, inb],
            out_dtype: out,
            expr: gelu_plus(0, 1),
        });
        keys.push(Key::Ew(
            spec.clone(),
            vec![IdxMode::Flat, IdxMode::Scalar],
            IdxMode::Flat,
        ));
        keys.push(Key::Ew(
            spec,
            vec![IdxMode::Strided, IdxMode::Strided],
            IdxMode::Strided,
        ));
    }
    let all_ops = {
        let x = || load(0);
        let mut e = Expr::konst(0.0);
        for op in [
            UnaryOp::Neg,
            UnaryOp::Exp,
            UnaryOp::Log,
            UnaryOp::Sqrt,
            UnaryOp::Rsqrt,
            UnaryOp::Recip,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tanh,
            UnaryOp::Erf,
            UnaryOp::Abs,
        ] {
            e = Expr::Binary(BinaryOp::Add, Box::new(e), Box::new(Expr::Unary(op, x())));
        }
        for op in [
            BinaryOp::Sub,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
            BinaryOp::Pow,
            BinaryOp::CmpLt,
            BinaryOp::CmpEq,
        ] {
            e = Expr::Binary(
                BinaryOp::Add,
                Box::new(e),
                Box::new(Expr::Binary(op, x(), load(1))),
            );
        }
        Expr::Where(
            load(1),
            Box::new(e),
            Box::new(Expr::Binary(BinaryOp::Pow, x(), Box::new(Expr::konst(2.0)))),
        )
    };
    keys.push(Key::Ew(
        Arc::new(EwSpec {
            rank: 1,
            in_dtypes: vec![DType::F32, DType::F32],
            out_dtype: DType::F32,
            expr: all_ops,
        }),
        vec![IdxMode::Flat; 2],
        IdxMode::Flat,
    ));
    let fill = Arc::new(EwSpec {
        rank: 0,
        in_dtypes: vec![],
        out_dtype: DType::F32,
        expr: Expr::konst(f32::NEG_INFINITY),
    });
    keys.push(Key::Ew(fill, vec![], IdxMode::Flat));
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        for rank in [1, 3] {
            let spec = Arc::new(ReduceSpec {
                op,
                rank,
                in_dtypes: vec![DType::F16, DType::F32],
                out_dtype: DType::F32,
                expr: Expr::Binary(BinaryOp::Mul, load(0), load(1)),
            });
            keys.push(Key::Reduce(spec.clone(), true));
            keys.push(Key::Reduce(spec, false));
        }
    }
    let bias = Some(Expr::Binary(BinaryOp::Add, load(0), load(1)));
    use DType::{BF16, F16, F32, Int4, Int8, Q4K, Q5K, Q6K};
    for b in [F32, F16, BF16, Int8, Int4, Q4K, Q5K, Q6K] {
        for a in [DType::F32, DType::F16] {
            let s = mat_spec(a, b, true, vec![DType::F32], bias.clone());
            keys.push(Key::Gemm(s.clone()));
            keys.push(Key::Gemv(s.clone(), false, false, 4));
            keys.push(Key::Gemv(s.clone(), a == F32, matches!(b, F16 | F32), 1));
            if b == DType::F16 || b.is_quant() {
                keys.push(Key::GemmMma(s.clone(), false, false));
                keys.push(Key::GemmMma(s, a == DType::F32, b == DType::F16));
            }
        }
    }
    keys.push(Key::Gemm(mat_spec(
        DType::F32,
        DType::F16,
        false,
        vec![],
        None,
    )));
    keys.push(Key::GemmMma(
        mat_spec(DType::F32, DType::F16, false, vec![], None),
        true,
        false,
    ));
    for t in [F32, F16, Int8, Int4, Q4K, Q5K, Q6K] {
        keys.push(Key::Gather(Arc::new(GatherSpec {
            table_dtype: t,
            out_dtype: DType::F32,
        })));
    }
    for t in [DType::F32, DType::F16, DType::I32] {
        keys.push(Key::GatherAxis(Arc::new(GatherAxisSpec { dtype: t })));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: t }), true));
        keys.push(Key::TopK(Arc::new(TopKSpec { in_dtype: t }), false));
        keys.push(Key::TopKSmall(Arc::new(TopKSpec { in_dtype: t })));
    }
    keys.push(Key::TopKMerge);
    for s in [mat_spec(F32, Q6K, true, vec![], None), mat_spec(F16, Int4, true, vec![], None)] {
        keys.extend([Key::GemvId(s.clone(), true, false, 2), Key::GemmMmaId(s.clone(), false, false, false), Key::GemmMmaId(s, true, false, true)]);
    }
    keys.push(Key::MoeRoute);
    let gdn = Arc::new(GatedDeltaNetSpec { out_dtype: F16 });
    keys.extend([Key::Gdn(gdn.clone(), 20), Key::Gdn(gdn, 128)]);
    keys.push(Key::Ew(
        Arc::new(EwSpec {
            rank: 1,
            in_dtypes: vec![DType::I32, DType::F32],
            out_dtype: DType::I32,
            expr: Expr::IntBinary(
                IntOp::DivS,
                Box::new(Expr::LoadBits(0)),
                Box::new(Expr::Convert(Conv::F32ToI32, load(1))),
            ),
        }),
        vec![IdxMode::Flat; 2],
        IdxMode::Flat,
    ));
    keys.push(Key::Softmax(Arc::new(SoftmaxSpec {
        in_dtype: DType::F16,
        out_dtype: DType::F32,
    })));
    keys.push(Key::Sinkhorn(4));
    keys.push(Key::Sinkhorn(8));
    for rms in [false, true] {
        for wide in [false, true] {
            keys.push(Key::LayerNorm(
                Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F32, out_dtype: DType::F16 }),
                rms,
                wide,
            ));
        }
    }
    for causal in [false, true] {
        for mask in [None, Some(DType::F32)] {
            for dt in [DType::F16, DType::F32] {
                let s = Arc::new(AttentionSpec {
                    dtype: dt,
                    mask_dtype: mask,
                    causal,
                    out_dtype: DType::F32,
                });
                keys.push(Key::AttnSplit(s.clone(), true, 64, 1));
                keys.push(Key::AttnSplit(s.clone(), false, 80, 4));
                keys.push(Key::AttnMerge(s.clone()));
                keys.push(Key::AttnMma(s.clone(), 64, true, false));
                keys.push(Key::AttnMma(s.clone(), 128, false, true));
                keys.push(Key::AttnMma(s, 48, false, true));
            }
        }
    }
    keys
}

#[test]
fn sources_fill_every_slot() {
    let caps = Caps::test(121);
    for key in all_keys() {
        let src = codegen::source(&key, &caps);
        assert!(!src.contains("{{"), "{key:?}: unfilled slot\n{src}");
        assert!(!src.contains("// #if"), "{key:?}: unresolved conditional");
        assert!(src.contains(".entry lapid_main"), "{key:?}: no entry");
    }
}

#[test]
fn devices_never_panic() {
    match devices() {
        Ok(list) => {
            for (i, d) in list.iter().enumerate() {
                assert_eq!(d.index, i);
            }
        }
        Err(e) => assert_eq!(e.code, "cuda_unavailable"),
    }
}

// ------------------------------------------------------------------ device tests

#[test]
fn kernels_compile() {
    let Some(be) = cuda() else { return };
    for key in all_keys() {
        if let Err(e) = be.modules.get(&be.ctx, &key) {
            panic!("{key:?}:\n{}", e.message);
        }
    }
}

#[test]
fn upload_download_roundtrip() {
    let Some(be) = cuda() else { return };
    for n in [0usize, 1, 4093, 9 << 20] {
        let data: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
        let b = be.upload(&data).unwrap();
        assert_eq!(be.download(&b).unwrap(), data, "{n} bytes");
    }
}

#[test]
fn elementwise_views_and_dtypes() {
    let Some(be) = cuda() else { return };
    let shape = vec![3usize, 5, 7];
    let n: usize = shape.iter().product();
    for &(ina, inb, out) in &[
        (DType::F32, DType::F32, DType::F32),
        (DType::F16, DType::BF16, DType::F16),
        (DType::F32, DType::F16, DType::BF16),
        (DType::I32, DType::F32, DType::F32),
    ] {
        let spec = Arc::new(EwSpec {
            rank: 3,
            in_dtypes: vec![ina, inb],
            out_dtype: out,
            expr: gelu_plus(0, 1),
        });
        let a: Vec<f32> = rnd(n, 1).iter().map(|x| x * 3.0).collect();
        let b = rnd(5 * 7, 2);
        // a permuted ([7,5,3] storage viewed as [3,5,7]), b broadcast over dim 0
        let dims = EwDims {
            shape: shape.clone(),
            in_views: vec![
                View {
                    offset: 0,
                    strides: vec![1, 3, 15],
                },
                View {
                    offset: 0,
                    strides: vec![0, 7, 1],
                },
            ],
            out_view: View::contiguous(&shape),
        };
        let a = if ina == DType::I32 {
            a.iter().map(|x| x.round()).collect()
        } else {
            a
        };
        compare(
            &be,
            &format!("ew {ina:?} {inb:?} {out:?}"),
            Kernel::Elementwise(spec, dims),
            &[bytes(ina, &a), bytes(inb, &b)],
            out,
            n,
            2e-3,
        );
    }
}

#[test]
fn elementwise_all_ops() {
    let Some(be) = cuda() else { return };
    let x = || load(0);
    let mut cases: Vec<Expr> = [
        UnaryOp::Neg,
        UnaryOp::Exp,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Recip,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tanh,
        UnaryOp::Erf,
        UnaryOp::Abs,
        UnaryOp::Log,
    ]
    .into_iter()
    .map(|op| Expr::Unary(op, x()))
    .collect();
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::Pow,
        BinaryOp::CmpLt,
        BinaryOp::CmpEq,
    ] {
        cases.push(Expr::Binary(op, x(), load(1)));
    }
    cases.push(Expr::Where(
        Box::new(Expr::Binary(BinaryOp::CmpLt, x(), load(1))),
        x(),
        load(1),
    ));
    cases.push(Expr::Binary(BinaryOp::Pow, x(), Box::new(Expr::konst(3.0))));
    let n = 4096;
    let a: Vec<f32> = rnd(n, 3).iter().map(|v| v * 6.0).collect();
    let mut b: Vec<f32> = rnd(n, 4).iter().map(|v| v * 3.0).collect();
    b[..64].copy_from_slice(&a[..64]); // CmpEq hits
    for (i, e) in cases.into_iter().enumerate() {
        let positive = matches!(
            e,
            Expr::Unary(UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Log, _)
                | Expr::Binary(BinaryOp::Pow, _, _)
        );
        let a: Vec<f32> = if positive {
            a.iter().map(|v| v.abs() + 0.1).collect()
        } else {
            a.clone()
        };
        let spec = Arc::new(EwSpec {
            rank: 1,
            in_dtypes: vec![DType::F32, DType::F32],
            out_dtype: DType::F32,
            expr: e.clone(),
        });
        let dims = EwDims {
            shape: vec![n],
            in_views: vec![View::contiguous(&[n]); 2],
            out_view: View::contiguous(&[n]),
        };
        compare(
            &be,
            &format!("op {i}: {e:?}"),
            Kernel::Elementwise(spec, dims),
            &[bytes(DType::F32, &a), bytes(DType::F32, &b)],
            DType::F32,
            n,
            2e-6,
        );
    }
}

#[test]
fn reduce_row_and_thread() {
    let Some(be) = cuda() else { return };
    for op in [ReduceOp::Sum, ReduceOp::Max] {
        for &(outer, inner) in &[(3usize, 1000usize), (1000, 7), (1, 5)] {
            let spec = Arc::new(ReduceSpec {
                op,
                rank: 2,
                in_dtypes: vec![DType::F16, DType::F32],
                out_dtype: DType::F32,
                expr: Expr::Binary(BinaryOp::Mul, load(0), load(1)),
            });
            let shape = vec![outer, inner];
            // second input transposed storage [inner, outer]
            let dims = ReduceDims {
                shape: shape.clone(),
                in_views: vec![
                    View::contiguous(&shape),
                    View {
                        offset: 0,
                        strides: vec![1, outer as isize],
                    },
                ],
            };
            let a = rnd(outer * inner, 5);
            let b = rnd(outer * inner, 6);
            compare(
                &be,
                &format!("reduce {op:?} {outer}x{inner}"),
                Kernel::Reduce(spec, dims),
                &[bytes(DType::F16, &a), bytes(DType::F32, &b)],
                DType::F32,
                outer,
                1e-5,
            );
        }
    }
}

#[test]
fn gather_tables() {
    let Some(be) = cuda() else { return };
    let (rows, cols, count) = (50usize, 96usize, 7usize);
    let table = rnd(rows * cols, 10);
    let idx: Vec<u8> = [0i32, 49, 3, 3, 17, 2, 48]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    for t in [DType::F32, DType::F16, DType::Int8, DType::Int4] {
        let tb = if t.is_quant() {
            quant(t, &table, rows, cols)
        } else {
            bytes(t, &table)
        };
        let spec = Arc::new(GatherSpec {
            table_dtype: t,
            out_dtype: DType::F32,
        });
        compare(
            &be,
            &format!("gather {t:?}"),
            Kernel::Gather(spec, GatherDims { rows, cols, count }),
            &[tb, idx.clone()],
            DType::F32,
            count * cols,
            1e-6,
        );
    }
}

#[test]
fn softmax_and_layer_norm() {
    let Some(be) = cuda() else { return };
    for &(rows, cols) in &[(5usize, 1500usize), (3, 7), (2, 51865)] {
        let x: Vec<f32> = rnd(rows * cols, 11).iter().map(|v| v * 8.0).collect();
        let spec = Arc::new(SoftmaxSpec {
            in_dtype: DType::F32,
            out_dtype: DType::F32,
        });
        compare(
            &be,
            &format!("softmax {rows}x{cols}"),
            Kernel::Softmax(spec, RowDims { rows, cols }),
            &[bytes(DType::F32, &x)],
            DType::F32,
            rows * cols,
            1e-5,
        );
        let spec = Arc::new(LayerNormSpec {
            in_dtype: DType::F32,
            param_dtype: DType::F32,
            out_dtype: DType::F32,
        });
        let w = rnd(cols, 12);
        let b = rnd(cols, 13);
        let dims = LayerNormDims {
            rows,
            cols,
            eps: 1e-5f32.to_bits(),
        };
        compare(
            &be,
            &format!("layer_norm {rows}x{cols}"),
            Kernel::LayerNorm(spec, dims),
            &[
                bytes(DType::F32, &x),
                bytes(DType::F32, &w),
                bytes(DType::F32, &b),
            ],
            DType::F32,
            rows * cols,
            1e-5,
        );
    }
}

