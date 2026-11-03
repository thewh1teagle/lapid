//! Tests of the integer / indexing IR features on CUDA device 0, exact against the CPU
//! backend: typed integer expressions, GatherAxis, TopK, out-of-range index errors,
//! strided matmul outputs (every matmul path) and graph replay with in-place writes.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_cpu::CpuBackend;
use lapid_tensor::{DType, DeviceData};

use crate::tests::{bytes, check, cuda, floats, mat_spec, quant, rnd, run_on};

fn words(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn fwords(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run on CUDA and CPU and require identical output bytes.
fn exact(what: &str, kernel: Kernel, inputs: &[Vec<u8>], out_bytes: usize) {
    let Some(be) = cuda() else { return };
    let got = run_on(&be, &kernel, inputs, out_bytes);
    let want = run_on(&CpuBackend::new(), &kernel, inputs, out_bytes);
    if got != want {
        let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0) / 4;
        let w = |b: &[u8]| i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        if got.len() <= 64 {
            eprintln!("got  {:?}\nwant {:?}", got.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect::<Vec<_>>(), want.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect::<Vec<_>>());
        }
        panic!("{what}: first difference at word {i}: got {:#x}, want {:#x}", w(&got), w(&want));
    }
}

fn bin(op: IntOp, a: Expr, b: Expr) -> Expr {
    Expr::IntBinary(op, Box::new(a), Box::new(b))
}

fn conv(c: Conv, a: Expr) -> Expr {
    Expr::Convert(c, Box::new(a))
}

const INT_OPS: [IntOp; 20] = [
    IntOp::Add,
    IntOp::Sub,
    IntOp::Mul,
    IntOp::DivS,
    IntOp::DivU,
    IntOp::RemS,
    IntOp::RemU,
    IntOp::MinS,
    IntOp::MinU,
    IntOp::MaxS,
    IntOp::MaxU,
    IntOp::And,
    IntOp::Or,
    IntOp::Xor,
    IntOp::Shl,
    IntOp::ShrS,
    IntOp::ShrU,
    IntOp::CmpLtS,
    IntOp::CmpLtU,
    IntOp::CmpEq,
];

/// Interesting i32 pairs (zero divisors, i32::MIN / -1, shifts >= 32, negatives).
fn int_data(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let special = [0, 1, -1, 2, -2, 31, 32, 33, 63, i32::MIN, i32::MAX, 7, -7, 1 << 24, (1 << 24) + 1];
    let r = rnd(3 * n, 77);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    for i in 0..n {
        a.push(if i % 3 == 0 { special[i % special.len()] } else { (r[i] * 2e9) as i32 });
        b.push(if i % 2 == 0 { special[(i / 2) % special.len()] } else { (r[n + i] * 100.0) as i32 });
        let fs = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e9, -3e9, 5e9, -0.0, 2.5, -2.5, 0.99];
        c.push(if i % 4 == 1 { fs[(i / 4) % fs.len()] } else { r[2 * n + i] * 1e6 });
    }
    (a, b, c)
}

#[test]
fn int_exprs_elementwise_exact() {
    let n = 1000;
    let (a, b, c) = (Expr::LoadBits(0), Expr::LoadBits(1), Expr::Load(2));
    let (av, bv, cv) = int_data(n);
    let inputs = [words(&av), words(&bv), fwords(&cv)];
    let mk = |expr: Expr, out: DType| {
        Kernel::Elementwise(
            Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32, DType::I32, DType::F32], out_dtype: out, expr }),
            EwDims {
                shape: vec![n],
                in_views: vec![View::contiguous(&[n]); 3],
                out_view: View::contiguous(&[n]),
            },
        )
    };
    // every binary op on its own (raw i32 store)
    for op in INT_OPS {
        exact(&format!("int {op:?}"), mk(bin(op, a.clone(), b.clone()), DType::I32), &inputs, 4 * n);
    }
    // unary ops, conversions, Where with an int condition
    let un = |op| Expr::IntUnary(op, Box::new(a.clone()));
    for (name, e) in [
        ("neg", un(IntUnaryOp::Neg)),
        ("abs", un(IntUnaryOp::Abs)),
        ("not", un(IntUnaryOp::Not)),
        ("f32_to_i32", conv(Conv::F32ToI32, c.clone())),
        ("f32_to_u32", conv(Conv::F32ToU32, c.clone())),
        ("f32_bits", conv(Conv::F32Bits, c.clone())),
        (
            "where_int",
            Expr::Where(Box::new(bin(IntOp::CmpLtS, a.clone(), b.clone())), Box::new(a.clone()), Box::new(Expr::konst_int(-5))),
        ),
        ("where_f32_cond", Expr::Where(Box::new(c.clone()), Box::new(b.clone()), Box::new(a.clone()))),
    ] {
        exact(&format!("int {name}"), mk(e, DType::I32), &inputs, 4 * n);
    }
    // f32 roots built from integer math
    for (name, e) in [
        ("i32_to_f32", conv(Conv::I32ToF32, bin(IntOp::Add, a.clone(), b.clone()))),
        ("u32_to_f32", conv(Conv::U32ToF32, a.clone())),
        ("bits_f32", conv(Conv::BitsF32, conv(Conv::F32Bits, c.clone()))),
        ("load_i32_as_f32", Expr::Binary(lapid_tensor::BinaryOp::Add, Box::new(Expr::Load(0)), Box::new(Expr::Load(1)))),
    ] {
        exact(&format!("int {name}"), mk(e, DType::F32), &inputs, 4 * n);
    }
    // f32 root truncated into an i32 store; plain i32 copy keeps every bit
    exact("f32 root to i32", mk(c.clone(), DType::I32), &inputs, 4 * n);
    exact("i32 copy", mk(Expr::Load(0), DType::I32), &inputs, 4 * n);
}

#[test]
fn int_exprs_in_reduce_and_epilogue() {
    // reduce: sum over rows of f32(a & 0xff)
    let (rows, cols) = (37, 300);
    let (av, _, _) = int_data(rows * cols);
    let e = conv(Conv::I32ToF32, bin(IntOp::And, Expr::LoadBits(0), Expr::konst_int(255)));
    let spec = Arc::new(ReduceSpec { op: lapid_tensor::ReduceOp::Sum, rank: 2, in_dtypes: vec![DType::I32], out_dtype: DType::F32, expr: e });
    let k = Kernel::Reduce(spec, ReduceDims { shape: vec![rows, cols], in_views: vec![View::contiguous(&[rows, cols])] });
    exact("reduce int prologue", k, &[words(&av)], 4 * rows);
    // matmul epilogue: clear the low mantissa bits of acc
    let Some(be) = cuda() else { return };
    let (m, n, kk) = (5, 40, 64);
    let epi = conv(Conv::BitsF32, bin(IntOp::And, Expr::LoadBits(0), Expr::konst_int(!0xff)));
    let spec = mat_spec(DType::F32, DType::F32, true, vec![], Some(epi));
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k: kk,
        a: MatView::contiguous(m, kk),
        b: MatView::contiguous(n, kk),
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let inputs = [bytes(DType::F32, &rnd(m * kk, 1)), bytes(DType::F32, &rnd(n * kk, 2))];
    let got = floats(DType::F32, &run_on(&be, &Kernel::MatMul(spec.clone(), dims.clone()), &inputs, 4 * m * n));
    assert!(got.iter().all(|v| v.to_bits() & 0xff == 0), "epilogue bits not cleared");
    // reference: the plain product with the same bits cleared
    let plain = mat_spec(DType::F32, DType::F32, true, vec![], None);
    let want: Vec<f32> = floats(DType::F32, &run_on(&CpuBackend::new(), &Kernel::MatMul(plain, dims), &inputs, 4 * m * n))
        .iter()
        .map(|v| f32::from_bits(v.to_bits() & !0xff))
        .collect();
    check("matmul int epilogue", &got, &want, 1e-5);
}

#[test]
fn gather_axis_exact() {
    let (outer, len, n, inner) = (3usize, 11usize, 5usize, 7usize);
    let x = rnd(outer * len * inner, 3);
    let idx: Vec<i32> = (0..outer * n * inner).map(|i| ((i * 7 + i / 5) % len) as i32).collect();
    for dt in [DType::F32, DType::F16, DType::BF16, DType::I32] {
        let xs: Vec<f32> = if dt == DType::I32 { x.iter().map(|v| (v * 1e6).round()).collect() } else { x.clone() };
        // contiguous x [outer, len, inner]
        let d = GatherAxisDims {
            outer,
            n,
            inner,
            axis_len: len,
            x: View { offset: 0, strides: vec![(len * inner) as isize, inner as isize, 1] },
            idx: View { offset: 0, strides: vec![(n * inner) as isize, inner as isize, 1] },
        };
        let k = Kernel::GatherAxis(Arc::new(GatherAxisSpec { dtype: dt }), d.clone());
        exact(&format!("gather_axis {dt:?}"), k, &[bytes(dt, &xs), words(&idx)], outer * n * inner * dt.size());
        // transposed x view (x stored [inner, len, outer]) and broadcast indices over inner
        let d2 = GatherAxisDims {
            x: View { offset: 0, strides: vec![1, outer as isize, (len * outer) as isize] },
            idx: View { offset: 2, strides: vec![(n * inner) as isize, inner as isize, 0] },
            ..d
        };
        let k = Kernel::GatherAxis(Arc::new(GatherAxisSpec { dtype: dt }), d2);
        exact(&format!("gather_axis strided {dt:?}"), k, &[bytes(dt, &xs), words(&idx)], outer * n * inner * dt.size());
    }
}

fn topk_case(dt: DType, rows: usize, cols: usize, k: usize, ties: bool) {
    let mut x = rnd(rows * cols, (rows * 31 + cols + k) as u64);
    if ties {
        // few distinct values (incl. -0 / +0) so ties dominate
        for (i, v) in x.iter_mut().enumerate() {
            *v = [0.5, -0.0, 0.0, 0.25, -1.0, 0.5][(i * 7 + i / 3) % 6];
        }
    }
    if dt == DType::I32 {
        x.iter_mut().for_each(|v| *v = (*v * 1000.0).round());
    }
    let spec = Arc::new(TopKSpec { in_dtype: dt });
    let kernel = Kernel::TopK(spec, TopKDims { rows, cols, k });
    exact(&format!("topk {dt:?} {rows}x{cols} k={k} ties={ties}"), kernel, &[bytes(dt, &x)], rows * k * 4);
}

#[test]
fn topk_exact() {
    topk_case(DType::F32, 1, 16, 16, false);
    topk_case(DType::F32, 1, 7, 2, false);
    topk_case(DType::F32, 1, 16, 3, false);
    for dt in [DType::F32, DType::F16, DType::I32] {
        for &(rows, cols) in &[(1usize, 1usize), (3, 7), (2, 384), (2, 1000), (1, 1024), (1, 1025), (1, 51865)] {
            for k in [1usize, 2, 5, 64, 300] {
                if k <= cols {
                    topk_case(dt, rows, cols, k, false);
                    topk_case(dt, rows, cols, k, true);
                }
            }
        }
    }
    topk_case(DType::BF16, 4, 5000, 1, true);
    topk_case(DType::F32, 2, 3000, 2048, false);
    topk_case(DType::F32, 1, 262144, 1, false);
    topk_case(DType::F32, 1, 262144, 50, true);
}

#[test]
fn index_out_of_range_sets_error() {
    let Some(be) = cuda() else { return };
    let (rows, cols) = (10usize, 32usize);
    let table = be.upload(&bytes(DType::F32, &rnd(rows * cols, 4))).unwrap();
    let gather = |idx: &[i32]| -> lapid_tensor::Result<()> {
        let ib = be.upload(&words(idx)).unwrap();
        let out = be.alloc(idx.len() * cols * 4).unwrap();
        let k = Kernel::Gather(
            Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 }),
            GatherDims { rows, cols, count: idx.len() },
        );
        be.execute(&[Launch { kernel: k, inputs: vec![0, 1], output: 2 }], &[table.clone(), ib, out])
    };
    assert_eq!(gather(&[1, 10, 2]).unwrap_err().code, "index_out_of_range");
    gather(&[0, 9, 3]).unwrap();
    assert_eq!(gather(&[-1]).unwrap_err().code, "index_out_of_range");
    gather(&[5]).unwrap();
    // GatherAxis, also inside a multi-launch (graph-captured) execute
    let x = be.upload(&bytes(DType::F32, &rnd(4 * 6, 5))).unwrap();
    let ga = |idx: &[i32]| -> lapid_tensor::Result<()> {
        let ib = be.upload(&words(idx)).unwrap();
        let out = be.alloc(idx.len() * 4).unwrap();
        let k = Kernel::GatherAxis(
            Arc::new(GatherAxisSpec { dtype: DType::F32 }),
            GatherAxisDims {
                outer: 4,
                n: idx.len() / 4,
                inner: 1,
                axis_len: 6,
                x: View { offset: 0, strides: vec![6, 1, 0] },
                idx: View { offset: 0, strides: vec![(idx.len() / 4) as isize, 1, 0] },
            },
        );
        let l = Launch { kernel: k, inputs: vec![0, 1], output: 2 };
        be.execute(&[l.clone(), l], &[x.clone(), ib, out])
    };
    ga(&[0, 5, 1, 2, 3, 4, 5, 0]).unwrap();
    assert_eq!(ga(&[0, 5, 1, 6, 3, 4, 5, 0]).unwrap_err().code, "index_out_of_range");
    ga(&[0, 5, 1, 2, 3, 4, 5, 0]).unwrap();
}

/// Matmul writing through a strided `out` into a pre-filled larger buffer (KV-cache
/// layout: rows of `n` at row stride `rs`, batch stride `bs`, offset `off`) on every path.
#[allow(clippy::too_many_arguments)]
fn strided_out_case(a_t: DType, b_t: DType, bt: bool, batch: usize, m: usize, n: usize, k: usize, tol: f32) {
    let Some(be) = cuda() else { return };
    let is_q = b_t.is_quant();
    let (rs, off) = (n + 24, 5usize);
    let bs = (m + 3) * rs;
    let total = off + batch * bs;
    let spec = mat_spec(a_t, b_t, bt, vec![DType::F32], Some(crate::tests::gelu_plus(0, 1)));
    let a = rnd(batch * m * k, 21);
    let bw: Vec<f32> = rnd(if is_q { n * k } else { batch * n * k }, 22).iter().map(|v| v * 0.2).collect();
    let b_bytes = if is_q { quant(b_t, &bw, n, k) } else { bytes(b_t, &bw) };
    let dims = MatMulDims {
        batch,
        m,
        n,
        k,
        a: MatView::contiguous(m, k),
        b: MatView {
            offset: 0,
            batch_stride: if is_q { 0 } else { n * k },
            row_stride: if bt { k } else { n },
        },
        extra_views: vec![View { offset: 0, strides: vec![0, 0, 1] }],
        out: MatView { offset: off, batch_stride: bs, row_stride: rs },
    };
    let fill = rnd(total, 23);
    let inputs = [bytes(a_t, &a), b_bytes, bytes(DType::F32, &rnd(n, 24))];
    let run = |be: &dyn Backend, spec: Arc<MatMulSpec>, inputs: &[Vec<u8>]| {
        let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
        bufs.push(be.upload(&bytes(DType::F32, &fill)).unwrap());
        let l = Launch { kernel: Kernel::MatMul(spec, dims.clone()), inputs: vec![0, 1, 2], output: 3 };
        be.execute(&[l], &bufs).unwrap();
        floats(DType::F32, &be.download(&bufs[3]).unwrap())
    };
    let got = run(&be, spec.clone(), &inputs);
    let (ref_spec, ref_inputs) = if is_q {
        let f = if b_t == DType::Int8 { lapid_quant::Format::Int8 } else { lapid_quant::Format::Int4 };
        let deq = lapid_quant::dequantize(f, &inputs[1], n, k);
        let s = mat_spec(a_t, DType::F32, true, vec![DType::F32], spec.epilogue.clone());
        (s, vec![inputs[0].clone(), bytes(DType::F32, &deq), inputs[2].clone()])
    } else {
        (spec, inputs.to_vec())
    };
    let want = run(&CpuBackend::new(), ref_spec, &ref_inputs);
    let what = format!("strided out {a_t:?}x{b_t:?} bt={bt} {batch}x{m}x{n}x{k}");
    // untouched elements keep their bytes exactly
    let mut written = vec![false; total];
    for bt_ in 0..batch {
        for i in 0..m {
            for j in 0..n {
                written[off + bt_ * bs + i * rs + j] = true;
            }
        }
    }
    for (e, &w) in written.iter().enumerate() {
        if !w {
            assert_eq!(got[e].to_bits(), fill[e].to_bits(), "{what}: element {e} clobbered");
        }
    }
    check(&what, &got, &want, tol);
}

#[test]
fn matmul_strided_out_all_paths() {
    // scalar f32 GEMM, tensor-core GEMM (f16 / int8 / int4), GEMV (m <= 4)
    strided_out_case(DType::F32, DType::F32, true, 2, 70, 130, 100, 1e-5);
    strided_out_case(DType::F32, DType::F32, false, 1, 33, 64, 40, 1e-5);
    for b in [DType::F16, DType::Int8, DType::Int4] {
        strided_out_case(DType::F32, b, true, 1, 150, 200, 96, 3e-3);
        strided_out_case(DType::F32, b, true, 1, 1, 300, 768, 2e-5);
        strided_out_case(DType::F32, b, true, 1, 3, 64, 128, 2e-5);
    }
    strided_out_case(DType::F16, DType::F16, true, 2, 40, 136, 64, 3e-3);
    strided_out_case(DType::F32, DType::F16, true, 2, 2, 45, 100, 2e-5);
}

#[test]
fn graph_replay_with_changing_buffers() {
    // the same 3-launch sequence (graph-captured, then updated) over different buffers and
    // offsets, writing in place into a persistent device buffer like a KV-cache append
    let Some(be) = cuda() else { return };
    let cpu = CpuBackend::new();
    let (cap, d) = (16usize, 8usize);
    let spec = Arc::new(EwSpec {
        rank: 2,
        in_dtypes: vec![DType::F32],
        out_dtype: DType::F32,
        expr: Expr::Binary(lapid_tensor::BinaryOp::Mul, Box::new(Expr::Load(0)), Box::new(Expr::konst(2.0))),
    });
    let mm = mat_spec(DType::F32, DType::F32, true, vec![], None);
    let w = rnd(d * d, 30);
    let mut caches: Vec<(DeviceData, DeviceData)> = Vec::new();
    for be_ in [&be as &dyn Backend, &cpu as &dyn Backend] {
        caches.push((be_.upload(&vec![0u8; cap * d * 4]).unwrap(), be_.upload(&bytes(DType::F32, &w)).unwrap()));
    }
    for step in 0..6usize {
        let x = rnd(d, 40 + step as u64);
        let mut outs = Vec::new();
        for (bi, be_) in [&be as &dyn Backend, &cpu as &dyn Backend].into_iter().enumerate() {
            let (cache, wb) = &caches[bi];
            let xb = be_.upload(&bytes(DType::F32, &x)).unwrap();
            let tmp = be_.alloc(d * 4).unwrap();
            let out = be_.alloc(d * 4).unwrap();
            let launches = vec![
                // tmp = 2 x
                Launch {
                    kernel: Kernel::Elementwise(
                        spec.clone(),
                        EwDims { shape: vec![1, d], in_views: vec![View::contiguous(&[1, d])], out_view: View::contiguous(&[1, d]) },
                    ),
                    inputs: vec![0],
                    output: 1,
                },
                // cache[step] = tmp @ W^T (strided out at the step's row)
                Launch {
                    kernel: Kernel::MatMul(
                        mm.clone(),
                        MatMulDims {
                            batch: 1,
                            m: 1,
                            n: d,
                            k: d,
                            a: MatView::contiguous(1, d),
                            b: MatView::contiguous(d, d),
                            extra_views: vec![],
                            out: MatView { offset: step * d, batch_stride: cap * d, row_stride: d },
                        },
                    ),
                    inputs: vec![1, 2],
                    output: 3,
                },
                // out = sum of cache rows 0..=step (reduce over a strided view)
                Launch {
                    kernel: Kernel::Reduce(
                        Arc::new(ReduceSpec {
                            op: lapid_tensor::ReduceOp::Sum,
                            rank: 2,
                            in_dtypes: vec![DType::F32],
                            out_dtype: DType::F32,
                            expr: Expr::Load(0),
                        }),
                        ReduceDims { shape: vec![d, step + 1], in_views: vec![View { offset: 0, strides: vec![1, d as isize] }] },
                    ),
                    inputs: vec![3],
                    output: 4,
                },
            ];
            be_.execute(&launches, &[xb, tmp, wb.clone(), cache.clone(), out.clone()]).unwrap();
            outs.push(floats(DType::F32, &be_.download(&out).unwrap()));
        }
        check(&format!("graph replay step {step}"), &outs[0], &outs[1], 1e-5);
    }
}

/// `cargo test --release -p lapid-cuda bench_topk -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_topk() {
    let Some(be) = cuda() else { return };
    for &(rows, cols, k) in &[(1usize, 51865usize, 1usize), (1, 262144, 1), (1, 51865, 50), (1, 262144, 300), (8, 51865, 1)] {
        let x = be.upload(&bytes(DType::F32, &rnd(rows * cols, 9))).unwrap();
        let out = be.alloc(rows * k * 4).unwrap();
        let kernel = Kernel::TopK(Arc::new(TopKSpec { in_dtype: DType::F32 }), TopKDims { rows, cols, k });
        let l = [Launch { kernel, inputs: vec![0], output: 1 }];
        let bufs = [x, out];
        let _ = be.time_launches(&l, &bufs).unwrap();
        let mut t: Vec<f64> = (0..20).map(|_| be.time_launches(&l, &bufs).unwrap()[0]).collect();
        t.sort_by(f64::total_cmp);
        eprintln!("topk {rows}x{cols} k={k}: {:.1} us", 1e3 * t[10]);
    }
}
