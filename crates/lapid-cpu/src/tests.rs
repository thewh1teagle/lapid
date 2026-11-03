//! End-to-end kernel tests through the `Backend` API against naive f64 references.
//! Every test runs on a multi-threaded SIMD backend and a single-threaded scalar backend.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_quant::{Format, GROUP};
use lapid_tensor::{BinaryOp, DType, DeviceData, ReduceOp, UnaryOp};
use half::{bf16, f16};

use super::*;

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

fn backends() -> Vec<CpuBackend> {
    vec![CpuBackend::with_threads(4), CpuBackend::build(1, Isa::default())]
}

fn up(be: &CpuBackend, v: &[f32], dt: DType) -> DeviceData {
    let bytes: Vec<u8> = match dt {
        DType::F32 => bytemuck::cast_slice(v).to_vec(),
        DType::F16 => {
            let h: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
            bytemuck::cast_slice(&h).to_vec()
        }
        DType::BF16 => {
            let h: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
            bytemuck::cast_slice(&h).to_vec()
        }
        DType::I32 => {
            let h: Vec<i32> = v.iter().map(|&x| x as i32).collect();
            bytemuck::cast_slice(&h).to_vec()
        }
        _ => panic!("quantized"),
    };
    be.upload(&bytes).unwrap()
}

/// Round-trip through a storage dtype (what the kernel actually sees).
fn rt(v: &[f32], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        DType::BF16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        _ => v.to_vec(),
    }
}

fn down(be: &CpuBackend, b: &DeviceData, dt: DType) -> Vec<f32> {
    let bytes = be.download(b).unwrap();
    match dt {
        DType::F32 => bytemuck::pod_collect_to_vec::<u8, f32>(&bytes),
        DType::F16 => bytemuck::pod_collect_to_vec::<u8, f16>(&bytes).iter().map(|x| x.to_f32()).collect(),
        DType::BF16 => bytemuck::pod_collect_to_vec::<u8, bf16>(&bytes).iter().map(|x| x.to_f32()).collect(),
        DType::I32 => bytemuck::pod_collect_to_vec::<u8, i32>(&bytes).iter().map(|&x| x as f32).collect(),
        _ => panic!("quantized"),
    }
}

fn run(be: &CpuBackend, kernel: Kernel, inputs: Vec<DeviceData>, out_bytes: usize) -> DeviceData {
    let out = be.alloc(out_bytes).unwrap();
    let n = inputs.len();
    let mut bufs = inputs;
    bufs.push(out.clone());
    be.prepare(&[Launch { kernel: kernel.clone(), inputs: (0..n).collect(), output: n }]).unwrap();
    be.execute(&[Launch { kernel, inputs: (0..n).collect(), output: n }], &bufs).unwrap();
    out
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * w.abs().max(1.0), "element {i}: got {g}, want {w}");
    }
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(&g, &w)| ((g - w) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|&w| (w as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn bx(e: Expr) -> Box<Expr> {
    Box::new(e)
}

#[test]
fn elementwise_broadcast_permute_offset() {
    let (a, b, c) = (3, 70, 50); // inner dim >= 48: row-wise path
    let mut rng = Rng(1);
    let x = rng.vec(b * a * c); // stored as [b, a, c], viewed as [a, b, c]
    let bias = rng.vec(c);
    let expr = Expr::Binary(BinaryOp::Add, bx(Expr::Load(0)), bx(Expr::Binary(BinaryOp::Mul, bx(Expr::Load(1)), bx(Expr::konst(2.0)))));
    for be in backends() {
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::F32, DType::F32], out_dtype: DType::F16, expr: expr.clone() });
        let dims = EwDims {
            shape: vec![a, b, c],
            in_views: vec![View { offset: 0, strides: vec![c as isize, (a * c) as isize, 1] }, View { offset: 0, strides: vec![0, 0, 1] }],
            out_view: View { offset: 7, strides: vec![(b * c) as isize, c as isize, 1] },
        };
        let out = run(&be, Kernel::Elementwise(spec, dims), vec![up(&be, &x, DType::F32), up(&be, &bias, DType::F32)], (7 + a * b * c) * 2);
        let got = down(&be, &out, DType::F16);
        let mut want = vec![0f32; a * b * c];
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    want[(i * b + j) * c + k] = x[(j * a + i) * c + k] + 2.0 * bias[k];
                }
            }
        }
        close(&got[7..], &rt(&want, DType::F16), 2e-3);
        assert!(got[..7].iter().all(|&v| v == 0.0));
    }
}

#[test]
fn elementwise_linear_path_gelu_bf16() {
    // Transposed small inner dims force the offset-list path.
    let shape = [50usize, 3, 2];
    let mut rng = Rng(2);
    let x = rt(&rng.vec(100 * 3), DType::BF16);
    let s2 = bx(Expr::konst(std::f32::consts::FRAC_1_SQRT_2));
    let lapid = Expr::Binary(
        BinaryOp::Mul,
        bx(Expr::Binary(BinaryOp::Mul, bx(Expr::Load(0)), bx(Expr::konst(0.5)))),
        bx(Expr::Binary(BinaryOp::Add, bx(Expr::konst(1.0)), bx(Expr::Unary(UnaryOp::Erf, bx(Expr::Binary(BinaryOp::Mul, bx(Expr::Load(0)), s2)))))),
    );
    for be in backends() {
        let spec = Arc::new(EwSpec { rank: 3, in_dtypes: vec![DType::BF16], out_dtype: DType::F32, expr: lapid.clone() });
        // element (i, j, k) at k * 150 + j * 50 + i
        let dims = EwDims {
            shape: shape.to_vec(),
            in_views: vec![View { offset: 0, strides: vec![1, 50, 150] }],
            out_view: View::contiguous(&shape),
        };
        let out = run(&be, Kernel::Elementwise(spec, dims), vec![up(&be, &x, DType::BF16)], 300 * 4);
        let got = down(&be, &out, DType::F32);
        let mut want = vec![0f32; 300];
        for i in 0..50 {
            for j in 0..3 {
                for k in 0..2 {
                    let v = x[k * 150 + j * 50 + i] as f64;
                    want[(i * 3 + j) * 2 + k] = (0.5 * v * (1.0 + math::erf((v / 2f64.sqrt()) as f32) as f64)) as f32;
                }
            }
        }
        close(&got, &want, 1e-5);
    }
}

#[test]
fn elementwise_i32_copy_is_exact() {
    let vals: Vec<i32> = (0..1000).map(|i| (1 << 25) + i * 3 - 7).collect();
    for be in backends() {
        let buf = be.upload(bytemuck::cast_slice(&vals)).unwrap();
        let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::I32], out_dtype: DType::I32, expr: Expr::Load(0) });
        let dims = EwDims { shape: vec![500], in_views: vec![View { offset: 999, strides: vec![-2] }], out_view: View::contiguous(&[500]) };
        let out = run(&be, Kernel::Elementwise(spec, dims), vec![buf], 500 * 4);
        let got = bytemuck::pod_collect_to_vec::<u8, i32>(&be.download(&out).unwrap());
        for t in 0..500 {
            assert_eq!(got[t], vals[999 - 2 * t]);
        }
    }
}

#[test]
fn reduce_sum_and_max() {
    let mut rng = Rng(3);
    for (rows, len) in [(7usize, 3000usize), (1, 100_000), (300, 3)] {
        let x = rng.vec(rows * len); // stored transposed: element (r, j) at j * rows + r
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            for be in backends() {
                let expr = Expr::Binary(BinaryOp::Mul, bx(Expr::Load(0)), bx(Expr::Load(0)));
                let spec = Arc::new(ReduceSpec { op, rank: 2, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr });
                let dims = ReduceDims { shape: vec![rows, len], in_views: vec![View { offset: 0, strides: vec![1, rows as isize] }] };
                let out = run(&be, Kernel::Reduce(spec, dims), vec![up(&be, &x, DType::F32)], rows * 4);
                let got = down(&be, &out, DType::F32);
                let want: Vec<f32> = (0..rows)
                    .map(|r| {
                        let it = (0..len).map(|j| (x[j * rows + r] as f64).powi(2));
                        match op {
                            ReduceOp::Sum => it.sum::<f64>() as f32,
                            ReduceOp::Max => it.fold(f64::NEG_INFINITY, f64::max) as f32,
                        }
                    })
                    .collect();
                close(&got, &want, 1e-4);
            }
        }
    }
}

#[test]
fn softmax_and_layer_norm() {
    let (rows, cols) = (37, 517);
    let mut rng = Rng(4);
    let mut x: Vec<f32> = rng.vec(rows * cols).iter().map(|v| v * 8.0).collect();
    for c in 0..cols {
        x[5 * cols + c] = f32::NEG_INFINITY; // fully masked row
    }
    x[6 * cols + 3] = f32::NEG_INFINITY;
    let w = rng.vec(cols);
    let b = rng.vec(cols);
    for be in backends() {
        let spec = Arc::new(SoftmaxSpec { in_dtype: DType::F32, out_dtype: DType::F32 });
        let out = run(&be, Kernel::Softmax(spec, RowDims { rows, cols }), vec![up(&be, &x, DType::F32)], rows * cols * 4);
        let got = down(&be, &out, DType::F32);
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let want: Vec<f32> = if m == f64::NEG_INFINITY {
                vec![0.0; cols]
            } else {
                let e: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|v| (v / s) as f32).collect()
            };
            close(&got[r * cols..(r + 1) * cols], &want, 1e-5);
        }

        let xs: Vec<f32> = x.iter().map(|&v| if v.is_finite() { v } else { 1.0 }).collect();
        let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F16, out_dtype: DType::F32 });
        let dims = LayerNormDims { rows, cols, eps: 1e-5f32.to_bits() };
        let out = run(&be, Kernel::LayerNorm(spec, dims), vec![up(&be, &xs, DType::F32), up(&be, &w, DType::F16), up(&be, &b, DType::F16)], rows * cols * 4);
        let got = down(&be, &out, DType::F32);
        let (wr, br) = (rt(&w, DType::F16), rt(&b, DType::F16));
        for r in 0..rows {
            let row = &xs[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
            let rstd = 1.0 / (var + 1e-5).sqrt();
            let want: Vec<f32> = (0..cols).map(|c| ((row[c] as f64 - mean) * rstd * wr[c] as f64 + br[c] as f64) as f32).collect();
            close(&got[r * cols..(r + 1) * cols], &want, 1e-4);
        }

        let spec = Arc::new(LayerNormSpec { in_dtype: DType::F32, param_dtype: DType::F16, out_dtype: DType::F32 });
        let dims = LayerNormDims { rows, cols, eps: 1e-6f32.to_bits() };
        let out = run(&be, Kernel::RmsNorm(spec, dims), vec![up(&be, &xs, DType::F32), up(&be, &w, DType::F16)], rows * cols * 4);
        let got = down(&be, &out, DType::F32);
        for r in 0..rows {
            let row = &xs[r * cols..(r + 1) * cols];
            let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
            let rstd = 1.0 / (ms + 1e-6).sqrt();
            let want: Vec<f32> = (0..cols).map(|c| (row[c] as f64 * rstd * wr[c] as f64) as f32).collect();
            close(&got[r * cols..(r + 1) * cols], &want, 1e-4);
        }
    }
}

/// Test quantizer (independent of lapid-quant's implementation): returns SoA bytes and the
/// dequantized matrix.
fn quantize(fmt: Format, x: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let l = lapid_quant::layout(fmt, rows, cols);
    let mut bytes = vec![0u8; l.total];
    let mut deq = vec![0f32; rows * cols];
    for r in 0..rows {
        for g in 0..cols / GROUP {
            let xs = &x[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = f16::from_f32(match fmt {
                Format::Int8 => amax / 127.0,
                _ => amax / 7.0,
            });
            let df = d.to_f32();
            let id = if df > 0.0 { 1.0 / df } else { 0.0 };
            let si = 2 * (r * (cols / GROUP) + g);
            bytes[l.scales_offset + si..l.scales_offset + si + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            for t in 0..GROUP {
                let e = r * cols + g * GROUP + t;
                match fmt {
                    Format::Int8 => {
                        let q = (xs[t] * id).round().clamp(-127.0, 127.0) as i8;
                        bytes[e] = q as u8;
                        deq[e] = q as f32 * df;
                    }
                    _ => {
                        let q = ((xs[t] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                        let byte = (r * cols + g * GROUP) / 2 + t % 16;
                        bytes[byte] |= if t < 16 { q } else { q << 4 };
                        deq[e] = (q as f32 - 8.0) * df;
                    }
                }
            }
        }
    }
    (bytes, deq)
}

#[test]
fn gather_plain_and_quant() {
    let (rows, cols) = (20, 64);
    let mut rng = Rng(5);
    let table = rng.vec(rows * cols);
    let ids: Vec<i32> = vec![3, 0, 19, 3, 7];
    for be in backends() {
        let idx = be.upload(bytemuck::cast_slice(&ids)).unwrap();
        let dims = GatherDims { rows, cols, count: ids.len() };
        let spec = Arc::new(GatherSpec { table_dtype: DType::F16, out_dtype: DType::F32 });
        let out = run(&be, Kernel::Gather(spec, dims.clone()), vec![up(&be, &table, DType::F16), idx.clone()], ids.len() * cols * 4);
        let got = down(&be, &out, DType::F32);
        let tr = rt(&table, DType::F16);
        let want: Vec<f32> = ids.iter().flat_map(|&i| tr[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        close(&got, &want, 0.0);
        for fmt in [Format::Int8, Format::Int4] {
            let (bytes, deq) = quantize(fmt, &table, rows, cols);
            let dt = if fmt == Format::Int8 { DType::Int8 } else { DType::Int4 };
            let spec = Arc::new(GatherSpec { table_dtype: dt, out_dtype: DType::F32 });
            let out = run(&be, Kernel::Gather(spec, dims.clone()), vec![be.upload(&bytes).unwrap(), idx.clone()], ids.len() * cols * 4);
            let got = down(&be, &out, DType::F32);
            let want: Vec<f32> = ids.iter().flat_map(|&i| deq[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
            close(&got, &want, 1e-6);
        }
        // MXFP4 (AoS 17-byte blocks): rows dequantized exactly as lapid-quant's reference.
        let bytes = lapid_quant::quantize(Format::Mxfp4, &table, rows, cols);
        let deq = lapid_quant::dequantize(Format::Mxfp4, &bytes, rows, cols);
        let spec = Arc::new(GatherSpec { table_dtype: DType::Mxfp4, out_dtype: DType::F32 });
        let out = run(&be, Kernel::Gather(spec, dims.clone()), vec![be.upload(&bytes).unwrap(), idx.clone()], ids.len() * cols * 4);
        let want: Vec<f32> = ids.iter().flat_map(|&i| deq[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        close(&down(&be, &out, DType::F32), &want, 0.0);
        // Out-of-range index is an error, not UB.
        let bad = be.upload(bytemuck::cast_slice(&[0i32, 20])).unwrap();
        let spec = Arc::new(GatherSpec { table_dtype: DType::F32, out_dtype: DType::F32 });
        let out = be.alloc(2 * cols * 4).unwrap();
        let l = Launch { kernel: Kernel::Gather(spec, GatherDims { rows, cols, count: 2 }), inputs: vec![0, 1], output: 2 };
        let err = be.execute(&[l], &[up(&be, &table, DType::F32), bad, out]).unwrap_err();
        assert_eq!(err.code, "index_out_of_range");
    }
}

/// MatMulId over MXFP4 and i-quant expert stacks `[E, N, K]` (each expert `layout(..).total` bytes):
/// decode-sized (int8 activations) and prefill-sized (exact) row counts per expert.
#[test]
fn matmul_id_mxfp4() {
    for (dt, k) in [(DType::Mxfp4, 96), (DType::Iq2Xxs, 256), (DType::Iq2Xs, 256), (DType::Iq2S, 512), (DType::Iq3Xxs, 256)] {
        matmul_id_quant(dt, k);
    }
}

fn matmul_id_quant(dt: DType, k: usize) {
    let fmt = dt.quant_format().unwrap();
    let (e, n) = (5, 24);
    let mut rng = Rng(9);
    let one = lapid_quant::layout(fmt, n, k).total;
    let mut wb = vec![0u8; e * one];
    let mut wf = Vec::new();
    for x in 0..e {
        let w: Vec<f32> = rng.vec(n * k).iter().map(|v| v * 0.1).collect();
        let b = lapid_quant::quantize(fmt, &w, n, k);
        wb[x * one..x * one + b.len()].copy_from_slice(&b);
        wf.extend(lapid_quant::dequantize(fmt, &b, n, k));
    }
    for (t, r) in [(1usize, 2usize), (40, 3)] {
        let x = rng.vec(t * k);
        let ids: Vec<i32> = (0..t * r).map(|i| ((i * 7 + i / r) % e) as i32).collect();
        let mut want = vec![0f32; t * r * n];
        for i in 0..t * r {
            let ex = ids[i] as usize;
            for j in 0..n {
                let s: f64 = (0..k).map(|p| x[(i / r) * k + p] as f64 * wf[(ex * n + j) * k + p] as f64).sum();
                want[i * n + j] = s as f32;
            }
        }
        let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: dt, out_dtype: DType::F32 });
        let kernel = Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k });
        for be in backends() {
            let ins = vec![up(&be, &x, DType::F32), be.upload(&wb).unwrap(), be.upload(bytemuck::cast_slice(&ids)).unwrap()];
            let got = down(&be, &run(&be, kernel.clone(), ins, t * r * n * 4), DType::F32);
            let err = rel_err(&got, &want);
            assert!(err < if t * r <= 8 { 2e-2 } else { 1e-5 }, "{dt:?} t={t} r={r}: rel err {err}");
        }
    }
}

/// Reference matmul in f64 over already-rounded operands. `b(bt, p, j)` gives B values.
fn mm_ref(batch: usize, m: usize, n: usize, k: usize, a: impl Fn(usize, usize, usize) -> f32, b: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
    let mut out = vec![0f32; batch * m * n];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..n {
                out[(bt * m + i) * n + j] = (0..k).map(|p| a(bt, i, p) as f64 * b(bt, p, j) as f64).sum::<f64>() as f32;
            }
        }
    }
    out
}

#[test]
fn matmul_float_paths() {
    let mut rng = Rng(6);
    // (batch, m, n, k, transposed, b dtype, b batched, epilogue)
    let cases = [
        (1, 1, 37, 64, true, DType::F32, false, false),
        (1, 1, 37, 77, true, DType::F16, false, true),
        (2, 3, 70, 96, true, DType::F16, true, false),
        (1, 5, 1031, 64, true, DType::F16, false, true),
        (1, 3, 100, 33, false, DType::F16, false, true),
        (2, 7, 40, 70, true, DType::BF16, true, false),
        (2, 50, 70, 40, true, DType::F32, true, true),
        (1, 130, 200, 96, true, DType::F16, false, true),
        (3, 20, 33, 17, false, DType::F32, true, false),
    ];
    for &(batch, m, n, k, tr, bdt, b_batched, epi) in &cases {
        let a_rs = k + 3; // padded rows
        let a_bs = m * a_rs;
        let a = rng.vec(batch * a_bs);
        let b_rows = if tr { n } else { k };
        let b_cols = if tr { k } else { n };
        let b_bs = if b_batched { b_rows * b_cols } else { 0 };
        let b_count = if b_batched { batch } else { 1 };
        let b = rt(&rng.vec(b_count * b_rows * b_cols), bdt);
        let bias = rng.vec(n);
        let want = mm_ref(
            batch,
            m,
            n,
            k,
            |bt, i, p| a[bt * a_bs + i * a_rs + p],
            |bt, p, j| if tr { b[bt * b_bs + j * k + p] } else { b[bt * b_bs + p * n + j] },
        );
        let want: Vec<f32> = if epi {
            want.iter().enumerate().map(|(e, &v)| (v + bias[e % n]).max(0.0)).collect()
        } else {
            want
        };
        for be in backends() {
            let epilogue = epi.then(|| {
                Expr::Binary(BinaryOp::Max, bx(Expr::Binary(BinaryOp::Add, bx(Expr::Load(0)), bx(Expr::Load(1)))), bx(Expr::konst(0.0)))
            });
            let spec = Arc::new(MatMulSpec {
                a_dtype: DType::F32,
                b_dtype: bdt,
                b_transposed: tr,
                extra_dtypes: if epi { vec![DType::F32] } else { vec![] },
                out_dtype: DType::F32,
                epilogue,
            });
            let dims = MatMulDims {
                batch,
                m,
                n,
                k,
                a: MatView { offset: 0, batch_stride: a_bs, row_stride: a_rs },
                b: MatView { offset: 0, batch_stride: b_bs, row_stride: b_cols },
                extra_views: if epi { vec![View { offset: 0, strides: vec![0, 0, 1] }] } else { vec![] },
                out: MatView::contiguous(m, n),
            };
            let mut inputs = vec![up(&be, &a, DType::F32), up(&be, &b, bdt)];
            if epi {
                inputs.push(up(&be, &bias, DType::F32));
            }
            let out = run(&be, Kernel::MatMul(spec, dims), inputs, batch * m * n * 4);
            let got = down(&be, &out, DType::F32);
            close(&got, &want, 1e-4);
        }
    }
}

#[test]
fn matmul_quantized() {
    let mut rng = Rng(7);
    let (n, k) = (45, 96);
    for fmt in [Format::Int8, Format::Int4] {
        let w = rng.vec(n * k);
        let (bytes, deq) = quantize(fmt, &w, n, k);
        let dt = if fmt == Format::Int8 { DType::Int8 } else { DType::Int4 };
        for (batch, m) in [(1, 1), (1, 5), (2, 35)] {
            let a = rng.vec(batch * m * k);
            let want = mm_ref(batch, m, n, k, |bt, i, p| a[(bt * m + i) * k + p], |_, p, j| deq[j * k + p]);
            let mut results = Vec::new();
            for be in backends() {
                let spec = Arc::new(MatMulSpec {
                    a_dtype: DType::F32,
                    b_dtype: dt,
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
                    a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
                    b: MatView { offset: 0, batch_stride: 0, row_stride: k },
                    extra_views: vec![],
                    out: MatView::contiguous(m, n),
                };
                let out = run(&be, Kernel::MatMul(spec, dims), vec![up(&be, &a, DType::F32), be.upload(&bytes).unwrap()], batch * m * n * 4);
                let got = down(&be, &out, DType::F32);
                // Activation quantization error only (B is exactly the dequantized matrix).
                assert!(rel_err(&got, &want) < 1.5e-2, "{fmt:?} m={m}: rel err {}", rel_err(&got, &want));
                results.push(got);
            }
            // SIMD and scalar kernels agree closely (bf16 large-m path: ~2e-3).
            assert!(rel_err(&results[0], &results[1]) < 5e-3);
        }
    }
}

#[test]
fn attention_matches_reference() {
    let mut rng = Rng(8);
    // (b, h, sq, sk, d, causal, mask)
    let cases = [
        (1, 2, 70, 90, 40, false, true),
        (1, 3, 5, 9, 64, true, false),
        (2, 2, 100, 100, 16, true, true),
        (1, 1, 1, 130, 64, false, false),
        // decode path: split keys over threads, causal with a cache, masks
        (1, 3, 1, 1500, 64, false, false),
        (1, 2, 1, 700, 64, true, true),
        (1, 2, 4, 40, 64, true, true),
        (2, 2, 3, 900, 32, true, false),
    ];
    for &(bsz, h, sq, sk, d, causal, has_mask) in &cases {
        let scale = 1.0 / (d as f32).sqrt();
        // q stored as [B, S, H, D] (un-permuted projection output), k/v contiguous.
        let q_store = rng.vec(bsz * sq * h * d);
        let k = rng.vec(bsz * h * sk * d);
        let v = rng.vec(bsz * h * sk * d);
        let mask: Vec<f32> = (0..sq * sk).map(|e| if e % 7 == 3 { f32::NEG_INFINITY } else { rng.f() }).collect();
        let mut q = vec![0f32; bsz * h * sq * d];
        for b in 0..bsz {
            for hh in 0..h {
                for s in 0..sq {
                    for dd in 0..d {
                        q[((b * h + hh) * sq + s) * d + dd] = q_store[((b * sq + s) * h + hh) * d + dd];
                    }
                }
            }
        }
        let full_mask: Vec<f32> = (0..bsz * h).flat_map(|_| mask.clone()).collect();
        let want = attention::reference(&q, &k, &v, has_mask.then_some(&full_mask[..]), (bsz, h, sq, sk, d), scale, causal);
        for be in backends() {
            let spec = Arc::new(AttentionSpec {
                dtype: DType::F32,
                mask_dtype: has_mask.then_some(DType::F32),
                causal,
                out_dtype: DType::F32,
            });
            let dims = AttentionDims {
                batch: bsz,
                heads: h,
                sq,
                sk,
                d,
                scale: scale.to_bits(),
                q: View { offset: 0, strides: vec![(sq * h * d) as isize, d as isize, (h * d) as isize, 1] },
                k: View::contiguous(&[bsz, h, sk, d]),
                v: View::contiguous(&[bsz, h, sk, d]),
                mask: has_mask.then(|| View { offset: 0, strides: vec![0, 0, sk as isize, 1] }),
            };
            let mut inputs = vec![up(&be, &q_store, DType::F32), up(&be, &k, DType::F32), up(&be, &v, DType::F32)];
            if has_mask {
                inputs.push(up(&be, &mask, DType::F32));
            }
            let out = run(&be, Kernel::Attention(spec, dims), inputs, bsz * h * sq * d * 4);
            let got = down(&be, &out, DType::F32);
            close(&got, &want, 2e-5);
        }
    }
}

#[test]
fn launch_validation() {
    let be = CpuBackend::with_threads(2);
    let x = up(&be, &[1.0, 2.0, 3.0], DType::F32);
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) });
    // Aliasing input and output.
    let l = Launch {
        kernel: Kernel::Elementwise(spec.clone(), EwDims { shape: vec![3], in_views: vec![View::contiguous(&[3])], out_view: View::contiguous(&[3]) }),
        inputs: vec![0],
        output: 1,
    };
    assert_eq!(be.execute(std::slice::from_ref(&l), &[x.clone(), x.clone()]).unwrap_err().code, "buffer_aliasing");
    // Out-of-bounds view.
    let out = be.alloc(12).unwrap();
    let l = Launch {
        kernel: Kernel::Elementwise(spec, EwDims { shape: vec![4], in_views: vec![View::contiguous(&[4])], out_view: View::contiguous(&[4]) }),
        inputs: vec![0],
        output: 1,
    };
    assert_eq!(be.execute(&[l], &[x, out]).unwrap_err().code, "out_of_bounds");
    assert!(be.device_name().starts_with("cpu: 2 threads"));
    assert_eq!(be.name(), "cpu");
    assert!(!CpuBackend::features().is_empty());
}
