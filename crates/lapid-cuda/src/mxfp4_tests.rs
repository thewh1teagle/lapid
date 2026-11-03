//! MXFP4 matmul (GEMV, tensor-core GEMM, scalar GEMM), gather and MatMulId tests on CUDA
//! device 0 against f64 references over `lapid_quant`'s reference dequantization.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_quant::Format;
use lapid_tensor::{DType, DeviceData};

use crate::CudaBackend;
use crate::tests::{bytes, check, cuda, floats, mat_spec, rnd, run_on};

/// Random MXFP4 matrix: random nibbles, scale bytes `e` in 118..134 (or any byte).
fn random(rows: usize, cols: usize, seed: u64, any_scale: bool) -> Vec<u8> {
    let total = lapid_quant::layout(Format::Mxfp4, rows, cols).total;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut b: Vec<u8> = (0..total)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    if !any_scale {
        for blk in b[..rows * cols / 32 * 17].as_chunks_mut::<17>().0 {
            blk[0] = 118 + blk[0] % 16;
        }
    }
    b
}

fn deq(b: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    lapid_quant::dequantize(Format::Mxfp4, b, rows, cols)
}

fn naive(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: f64 = (0..k).map(|t| a[i * k + t] as f64 * w[j * k + t] as f64).sum();
            out[i * n + j] = s as f32;
        }
    }
    out
}

fn matmul_case(be: &dyn Backend, a_t: DType, m: usize, n: usize, k: usize, tol: f32) {
    let spec = mat_spec(a_t, DType::Mxfp4, true, vec![], None);
    let a = floats(a_t, &bytes(a_t, &rnd(m * k, 3)));
    let wb = random(n, k, 4, false);
    let w = deq(&wb, n, k);
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView { offset: 0, batch_stride: m * k, row_stride: k },
        b: MatView { offset: 0, batch_stride: 0, row_stride: k },
        extra_views: vec![],
        out: MatView::contiguous(m, n),
    };
    let got = floats(DType::F32, &run_on(be, &Kernel::MatMul(spec, dims), &[bytes(a_t, &a), wb], m * n * 4));
    check(&format!("matmul {a_t:?} x mxfp4 {m}x{n}x{k}"), &got, &naive(&a, &w, m, n, k), tol);
}

#[test]
fn mxfp4_gemv() {
    let Some(be) = cuda() else { return };
    // k not a multiple of 256 (partial warp steps), odd n (block alignment varies by row)
    for (m, n, k) in [(1, 67, 512), (2, 33, 96), (3, 41, 224), (1, 9, 32), (1, 300, 2048)] {
        matmul_case(&be, DType::F32, m, n, k, 1e-5);
    }
    matmul_case(&be, DType::F16, 1, 40, 768, 1e-5);
}

#[test]
fn mxfp4_gemm() {
    let Some(be) = cuda() else { return };
    // tensor-core path: f16 operands
    matmul_case(&be, DType::F32, 150, 200, 512, 2e-3);
    matmul_case(&be, DType::F32, 70, 130, 96, 2e-3);
    matmul_case(&be, DType::F16, 70, 131, 256, 2e-3);
    // scalar GEMM (bf16 activations): element loads
    matmul_case(&be, DType::BF16, 20, 30, 96, 1e-5);
}

#[test]
fn mxfp4_gather() {
    let Some(be) = cuda() else { return };
    for (rows, cols) in [(37, 512), (9, 96)] {
        // every scale byte: exact (denormal d for e < 2 included)
        let tb = random(rows, cols, 5, true);
        let t = deq(&tb, rows, cols);
        let idx = [3i32, 0, (rows - 1) as i32, 7, 3];
        let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec = Arc::new(GatherSpec { table_dtype: DType::Mxfp4, out_dtype: DType::F32 });
        let k = Kernel::Gather(spec, GatherDims { rows, cols, count: idx.len() });
        let got = floats(DType::F32, &run_on(&be, &k, &[tb, ib], idx.len() * cols * 4));
        let want: Vec<f32> = idx.iter().flat_map(|&i| t[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "gather {rows}x{cols} element {i}: {g} vs {w}");
        }
    }
}

fn run_id(be: &CudaBackend, x_t: DType, (t, r, e, n, k): (usize, usize, usize, usize, usize), ids: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let x = floats(x_t, &bytes(x_t, &rnd(t * k, 11)));
    let one = lapid_quant::layout(Format::Mxfp4, n, k).total;
    let mut wb = Vec::new();
    let mut wf = Vec::new();
    for ex in 0..e {
        let b = random(n, k, 20 + ex as u64, false);
        assert_eq!(b.len(), one);
        wf.extend(deq(&b, n, k));
        wb.extend(b);
    }
    let spec = Arc::new(MatMulIdSpec { x_dtype: x_t, w_dtype: DType::Mxfp4, out_dtype: DType::F32 });
    let kernel = Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k });
    let ib: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs = [bytes(x_t, &x), wb, ib];
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.upload(&vec![0u8; t * r * n * 4]).unwrap());
    be.execute(&[Launch { kernel, inputs: vec![0, 1, 2], output: 3 }], &bufs).unwrap();
    let got = floats(DType::F32, &be.download(&bufs[3]).unwrap());
    let mut want = vec![0f32; t * r * n];
    for i in 0..t * r {
        let ex = ids[i] as usize;
        let o = naive(&x[(i / r) * k..(i / r + 1) * k], &wf[ex * n * k..(ex + 1) * n * k], 1, n, k);
        want[i * n..(i + 1) * n].copy_from_slice(&o);
    }
    (got, want)
}

#[test]
fn mxfp4_matmul_id() {
    let Some(be) = cuda() else { return };
    for (t, r, e, n, k, tol) in [(1, 2, 5, 70, 256, 1e-5), (3, 2, 4, 33, 96, 1e-5), (70, 3, 7, 150, 256, 2e-3), (64, 8, 64, 96, 160, 2e-3), (8, 1, 6, 40, 256, 1e-5)] {
        let ids: Vec<i32> = (0..t * r).map(|i| ((i * 7 + (i / r) * 3) % e) as i32).collect();
        let (got, want) = run_id(&be, DType::F32, (t, r, e, n, k), &ids);
        check(&format!("matmul_id mxfp4 t{t} r{r} e{e} n{n} k{k}"), &got, &want, tol);
    }
}
