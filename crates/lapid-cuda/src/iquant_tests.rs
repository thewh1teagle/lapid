//! i-quant (IQ2_XXS / IQ2_S / IQ3_XXS) matmul (GEMV over int8 activations, tensor-core
//! GEMM, scalar GEMM), gather and MatMulId tests on CUDA device 0 against f64 references over
//! `lapid_quant`'s reference dequantization.

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{DType, DeviceData};

use crate::CudaBackend;
use crate::tests::{bytes, check, cuda, floats, mat_spec, rnd, run_on};

const IQ: [DType; 3] = [DType::Iq2Xxs, DType::Iq2S, DType::Iq3Xxs];

/// Random blocks: random grid indices, signs and sub-scales, f16 `d` in [0.002, 0.02).
fn random(d: DType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let f = d.quant_format().unwrap();
    let l = lapid_quant::layout(f, rows, cols);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut b: Vec<u8> = (0..l.total)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    for blk in b[..l.qs_bytes].chunks_exact_mut(f.block_bytes()) {
        let v = 0.002 + f32::from(blk[0]) / 255.0 * 0.018;
        blk[..2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    b
}

fn deq(d: DType, b: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    lapid_quant::dequantize(d.quant_format().unwrap(), b, rows, cols)
}

/// The GEMV pre-pass's int8 activations, dequantized: per 32 values d = amax / 127,
/// q = round_half_even(x * 127 / amax).
fn q8(a: &[f32]) -> Vec<f32> {
    a.chunks(32)
        .flat_map(|g| {
            let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (d, id) = (amax / 127.0, if amax > 0.0 { 127.0 / amax } else { 0.0 });
            g.iter().map(move |&x| (x * id).round_ties_even().clamp(-127.0, 127.0) * d).collect::<Vec<_>>()
        })
        .collect()
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

/// `exact_q8`: compare against the int8-quantized activations (GEMV path) at `tol`.
fn matmul_case(be: &dyn Backend, w_t: DType, a_t: DType, (m, n, k): (usize, usize, usize), exact_q8: bool, tol: f32) {
    let spec = mat_spec(a_t, w_t, true, vec![], None);
    let a = floats(a_t, &bytes(a_t, &rnd(m * k, 3)));
    let wb = random(w_t, n, k, 4);
    let w = deq(w_t, &wb, n, k);
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
    let ar = if exact_q8 { q8(&a) } else { a };
    check(&format!("matmul {a_t:?} x {w_t:?} {m}x{n}x{k}"), &got, &naive(&ar, &w, m, n, k), tol);
}

#[test]
fn iquant_gemv() {
    let Some(be) = cuda() else { return };
    for w in IQ {
        // k = 256 (one block, partial warp step), odd n (block alignment varies by row)
        for (m, n, k) in [(1, 67, 512), (2, 33, 256), (3, 41, 2304), (1, 300, 5120), (4, 9, 768)] {
            matmul_case(&be, w, DType::F32, (m, n, k), true, 1e-5);
        }
        matmul_case(&be, w, DType::F16, (1, 40, 768), true, 1e-5);
        // vs the unquantized activations
        matmul_case(&be, w, DType::F32, (1, 64, 1024), false, 2e-2);
    }
}

#[test]
fn iquant_gemm() {
    let Some(be) = cuda() else { return };
    for w in IQ {
        // tensor-core path: f16 operands
        matmul_case(&be, w, DType::F32, (150, 200, 512), false, 2e-3);
        matmul_case(&be, w, DType::F16, (70, 131, 256), false, 2e-3);
        // scalar GEMM (bf16 activations): element loads
        matmul_case(&be, w, DType::BF16, (70, 130, 256), false, 1e-5);
        // small outputs run as GEMV (int8 activations)
        matmul_case(&be, w, DType::BF16, (20, 30, 256), true, 1e-5);
    }
}

#[test]
fn iquant_gather() {
    let Some(be) = cuda() else { return };
    for w in IQ {
        let (rows, cols) = (37, 512);
        let tb = random(w, rows, cols, 5);
        let t = deq(w, &tb, rows, cols);
        let idx = [3i32, 0, (rows - 1) as i32, 7, 3];
        let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec = Arc::new(GatherSpec { table_dtype: w, out_dtype: DType::F32 });
        let k = Kernel::Gather(spec, GatherDims { rows, cols, count: idx.len() });
        let got = floats(DType::F32, &run_on(&be, &k, &[tb, ib], idx.len() * cols * 4));
        let want: Vec<f32> = idx.iter().flat_map(|&i| t[i as usize * cols..(i as usize + 1) * cols].to_vec()).collect();
        for (i, (g, x)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), x.to_bits(), "{w:?} gather element {i}: {g} vs {x}");
        }
    }
}

fn run_id(be: &CudaBackend, w_t: DType, (t, r, e, n, k): (usize, usize, usize, usize, usize), ids: &[i32], exact_q8: bool) -> (Vec<f32>, Vec<f32>) {
    let x = rnd(t * k, 11);
    let mut wb = Vec::new();
    let mut wf = Vec::new();
    for ex in 0..e {
        let b = random(w_t, n, k, 20 + ex as u64);
        wf.extend(deq(w_t, &b, n, k));
        wb.extend(b);
    }
    let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: w_t, out_dtype: DType::F32 });
    let kernel = Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k });
    let ib: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs = [bytes(DType::F32, &x), wb, ib];
    let mut bufs: Vec<DeviceData> = inputs.iter().map(|b| be.upload(b).unwrap()).collect();
    bufs.push(be.upload(&vec![0u8; t * r * n * 4]).unwrap());
    be.execute(&[Launch { kernel, inputs: vec![0, 1, 2], output: 3 }], &bufs).unwrap();
    let got = floats(DType::F32, &be.download(&bufs[3]).unwrap());
    let xr = if exact_q8 { q8(&x) } else { x };
    let mut want = vec![0f32; t * r * n];
    for i in 0..t * r {
        let ex = ids[i] as usize;
        let o = naive(&xr[(i / r) * k..(i / r + 1) * k], &wf[ex * n * k..(ex + 1) * n * k], 1, n, k);
        want[i * n..(i + 1) * n].copy_from_slice(&o);
    }
    (got, want)
}

#[test]
fn iquant_matmul_id() {
    let Some(be) = cuda() else { return };
    for w in IQ {
        // decode GEMV rows (int8 activations, exact vs the quantized reference) and routed GEMM
        for (t, r, e, n, k, q, tol) in [(1, 6, 9, 70, 512, true, 1e-5), (3, 2, 4, 33, 256, true, 1e-5), (70, 3, 7, 150, 256, false, 2e-3), (64, 8, 64, 96, 512, false, 2e-3)] {
            let ids: Vec<i32> = (0..t * r).map(|i| ((i * 7 + (i / r) * 3) % e) as i32).collect();
            let (got, want) = run_id(&be, w, (t, r, e, n, k), &ids, q);
            check(&format!("matmul_id {w:?} t{t} r{r} e{e} n{n} k{k}"), &got, &want, tol);
        }
    }
}

/// Many experts (an expert cache's pool of slots), ids spread over the whole range: the
/// decode GEMV rows and the routed GEMM.
#[test]
fn iquant_matmul_id_many_experts() {
    let Some(be) = cuda() else { return };
    for (t, r, q, tol) in [(1, 6, true, 1e-5), (70, 4, false, 2e-3)] {
        let (e, n, k) = (5000, 24, 256);
        let ids: Vec<i32> = (0..t * r).map(|i| ((i * 997 + 13) % e) as i32).collect();
        let (got, want) = run_id(&be, DType::Iq2Xxs, (t, r, e, n, k), &ids, q);
        check(&format!("matmul_id iq2_xxs t{t} r{r} e{e}"), &got, &want, tol);
    }
}
