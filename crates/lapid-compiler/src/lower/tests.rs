//! Lowering tests: build small graphs, assert on the emitted launches.

use lapid_tensor::{DType, Tensor};

use crate::ir::{BufferSource, Kernel, Schedule};
use crate::{CompileOptions, compile, dump};

fn host(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    Tensor::from_f32(&data, shape).unwrap()
}

fn run(outs: &[Tensor]) -> Schedule {
    compile(outs, &CompileOptions::default()).unwrap()
}

fn run_with(outs: &[Tensor], fuse: bool, native: bool) -> Schedule {
    compile(outs, &CompileOptions { fuse, native_composites: native, native_sinkhorn: false }).unwrap()
}

fn kinds(s: &Schedule) -> Vec<&'static str> {
    s.launches.iter().map(|l| l.kernel.kind()).collect()
}

fn is_output(s: &Schedule, buf: usize) -> bool {
    matches!(s.buffers[buf].source, BufferSource::Output)
}

#[test]
fn fused_elementwise_chain_is_one_launch() {
    let a = host(&[4, 8]);
    let b = host(&[4, 8]);
    let y = a.add(&b).unwrap().exp().unwrap().mul_scalar(2.0).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["elementwise"]);
    let l = &s.launches[0];
    assert_eq!(l.inputs.len(), 2);
    assert!(is_output(&s, l.output));
    assert_eq!(s.outputs[0].buf, l.output);
    let Kernel::Elementwise(spec, dims) = &l.kernel else { panic!() };
    assert_eq!(spec.expr.num_loads(), 2);
    // Both inputs and the output are dense: coalesced to rank 1.
    assert_eq!(dims.shape, vec![32]);
}

#[test]
fn no_fuse_emits_one_kernel_per_op_and_reuses_slots() {
    let a = host(&[64]);
    let mut y = a.clone();
    for _ in 0..5 {
        y = y.exp().unwrap();
    }
    let s = run_with(&[y], false, true);
    assert_eq!(kinds(&s), ["elementwise"; 5]);
    // 4 temps in a chain need at most 2 slots.
    assert!(s.slot_bytes.len() <= 2, "{}", dump(&s));
    assert!(is_output(&s, s.launches[4].output));
}

#[test]
fn broadcast_bias_uses_zero_stride() {
    let a = host(&[4, 8]);
    let bias = host(&[8]);
    let s = run(&[a.add(&bias).unwrap()]);
    assert_eq!(kinds(&s), ["elementwise"]);
    let Kernel::Elementwise(_, dims) = &s.launches[0].kernel else { panic!() };
    assert_eq!(dims.shape, vec![4, 8]);
    assert!(dims.in_views.iter().any(|v| v.strides == vec![0, 1]));
}

#[test]
fn matmul_bias_becomes_epilogue() {
    let x = host(&[3, 8]);
    let w = host(&[16, 8]);
    let b = host(&[16]);
    let y = x.matmul_t(&w).unwrap().add(&b).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["matmul"], "{}", dump(&s));
    let Kernel::MatMul(spec, dims) = &s.launches[0].kernel else { panic!() };
    assert!(spec.b_transposed);
    assert!(spec.epilogue.is_some());
    assert_eq!((dims.batch, dims.m, dims.n, dims.k), (1, 3, 16, 8));
    assert_eq!(dims.extra_views.len(), 1);
    assert_eq!(dims.extra_views[0].strides, vec![0, 0, 1]);
    assert!(is_output(&s, s.launches[0].output));
}

#[test]
fn matmul_gelu_fuses_into_at_most_two_launches() {
    let x = host(&[3, 8]);
    let w = host(&[16, 8]);
    let b = host(&[16]);
    let y = x.matmul_t(&w).unwrap().add(&b).unwrap().gelu().unwrap();
    let s = run(&[y]);
    assert_eq!(s.launches[0].kernel.kind(), "matmul");
    assert!(s.launches.len() <= 2, "{}", dump(&s));
}

#[test]
fn transposed_weight_view_needs_no_copy() {
    let a = host(&[3, 8]);
    let w = host(&[8, 16]);
    let wt = w.permute(&[1, 0]).unwrap(); // [16, 8], strides (1, 16)
    let s = run(&[a.matmul_t(&wt).unwrap()]);
    assert_eq!(kinds(&s), ["matmul"]);
    let Kernel::MatMul(spec, dims) = &s.launches[0].kernel else { panic!() };
    assert!(!spec.b_transposed);
    assert_eq!(dims.b.row_stride, 16);
}

#[test]
fn batched_matmul_broadcasts_b() {
    let a = host(&[2, 3, 8]);
    let w = host(&[8, 5]);
    let s = run(&[a.matmul(&w).unwrap()]);
    assert_eq!(kinds(&s), ["matmul"]);
    let Kernel::MatMul(_, dims) = &s.launches[0].kernel else { panic!() };
    assert_eq!((dims.batch, dims.m, dims.n, dims.k), (2, 3, 5, 8));
    assert_eq!(dims.a.batch_stride, 24);
    assert_eq!(dims.b.batch_stride, 0);
}

#[test]
fn attention_reads_head_views_directly() {
    let heads = |t: &Tensor| t.reshape(&[1, 5, 2, 4]).unwrap().permute(&[0, 2, 1, 3]).unwrap();
    let (q, k, v) = (heads(&host(&[1, 5, 8])), heads(&host(&[1, 5, 8])), heads(&host(&[1, 5, 8])));
    let y = Tensor::attention(&q, &k, &v, None, 0.5, true).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["attention"], "{}", dump(&s));
    let Kernel::Attention(spec, dims) = &s.launches[0].kernel else { panic!() };
    assert!(spec.causal);
    assert_eq!((dims.batch, dims.heads, dims.sq, dims.sk, dims.d), (1, 2, 5, 5, 4));
    // Size-1 batch dim gets stride 0 from the reshape.
    assert_eq!(dims.q.strides, vec![0, 4, 8, 1]);
    assert!(dims.mask.is_none());
}

#[test]
fn projection_then_heads_uses_matmul_layout() {
    // q = (x @ Wq^T + bq) reshaped to heads: one epilogue matmul + attention, no copies.
    let x = host(&[1, 5, 8]);
    let proj = |w: &Tensor, b: &Tensor| {
        x.matmul_t(w).unwrap().add(b).unwrap().reshape(&[1, 5, 2, 4]).unwrap().permute(&[0, 2, 1, 3]).unwrap()
    };
    let q = proj(&host(&[8, 8]), &host(&[8]));
    let k = proj(&host(&[8, 8]), &host(&[8]));
    let v = proj(&host(&[8, 8]), &host(&[8]));
    let y = Tensor::attention(&q, &k, &v, None, 0.5, false).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["matmul", "matmul", "matmul", "attention"], "{}", dump(&s));
}

#[test]
fn softmax_native_single_launch() {
    let x = host(&[2, 3, 5]);
    let s = run(&[x.softmax().unwrap()]);
    assert_eq!(kinds(&s), ["softmax"]);
    let Kernel::Softmax(_, d) = &s.launches[0].kernel else { panic!() };
    assert_eq!((d.rows, d.cols), (6, 5));
}

#[test]
fn softmax_decomposed_without_native_composites() {
    let x = host(&[2, 5]);
    let s = run_with(&[x.softmax().unwrap()], true, false);
    assert!(!kinds(&s).contains(&"softmax"));
    assert!(kinds(&s).contains(&"reduce"));
}

#[test]
fn layer_norm_native() {
    let x = host(&[2, 8]);
    let s = run(&[x.layer_norm(&host(&[8]), &host(&[8]), 1e-5).unwrap()]);
    assert_eq!(kinds(&s), ["layer_norm"]);
}

#[test]
fn reduce_moves_axis_last() {
    let x = host(&[4, 6]);
    let s = run(&[x.sum(0, false).unwrap()]);
    assert_eq!(kinds(&s), ["reduce"]);
    let Kernel::Reduce(_, d) = &s.launches[0].kernel else { panic!() };
    assert_eq!(d.shape, vec![6, 4]);
    assert_eq!(d.in_views[0].strides, vec![1, 6]);
}

#[test]
fn pad_emits_fills_and_interior() {
    let x = host(&[2, 3]);
    let s = run(&[x.pad(1, 1, 2, 0.0).unwrap()]);
    assert_eq!(kinds(&s), ["elementwise"; 3]);
    let out = s.launches[0].output;
    assert!(s.launches.iter().all(|l| l.output == out));
    let Kernel::Elementwise(_, d) = &s.launches[2].kernel else { panic!() };
    assert_eq!(d.out_view.offset, 1);
}

#[test]
fn cat_writes_each_piece() {
    let a = host(&[2, 3]);
    let b = host(&[2, 1]);
    let s = run(&[Tensor::cat(&[&a, &b], 1).unwrap()]);
    assert_eq!(kinds(&s), ["elementwise"; 2]);
    let Kernel::Elementwise(_, d) = &s.launches[1].kernel else { panic!() };
    assert_eq!(d.out_view.offset, 3);
}

#[test]
fn gather_from_table() {
    let table = host(&[10, 4]);
    let idx = Tensor::from_i32(&[1, 7, 3], &[3]).unwrap();
    let s = run(&[table.gather(&idx).unwrap()]);
    assert_eq!(kinds(&s), ["gather"]);
    let Kernel::Gather(spec, d) = &s.launches[0].kernel else { panic!() };
    assert_eq!(spec.table_dtype, DType::F32);
    assert_eq!((d.rows, d.cols, d.count), (10, 4, 3));
}

#[test]
fn reshape_of_transpose_copies_once() {
    let x = host(&[4, 6]);
    let y = x.permute(&[1, 0]).unwrap().reshape(&[24]).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["elementwise"], "{}", dump(&s));
    assert!(is_output(&s, s.launches[0].output));
}

#[test]
fn leaf_output_is_copied() {
    let x = host(&[3, 3]);
    let s = run(&[x]);
    assert_eq!(kinds(&s), ["elementwise"]);
    assert!(matches!(s.buffers[s.launches[0].inputs[0]].source, BufferSource::Host(_)));
    assert!(is_output(&s, s.launches[0].output));
}

#[test]
fn duplicate_outputs_get_distinct_buffers() {
    let y = host(&[5]).exp().unwrap();
    let s = run(&[y.clone(), y]);
    assert_eq!(s.outputs.len(), 2);
    assert_ne!(s.outputs[0].buf, s.outputs[1].buf);
    assert!(is_output(&s, s.outputs[0].buf) && is_output(&s, s.outputs[1].buf));
}

#[test]
fn cast_to_output_dtype_is_the_store() {
    let x = host(&[4]);
    let y = x.exp().unwrap().cast(DType::F16).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["elementwise"]);
    let Kernel::Elementwise(spec, _) = &s.launches[0].kernel else { panic!() };
    assert_eq!(spec.out_dtype, DType::F16);
    assert_eq!(spec.in_dtypes, vec![DType::F32]);
}

#[test]
fn shared_intermediate_is_materialized_once() {
    let x = host(&[8]);
    let h = x.exp().unwrap().tanh().unwrap(); // 2 ops, 3 users
    let y = h.add(&h.mul(&h).unwrap()).unwrap().add(&h).unwrap();
    let s = run(&[y]);
    assert_eq!(kinds(&s), ["elementwise", "elementwise"], "{}", dump(&s));
}

#[test]
fn dump_is_readable() {
    let x = host(&[3, 8]);
    let w = host(&[16, 8]);
    let s = run(&[x.matmul_t(&w).unwrap()]);
    let d = dump(&s);
    assert!(d.starts_with("#0 matmul f32 x f32^T -> f32"), "{d}");
    assert!(d.contains("buffers:") && d.contains("outputs:"));
}
