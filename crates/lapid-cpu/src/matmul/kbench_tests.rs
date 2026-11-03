//! Quantized GEMV / GEMM micro-benchmarks on Qwen3-0.6B shapes (ignored tests).
//! `LAPID_THREADS=10 cargo test --release -p lapid-cpu kbench_ -- --ignored --nocapture --test-threads=1`

use std::sync::Arc;
use std::time::Instant;

use lapid_compiler::{Backend, Kernel, Launch, MatMulDims, MatMulSpec, MatView};
use lapid_quant::Format;
use lapid_tensor::{DType, DeviceData};

use crate::CpuBackend;

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

/// Random quantized bytes with sane scales (no reference quantizer: fast for 150M elems).
pub(crate) fn rand_weight(fmt: Format, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let l = lapid_quant::layout(fmt, n, k);
    let mut v = vec![0u8; l.total];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = (s >> 32) as u8;
    }
    let d = half::f16::from_f32(0.01).to_le_bytes();
    let dm = half::f16::from_f32(0.004).to_le_bytes();
    match fmt {
        Format::Int8 | Format::Int4 => {
            for g in 0..l.scales_bytes / 2 {
                v[l.scales_offset + 2 * g..][..2].copy_from_slice(&d);
            }
            if fmt == Format::Int8 {
                v[..l.qs_bytes].iter_mut().for_each(|b| *b = (*b as i8).max(-127) as u8);
            }
        }
        _ => {
            for b in v[..l.qs_bytes].chunks_exact_mut(fmt.block_bytes()) {
                if fmt == Format::Q6K {
                    b[208..210].copy_from_slice(&d);
                } else {
                    b[0..2].copy_from_slice(&d);
                    b[2..4].copy_from_slice(&dm);
                }
            }
        }
    }
    v
}

pub(crate) fn dtype(fmt: Format) -> DType {
    match fmt {
        Format::Int8 => DType::Int8,
        Format::Int4 => DType::Int4,
        Format::Q4K => DType::Q4K,
        Format::Q5K => DType::Q5K,
        Format::Q6K => DType::Q6K,
        Format::Mxfp4 => DType::Mxfp4,
    }
}

pub(crate) fn mm(b_dtype: DType, m: usize, n: usize, k: usize) -> Kernel {
    Kernel::MatMul(
        Arc::new(MatMulSpec {
            a_dtype: DType::F32,
            b_dtype,
            b_transposed: true,
            extra_dtypes: vec![],
            out_dtype: DType::F32,
            epilogue: None,
        }),
        MatMulDims {
            batch: 1,
            m,
            n,
            k,
            a: MatView { offset: 0, batch_stride: 0, row_stride: k },
            b: MatView { offset: 0, batch_stride: 0, row_stride: k },
            extra_views: vec![],
            out: MatView::contiguous(m, n),
        },
    )
}

/// Median ms per launch over `reps` executions of `launches`.
fn time(be: &CpuBackend, launches: &[Launch], bufs: &[DeviceData], reps: usize) -> (f64, f64) {
    be.prepare(launches).unwrap();
    for _ in 0..3 {
        be.execute(launches, bufs).unwrap();
    }
    let mut v = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        be.execute(launches, bufs).unwrap();
        v.push(t.elapsed().as_secs_f64() * 1e3 / launches.len() as f64);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    (v[v.len() / 2], v[0])
}

const SHAPES: [(usize, usize, Format); 8] = [
    (2048, 1024, Format::Q4K),
    (1024, 1024, Format::Q4K),
    (1024, 1024, Format::Q6K),
    (1024, 2048, Format::Q4K),
    (3072, 1024, Format::Q4K),
    (1024, 3072, Format::Q4K),
    (1024, 3072, Format::Q6K),
    (151936, 1024, Format::Q6K),
];

#[test]
#[ignore]
fn kbench_gemv() {
    let be = CpuBackend::new();
    eprintln!("{}", be.device_name());
    let mut shapes: Vec<(usize, usize, Format)> = SHAPES.to_vec();
    shapes.extend([(3072, 1024, Format::Q5K), (3072, 1024, Format::Int8), (3072, 1024, Format::Int4), (64, 1024, Format::Q4K), (256, 1024, Format::Q4K)]);
    for m in [1usize, 2] {
        for &(n, k, fmt) in &shapes {
            let a = be.upload(bytemuck::cast_slice(&rnd(m * k, 3))).unwrap();
            let out = be.alloc(m * n * 4).unwrap();
            let wbytes = lapid_quant::layout(fmt, n, k).total;
            // Enough distinct copies (cycled) that the set does not live in cache.
            let copies = std::env::var("KB_COPIES").ok().and_then(|v| v.parse().ok()).unwrap_or((96usize << 20).div_ceil(wbytes).clamp(1, 64));
            let mut bufs = vec![a, out];
            let mut launches = Vec::new();
            for c in 0..copies {
                bufs.push(be.upload(&rand_weight(fmt, n, k, 7 + c as u64)).unwrap());
                launches.push(Launch { kernel: mm(dtype(fmt), m, n, k), inputs: vec![0, bufs.len() - 1], output: 1 });
            }
            let (ms, min) = time(&be, &launches, &bufs, if copies == 1 { 10 } else { 20 });
            eprintln!(
                "gemv m={m} {fmt:?} n={n:<6} k={k:<5} {:>8.1} us (min {:>7.1})  {:>6.1} GB/s",
                ms * 1e3,
                min * 1e3,
                wbytes as f64 / ms / 1e6
            );
        }
    }
}

#[test]
#[ignore]
fn kbench_gemm() {
    let be = CpuBackend::new();
    eprintln!("{}", be.device_name());
    let ms: Vec<usize> = std::env::var("KB_M").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![64, 512]);
    for m in ms {
        for &(n, k, fmt) in &SHAPES[..7] {
            let a = be.upload(bytemuck::cast_slice(&rnd(m * k, 3))).unwrap();
            let out = be.alloc(m * n * 4).unwrap();
            let b = be.upload(&rand_weight(fmt, n, k, 7)).unwrap();
            let l = [Launch { kernel: mm(dtype(fmt), m, n, k), inputs: vec![0, 2], output: 1 }];
            let (ms, _) = time(&be, &l, &[a, out, b], 10);
            eprintln!("gemm m={m} {fmt:?} n={n:<6} k={k:<5} {:>8.1} us  {:>6.0} GFLOP/s", ms * 1e3, (2 * m * n * k) as f64 / ms / 1e6);
        }
    }
}

/// Read-only roofline: sum the bytes of `size`-byte buffers (cycled out of cache) through the
/// pool, split into `tasks` contiguous chunks.
#[test]
#[ignore]
fn kbench_read() {
    use crate::par;
    let threads: usize = std::env::var("LAPID_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let pool = par::Pool::new(threads);
    for size in [600usize << 10, 1200 << 10, 1800 << 10, 2600 << 10, 128 << 20] {
        let copies = (96usize << 20).div_ceil(size).clamp(1, 64);
        let bufs: Vec<Vec<u8>> = (0..copies).map(|c| vec![c as u8 + 1; size]).collect();
        for tasks in [threads, threads * 2, threads * 4] {
            let chunk = size.div_ceil(tasks).div_ceil(64) * 64;
            let mut v = Vec::new();
            pool.install(|| {
                for rep in 0..23 {
                    let t = Instant::now();
                    for b in &bufs {
                        let s = par::map(true, tasks, || (), |_, i| {
                            let lo = (i * chunk).min(size);
                            let hi = ((i + 1) * chunk).min(size);
                            let mut acc = [0u64; 2];
                            for c in b[lo..hi].as_chunks::<16>().0 {
                                let w: [u64; 2] = bytemuck::cast(*c);
                                acc[0] = acc[0].wrapping_add(w[0]);
                                acc[1] = acc[1].wrapping_add(w[1]);
                            }
                            acc[0] ^ acc[1]
                        });
                        std::hint::black_box(s);
                    }
                    if rep >= 3 {
                        v.push(t.elapsed().as_secs_f64() / copies as f64);
                    }
                }
            });
            v.sort_by(|a, b| a.total_cmp(b));
            let s = v[v.len() / 2];
            eprintln!("read size={:>6}K tasks={tasks:<3} {:>8.1} us {:>6.1} GB/s", size >> 10, s * 1e6, size as f64 / s / 1e9);
        }
    }
}

#[test]
#[ignore]
fn kbench_quant() {
    for k in [1024usize, 3072] {
        let x = rnd(k, 1);
        let mut q = vec![0i8; k];
        let mut s = vec![0f32; k / 32];
        let mut b = vec![0f32; k / 32];
        let t = Instant::now();
        for _ in 0..10000 {
            super::quant::quantize_row_sums(std::hint::black_box(&x), &mut q, &mut s, Some(&mut b));
        }
        eprintln!("read quant k={k}: {:.3} us", t.elapsed().as_secs_f64() * 1e6 / 10000.0);
    }
}

/// Per-launch overhead: many small in-cache GEMVs (for `perf record`).
#[test]
#[ignore]
fn kbench_tiny() {
    let be = CpuBackend::new();
    let (n, k) = (256usize, 1024usize);
    let fmt = Format::Q4K;
    let a = be.upload(bytemuck::cast_slice(&rnd(k, 3))).unwrap();
    let out = be.alloc(n * 4).unwrap();
    let w = be.upload(&rand_weight(fmt, n, k, 7)).unwrap();
    let l: Vec<Launch> = (0..200).map(|_| Launch { kernel: mm(dtype(fmt), 1, n, k), inputs: vec![0, 2], output: 1 }).collect();
    let bufs = [a, out, w];
    let reps: usize = std::env::var("KB_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let (ms, min) = time(&be, &l, &bufs, reps);
    eprintln!("gemv tiny n={n} k={k}: {:.2} us/launch (min {:.2})", ms * 1e3, min * 1e3);
}
