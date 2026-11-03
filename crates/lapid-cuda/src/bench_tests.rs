//! Kernel micro-benchmarks on the Whisper-small launch shapes, timed with CUDA events.
//! Ignored by default; run with
//! `cargo test -p lapid-cuda --release bench_kernels -- --ignored --nocapture`
//! (`LAPID_BENCH_FILTER=gemv` narrows the set).

use std::sync::Arc;

use lapid_compiler::*;
use lapid_tensor::{BinaryOp, DType, DeviceData, UnaryOp};

use crate::{CudaBackend, devices};

fn load(i: usize) -> Box<Expr> {
    Box::new(Expr::Load(i))
}

fn add01() -> Expr {
    Expr::Binary(BinaryOp::Add, load(0), load(1))
}

/// gelu(acc + bias)
fn gelu_bias() -> Expr {
    let x = || Box::new(add01());
    Expr::Binary(
        BinaryOp::Mul,
        x(),
        Box::new(Expr::Binary(
            BinaryOp::Mul,
            Box::new(Expr::Binary(
                BinaryOp::Add,
                Box::new(Expr::Unary(
                    UnaryOp::Erf,
                    Box::new(Expr::Binary(
                        BinaryOp::Mul,
                        x(),
                        Box::new(Expr::konst(0.70710677)),
                    )),
                )),
                Box::new(Expr::konst(1.0)),
            )),
            Box::new(Expr::konst(0.5)),
        )),
    )
}

fn bytes_of(dt: DType, rows: usize, cols: usize) -> usize {
    match dt.quant_format() {
        Some(f) => lapid_quant::layout(f, rows, cols).total,
        None => rows * cols * dt.size(),
    }
}

struct Case {
    name: String,
    /// Inputs that are weights / caches: replicated so every launch reads them cold
    /// (from DRAM), as in a real model run.
    cold: Vec<usize>,
    /// Buffers with fixed contents (e.g. valid expert ids), else filler bytes.
    init: Vec<(usize, Vec<u8>)>,
    launch: Launch,
    /// Buffer sizes in bytes (inputs..., output).
    sizes: Vec<usize>,
    flops: f64,
    bytes: f64,
}

fn matmul_case(m: usize, n: usize, k: usize, b: DType, epi: Option<Expr>) -> Case {
    let extras = if epi.is_some() {
        vec![DType::F32]
    } else {
        vec![]
    };
    let spec = Arc::new(MatMulSpec {
        a_dtype: DType::F32,
        b_dtype: b,
        b_transposed: true,
        extra_dtypes: extras.clone(),
        out_dtype: DType::F32,
        epilogue: epi,
    });
    let dims = MatMulDims {
        batch: 1,
        m,
        n,
        k,
        a: MatView {
            offset: 0,
            batch_stride: 0,
            row_stride: k,
        },
        b: MatView {
            offset: 0,
            batch_stride: 0,
            row_stride: k,
        },
        extra_views: extras
            .iter()
            .map(|_| View {
                offset: 0,
                strides: vec![0, 0, 1],
            })
            .collect(),
        out: MatView::contiguous(m, n),
    };
    let mut sizes = vec![m * k * 4, bytes_of(b, n, k)];
    sizes.extend(extras.iter().map(|_| n * 4));
    sizes.push(m * n * 4);
    let ni = sizes.len() - 1;
    Case {
        name: format!("matmul {m}x{n}x{k} {}", b.name()),
        launch: Launch {
            kernel: Kernel::MatMul(spec, dims),
            inputs: (0..ni).collect(),
            output: ni,
        },
        cold: vec![1],
        init: vec![],
        flops: 2.0 * (m * n * k) as f64,
        bytes: (sizes.iter().sum::<usize>()) as f64,
        sizes,
    }
}

fn attention_case(sq: usize, sk: usize, encoder: bool, dt: DType, d: usize) -> Case {
    let h = 12usize;
    let spec = Arc::new(AttentionSpec {
        dtype: dt,
        mask_dtype: None,
        causal: false,
        out_dtype: DType::F32,
    });
    let (qv, kv) = if encoder {
        // [1500, 12, 64] activations viewed as [B, H, S, D]
        let v = View {
            offset: 0,
            strides: vec![0, d as isize, (h * d) as isize, 1],
        };
        (v.clone(), v)
    } else {
        (
            View {
                offset: 0,
                strides: vec![0, d as isize, 0, 1],
            },
            View {
                offset: 0,
                strides: vec![(h * sk * d) as isize, (sk * d) as isize, d as isize, 1],
            },
        )
    };
    let dims = AttentionDims {
        batch: 1,
        heads: h,
        sq,
        sk,
        d,
        scale: (1.0 / (d as f32).sqrt()).to_bits(),
        q: qv,
        k: kv.clone(),
        v: kv,
        mask: None,
    };
    let es = dt.size();
    let sizes = vec![
        h * sq * d * es,
        h * sk * d * es,
        h * sk * d * es,
        h * sq * d * 4,
    ];
    Case {
        name: format!("attention sq={sq} sk={sk} d={d} {}", dt.name()),
        launch: Launch {
            kernel: Kernel::Attention(spec, dims),
            inputs: vec![0, 1, 2],
            output: 3,
        },
        cold: if encoder { vec![] } else { vec![1, 2] },
        init: vec![],
        flops: 4.0 * (h * sq * sk * d) as f64,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

fn layer_norm_case(rows: usize) -> Case {
    let spec = Arc::new(LayerNormSpec {
        in_dtype: DType::F32,
        param_dtype: DType::F32,
        out_dtype: DType::F32,
    });
    let dims = LayerNormDims {
        rows,
        cols: 768,
        eps: 1e-5f32.to_bits(),
    };
    let sizes = vec![rows * 768 * 4, 768 * 4, 768 * 4, rows * 768 * 4];
    Case {
        name: format!("layer_norm {rows}x768"),
        launch: Launch {
            kernel: Kernel::LayerNorm(spec, dims),
            inputs: vec![0, 1, 2],
            output: 3,
        },
        cold: vec![],
        init: vec![],
        flops: 0.0,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

/// Plain copy of `n` f32 (read + write bandwidth reference).
fn copy_case(n: usize) -> Case {
    let spec = Arc::new(EwSpec { rank: 1, in_dtypes: vec![DType::F32], out_dtype: DType::F32, expr: Expr::Load(0) });
    let dims = EwDims { shape: vec![n], in_views: vec![View::contiguous(&[n])], out_view: View::contiguous(&[n]) };
    let sizes = vec![n * 4, n * 4];
    Case {
        name: format!("copy f32 {n}"),
        launch: Launch { kernel: Kernel::Elementwise(spec, dims), inputs: vec![0], output: 1 },
        cold: vec![],
        init: vec![],
        flops: 0.0,
        bytes: (n * 8) as f64,
        sizes,
    }
}

/// MatMulId: t tokens x r experts of e, [n, k] experts of dtype `w` (ids spread evenly).
fn moe_case(t: usize, r: usize, e: usize, n: usize, k: usize, w: DType) -> Case {
    let spec = Arc::new(MatMulIdSpec { x_dtype: DType::F32, w_dtype: w, out_dtype: DType::F32 });
    let sizes = vec![t * k * 4, e * bytes_of(w, n, k), t * r * 4, t * r * n * 4];
    Case {
        name: format!("moe t{t} r{r} e{e} {n}x{k} {}", w.name()),
        launch: Launch {
            kernel: Kernel::MatMulId(spec, MatMulIdDims { t, r, e, n, k }),
            inputs: vec![0, 1, 2],
            output: 3,
        },
        cold: vec![1],
        init: vec![(2, (0..t * r).flat_map(|i| ((i * 7 % e) as i32).to_le_bytes()).collect())],
        flops: 2.0 * (t * r * n * k) as f64,
        // decode reads only the selected experts
        bytes: (sizes[0] + sizes[1] * (t * r).min(e) / e + sizes[3]) as f64,
        sizes,
    }
}

/// Gated delta rule over t tokens (state in + state out + q/k/v traffic).
fn gdn_case(t: usize, h: usize, d: usize) -> Case {
    let spec = Arc::new(GatedDeltaNetSpec { out_dtype: DType::F32 });
    let dims = GatedDeltaNetDims { b: 1, t, h, dk: d, dv: d, scale: (1.0f32 / (d as f32).sqrt()).to_bits() };
    let tok = t * h * d * 4;
    let sizes = vec![tok, tok, tok, t * h * 4, t * h * 4, h * d * d * 4, h * (t + d) * d * 4];
    Case {
        name: format!("gdn t{t} h{h} d{d}"),
        launch: Launch { kernel: Kernel::GatedDeltaNet(spec, dims), inputs: (0..6).collect(), output: 6 },
        cold: vec![],
        init: vec![],
        flops: 6.0 * (t * h * d * d) as f64,
        bytes: sizes.iter().sum::<usize>() as f64,
        sizes,
    }
}

fn cases() -> Vec<Case> {
    let mut v = vec![copy_case(64 << 20)];
    for b in [DType::Int8, DType::F16] {
        // encoder
        v.push(matmul_case(1500, 768, 768, b, Some(add01())));
        v.push(matmul_case(1500, 3072, 768, b, Some(gelu_bias())));
        v.push(matmul_case(1500, 768, 3072, b, Some(add01())));
        // decoder
        v.push(matmul_case(1, 768, 768, b, Some(add01())));
        v.push(matmul_case(1, 3072, 768, b, Some(gelu_bias())));
        v.push(matmul_case(1, 768, 3072, b, Some(add01())));
        v.push(matmul_case(1, 51865, 768, b, None));
        v.push(matmul_case(3, 768, 768, b, Some(add01())));
    }
    // LFM3-1.7B decode GEMVs (hidden 2048, ffn 7680, conv in_proj 6144, kv 512, vocab)
    for b in [DType::Int4, DType::Int8, DType::F16, DType::Q4K, DType::Q6K, DType::Mxfp4] {
        for (n, k) in [(2048, 2048), (512, 2048), (6144, 2048), (7680, 2048), (2048, 7680), (128000, 2048)] {
            let mut c = matmul_case(1, n, k, b, None);
            c.name = format!("lfm3 {}", c.name);
            v.push(c);
        }
    }
    for w in [DType::Q4K, DType::Int4] {
        v.push(moe_case(1, 8, 128, 768, 2048, w));
        v.push(moe_case(512, 8, 128, 768, 2048, w));
    }
    // Qwen3.5-35B-A3B MoE (256 experts, top-8, hidden 2048, expert ffn 512)
    for w in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Int8] {
        v.push(moe_case(1, 8, 256, 512, 2048, w));
        v.push(moe_case(1, 8, 256, 2048, 512, w));
        v.push(moe_case(512, 8, 256, 512, 2048, w));
    }
    // DeepSeek-V4-Flash-like MoE (32 of its 256 experts to bound the cold copies, top-6, hidden 4096, expert ffn 2048) and
    // decode GEMVs: MXFP4 vs Q4K
    for w in [DType::Mxfp4, DType::Q4K] {
        v.push(moe_case(1, 6, 32, 2048, 4096, w));
        v.push(moe_case(1, 6, 32, 4096, 2048, w));
        v.push(moe_case(512, 6, 32, 2048, 4096, w));
        for (n, k) in [(4096, 4096), (2048, 4096), (4096, 2048)] {
            let mut c = matmul_case(1, n, k, w, None);
            c.name = format!("ds4 {}", c.name);
            v.push(c);
        }
        let mut c = matmul_case(512, 4096, 4096, w, None);
        c.name = format!("ds4 {}", c.name);
        v.push(c);
    }
    // Qwen3.5-0.8B prefill GEMMs (m = 512)
    for b in [DType::Q4K, DType::Q6K, DType::Int8, DType::F16] {
        for (n, k) in [(3584, 1024), (1024, 3584), (6144, 1024), (248320, 1024)] {
            let mut c = matmul_case(512, n, k, b, None);
            c.name = format!("q35 {}", c.name);
            v.push(c);
        }
    }
    v.push(gdn_case(1, 32, 128));
    v.push(gdn_case(512, 32, 128));
    for dt in [DType::F32, DType::F16] {
        v.push(attention_case(1500, 1500, true, dt, 64));
        v.push(attention_case(512, 512, true, dt, 128));
    }
    v.push(attention_case(1, 1500, false, DType::F32, 64));
    v.push(attention_case(1, 24, false, DType::F32, 64));
    v.push(attention_case(1, 1024, false, DType::F32, 512));
    v.push(attention_case(512, 512, true, DType::F32, 512));
    v.push(attention_case(512, 512, true, DType::F32, 256));
    v.push(attention_case(1, 1024, false, DType::F32, 256));
    v.push(layer_norm_case(1));
    v.push(layer_norm_case(1500));
    v
}

#[test]
#[ignore]
fn bench_kernels() {
    let Ok(list) = devices() else { return };
    if list.is_empty() {
        return;
    }
    let be = CudaBackend::new(0).unwrap();
    let filter = std::env::var("LAPID_BENCH_FILTER").unwrap_or_default();
    let reps = 20usize;
    println!(
        "{:<34} {:>10} {:>10} {:>9} {:>9}",
        "case", "median us", "min us", "TFLOP/s", "GB/s"
    );
    for c in cases() {
        if !filter.is_empty() && !c.name.contains(&filter) {
            continue;
        }
        let mut bufs: Vec<DeviceData> = c
            .sizes
            .iter()
            .enumerate()
            .map(|(b, &s)| {
                if let Some((_, d)) = c.init.iter().find(|x| x.0 == b) {
                    return be.upload(d).unwrap();
                }
                let data: Vec<u8> = (0..s)
                    .map(|i| ((i * 2654435761usize) >> 13) as u8 & 0x3f)
                    .collect();
                be.upload(&data).unwrap()
            })
            .collect();
        // cold inputs: enough copies that one pass touches >= 96 MiB (beyond L2)
        let cold_bytes: usize = c.cold.iter().map(|&i| c.sizes[i]).sum();
        let n = if cold_bytes > 0 {
            reps.max((96usize << 20).div_ceil(cold_bytes))
        } else {
            reps
        };
        let mut launches: Vec<Launch> = Vec::with_capacity(n);
        for r in 0..n {
            let mut l = c.launch.clone();
            if r > 0 {
                for &i in &c.cold {
                    bufs.push(be.upload(&vec![0x11u8; c.sizes[i]]).unwrap());
                    l.inputs[i] = bufs.len() - 1;
                }
            }
            launches.push(l);
        }
        be.prepare(&launches[..1]).unwrap();
        let _ = be.time_launches(&launches[..2], &bufs).unwrap();
        let _ = be.time_launches(&launches, &bufs).unwrap();
        let mut t = be.time_launches(&launches, &bufs).unwrap();
        t.sort_by(|a, b| a.total_cmp(b));
        let med = t[t.len() / 2];
        println!(
            "{:<34} {:>10.2} {:>10.2} {:>9.2} {:>9.1}",
            c.name,
            med * 1e3,
            t[0] * 1e3,
            c.flops / (med * 1e-3) / 1e12,
            c.bytes / (med * 1e-3) / 1e9
        );
    }
}

/// Host cost of recording + submitting decode-step-sized executes, wall time per execute
/// (min of 50) vs. GPU time: 200 tiny dependent launches, and a mixed step (500 launches:
/// every 5th an int4 2048x2048 GEMV, the rest tiny).
#[test]
#[ignore]
fn bench_host_overhead() {
    let Ok(list) = devices() else { return };
    if list.is_empty() {
        return;
    }
    let be = CudaBackend::new(0).unwrap();
    let tiny = matmul_case(1, 64, 64, DType::Int8, Some(add01()));
    let big = matmul_case(1, 2048, 2048, DType::Int4, None);
    let mut bufs: Vec<DeviceData> = Vec::new();
    let mut bind = |c: &Case| -> Launch {
        let base = bufs.len();
        for &s in &c.sizes {
            bufs.push(be.upload(&vec![1u8; s]).unwrap());
        }
        let mut l = c.launch.clone();
        l.inputs.iter_mut().for_each(|i| *i += base);
        l.output += base;
        l
    };
    let (lt, lb) = (bind(&tiny), bind(&big));
    let sets: [(&str, Vec<Launch>); 2] = [
        ("200 tiny", (0..200).map(|_| lt.clone()).collect()),
        ("500 mixed", (0..500).map(|i| if i % 5 == 0 { lb.clone() } else { lt.clone() }).collect()),
    ];
    for (name, launches) in &sets {
        be.execute(launches, &bufs).unwrap();
        let mut best = f64::MAX;
        for _ in 0..50 {
            let t0 = std::time::Instant::now();
            be.execute(launches, &bufs).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        let gpu: f64 = be.time_launches(launches, &bufs).unwrap().iter().sum();
        println!("{name}: wall {best:.3} ms / execute (min of 50), gpu (event-timed) {gpu:.3} ms");
    }
}
