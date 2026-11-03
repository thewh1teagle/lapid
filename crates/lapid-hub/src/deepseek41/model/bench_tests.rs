//! Expert-cache benchmark on a synthetic DeepSeek V4.1 Flash file: random weights with the
//! real shapes (d 5120, 384 MXFP4 experts of 2304, int8 dense) but 4 layers (sliding,
//! ratio-2 kv/index source, ratio-2 reader, ratio-1 kv/index/candidate source), no Engram.
//! Both tests are ignored and need `LAPID_DS41_SYNTH=<file>` (~31 GB):
//!
//! ```text
//! LAPID_DS41_SYNTH=/tmp/x/synth.lapid cargo test --release -p lapid-hub synth_generate -- --ignored
//! LAPID_DS41_SYNTH=/tmp/x/synth.lapid LAPID_EXPERT_CACHE_SLOTS=430 LAPID_DS41_ALPHA=0.8 \
//!     cargo test --release -p lapid-hub synth_bench -- --ignored --nocapture
//! ```
//! `LAPID_DS41_ALPHA=a`: decode tokens pick their experts from Zipf(a) per layer (0 =
//! uniform: the worst case); unset: the random router (degenerate: always the same experts).
//! `LAPID_DS41_COLD=1` drops the file from the page cache first; `LAPID_DS41_PROMPT` /
//! `LAPID_DS41_GEN` / `LAPID_DS41_CHUNK`: prompt length, decode tokens, prefill chunk.

use std::path::PathBuf;
use std::time::Instant;

use lapid::{DType, Device, Tensor};
use serde_json::{Value, json};

use crate::deepseek41::Deepseek41;

const D: usize = 5120;
const QL: usize = 1280;
const H: usize = 64;
const HD: usize = 512;
const G: usize = 8;
const OL: usize = 1024;
const E: usize = 384;
const I: usize = 2304;
const IH: usize = 32;
const ID: usize = 128;
const HC: usize = 4;
const VOCAB: usize = 129280;
const LAYERS: usize = 4;

fn synth_path() -> Option<PathBuf> {
    std::env::var_os("LAPID_DS41_SYNTH").map(PathBuf::from)
}

/// Deterministic uniform values in `[-a, a)` (splitmix64), filled in parallel.
fn uniform(n: usize, seed: u64, a: f32) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let chunk = 1 << 20;
    std::thread::scope(|s| {
        for (ci, c) in v.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (ci as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                for o in c.iter_mut() {
                    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = x;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    *o = ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a;
                }
            });
        }
    });
    v
}

/// Random MXFP4 stack `[e, n, k]`: scale exponents 2^-7 .. 2^-5, random nibbles.
fn mxfp4_stack(e: usize, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let per = lapid::quant::layout(lapid::quant::Format::Mxfp4, n, k).total;
    let r = uniform(e * per / 4 + 1, seed, 1.0);
    let mut out: Vec<u8> = r.iter().flat_map(|f| f.to_bits().to_le_bytes()).take(e * per).collect();
    for m in 0..e {
        for b in 0..(n * k / 32) {
            out[m * per + b * 17] = 121 + (out[m * per + b * 17] % 3);
        }
    }
    out
}

fn int8(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let a = 1.7 / (cols as f32).sqrt();
    lapid::quant::quantize(lapid::quant::Format::Int8, &uniform(rows * cols, seed, a), rows, cols)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The deepseek41 metadata of the synthetic model (real V4.1 Flash text config, 4 layers).
fn metadata(template: &Value) -> Value {
    let mut m = template.clone();
    let c = &mut m["deepseek41"];
    let set = |c: &mut Value, k: &str, v: Value| c[k] = v;
    for (k, v) in [
        ("hidden_size", json!(D)),
        ("moe_intermediate_size", json!(I)),
        ("num_hidden_layers", json!(LAYERS)),
        ("num_attention_heads", json!(H)),
        ("q_lora_rank", json!(QL)),
        ("o_lora_rank", json!(OL)),
        ("o_groups", json!(G)),
        ("swiglu_limit", json!(10.0)),
        ("n_routed_experts", json!(E)),
        ("num_experts_per_tok", json!(6)),
        ("sliding_window", json!(128)),
        ("compress_ratios", json!([0, 2, 2, 1, 0])),
        ("kv_source_layer_ids", json!([1, 3])),
        ("index_source_layer_ids", json!([1, 3])),
        ("index_n_heads", json!(IH)),
        ("index_head_dim", json!(ID)),
        ("index_topk", json!(512)),
        ("candidate_source_layer_id", json!(3)),
        ("candidate_topk_blocks", json!(2048)),
        ("candidate_block_size", json!(8)),
        ("engram_layer_ids", json!([])),
        ("engram_num_embeddings", json!([])),
    ] {
        set(c, k, v);
    }
    if let Some(o) = c.as_object_mut() {
        o.remove("engram_hash");
    }
    m["name"] = json!("deepseek41-synthetic-4l");
    m
}

/// (name, dtype, shape) of every tensor; dense linears int8, norms / router / mHC f32.
fn tensors() -> Vec<(String, DType, Vec<usize>)> {
    let mut t: Vec<(String, DType, Vec<usize>)> = Vec::new();
    let mut add = |n: String, dt: DType, s: &[usize]| t.push((n, dt, s.to_vec()));
    add("embed.weight".into(), DType::Int8, &[VOCAB, D]);
    add("head.weight".into(), DType::Int8, &[VOCAB, D]);
    add("norm.weight".into(), DType::F32, &[D]);
    let mix = (2 + HC) * HC;
    for i in 0..LAYERS {
        let p = format!("layers.{i}");
        for k in ["attn", "ffn"] {
            add(format!("{p}.hc_{k}_fn"), DType::F32, &[mix, HC * D]);
            add(format!("{p}.hc_{k}_base"), DType::F32, &[mix]);
            add(format!("{p}.hc_{k}_scale"), DType::F32, &[3]);
        }
        add(format!("{p}.attn_norm.weight"), DType::F32, &[D]);
        add(format!("{p}.ffn_norm.weight"), DType::F32, &[D]);
        let a = format!("{p}.attn");
        add(format!("{a}.attn_sink"), DType::F32, &[H]);
        add(format!("{a}.kv_norm.weight"), DType::F32, &[HD]);
        add(format!("{a}.q_norm.weight"), DType::F32, &[QL]);
        add(format!("{a}.wkv.weight"), DType::Int8, &[HD, D]);
        add(format!("{a}.wo_a.weight"), DType::Int8, &[G, OL, H / G * HD]);
        add(format!("{a}.wo_b.weight"), DType::Int8, &[D, G * OL]);
        add(format!("{a}.wq_a.weight"), DType::Int8, &[QL, D]);
        add(format!("{a}.wq_b.weight"), DType::Int8, &[H * HD, QL]);
        if i == 1 || i == 3 {
            add(format!("{a}.compressor.norm.weight"), DType::F32, &[HD]);
            add(format!("{a}.compressor.wkv.weight"), DType::Int8, &[HD, D]);
            if i == 1 {
                add(format!("{a}.compressor.wgate.weight"), DType::Int8, &[HD, D]);
            }
            add(format!("{a}.indexer.k_norm.weight"), DType::F32, &[ID]);
            add(format!("{a}.indexer.wk.weight"), DType::Int8, &[ID, HD]);
            add(format!("{a}.indexer.weights_proj.weight"), DType::Int8, &[IH, D]);
            add(format!("{a}.indexer.wq_b.weight"), DType::Int8, &[IH * ID, QL]);
        }
        let f = format!("{p}.ffn");
        add(format!("{f}.gate.weight"), DType::F32, &[E, D]);
        add(format!("{f}.gate.bias"), DType::F32, &[E]);
        add(format!("{f}.experts.w1"), DType::Mxfp4, &[E, I, D]);
        add(format!("{f}.experts.w3"), DType::Mxfp4, &[E, I, D]);
        add(format!("{f}.experts.w2"), DType::Mxfp4, &[E, D, I]);
        add(format!("{f}.shared_experts.w1.weight"), DType::Int8, &[I, D]);
        add(format!("{f}.shared_experts.w3.weight"), DType::Int8, &[I, D]);
        add(format!("{f}.shared_experts.w2.weight"), DType::Int8, &[D, I]);
    }
    t
}

fn data(name: &str, dt: DType, s: &[usize], seed: u64) -> Vec<u8> {
    let ones = |n: usize| f32_bytes(&vec![1.0; n]);
    match dt {
        DType::Mxfp4 => mxfp4_stack(s[0], s[1], s[2], seed),
        DType::Int8 if s.len() == 3 => (0..s[0]).flat_map(|m| int8(s[1], s[2], seed * 31 + m as u64)).collect(),
        DType::Int8 => int8(s[0], s[1], seed),
        _ if name.ends_with("norm.weight") => ones(s[0]),
        _ if name.ends_with("_scale") => f32_bytes(&[0.5, 0.5, 0.5]),
        _ if name.ends_with("_base") || name.ends_with("gate.bias") || name.ends_with("attn_sink") => f32_bytes(&vec![0.0; s.iter().product()]),
        _ if name.ends_with("gate.weight") => f32_bytes(&uniform(s.iter().product(), seed, 1.7 / (D as f32).sqrt())),
        _ => f32_bytes(&uniform(s.iter().product(), seed, 0.01)),
    }
}

/// Write the synthetic file (template: the tiny reference converted to int8, for its
/// metadata and tokenizer).
#[test]
#[ignore]
fn synth_generate() {
    let Some(out) = synth_path() else { return };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/deepseek41/tiny");
    let tiny = out.with_file_name("tiny-template.lapid");
    crate::deepseek41::convert::CONVERT.convert(&root.join("model").display().to_string(), lapid_convert::OutDType::Int8, &tiny).expect("convert tiny");
    let tf = lapid::LapidFile::open(&tiny).expect("template");
    let mut w = lapid::format::Writer::create(&out, metadata(&tf.metadata)).expect("create");
    let tok = "tokenizer/tokenizer.json";
    w.add_blob(tok, "application/json", tf.blob(tok).expect("tokenizer")).expect("blob");
    let t0 = Instant::now();
    for (i, (name, dt, shape)) in tensors().into_iter().enumerate() {
        let bytes = data(&name, dt, &shape, 1000 + i as u64);
        assert_eq!(bytes.len(), dt.storage_bytes(&shape), "{name}");
        w.add(&name, dt, &shape, &bytes).expect("add");
        eprintln!("{name} {:?} {:.1} s", shape, t0.elapsed().as_secs_f64());
    }
    w.finish().expect("finish");
    drop(tf);
    let _ = std::fs::remove_file(&tiny);
}

#[cfg(unix)]
fn drop_page_cache(path: &std::path::Path) {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    if let Ok(f) = std::fs::File::open(path) {
        // SAFETY: a valid descriptor; advisory (4 = POSIX_FADV_DONTNEED).
        unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4) };
    }
}

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Prefill + decode timing, expert-cache hit rate and miss bandwidth on CUDA.
#[test]
#[ignore]
fn synth_bench() {
    let Some(path) = synth_path() else { return };
    #[cfg(unix)]
    if env_f("LAPID_DS41_COLD", 0.0) > 0.0 {
        drop_page_cache(&path);
    }
    let dev = Device::from_name(&std::env::var("LAPID_DEVICE").unwrap_or_else(|_| "cuda".into())).expect("device");
    let t0 = Instant::now();
    let mut llm = Deepseek41::load(&path, &dev).expect("load");
    eprintln!("load {:.1} s", t0.elapsed().as_secs_f64());
    // Decode routing: `LAPID_DS41_ALPHA` set -> each token's experts are drawn per layer
    // from Zipf(alpha) over a random expert order (Gumbel top-k through a dominant
    // selection bias; alpha 0 = uniform), else the random router alone.
    let alpha = std::env::var("LAPID_DS41_ALPHA").ok().and_then(|v| v.parse::<f32>().ok());
    let orders: Vec<Vec<f32>> = (0..LAYERS)
        .map(|li| {
            let r = uniform(E, 77 + li as u64, 1.0);
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_by(|&a, &b| r[a].total_cmp(&r[b]));
            let mut logw = vec![0f32; E];
            for (rank, &e) in order.iter().enumerate() {
                logw[e] = -alpha.unwrap_or(0.0) * (1.0 + rank as f32).ln();
            }
            logw
        })
        .collect();
    let mut step_no = 0u64;
    let mut route = |llm: &mut Deepseek41| {
        let Some(_) = alpha else { return };
        step_no += 1;
        for (li, layer) in llm.model_mut().layers.iter_mut().enumerate() {
            let u = uniform(E, step_no * 1000 + li as u64, 0.5);
            let bias: Vec<f32> = orders[li].iter().zip(&u).map(|(lw, u)| 100.0 * (lw - (-(u + 0.5).max(1e-7).ln()).ln())).collect();
            layer.moe.bias = dev.upload(&Tensor::from_f32(&bias, &[E]).expect("bias")).expect("upload");
        }
    };
    let prompt_len = env_f("LAPID_DS41_PROMPT", 256.0) as usize;
    if let Some(c) = std::env::var("LAPID_DS41_CHUNK").ok().and_then(|v| v.parse().ok()) {
        llm.model_mut().prefill_chunk = c;
    }
    let mem0 = lapid::residency::available_memory().unwrap_or(0);
    let n_gen = env_f("LAPID_DS41_GEN", 64.0) as usize;
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + (i * 7919) % 100_000).collect();
    let mut s = llm.new_session();
    let stats = |llm: &Deepseek41| llm.model().expert_cache().map(|c| c.stats());
    let t1 = Instant::now();
    let mut logits = llm.step(&mut s, &prompt).expect("prefill");
    let tp = t1.elapsed().as_secs_f64();
    let sp = stats(&llm);
    let mem1 = lapid::residency::available_memory().unwrap_or(0);
    eprintln!("prefill {prompt_len} tokens: {:.2} s ({:.1} tok/s), {:.2} GiB less available; cache {sp:?}", tp, prompt_len as f64 / tp, (mem0 as f64 - mem1 as f64) / (1u64 << 30) as f64);
    if let Some(c) = llm.model().expert_cache() {
        c.reset_stats();
    }
    let mut times = Vec::new();
    for _ in 0..n_gen {
        let next = lapid_nn::argmax(&logits) as u32;
        route(&mut llm);
        let t = Instant::now();
        logits = llm.step(&mut s, &[next]).expect("decode");
        times.push(t.elapsed().as_secs_f64());
    }
    let warm = &times[times.len() / 4..];
    let avg = warm.iter().sum::<f64>() / warm.len() as f64;
    let first = times.iter().take(8).sum::<f64>() / times.len().min(8) as f64;
    eprintln!("decode {n_gen} tokens: first 8 avg {:.1} ms, last 3/4 avg {:.1} ms ({:.2} tok/s) = {:.2} ms/layer", first * 1e3, avg * 1e3, 1.0 / avg, avg * 1e3 / LAYERS as f64);
    if let Some(st) = stats(&llm) {
        let gb = st.loaded_bytes as f64 / 1e9;
        eprintln!(
            "decode cache: hit rate {:.3} ({} hits, {} misses), {:.2} GB loaded, read {:.2} s ({:.1} GB/s), write {:.2} s ({:.1} GB/s), {:.1} MB/token",
            st.hit_rate(),
            st.hits,
            st.misses,
            gb,
            st.read_time.as_secs_f64(),
            gb / st.read_time.as_secs_f64().max(1e-9),
            st.write_time.as_secs_f64(),
            gb / st.write_time.as_secs_f64().max(1e-9),
            gb * 1e3 / n_gen as f64
        );
    }
}
