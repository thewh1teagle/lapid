//! Large synthetic conversion (ignored): DeepSeek V4.1-shaped FP8 / FP4 / Engram tensors at
//! real sizes, to measure time and memory. `LAPID_BIG_DIR=<scratch dir>` (removed after),
//! `LAPID_BIG_LAYERS` expert layers (default 1), `LAPID_BIG_ENGRAM_ROWS` (default 24M);
//! `LAPID_BIG_KEEP=1` keeps the directory (and reuses existing shards on the next run).

use std::io::Write;
use std::time::Instant;

use super::*;
use crate::mapping::RoleRule;

const E: usize = 384;

struct Hooks;
impl ConvertHooks for Hooks {
    fn expected(&self, _cfg: &Value) -> Result<Vec<String>> {
        Ok(vec![])
    }
    fn hf_unpaired(&self, n: &str) -> bool {
        n.contains(".engram.embed.")
    }
    fn transform(&self, name: &str, _src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        if let Some(p) = name.strip_suffix(".engram.embed.weight") {
            return Ok(Emit::Blob { name: format!("{p}.engram.embed.fp8"), media_type: "application/octet-stream" });
        }
        if let Some(p) = name.strip_suffix(".engram.embed.scale") {
            return Ok(Emit::Blob { name: format!("{p}.engram.embed.e8m0"), media_type: "application/octet-stream" });
        }
        if let Some((p, rest)) = name.split_once(".ffn.experts.") {
            let (e, w) = rest.split_once('.').unwrap_or_default();
            if e == "0" {
                let w = w.trim_end_matches(".weight");
                let parts = (0..E).map(|x| format!("{p}.ffn.experts.{x}.{w}.weight")).collect();
                return Ok(Emit::Stack { name: format!("{p}.ffn.experts.{w}"), parts });
            }
            return Ok(Emit::Drop);
        }
        Ok(Emit::Keep)
    }
}

static SPEC: ConvertSpec = ConvertSpec {
    architecture: "toy",
    license: "mit",
    hf_model_types: &["toy"],
    gguf_architectures: &[],
    hf_names: &[],
    gguf_names: &[],
    roles: &[RoleRule::new("embed.weight", Role::LinearInt8), RoleRule::new("*.ffn.experts.*", Role::Linear)],
    text_assets: false,
    hooks: &Hooks,
};

/// Streaming safetensors writer: header first, then each tensor's bytes from `fill`.
fn write_big(path: &Path, tensors: &[(String, &str, Vec<usize>, usize)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (n, dt, shape, nb) in tensors {
        header.insert(n.clone(), json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}));
        off += nb;
    }
    let h = serde_json::to_vec(&Value::Object(header)).expect("json");
    let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path).expect("create"));
    f.write_all(&(h.len() as u64).to_le_bytes()).expect("w");
    f.write_all(&h).expect("w");
    let mut buf = vec![0u8; 16 << 20];
    let mut seed = 0x9e3779b97f4a7c15u64;
    for (n, dt, _, nb) in tensors {
        let mut left = *nb;
        while left > 0 {
            let k = left.min(buf.len());
            for c in buf[..k].chunks_mut(8) {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let mut v = (seed >> 11).to_le_bytes();
                if *dt == "F8_E8M0" {
                    v.iter_mut().for_each(|b| *b = 110 + *b % 20);
                } else if *dt == "F8_E4M3" {
                    v.iter_mut().for_each(|b| *b &= 0xf7); // no NaN
                } else if *dt == "BF16" {
                    v[1] = 0x3c | (v[1] & 0x80);
                    v[3] = 0x3c | (v[3] & 0x80);
                    v[5] = 0x3c | (v[5] & 0x80);
                    v[7] = 0x3c | (v[7] & 0x80);
                }
                c.copy_from_slice(&v[..c.len()]);
            }
            f.write_all(&buf[..k]).expect("w");
            left -= k;
        }
        let _ = n;
    }
    f.flush().expect("flush");
}

#[test]
#[ignore]
fn big_synthetic() {
    let Some(dir) = std::env::var_os("LAPID_BIG_DIR").map(PathBuf::from) else { return };
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (layers, rows) = (env("LAPID_BIG_LAYERS", 1), env("LAPID_BIG_ENGRAM_ROWS", 24_000_000));
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("config.json"), r#"{"model_type": "toy"}"#).expect("cfg");
    let t0 = Instant::now();
    let fp8 = |v: &mut Vec<(String, &str, Vec<usize>, usize)>, n: String, r: usize, c: usize| {
        v.push((format!("{n}.weight"), "F8_E4M3", vec![r, c], r * c));
        v.push((format!("{n}.scale"), "F8_E8M0", vec![r / 32, c / 32], r * c / 1024));
    };
    let mut shards = Vec::new();
    let mut a = vec![("embed.weight".to_string(), "BF16", vec![129280, 5120], 129280 * 5120 * 2)];
    a.push(("layers.1.engram.embed.weight".into(), "F8_E4M3", vec![rows, 256], rows * 256));
    a.push(("layers.1.engram.embed.scale".into(), "F8_E8M0", vec![rows, 8], rows * 8));
    shards.push(a);
    for l in 0..layers {
        let mut v = Vec::new();
        for (n, r, c) in [("attn.wq_a", 1280, 5120), ("attn.wq_b", 32768, 1280), ("attn.wkv", 512, 5120), ("attn.wo_a", 8192, 4096), ("attn.wo_b", 5120, 8192), ("ffn.shared_experts.w1", 2304, 5120), ("ffn.shared_experts.w2", 5120, 2304), ("ffn.shared_experts.w3", 2304, 5120)] {
            fp8(&mut v, format!("layers.{l}.{n}"), r, c);
        }
        for e in 0..E {
            for (w, r, c) in [("w1", 2304, 5120), ("w2", 5120, 2304), ("w3", 2304, 5120)] {
                let p = format!("layers.{l}.ffn.experts.{e}.{w}");
                v.push((format!("{p}.weight"), "I8", vec![r, c / 2], r * c / 2));
                v.push((format!("{p}.scale"), "F8_E8M0", vec![r, c / 32], r * c / 32));
            }
        }
        shards.push(v);
    }
    let mut map = serde_json::Map::new();
    let mut total = 0usize;
    for (i, s) in shards.iter().enumerate() {
        let f = format!("model-{i:05}.safetensors");
        total += s.iter().map(|t| t.3).sum::<usize>();
        for t in s {
            map.insert(t.0.clone(), json!(f));
        }
        if !dir.join(&f).exists() {
            write_big(&dir.join(&f), s);
        }
    }
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&json!({"weight_map": map})).expect("json")).expect("index");
    eprintln!("generated {:.1} GB in {:.1}s", total as f64 / 1e9, t0.elapsed().as_secs_f64());
    let out = dir.join("out.lapid");
    let t1 = Instant::now();
    SPEC.convert_hf(dir.to_str().expect("utf8"), OutDType::Int8, &out).expect("convert");
    let secs = t1.elapsed().as_secs_f64();
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    eprintln!("converted {:.1} GB -> {:.1} GB in {secs:.1}s ({:.2} GB/s in)", total as f64 / 1e9, size as f64 / 1e9, total as f64 / 1e9 / secs);
    let r = lapid_format::Reader::open(&out).expect("open");
    assert_eq!(r.blob("layers.1.engram.embed.fp8").map(|b| b.len()), Some(rows * 256));
    assert_eq!(r.tensor("layers.0.ffn.experts.w1").map(|t| t.dtype), Some(lapid_tensor::DType::Mxfp4));
    if std::env::var_os("LAPID_BIG_KEEP").is_none() {
        std::fs::remove_dir_all(&dir).ok();
    }
}

