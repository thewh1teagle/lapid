use super::*;
use crate::ggml::GgmlType;
use crate::gguf::tests::GgufBuilder;
use crate::mapping::RoleRule;

/// Scratch directory of one test (removed by the test).
pub(crate) fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lapid-engine-test-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

/// Write a safetensors file of (name, dtype, shape, bytes).
pub(crate) fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (n, dt, shape, b) in tensors {
        header.insert(n.to_string(), json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + b.len()]}));
        off += b.len();
    }
    let h = serde_json::to_vec(&Value::Object(header)).expect("json");
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (_, _, _, b) in tensors {
        out.extend(b);
    }
    std::fs::write(path, out).expect("write");
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Hooks;
impl ConvertHooks for Hooks {
    fn gguf_config(&self, _f: &GgufFile) -> Result<Value> {
        Ok(json!({"model_type": "toy"}))
    }
    fn expected(&self, _cfg: &Value) -> Result<Vec<String>> {
        Ok(vec!["a.weight".into(), "ffn.experts.w1".into()])
    }
    fn transform(&self, name: &str, _src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        Ok(match name {
            "raw" => Emit::Blob { name: "raw.bin".into(), media_type: "application/octet-stream" },
            "ffn.experts.0.w1" => Emit::Stack { name: "ffn.experts.w1".into(), parts: (0..3).map(|e| format!("ffn.experts.{e}.w1")).collect() },
            n if n.starts_with("ffn.experts.") => Emit::Drop,
            _ => Emit::Keep,
        })
    }
}

static SPEC: ConvertSpec = ConvertSpec {
    architecture: "toy",
    license: "mit",
    hf_model_types: &["toy"],
    gguf_architectures: &["toy"],
    hf_names: &[Rule::drop("lm_head.weight")],
    gguf_names: &[Rule::new("{x}", "{x}"), Rule::new("{a}.{b}", "{a}.{b}"), Rule::new("{a}.{b}.{c}.{d}", "{a}.{b}.{c}.{d}")],
    roles: &[RoleRule::new("embed.weight", Role::LinearInt8), RoleRule::new("ffn.experts.*", Role::Linear)],
    text_assets: false,
    hooks: &Hooks,
};

#[test]
fn spec_basics() {
    assert!(SPEC.accepts_model_type("toy"));
    assert_eq!(SPEC.role("embed.weight", &[4, 4]), Role::LinearInt8);
    assert_eq!((SPEC.role_arc())("x.weight", &[4, 4]), Role::Linear);
    assert!(SPEC.convert_hf("/nonexistent/dir/x", OutDType::Int8, Path::new("/tmp/x.lapid")).is_err());
}

fn ramp(n: usize, k: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 3) % 23) as f32 * k - 1.0).collect()
}

#[test]
fn hf_blob_and_stack() {
    let d = scratch("hf-blob-stack");
    std::fs::write(d.join("config.json"), r#"{"model_type": "toy"}"#).expect("cfg");
    let raw: Vec<u8> = (0..1000u32).map(|i| (i * 31) as u8).collect();
    let experts: Vec<Vec<f32>> = (0..3).map(|e| ramp(4 * 64, 0.01 * (e + 1) as f32)).collect();
    let mut ts = vec![("a.weight", "F32", vec![4, 64], f32_bytes(&ramp(256, 0.1))), ("raw", "U8", vec![1000], raw.clone())];
    let names: Vec<String> = (0..3).map(|e| format!("ffn.experts.{e}.w1")).collect();
    for (n, x) in names.iter().zip(&experts) {
        ts.push((n.as_str(), "F32", vec![4, 64], f32_bytes(x)));
    }
    write_safetensors(&d.join("model.safetensors"), &ts);
    let out = d.join("out.lapid");
    SPEC.convert_hf(d.to_str().expect("utf8"), OutDType::Int8, &out).expect("convert");
    let r = lapid_format::Reader::open(&out).expect("open");
    r.verify_tensors().expect("hashes");
    assert_eq!(r.blob("raw.bin"), Some(&raw[..]));
    assert!(r.tensor("raw").is_none());
    let t = r.tensor("ffn.experts.w1").expect("stack");
    assert_eq!((t.dtype, t.shape.clone()), (lapid_tensor::DType::Int8, vec![3, 4, 64]));
    let b = r.tensor_bytes(t);
    let per = b.len() / 3;
    for (e, x) in experts.iter().enumerate() {
        let y = lapid_quant::dequantize(lapid_quant::Format::Int8, &b[e * per..(e + 1) * per], 4, 64);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 0.02), "expert {e}");
    }
    assert!(r.tensor("ffn.experts.1.w1").is_none());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn gguf_stack_keeps_blocks() {
    let d = scratch("gguf-stack");
    let mut g = GgufBuilder::default();
    g.kv_str("general.architecture", "toy");
    // Q8_0 blocks: f16 d + 32 i8.
    let block = |e: usize, r: usize| -> Vec<u8> {
        let mut b = half::f16::from_f32(0.5 * (e + 1) as f32).to_le_bytes().to_vec();
        b.extend((0..32).map(|j| ((j as i32 + r as i32 * 3 - 16) as i8) as u8));
        b
    };
    let mut want = Vec::new();
    for e in 0..3 {
        let data: Vec<u8> = (0..4).flat_map(|r| [block(e, r), block(e, r)].concat()).collect();
        want.push(crate::ggml::dequantize(GgmlType::Q8_0, &data, 4 * 64).expect("deq"));
        g.tensor(&format!("ffn.experts.{e}.w1"), &[64, 4], GgmlType::Q8_0, data);
    }
    g.tensor("a.weight", &[64, 4], GgmlType::F32, f32_bytes(&ramp(256, 0.1)));
    g.tensor("raw", &[8], GgmlType::F32, f32_bytes(&[1.0; 8]));
    let p = d.join("toy.gguf");
    std::fs::write(&p, g.build(3)).expect("write");
    let out = d.join("out.lapid");
    SPEC.convert_gguf(&p, OutDType::Int8, &out).expect("convert");
    let r = lapid_format::Reader::open(&out).expect("open");
    assert_eq!(r.blob("raw.bin"), Some(&f32_bytes(&[1.0; 8])[..]));
    let t = r.tensor("ffn.experts.w1").expect("stack");
    assert_eq!((t.dtype, t.shape.clone()), (lapid_tensor::DType::Int8, vec![3, 4, 64]));
    let b = r.tensor_bytes(t);
    let per = b.len() / 3;
    for (e, x) in want.iter().enumerate() {
        // Q8_0 -> int8 is a lossless repack (scale rounded to f16 both ways).
        assert_eq!(&lapid_quant::dequantize(lapid_quant::Format::Int8, &b[e * per..(e + 1) * per], 4, 64), x, "expert {e}");
    }
    std::fs::remove_dir_all(&d).ok();
}

struct MicroHooks;
impl ConvertHooks for MicroHooks {
    fn expected(&self, _cfg: &Value) -> Result<Vec<String>> {
        Ok(["wq.weight", "x.weight", "ffn.experts.w1", "eng.fp8", "eng.e8m0"].map(String::from).to_vec())
    }
    fn hf_unpaired(&self, n: &str) -> bool {
        n.starts_with("eng.")
    }
    fn transform(&self, name: &str, src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        let blob = |n: &str| Emit::Blob { name: n.into(), media_type: "application/octet-stream" };
        Ok(match name {
            "eng.embed.weight" => blob("eng.fp8"),
            "eng.embed.scale" => blob("eng.e8m0"),
            "ffn.experts.0.w1.weight" => {
                assert_eq!(src.stored_dtype(name).as_deref(), Some("F4_E2M1+E8M0"));
                assert_eq!(src.shape(name), Some(vec![4, 64]));
                Emit::Stack { name: "ffn.experts.w1".into(), parts: vec![name.into(), "ffn.experts.1.w1.weight".into()] }
            }
            "ffn.experts.1.w1.weight" => Emit::Drop,
            "wq.weight" => {
                assert_eq!(src.stored_dtype(name).as_deref(), Some("F8_E4M3+E8M0"));
                Emit::Keep
            }
            _ => Emit::Keep,
        })
    }
}

static MICRO: ConvertSpec = ConvertSpec {
    architecture: "toy",
    license: "mit",
    hf_model_types: &["toy"],
    gguf_architectures: &[],
    hf_names: &[],
    gguf_names: &[],
    roles: &[RoleRule::new("ffn.experts.*", Role::Linear)],
    text_assets: false,
    hooks: &MicroHooks,
};

#[test]
fn hf_fp8_fp4_scales() {
    let d = scratch("hf-micro");
    std::fs::write(d.join("config.json"), r#"{"model_type": "toy"}"#).expect("cfg");
    // FP8 [64, 64], 32x32 blocks.
    let wq: Vec<u8> = (0..64 * 64).map(|i| ((i * 13) % 0x7e) as u8 | if i % 3 == 0 { 0x80 } else { 0 }).collect();
    let wq_s = vec![127u8, 130, 120, 128];
    // FP4 [4, 64]: packed [4, 32], scales [4, 2].
    let fp4 = |k: usize| -> (Vec<u8>, Vec<u8>) { ((0..128).map(|i| (i * 29 + k * 7) as u8).collect(), (0..8).map(|i| 118 + (i + k) as u8 % 10).collect()) };
    let (x, xs) = fp4(0);
    let (e0, e0s) = fp4(1);
    let (e1, e1s) = fp4(2);
    let eng: Vec<u8> = (0..10 * 64).map(|i| (i % 251) as u8).collect();
    let eng_s: Vec<u8> = (0..20).map(|i| 100 + i as u8).collect();
    write_safetensors(&d.join("model-1.safetensors"), &[
        ("wq.weight", "F8_E4M3", vec![64, 64], wq.clone()),
        ("x.weight", "I8", vec![4, 32], x.clone()),
        ("eng.embed.weight", "F8_E4M3", vec![10, 64], eng.clone()),
        ("ffn.experts.0.w1.weight", "I8", vec![4, 32], e0.clone()),
        ("ffn.experts.1.w1.scale", "F8_E8M0", vec![4, 2], e1s.clone()),
    ]);
    write_safetensors(&d.join("model-2.safetensors"), &[
        ("wq.scale", "F8_E8M0", vec![2, 2], wq_s.clone()),
        ("x.scale", "F8_E8M0", vec![4, 2], xs.clone()),
        ("eng.embed.scale", "F8_E8M0", vec![10, 2], eng_s.clone()),
        ("ffn.experts.0.w1.scale", "F8_E8M0", vec![4, 2], e0s.clone()),
        ("ffn.experts.1.w1.weight", "I8", vec![4, 32], e1.clone()),
    ]);
    std::fs::write(
        d.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})).expect("json"),
    )
    .expect("index");
    let mx = |p: &[u8], s: &[u8]| {
        let mut b = lapid_quant::mxfp4::from_e2m1_pairs(p, s, 4, 64);
        b.resize(lapid_quant::layout(lapid_quant::Format::Mxfp4, 4, 64).total, 0);
        b
    };
    let want_wq = crate::microscale::fp8_to_f32(&wq, &wq_s, [64, 64], [2, 2]).expect("fp8");
    for dtype in [OutDType::F32, OutDType::Int8] {
        let out = d.join(format!("out-{}.lapid", dtype.name()));
        MICRO.convert_hf(d.to_str().expect("utf8"), dtype, &out).expect("convert");
        let r = lapid_format::Reader::open(&out).expect("open");
        r.verify_tensors().expect("hashes");
        let names: Vec<&str> = r.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(!names.iter().any(|n| n.ends_with(".scale") || n.starts_with("eng.")), "{names:?}");
        // FP8 -> exact f32 (fp32 output) / int8 by the policy.
        let t = r.tensor("wq.weight").expect("wq");
        if dtype == OutDType::F32 {
            assert_eq!(t.dtype, lapid_tensor::DType::F32);
            let got: Vec<f32> = r.tensor_bytes(t).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            assert_eq!(got, want_wq);
        } else {
            assert_eq!(t.dtype, lapid_tensor::DType::Int8);
        }
        // FP4 -> MXFP4 whatever the output dtype.
        let t = r.tensor("x.weight").expect("x");
        assert_eq!((t.dtype, t.shape.clone()), (lapid_tensor::DType::Mxfp4, vec![4, 64]));
        assert_eq!(r.tensor_bytes(t), &mx(&x, &xs)[..]);
        let t = r.tensor("ffn.experts.w1").expect("stack");
        assert_eq!((t.dtype, t.shape.clone()), (lapid_tensor::DType::Mxfp4, vec![2, 4, 64]));
        assert_eq!(r.tensor_bytes(t), &[mx(&e0, &e0s), mx(&e1, &e1s)].concat()[..]);
        // Unpaired engram tables as raw blobs.
        assert_eq!(r.blob("eng.fp8"), Some(&eng[..]));
        assert_eq!(r.blob("eng.e8m0"), Some(&eng_s[..]));
    }
    std::fs::remove_dir_all(&d).ok();
}
