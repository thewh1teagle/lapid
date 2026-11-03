use super::*;
use crate::OutDType;
use crate::ggml::tests::{Lcg, SUPPORTED, sample_block};
use crate::policy::Role;

/// Minimal GGUF writer for tests.
#[derive(Default)]
pub(crate) struct GgufBuilder {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    pub(crate) alignment: u64,
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

impl GgufBuilder {
    pub(crate) fn kv_raw(&mut self, key: &str, ty: u32, value: &[u8]) {
        put_str(&mut self.kv, key);
        self.kv.extend(ty.to_le_bytes());
        self.kv.extend(value);
        self.n_kv += 1;
    }
    pub(crate) fn kv_str(&mut self, key: &str, v: &str) {
        let mut b = Vec::new();
        put_str(&mut b, v);
        self.kv_raw(key, 8, &b);
    }
    pub(crate) fn kv_u32(&mut self, key: &str, v: u32) {
        self.kv_raw(key, 4, &v.to_le_bytes());
    }
    /// `ne` in ggml order (fastest first).
    pub(crate) fn tensor(&mut self, name: &str, ne: &[u64], ty: GgmlType, data: Vec<u8>) {
        self.tensors.push((name.into(), ne.to_vec(), ty.id(), data));
    }
    pub(crate) fn build(&self, version: u32) -> Vec<u8> {
        let align = if self.alignment == 0 { DEFAULT_ALIGNMENT } else { self.alignment };
        let mut b = b"GGUF".to_vec();
        b.extend(version.to_le_bytes());
        b.extend((self.tensors.len() as u64).to_le_bytes());
        b.extend(self.n_kv.to_le_bytes());
        b.extend(&self.kv);
        let mut off = 0u64;
        let mut offsets = Vec::new();
        for (name, ne, ty, data) in &self.tensors {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            ne.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend(ty.to_le_bytes());
            b.extend(off.to_le_bytes());
            offsets.push(off);
            off = (off + data.len() as u64).div_ceil(align) * align;
        }
        b.resize((b.len() as u64).div_ceil(align) as usize * align as usize, 0);
        let start = b.len();
        for ((_, _, _, data), o) in self.tensors.iter().zip(offsets) {
            b.resize(start + o as usize, 0);
            b.extend(data);
        }
        b
    }
}

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("lapid-gguf-test-{}-{tag}", std::process::id()))
}

#[test]
fn round_trip_every_type() {
    let mut rng = Lcg(42);
    let mut g = GgufBuilder::default();
    g.kv_str("general.architecture", "test");
    g.kv_u32("general.alignment", 64);
    g.alignment = 64;
    // Array of strings + nested scalar types.
    let mut arr = 8u32.to_le_bytes().to_vec();
    arr.extend(2u64.to_le_bytes());
    put_str(&mut arr, "a");
    put_str(&mut arr, "Ġb");
    g.kv_raw("tokenizer.ggml.tokens", 9, &arr);
    g.kv_raw("test.f", 6, &1.5f32.to_le_bytes());
    g.kv_raw("test.b", 7, &[1]);
    g.kv_raw("test.i64", 11, &(-3i64).to_le_bytes());
    let mut want = Vec::new();
    for t in SUPPORTED {
        // Two rows of two blocks each: ne = [2 * block_elems, 2].
        let blocks: Vec<_> = (0..4).map(|_| sample_block(t, &mut rng)).collect();
        let per = blocks[0].1.len();
        let data: Vec<u8> = blocks.iter().flat_map(|b| b.0.clone()).collect();
        let vals: Vec<f32> = blocks.iter().flat_map(|b| b.1.clone()).collect();
        g.tensor(&format!("t.{}", t.name()), &[2 * per as u64, 2], t, data);
        want.push((t, vec![2, 2 * per], vals));
    }
    g.tensor("bad", &[256], GgmlType::from_id(23), vec![0; 136]);
    for version in [2, 3] {
        let p = tmp(&format!("rt{version}.gguf"));
        std::fs::write(&p, g.build(version)).expect("write");
        assert!(is_gguf(&p).expect("magic"));
        let f = GgufFile::open(&p).expect("open");
        assert_eq!((f.version, f.alignment), (version, 64));
        assert_eq!(f.architecture(), Some("test"));
        assert_eq!(f.get("tokenizer.ggml.tokens").map(|v| v.to_json()), Some(json!(["a", "Ġb"])));
        assert_eq!(f.get("test.f").and_then(GgufValue::as_f64), Some(1.5));
        assert_eq!(f.get("test.b"), Some(&GgufValue::Bool(true)));
        assert_eq!(f.get("test.i64").and_then(GgufValue::as_f64), Some(-3.0));
        for (t, shape, vals) in &want {
            let info = f.tensor(&format!("t.{}", t.name())).expect("tensor");
            assert_eq!(&info.shape, shape);
            assert_eq!(info.offset % 64, 0);
            let got = f.to_f32(info).expect("dequantize");
            assert_eq!(&got, vals, "{}", t.name());
        }
        let bad = f.tensor("bad").expect("listed");
        let e = f.to_f32(bad).err().expect("iq4_xs unsupported");
        assert_eq!(e.code, "unsupported_gguf_type");
        assert!(e.message.contains("iq4_xs"));
        std::fs::remove_file(&p).ok();
    }
}

#[test]
fn malformed_rejected() {
    let mut g = GgufBuilder::default();
    g.kv_str("general.architecture", "x");
    g.tensor("w", &[32, 1], GgmlType::Q8_0, vec![0; 34]);
    let good = g.build(3);
    parse(&good).expect("valid");
    assert_eq!(parse(b"GGML\x03\0\0\0").err().map(|e| e.code), Some("format"));
    let mut v1 = good.clone();
    v1[4] = 1;
    assert_eq!(parse(&v1).err().map(|e| e.code), Some("unsupported_version"));
    assert_eq!(parse(&good[..good.len() - 1]).err().map(|e| e.code), Some("format"), "truncated data");
    assert_eq!(parse(&good[..30]).err().map(|e| e.code), Some("format"), "truncated header");
    let mut huge = good.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(parse(&huge).err().map(|e| e.code), Some("limit_exceeded"));
    // Q8_0 with a row length that is not a multiple of 32.
    let mut g = GgufBuilder::default();
    g.tensor("w", &[33], GgmlType::Q8_0, vec![0; 34]);
    assert_eq!(parse(&g.build(3)).err().map(|e| e.code), Some("format"));
    // Bad alignment value.
    let mut g = GgufBuilder::default();
    g.kv_u32("general.alignment", 3);
    assert_eq!(parse(&g.build(3)).err().map(|e| e.code), Some("format"));
}

fn role(name: &str, _: &[usize]) -> Role {
    if name.ends_with("bias") { Role::Param } else { Role::Linear }
}

#[test]
fn writes_through_policy() {
    let mut rng = Lcg(3);
    let mut g = GgufBuilder::default();
    let q8: Vec<_> = (0..4).map(|_| sample_block(GgmlType::Q8_0, &mut rng)).collect();
    let q4k: Vec<_> = (0..2).map(|_| sample_block(GgmlType::Q4K, &mut rng)).collect();
    g.tensor("q8", &[64, 2], GgmlType::Q8_0, q8.iter().flat_map(|b| b.0.clone()).collect());
    g.tensor("k4", &[256, 2], GgmlType::Q4K, q4k.iter().flat_map(|b| b.0.clone()).collect());
    g.tensor("bias", &[4], GgmlType::F32, sample_block(GgmlType::F32, &mut rng).0);
    let src = tmp("w.gguf");
    std::fs::write(&src, g.build(3)).expect("write");
    let f = GgufFile::open(&src).expect("open");
    let out = tmp("w.lapid");
    let meta = crate::writer::metadata("t", "test", OutDType::Int8, "mit", json!({"repo": "x/y"}));
    let mut w = ModelWriter::create(&out, meta, OutDType::Int8, role, 3).expect("create");
    for t in &f.tensors {
        f.write_tensor(&mut w, &t.name, t, None).expect("write");
    }
    w.finish(&[]).expect("finish");
    let r = lapid_format::Reader::open(&out).expect("open lapid");
    let q8t = r.tensor("q8").expect("q8");
    assert_eq!((q8t.dtype, q8t.shape.clone()), (DType::Int8, vec![2, 64]));
    // Lossless repack: dequantized int8 equals the ggml values.
    let want: Vec<f32> = q8.iter().flat_map(|b| b.1.clone()).collect();
    assert_eq!(lapid_quant::dequantize(Format::Int8, r.tensor_bytes(q8t), 2, 64), want);
    // Q4_K kept as-is (zero conversion), even with an int8 policy.
    let k4t = r.tensor("k4").expect("k4");
    assert_eq!((k4t.dtype, k4t.shape.clone()), (DType::Q4K, vec![2, 256]));
    let blocks: Vec<u8> = q4k.iter().flat_map(|b| b.0.clone()).collect();
    assert_eq!(&r.tensor_bytes(k4t)[..blocks.len()], &blocks[..]);
    let want: Vec<f32> = q4k.iter().flat_map(|b| b.1.clone()).collect();
    assert_eq!(lapid_quant::dequantize(Format::Q4K, r.tensor_bytes(k4t), 2, 256), want);
    assert_eq!(r.tensor("bias").map(|t| t.dtype), Some(DType::F32));
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&out).ok();
}

#[test]
fn mxfp4_kept_as_is() {
    // 3 rows x 64 = 6 ggml block_mxfp4 (17 B): stored unchanged as DType::Mxfp4.
    let blocks: Vec<u8> = (0..6 * 17).map(|i| if i % 17 == 0 { 120 + (i / 17) as u8 } else { (i * 37 % 251) as u8 }).collect();
    let mut g = GgufBuilder::default();
    g.tensor("mx", &[64, 3], GgmlType::from_id(39), blocks.clone());
    let src = tmp("mx.gguf");
    std::fs::write(&src, g.build(3)).expect("write");
    let f = GgufFile::open(&src).expect("open");
    let out = tmp("mx.lapid");
    let meta = crate::writer::metadata("t", "test", OutDType::Int8, "mit", json!({"repo": "x/y"}));
    let mut w = ModelWriter::create(&out, meta, OutDType::Int8, role, 3).expect("create");
    for t in &f.tensors {
        f.write_tensor(&mut w, &t.name, t, None).expect("write");
    }
    w.finish(&[]).expect("finish");
    let r = lapid_format::Reader::open(&out).expect("open lapid");
    let t = r.tensor("mx").expect("mx");
    assert_eq!((t.dtype, t.shape.clone()), (DType::Mxfp4, vec![3, 64]));
    assert_eq!(&r.tensor_bytes(t)[..blocks.len()], &blocks[..]);
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&out).ok();
}
