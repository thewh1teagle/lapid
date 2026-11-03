use super::*;
use crate::write::partial_path;
use std::path::PathBuf;
use crate::{Reader, Writer};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lapid-format-test-{}-{tag}-{n}.lapid", std::process::id()))
}

fn meta() -> serde_json::Value {
    json!({
        "name": "test", "architecture": "test", "dtype": "fp32", "license": "unknown",
        "source": {"repo": "local/test"}, "converter": "lapid-convert test",
        "extra": {"nested": [1, 2, 3]}
    })
}

fn write_sample(path: &Path) -> Vec<(String, DType, Vec<usize>, Vec<u8>)> {
    let q = lapid_quant_bytes();
    let tensors = vec![
        ("a".to_string(), DType::F32, vec![2, 3], (0..24u8).collect::<Vec<u8>>()),
        ("b.weight".to_string(), DType::F16, vec![5], vec![7u8; 10]),
        ("q".to_string(), DType::Int8, vec![1, 32], q),
        ("empty".to_string(), DType::F32, vec![0, 4], vec![]),
    ];
    let mut w = Writer::create(path, meta()).expect("create");
    for (n, d, s, b) in &tensors {
        w.add(n, *d, s, b).expect("add");
    }
    w.finish().expect("finish");
    tensors
}

fn lapid_quant_bytes() -> Vec<u8> {
    let n = DType::Int8.storage_bytes(&[1, 32]);
    (0..n).map(|i| (i % 251) as u8).collect()
}

#[test]
fn round_trip() {
    let p = tmp_path("rt");
    let tensors = write_sample(&p);
    let r = Reader::open(&p).expect("open");
    assert_eq!(r.metadata["format"], "lapid/1.1");
    assert!(r.blobs().is_empty());
    assert_eq!(r.components().len(), 1);
    assert_eq!(r.metadata["extra"]["nested"][2], 3);
    assert_eq!(r.tensors.len(), tensors.len());
    for (n, d, s, b) in &tensors {
        let t = r.tensor(n).expect("tensor present");
        assert_eq!(t.dtype, *d);
        assert_eq!(&t.shape, s);
        assert_eq!(t.offset % 256, 0);
        assert_eq!(t.block_size, block_size_of(*d));
        assert_eq!(r.tensor_bytes(t), &b[..]);
    }
    r.verify_tensors().expect("verify");
    // Data section starts 4096-aligned, after metadata + 64 KiB pad.
    let first = r.tensors.iter().map(|t| t.offset).min().unwrap_or(0);
    assert_eq!(first % 4096, 0);
    assert!(first >= METADATA_OFFSET + METADATA_PAD);
    assert!(!partial_path(&p).exists());
    std::fs::remove_file(&p).ok();
}

#[test]
fn missing_metadata_rejected() {
    let p = tmp_path("meta");
    let mut m = meta();
    m.as_object_mut().map(|o| o.remove("license"));
    let e = Writer::create(&p, m).err().expect("must fail");
    assert_eq!(e.code, "missing_metadata");
    assert!(!partial_path(&p).exists());
}

#[test]
fn bad_sizes_rejected() {
    let p = tmp_path("size");
    let mut w = Writer::create(&p, meta()).expect("create");
    assert_eq!(w.add("x", DType::F32, &[3], &[0u8; 8]).err().map(|e| e.code), Some("shape_mismatch"));
    assert_eq!(w.add("y", DType::Int4, &[2, 30], &[]).err().map(|e| e.code), Some("invalid_argument"));
    w.add("z", DType::F32, &[1], &[0u8; 4]).expect("add");
    assert_eq!(w.add("z", DType::F32, &[1], &[0u8; 4]).err().map(|e| e.code), Some("invalid_argument"));
    drop(w);
    assert!(!partial_path(&p).exists(), "unfinished writer removes its partial file");
    assert!(!p.exists());
}

fn corrupt(p: &Path, offset: usize, f: impl Fn(&mut u8)) {
    let mut b = std::fs::read(p).expect("read");
    f(&mut b[offset]);
    std::fs::write(p, b).expect("write");
}

#[test]
fn corrupt_header_rejected() {
    let p = tmp_path("hdr");
    write_sample(&p);
    let good = std::fs::read(&p).expect("read");
    let cases: [(usize, u8, &str); 5] = [
        (0, b'X', "format"),               // magic
        (4, 2, "unsupported_version"),     // version = 2
        (8, 1, "unsupported_feature"),     // unknown required feature bit
        (24, 0, "format"),                 // table_count = 0
        (23, 0x7f, "format"),              // table_offset far past EOF
    ];
    for (off, v, code) in cases {
        std::fs::write(&p, &good).expect("restore");
        corrupt(&p, off, |b| *b = v);
        let e = Reader::open(&p).err().expect("must fail");
        assert_eq!(e.code, code, "offset {off}: {e}");
    }
    // Truncated file.
    std::fs::write(&p, &good[..40]).expect("truncate");
    assert_eq!(Reader::open(&p).err().map(|e| e.code), Some("format"));
    // Metadata byte flipped -> section hash mismatch.
    std::fs::write(&p, &good).expect("restore");
    corrupt(&p, METADATA_OFFSET as usize + 2, |b| *b ^= 0x20);
    assert_eq!(Reader::open(&p).err().map(|e| e.code), Some("hash_mismatch"));
    // Unknown *required* section kind.
    std::fs::write(&p, &good).expect("restore");
    corrupt(&p, 64, |b| *b = 99);
    assert_eq!(Reader::open(&p).err().map(|e| e.code), Some("unsupported_section"));
    // Unknown optional section kind replaces METADATA -> metadata missing.
    std::fs::write(&p, &good).expect("restore");
    let mut b = good.clone();
    b[64] = 99;
    b[68] = 0;
    std::fs::write(&p, &b).expect("write");
    assert_eq!(Reader::open(&p).err().map(|e| e.code), Some("format"));
    std::fs::remove_file(&p).ok();
}

#[test]
fn corrupt_tensor_detected_by_verify() {
    let p = tmp_path("tensor");
    write_sample(&p);
    let r = Reader::open(&p).expect("open");
    let off = r.tensor("a").map(|t| t.offset as usize).expect("a");
    drop(r);
    corrupt(&p, off + 1, |b| *b ^= 0xff);
    let r = Reader::open(&p).expect("open still works (data hashes are lazy)");
    let e = r.verify_tensors().err().expect("must fail");
    assert_eq!(e.code, "hash_mismatch");
    assert!(e.message.contains("\"a\""));
    std::fs::remove_file(&p).ok();
}

#[test]
fn overlapping_ranges_rejected() {
    let mut r = vec![(0, 64, "h".to_string()), (60, 100, "s".to_string()), (100, 100, "e".to_string())];
    assert!(check_disjoint(&mut r).is_err());
    let mut r = vec![(0, 64, "h".to_string()), (64, 100, "s".to_string()), (50, 50, "e".to_string())];
    assert!(check_disjoint(&mut r).is_ok());
}

#[test]
fn block_size_checked() {
    let mut t = TensorInfo {
        name: "q".into(),
        dtype: DType::Int4,
        shape: vec![1, 32],
        offset: 0,
        nbytes: 0,
        hash: String::new(),
        block_size: Some(64),
    };
    assert_eq!(check_block_size(&mut t).err().map(|e| e.code), Some("unsupported"));
    t.block_size = None;
    check_block_size(&mut t).expect("absent means 32");
    assert_eq!(t.block_size, Some(32));
    t.dtype = DType::F32;
    assert_eq!(check_block_size(&mut t).err().map(|e| e.code), Some("format"));
    // Serialized index: present for quant, omitted otherwise.
    let j = serde_json::to_string(&t).expect("json");
    assert!(j.contains("\"block_size\":32"));
    t.block_size = None;
    assert!(!serde_json::to_string(&t).expect("json").contains("block_size"));
}

#[test]
fn kquant_expert_stack_round_trip() {
    let p = tmp_path("kq3");
    let mut w = Writer::create(&p, meta()).expect("create");
    let shape = [3usize, 2, 512];
    let n = DType::Q4K.storage_bytes(&shape);
    let bytes: Vec<u8> = (0..n).map(|i| (i % 253) as u8).collect();
    w.add("experts", DType::Q4K, &shape, &bytes).expect("3-D k-quant stack");
    let n8 = DType::Int8.storage_bytes(&[2, 4, 64]);
    w.add("experts8", DType::Int8, &[2, 4, 64], &vec![1u8; n8]).expect("3-D int8 stack");
    assert!(w.add("bad", DType::Q6K, &[2, 32], &[0u8; 8]).is_err(), "k-quant cols must be % 256");
    w.finish().expect("finish");
    let r = Reader::open(&p).expect("open");
    let t = r.tensor("experts").expect("tensor");
    assert_eq!(t.block_size, Some(256));
    assert_eq!(r.tensor_bytes(t), &bytes[..]);
    assert_eq!(r.tensor("experts8").expect("int8").block_size, Some(32));
    std::fs::remove_file(&p).ok();
    // Older writers emitted 32 for k-quants: read as 256.
    let mut info = TensorInfo {
        name: "k".into(),
        dtype: DType::Q5K,
        shape: vec![1, 256],
        offset: 0,
        nbytes: 0,
        hash: String::new(),
        block_size: Some(32),
    };
    check_block_size(&mut info).expect("k-quant block_size ignored");
    assert_eq!(info.block_size, Some(256));
}

#[test]
fn blobs_round_trip() {
    let p = tmp_path("blobs");
    let mut m = meta();
    m["components"] = json!({
        "lm": {"architecture": "llama", "config": {"d": 4}, "license": "apache-2.0", "tensor_prefix": "lm."},
        "codec": {"architecture": "codec", "config": "extra", "tensor_prefix": "codec."}
    });
    let mut w = Writer::create(&p, m).expect("create");
    w.add("lm.w", DType::F32, &[2], &[1u8; 8]).expect("add");
    w.add_blob("tokenizer/tokenizer.json", "application/json", b"{\"v\":1}").expect("blob");
    w.add_blob("voice/default.bin", "application/octet-stream", &[9u8; 300]).expect("blob");
    assert_eq!(w.add_blob("voice/default.bin", "application/octet-stream", &[]).err().map(|e| e.code), Some("invalid_argument"));
    assert_eq!(w.add_blob("x", "bad", &[]).err().map(|e| e.code), Some("invalid_argument"));
    w.finish().expect("finish");
    let r = Reader::open(&p).expect("open");
    assert_eq!(r.blob("tokenizer/tokenizer.json"), Some(&b"{\"v\":1}"[..]));
    assert_eq!(r.blob("voice/default.bin").map(|b| b.len()), Some(300));
    assert_eq!(r.blob("nope"), None);
    assert_eq!(r.blobs().len(), 2);
    assert_eq!(r.components().len(), 2);
    assert_eq!(r.kind(), crate::FileKind::Model);
    r.verify_tensors().expect("verify");
    // Flip a blob byte: the BLOBS section hash fails on open.
    let off = (r.parsed.blobs_offset + r.blobs()[1].offset) as usize;
    drop(r);
    corrupt(&p, off, |b| *b ^= 1);
    assert_eq!(Reader::open(&p).err().map(|e| e.code), Some("hash_mismatch"));
    std::fs::remove_file(&p).ok();
}

#[test]
fn shared_blobs_streamed() {
    let p = tmp_path("shared-blobs");
    // Larger than one copy chunk, from a shared image (as a source mmap would be).
    let image: Vec<u8> = (0..(70usize << 20) + 123).map(|i| (i * 7 + i / 4096) as u8).collect();
    let image: Bytes = Arc::new(image);
    let mut w = Writer::create(&p, meta()).expect("create");
    w.add("w", DType::F32, &[2], &[1u8; 8]).expect("add");
    w.add_blob_shared("big", "application/octet-stream", image.clone(), 100..(70 << 20) + 101).expect("blob");
    w.add_blob("small", "text/plain", b"hi").expect("blob");
    w.add_blob_shared("empty", "application/octet-stream", image.clone(), 5..5).expect("blob");
    let n = (*image).as_ref().len();
    assert_eq!(w.add_blob_shared("oob", "application/octet-stream", image.clone(), 0..n + 1).err().map(|e| e.code), Some("invalid_argument"));
    assert_eq!(w.add_blob_shared("big", "application/octet-stream", image.clone(), 0..1).err().map(|e| e.code), Some("invalid_argument"));
    w.finish().expect("finish");
    let r = Reader::open(&p).expect("open");
    assert_eq!(r.blob("big"), Some(&(*image).as_ref()[100..(70 << 20) + 101]));
    assert_eq!(r.blob("small"), Some(&b"hi"[..]));
    assert_eq!(r.blob("empty"), Some(&[][..]));
    r.verify_tensors().expect("verify");
    std::fs::remove_file(&p).ok();
}

#[test]
fn bad_components_rejected_by_writer() {
    let p = tmp_path("comp");
    let mut m = meta();
    m["components"] = json!({"lm": {"architecture": "llama", "config": {}, "tensor_prefix": "lm"}});
    assert_eq!(Writer::create(&p, m).err().map(|e| e.code), Some("format"));
    let mut m = meta();
    m["format"] = json!("lapid/2");
    assert_eq!(Writer::create(&p, m).err().map(|e| e.code), Some("format"));
    assert!(!partial_path(&p).exists());
}

#[test]
fn extras_file() {
    let p = tmp_path("extra");
    let mut m = meta();
    m["kind"] = json!("extra");
    m["extends"] = json!({"architecture": "test", "component": "voice"});
    let mut w = Writer::create(&p, m).expect("create");
    w.add("voice.mine.embedding", DType::F32, &[1], &[0u8; 4]).expect("add");
    w.finish().expect("finish");
    let r = Reader::open(&p).expect("open");
    assert_eq!(r.kind(), crate::FileKind::Extra);
    assert_eq!(r.extends().map(|e| e.component.as_str()), Some("voice"));
    assert!(r.components().is_empty());
    std::fs::remove_file(&p).ok();
}

/// A file laid out exactly as the format-1.0 writer did (METADATA at 256, three sections,
/// `"format": "lapid/1"`) must still open.
#[test]
fn format_1_0_file_reads() {
    let mut m = meta();
    m["format"] = json!("lapid/1");
    m["components"] = json!("free-form in 1.0");
    let meta_bytes = serde_json::to_vec(&m).expect("json");
    let data_off = align_up(256 + meta_bytes.len() as u64 + METADATA_PAD, DATA_ALIGN);
    let tensor = [3u8; 8];
    let info = TensorInfo {
        name: "w".into(),
        dtype: DType::F32,
        shape: vec![2],
        offset: data_off,
        nbytes: 8,
        hash: blake3::hash(&tensor).to_hex().to_string(),
        block_size: None,
    };
    let index = serde_json::to_vec(&vec![info]).expect("json");
    let index_off = align_up(data_off + 8, TENSOR_ALIGN);
    let mut b = vec![0u8; (index_off as usize) + index.len()];
    b[..64].copy_from_slice(&encode_header(64, 3));
    let secs = [
        Section { kind: 1, flags: 1, offset: 256, length: meta_bytes.len() as u64, hash: *blake3::hash(&meta_bytes).as_bytes() },
        Section { kind: 2, flags: 1, offset: index_off, length: index.len() as u64, hash: *blake3::hash(&index).as_bytes() },
        Section { kind: 3, flags: 1, offset: data_off, length: 8, hash: *blake3::hash(&tensor).as_bytes() },
    ];
    for (i, s) in secs.iter().enumerate() {
        b[64 + i * ENTRY_BYTES..64 + (i + 1) * ENTRY_BYTES].copy_from_slice(&s.encode());
    }
    b[256..256 + meta_bytes.len()].copy_from_slice(&meta_bytes);
    b[data_off as usize..data_off as usize + 8].copy_from_slice(&tensor);
    b[index_off as usize..].copy_from_slice(&index);
    let p = tmp_path("v10");
    std::fs::write(&p, &b).expect("write");
    let r = Reader::open(&p).expect("1.0 file opens");
    assert_eq!(r.metadata["format"], "lapid/1");
    assert_eq!(r.tensor_bytes(r.tensor("w").expect("w")), &tensor);
    assert!(r.blobs().is_empty());
    assert_eq!(r.components()[0].name, "test");
    r.verify_tensors().expect("verify");
    std::fs::remove_file(&p).ok();
}

#[test]
fn from_bytes_is_zero_copy_and_matches_open() {
    let p = tmp_path("bytes");
    let tensors = write_sample(&p);
    let image = std::fs::read(&p).expect("read");
    let _ = std::fs::remove_file(&p);
    let data: crate::Bytes = Arc::new(image);
    let r = Reader::from_bytes(data.clone()).expect("from_bytes");
    let base = (*data).as_ref().as_ptr() as usize;
    let len = (*data).as_ref().len();
    assert_eq!(r.as_bytes().as_ptr() as usize, base);
    for (n, _, _, b) in &tensors {
        let t = r.tensor(n).expect("tensor present");
        let got = r.tensor_bytes(t);
        assert_eq!(got, &b[..]);
        let p = got.as_ptr() as usize;
        assert!(p >= base && p + got.len() <= base + len, "tensor {n} not inside the image");
    }
    r.verify_tensors().expect("verify");
}

#[test]
fn from_static_and_bad_bytes() {
    let p = tmp_path("static");
    write_sample(&p);
    let image: &'static [u8] = Box::leak(std::fs::read(&p).expect("read").into_boxed_slice());
    let _ = std::fs::remove_file(&p);
    let r = Reader::from_static(image).expect("from_static");
    assert_eq!(r.tensors.len(), 4);
    assert_eq!(r.as_bytes().as_ptr(), image.as_ptr());

    let bad = Reader::from_bytes(Arc::new(vec![0u8; 16])).err().map(|e| e.code);
    assert_eq!(bad, Some("format"));
    let mut truncated = image.to_vec();
    truncated.truncate(image.len() / 2);
    assert!(Reader::from_bytes(Arc::new(truncated)).is_err());
}
