//! Writing helpers shared by every model converter: required metadata, a policy-applying
//! streaming writer with progress output, completeness checks and source hashing.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{IsTerminal, Write};
use std::path::Path;

use lapid_format::Writer;
use lapid_tensor::{DType, Error, Result, bail};
use serde_json::{Value, json};

use crate::OutDType;
use crate::policy::{Role, Target, encode, target_named};

/// Classifies a tensor (by name and shape) for the dtype policy.
pub type RoleFn = fn(&str, &[usize]) -> Role;

/// A shared role classifier (e.g. a closure over a declarative role table).
pub type RoleArc = std::sync::Arc<dyn Fn(&str, &[usize]) -> Role + Send + Sync>;

/// The required `.lapid` metadata keys. Models add their own config object
/// (e.g. `meta["whisper"] = ...`).
pub fn metadata(name: &str, architecture: &str, dtype: OutDType, license: &str, source: Value) -> Value {
    json!({
        "format": lapid_format::FORMAT_ID,
        "name": name,
        "architecture": architecture,
        "dtype": dtype.name(),
        "license": license,
        "source": source,
        "converter": concat!("lapid-convert ", env!("CARGO_PKG_VERSION")),
    })
}

/// Fails with "missing_tensor" listing (up to 8) names of `expected` absent from `have`.
pub fn check_complete(have: &BTreeSet<String>, expected: &[String]) -> Result<()> {
    let missing: Vec<&str> = expected.iter().filter(|n| !have.contains(*n)).map(|s| s.as_str()).collect();
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(8).copied().collect();
        bail!("missing_tensor", "{} expected tensors missing: {}", missing.len(), shown.join(", "));
    }
    Ok(())
}

/// `"blake3:<hex>"` of a source file (recorded in `metadata.source.file_hash`).
pub fn file_hash(path: &Path) -> Result<String> {
    let f = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    let mut h = blake3::Hasher::new();
    h.update_reader(f)?;
    Ok(format!("blake3:{}", h.finalize().to_hex()))
}

/// Read and parse a JSON file (errors "io" / "json").
pub fn read_json(path: &Path) -> Result<Value> {
    let s = std::fs::read(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&s).map_err(|e| Error::new("json", format!("{}: {e}", path.display())))
}

/// Wraps a `lapid_format::Writer`: applies the dtype policy through the model's [`RoleFn`],
/// tracks written names and prints progress on stderr.
pub struct ModelWriter {
    writer: Writer,
    dtype: OutDType,
    role: RoleArc,
    written: BTreeSet<String>,
    blobs: BTreeSet<String>,
    total: usize,
    tty: bool,
    n_layer: Option<usize>,
    prequantized: BTreeMap<String, (DType, Vec<u8>)>,
}

/// Keys whose integer value is taken as the layer count (for the q4_k_m mix).
const LAYER_KEYS: [&str; 5] = ["num_hidden_layers", "n_layer", "n_layers", "num_layers", "block_count"];

/// First integer under one of `LAYER_KEYS`, searching `meta` breadth-first.
fn find_layer_count(meta: &Value) -> Option<usize> {
    let mut queue = std::collections::VecDeque::from([meta]);
    while let Some(v) = queue.pop_front() {
        if let Value::Object(m) = v {
            for k in LAYER_KEYS {
                if let Some(n) = m.get(k).and_then(Value::as_u64) {
                    return Some(n as usize);
                }
            }
            queue.extend(m.values());
        }
    }
    None
}

impl ModelWriter {
    pub fn create(out: &Path, metadata: Value, dtype: OutDType, role: RoleFn, total: usize) -> Result<ModelWriter> {
        ModelWriter::create_shared(out, metadata, dtype, std::sync::Arc::new(role), total)
    }

    /// [`create`](Self::create) with a shared classifier.
    pub fn create_shared(out: &Path, metadata: Value, dtype: OutDType, role: RoleArc, total: usize) -> Result<ModelWriter> {
        if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("create {}: {e}", dir.display())))?;
        }
        let n_layer = find_layer_count(&metadata);
        Ok(ModelWriter {
            writer: Writer::create(out, metadata)?,
            dtype,
            role,
            written: BTreeSet::new(),
            blobs: BTreeSet::new(),
            total,
            tty: std::io::stderr().is_terminal(),
            n_layer,
            prequantized: BTreeMap::new(),
        })
    }

    /// Layer count used by the q4_k_m mix (default: the first `num_hidden_layers` /
    /// `n_layer` / `n_layers` / `num_layers` / `block_count` integer in the metadata).
    pub fn set_layer_count(&mut self, n: Option<usize>) {
        self.n_layer = n;
    }

    /// Already-encoded tensors (e.g. calibrated int4 from `calibrate`) that `add_f32` writes
    /// instead of policy-encoding the f32 values it is given.
    pub fn set_prequantized(&mut self, tensors: BTreeMap<String, (DType, Vec<u8>)>) {
        self.prequantized = tensors;
    }

    fn progress(&self, name: &str, dtype: DType) {
        let n = self.written.len();
        let mut err = std::io::stderr().lock();
        if self.tty {
            let _ = write!(err, "\r\x1b[2Kconverting [{n}/{}] {name} -> {}", self.total, dtype.name());
            let _ = err.flush();
        } else if n == self.total || n % 50 == 0 {
            let _ = writeln!(err, "converting [{n}/{}]", self.total);
        }
    }

    /// Storage target of `name` under this writer's dtype policy.
    pub fn target(&self, name: &str, shape: &[usize]) -> Target {
        target_named(name, (self.role)(name, shape), shape, self.dtype, self.n_layer)
    }

    /// Policy-encode and write an f32 tensor (or its prequantized bytes, if set).
    pub fn add_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        if let Some((dt, bytes)) = self.prequantized.remove(name) {
            return self.add_raw(name, dt, shape, &bytes);
        }
        let (dt, bytes) = encode(self.target(name, shape), data, shape)?;
        self.add_raw(name, dt, shape, &bytes)
    }

    /// Whether `add_f32(name, ..)` will write prequantized bytes (the f32 data is ignored).
    pub fn is_prequantized(&self, name: &str) -> bool {
        self.prequantized.contains_key(name)
    }

    /// Write already-encoded bytes.
    pub fn add_raw(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<()> {
        self.writer.add(name, dtype, shape, bytes)?;
        self.written.insert(name.to_string());
        self.progress(name, dtype);
        Ok(())
    }

    /// Store a named byte blob (tokenizer files, templates, ...) in the BLOBS section.
    pub fn add_blob(&mut self, name: &str, media_type: &str, bytes: &[u8]) -> Result<()> {
        self.writer.add_blob(name, media_type, bytes)?;
        self.blobs.insert(name.to_string());
        Ok(())
    }

    /// Store `data[range]` of a shared image (e.g. a source mmap) as a blob without copying
    /// it (streamed into the file by `finish`).
    pub fn add_blob_shared(&mut self, name: &str, media_type: &str, data: lapid_format::Bytes, range: std::ops::Range<usize>) -> Result<()> {
        self.writer.add_blob_shared(name, media_type, data, range)?;
        self.blobs.insert(name.to_string());
        Ok(())
    }

    /// Check that every `expected` tensor (or blob) was written, then finish the file
    /// atomically (streaming the shared blobs).
    pub fn finish(mut self, expected: &[String]) -> Result<()> {
        if self.tty {
            let _ = writeln!(std::io::stderr());
        }
        let blobs = std::mem::take(&mut self.blobs);
        self.written.extend(blobs);
        check_complete(&self.written, expected)?;
        self.writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_listed() {
        let have: BTreeSet<String> = ["a".to_string()].into();
        assert!(check_complete(&have, &["a".to_string()]).is_ok());
        let e = check_complete(&have, &["a".to_string(), "b".to_string()]).err().expect("missing");
        assert_eq!(e.code, "missing_tensor");
        assert!(e.message.contains('b'));
    }

    #[test]
    fn layer_count_from_metadata() {
        assert_eq!(find_layer_count(&json!({"name": "x", "lfm3": {"num_hidden_layers": 24}})), Some(24));
        assert_eq!(find_layer_count(&json!({"a": {"b": {"n_layer": 6}}, "c": 1})), Some(6));
        assert_eq!(find_layer_count(&json!({"a": 1})), None);
    }

    #[test]
    fn required_keys() {
        let m = metadata("m", "arch", OutDType::Int8, "mit", json!({"repo": "x/y"}));
        assert_eq!(m["dtype"], "int8");
        assert_eq!(m["architecture"], "arch");
        assert_eq!(m["format"], "lapid/1.1");
    }
}
