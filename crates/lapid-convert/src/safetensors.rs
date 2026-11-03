//! Minimal, validating safetensors reader (mmap-backed).
//!
//! Layout: u64 LE header length N | N bytes JSON header | data. The header maps tensor
//! names to `{"dtype", "shape", "data_offsets": [begin, end]}` (offsets relative to the
//! data start) plus an optional `"__metadata__"` string map.

use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use lapid_tensor::{Error, Result, bail};
use half::{bf16, f16};

const MAX_HEADER: u64 = 100 << 20;

#[derive(Clone, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

pub struct SafeTensors {
    mmap: Arc<memmap2::Mmap>,
    data_start: usize,
    entries: BTreeMap<String, Entry>,
    metadata: BTreeMap<String, String>,
}

fn st_err(msg: impl Into<String>) -> Error {
    Error::new("safetensors", msg)
}

fn elem_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        _ => return None,
    })
}

impl SafeTensors {
    pub fn open(p: &Path) -> Result<SafeTensors> {
        let file = File::open(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?;
        let size = file.metadata()?.len();
        if size < 8 {
            bail!("safetensors", "{}: file too small ({size} B)", p.display());
        }
        // SAFETY: read-only mapping; the file must not be truncated while open.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::new("io", format!("mmap {}: {e}", p.display())))?;
        Self::parse(mmap).map_err(|e| Error::new(e.code, format!("{}: {}", p.display(), e.message)))
    }

    fn parse(mmap: memmap2::Mmap) -> Result<SafeTensors> {
        let mut n8 = [0u8; 8];
        n8.copy_from_slice(&mmap[0..8]);
        let n = u64::from_le_bytes(n8);
        if n > MAX_HEADER || n + 8 > mmap.len() as u64 {
            bail!("safetensors", "header length {n} invalid for file of {} B", mmap.len());
        }
        let data_start = 8 + n as usize;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_start]).map_err(|e| st_err(format!("header JSON: {e}")))?;
        let data_len = mmap.len() - data_start;
        let mut entries = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, val) in m {
                        if let Some(s) = val.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = v.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| st_err(format!("{name}: missing dtype")))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| st_err(format!("{name}: missing shape")))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| st_err(format!("{name}: bad shape")))?;
            let offs: Vec<u64> = v
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            if offs.len() != 2 || offs[0] > offs[1] || offs[1] > data_len as u64 {
                bail!("safetensors", "{name}: bad data_offsets (data is {data_len} B)");
            }
            let (begin, end) = (offs[0] as usize, offs[1] as usize);
            if let Some(es) = elem_size(dtype) {
                let numel = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
                if numel.and_then(|n| n.checked_mul(es)) != Some(end - begin) {
                    bail!("safetensors", "{name}: {dtype} {shape:?} does not match {} data bytes", end - begin);
                }
            }
            entries.insert(name, Entry { dtype: dtype.to_string(), shape, begin, end });
        }
        Ok(SafeTensors { mmap: Arc::new(mmap), data_start, entries, metadata })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The `__metadata__` string map (empty if absent).
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| Error::new("missing_tensor", format!("safetensors: no tensor {name:?}")))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.entry(name)?.shape.clone())
    }

    /// safetensors dtype string ("F32", "F16", "BF16", ...).
    pub fn dtype(&self, name: &str) -> Result<String> {
        Ok(self.entry(name)?.dtype.clone())
    }

    /// Raw little-endian bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entry(name)?;
        Ok(&self.mmap[self.data_start + e.begin..self.data_start + e.end])
    }

    /// The whole file mapping and the byte range of a tensor in it (zero-copy streaming,
    /// e.g. into a blob).
    pub fn shared(&self, name: &str) -> Result<(lapid_format::Bytes, Range<usize>)> {
        let e = self.entry(name)?;
        Ok((self.mmap.clone(), self.data_start + e.begin..self.data_start + e.end))
    }

    /// Tensor converted to f32: floats (F32, F16, BF16, F64, F8_E4M3; F8_E8M0 as the power of
    /// two it encodes) and integers (exact up to 2^24). Large tensors convert on all cores.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let b = self.bytes(name)?;
        fn conv<const N: usize>(b: &[u8], f: impl Fn([u8; N]) -> f32 + Sync) -> Vec<f32> {
            let n = b.len() / N;
            let mut out = vec![0f32; n];
            crate::parallel::fill_rows(&mut out, 1, 1 << 16, |r, o| {
                for (y, c) in o.iter_mut().zip(b[r.start * N..r.end * N].chunks_exact(N)) {
                    *y = f(c.try_into().unwrap_or([0; N]));
                }
            });
            out
        }
        Ok(match e.dtype.as_str() {
            "F32" => conv::<4>(b, f32::from_le_bytes),
            "F16" => conv::<2>(b, |c| f16::from_le_bytes(c).to_f32()),
            "BF16" => conv::<2>(b, |c| bf16::from_le_bytes(c).to_f32()),
            "F8_E4M3" => conv::<1>(b, |c| crate::microscale::e4m3(c[0])),
            "F8_E8M0" => conv::<1>(b, |c| crate::microscale::e8m0(c[0])),
            "I8" => conv::<1>(b, |c| c[0] as i8 as f32),
            "U8" => conv::<1>(b, |c| c[0] as f32),
            "F64" => b
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(c);
                    f64::from_le_bytes(a) as f32
                })
                .collect(),
            // integer tables (e.g. hash routing ids), exact up to 2^24
            "I32" => b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
            "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f32).collect(),
            d => bail!("unsupported_dtype", "safetensors tensor {name:?} has dtype {d}, expected a float or integer type"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(header: &str, data: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "lapid-st-test-{}-{}.safetensors",
            std::process::id(),
            header.len() + data.len()
        ));
        let mut f = File::create(&p).expect("create");
        f.write_all(&(header.len() as u64).to_le_bytes()).expect("w");
        f.write_all(header.as_bytes()).expect("w");
        f.write_all(data).expect("w");
        p
    }

    #[test]
    fn parse_and_convert() {
        let mut data = Vec::new();
        for v in [1.0f32, -2.5] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        data.extend_from_slice(&bf16::from_f32(-4.0).to_le_bytes());
        let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[1],"data_offsets":[8,10]},"c":{"dtype":"BF16","shape":[1,1],"data_offsets":[10,12]}}"#;
        let p = write_file(header, &data);
        let st = SafeTensors::open(&p).expect("open");
        assert_eq!(st.names(), vec!["a", "b", "c"]);
        assert_eq!(st.metadata().get("format").map(String::as_str), Some("pt"));
        assert_eq!(st.to_f32("a").expect("a"), vec![1.0, -2.5]);
        assert_eq!(st.to_f32("b").expect("b"), vec![0.5]);
        assert_eq!(st.to_f32("c").expect("c"), vec![-4.0]);
        assert_eq!(st.shape("c").expect("c"), vec![1, 1]);
        assert_eq!(st.dtype("b").expect("b"), "F16");
        assert_eq!(st.to_f32("zz").err().map(|e| e.code), Some("missing_tensor"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_bad_offsets() {
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let p = write_file(header, &[0u8; 8]);
        assert_eq!(SafeTensors::open(&p).err().map(|e| e.code), Some("safetensors"));
        std::fs::remove_file(&p).ok();
    }
}
