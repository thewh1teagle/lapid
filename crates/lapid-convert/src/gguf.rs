//! GGUF reader (llama.cpp / ggml container, versions 2 and 3, little endian).
//!
//! ```text
//! "GGUF" | u32 version | u64 n_tensors | u64 n_kv
//! n_kv x      (string key, u32 value_type, value)
//! n_tensors x (string name, u32 n_dims, n_dims x u64 ne, u32 ggml_type, u64 offset)
//! pad to `general.alignment` (default 32) | tensor data (offsets relative to here)
//! ```
//! Strings are `u64 len` + UTF-8 bytes. Arrays are `u32 elem_type, u64 len, elems`.
//! `ne[0]` is the fastest-varying dimension: the row-major shape is `ne` reversed.
//! Model-specific tensor naming / metadata mapping lives with each model in `lapid-hub`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use lapid_quant::Format;
use lapid_tensor::{DType, Error, Result, bail};
use serde_json::{Value, json};

use crate::ggml::{self, GgmlType};
use crate::policy::Target;
use crate::writer::ModelWriter;

pub const MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u32 = 4;
const MAX_STRING: u64 = 1 << 30;

/// A typed GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Integer value (any integer type, non-negative), else `None`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) => u64::try_from(v).ok(),
            GgufValue::I16(v) => u64::try_from(v).ok(),
            GgufValue::I32(v) => u64::try_from(v).ok(),
            GgufValue::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            GgufValue::F32(v) => Some(v as f64),
            GgufValue::F64(v) => Some(v),
            GgufValue::I8(v) => Some(v as f64),
            GgufValue::I16(v) => Some(v as f64),
            GgufValue::I32(v) => Some(v as f64),
            GgufValue::I64(v) => Some(v as f64),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(s) = self { Some(s) } else { None }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        if let GgufValue::Array(a) = self { Some(a) } else { None }
    }

    /// JSON form (numbers, bools, strings, arrays); non-finite floats become null.
    pub fn to_json(&self) -> Value {
        match self {
            GgufValue::Bool(b) => json!(b),
            GgufValue::String(s) => json!(s),
            GgufValue::Array(a) => Value::Array(a.iter().map(GgufValue::to_json).collect()),
            GgufValue::F32(v) => json!(v),
            GgufValue::F64(v) => json!(v),
            GgufValue::I8(v) => json!(v),
            GgufValue::I16(v) => json!(v),
            GgufValue::I32(v) => json!(v),
            GgufValue::I64(v) => json!(v),
            v => json!(v.as_u64()),
        }
    }
}

/// One tensor of a GGUF file.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufTensor {
    pub name: String,
    pub ggml_type: GgmlType,
    /// Row-major shape (`ne` reversed: `[rows, cols]` for a matrix with `ne = [cols, rows]`).
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub nbytes: u64,
}

impl GgufTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// An opened (mmapped) GGUF file.
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensor>,
    pub alignment: u64,
    mmap: Arc<memmap2::Mmap>,
}

/// True when the file starts with the GGUF magic.
pub fn is_gguf(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    let mut m = [0u8; 4];
    Ok(f.read_exact(&mut m).is_ok() && m == MAGIC)
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        let file = std::fs::File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
        // SAFETY: read-only mapping; the file must not be truncated while open.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::new("io", format!("mmap {}: {e}", path.display())))?;
        let (version, metadata, tensors, alignment) =
            parse(&mmap).map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))?;
        Ok(GgufFile { version, metadata, tensors, alignment, mmap: Arc::new(mmap) })
    }

    /// `general.architecture` (e.g. `"llama"`, `"whisper"`).
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw (ggml-encoded) bytes of a tensor, zero copy.
    pub fn tensor_data(&self, t: &GgufTensor) -> &[u8] {
        &self.mmap[t.offset as usize..(t.offset + t.nbytes) as usize]
    }

    /// Dequantized f32 values of a tensor (error "unsupported_gguf_type" for other types).
    pub fn to_f32(&self, t: &GgufTensor) -> Result<Vec<f32>> {
        ggml::dequantize(t.ggml_type, self.tensor_data(t), t.numel())
            .map_err(|e| Error::new(e.code, format!("tensor {:?}: {}", t.name, e.message)))
    }

    /// Write tensor `t` as `name` through the writer's dtype policy: Q8_0 → int8 and
    /// Q4_0 → int4 are repacked losslessly when the policy picks that format; Q4_K / Q5_K /
    /// Q6_K blocks are kept as-is whenever the policy quantizes the tensor at all (zero
    /// conversion); f16/f32 are copied when the target matches; everything else is
    /// dequantized and re-encoded. `shape` overrides the stored shape (same element count),
    /// e.g. to flatten biases. 3-D tensors are expert stacks `[E, rows, cols]`.
    pub fn write_tensor(&self, w: &mut ModelWriter, name: &str, t: &GgufTensor, shape: Option<&[usize]>) -> Result<()> {
        write_ggml_tensor(w, name, t, self.tensor_data(t), shape)
    }

    /// All metadata as a JSON object (arrays included; large token arrays can be big).
    pub fn metadata_json(&self) -> Value {
        Value::Object(self.metadata.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
    }
}

/// [`GgufFile::write_tensor`] with the tensor's (possibly edited, e.g. row-permuted) raw bytes.
pub fn write_ggml_tensor(w: &mut ModelWriter, name: &str, t: &GgufTensor, data: &[u8], shape: Option<&[usize]>) -> Result<()> {
    {
        let shape = shape.map_or_else(|| t.shape.clone(), <[usize]>::to_vec);
        if shape.iter().product::<usize>() != t.numel() {
            bail!("shape_mismatch", "tensor {:?}: shape {shape:?} does not match {:?}", t.name, t.shape);
        }
        let target = w.target(name, &shape);
        let keep = match (t.ggml_type.lapid_format(), target) {
            (Some(f @ (Format::Int8 | Format::Int4)), Target::Quant(g)) if f == g => Some(f),
            (Some(f @ (Format::Q4K | Format::Q5K | Format::Q6K)), Target::Quant(_)) => Some(f),
            _ => None,
        };
        // ggml blocks run along ne[0] = the last row-major axis, which the policy quantizes.
        if let Some(f) = keep
            && matches!(shape.len(), 2 | 3)
            && shape.last() == t.shape.last()
        {
            let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let per = data.len() / shape[..shape.len() - 2].iter().product::<usize>().max(1);
            let mut bytes = Vec::new();
            for m in data.chunks_exact(per) {
                bytes.extend(lapid_quant::from_ggml_blocks(f, m, rows, cols));
            }
            let dt = Target::Quant(f).dtype();
            return w.add_raw(name, dt, &shape, &bytes);
        }
        match (t.ggml_type, target) {
            (GgmlType::F16, Target::F16) => w.add_raw(name, DType::F16, &shape, data),
            (GgmlType::F32, Target::F32) => w.add_raw(name, DType::F32, &shape, data),
            _ => {
                let v = ggml::dequantize(t.ggml_type, data, t.numel())
                    .map_err(|e| Error::new(e.code, format!("tensor {:?}: {}", t.name, e.message)))?;
                w.add_f32(name, &shape, &v)
            }
        }
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let end = (self.pos as u64).checked_add(n).filter(|&e| e <= self.b.len() as u64);
        let Some(end) = end else {
            bail!("format", "gguf: truncated at byte {} (need {n} more)", self.pos);
        };
        let s = &self.b[self.pos..end as usize];
        self.pos = end as usize;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N as u64)?);
        Ok(a)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            bail!("limit_exceeded", "gguf: string of {n} bytes");
        }
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::new("format", format!("gguf: invalid UTF-8 string at byte {}", self.pos)))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<GgufValue> {
        Ok(match ty {
            0 => GgufValue::U8(self.arr::<1>()?[0]),
            1 => GgufValue::I8(self.arr::<1>()?[0] as i8),
            2 => GgufValue::U16(u16::from_le_bytes(self.arr()?)),
            3 => GgufValue::I16(i16::from_le_bytes(self.arr()?)),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(i32::from_le_bytes(self.arr()?)),
            6 => GgufValue::F32(f32::from_le_bytes(self.arr()?)),
            7 => match self.arr::<1>()?[0] {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                b => bail!("format", "gguf: invalid bool byte {b}"),
            },
            8 => GgufValue::String(self.string()?),
            9 => {
                if depth > 0 {
                    bail!("format", "gguf: nested arrays are not supported");
                }
                let et = self.u32()?;
                let n = self.u64()?;
                // Every element takes at least one byte: bounds the allocation.
                if n > (self.b.len() - self.pos) as u64 {
                    bail!("format", "gguf: array of {n} elements exceeds the file");
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                GgufValue::Array(v)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(i64::from_le_bytes(self.arr()?)),
            12 => GgufValue::F64(f64::from_le_bytes(self.arr()?)),
            t => bail!("format", "gguf: unknown value type {t}"),
        })
    }
}

type Parsed = (u32, BTreeMap<String, GgufValue>, Vec<GgufTensor>, u64);

/// Parses and validates a complete GGUF image.
pub fn parse(b: &[u8]) -> Result<Parsed> {
    let mut c = Cursor { b, pos: 0 };
    if c.take(4).ok() != Some(&MAGIC[..]) {
        bail!("format", "not a GGUF file (bad magic)");
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        // A byte-swapped version means a big-endian file.
        let hint = if version.swap_bytes() <= 3 { " (big-endian GGUF is not supported)" } else { "" };
        bail!("unsupported_version", "gguf version {version}{hint} (supported: 2, 3)");
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
        bail!("limit_exceeded", "gguf: {n_tensors} tensors / {n_kv} metadata entries");
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let v = c.value(ty, 0)?;
        if metadata.insert(key.clone(), v).is_some() {
            bail!("format", "gguf: duplicate metadata key {key:?}");
        }
    }
    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(GgufValue::U32(a)) if *a > 0 && a.is_power_of_two() => *a as u64,
        Some(v) => bail!("format", "gguf: invalid general.alignment {v:?}"),
    };
    let mut raw = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            bail!("format", "gguf: tensor {name:?} has {n_dims} dims");
        }
        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            ne.push(c.u64()?);
        }
        let ty = GgmlType::from_id(c.u32()?);
        let offset = c.u64()?;
        raw.push((name, ne, ty, offset));
    }
    let data_start = (c.pos as u64).div_ceil(alignment) * alignment;
    let size = b.len() as u64;
    let mut tensors = Vec::with_capacity(raw.len());
    let mut names = std::collections::HashSet::new();
    for (name, ne, ty, offset) in raw {
        if !names.insert(name.clone()) {
            bail!("format", "gguf: duplicate tensor {name:?}");
        }
        let numel = ne.iter().try_fold(1u64, |a, &d| a.checked_mul(d));
        let numel = numel.filter(|&n| n <= 1 << 48).ok_or_else(|| Error::new("format", format!("gguf: tensor {name:?} too large")))?;
        // Unsupported types still get listed (with their size unknown) so callers can name them.
        let nbytes = match ty.block() {
            Some(_) => ty.nbytes(numel).map_err(|e| Error::new(e.code, format!("gguf: tensor {name:?}: {}", e.message)))?,
            None => 0,
        };
        if offset % alignment != 0 {
            bail!("format", "gguf: tensor {name:?} offset {offset} not {alignment}-aligned");
        }
        let abs = data_start.checked_add(offset).filter(|a| a.checked_add(nbytes).is_some_and(|e| e <= size));
        let Some(abs) = abs else {
            bail!("format", "gguf: tensor {name:?} [{offset}, +{nbytes}) outside the file");
        };
        let shape = ne.iter().rev().map(|&d| d as usize).collect();
        tensors.push(GgufTensor { name, ggml_type: ty, shape, offset: abs, nbytes });
    }
    Ok((version, metadata, tensors, alignment))
}

#[cfg(test)]
#[path = "gguf_tests.rs"]
pub(crate) mod tests;
