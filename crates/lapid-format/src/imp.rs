//! `.lapid` v1 reader / streaming writer. Byte layout: `lib.rs` doc comment and docs/spec/format.md.
//!
//! Writer (`crate::write`) file order: header (64 B) | section table (3-4 x 56 B) |
//! METADATA (at 1024) | 64 KiB zero pad | TENSOR_DATA (4096-aligned, tensors 256-aligned) |
//! TENSOR_INDEX | BLOBS (4096-aligned, only when blobs were added).
//! The output is written to `<path>.partial` and atomically renamed by `finish`.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use lapid_tensor::{DType, Error, Result, bail};

use crate::blobs::{self, MAX_BLOBS_SECTION};
use crate::meta::{check_metadata, parse_components, parse_kind};
use crate::{
    Bytes, MAGIC, Parsed, Reader, SECTION_BLOBS, SECTION_METADATA, SECTION_TENSOR_DATA,
    SECTION_TENSOR_INDEX, TensorInfo, VERSION,
};

pub(crate) const HEADER_BYTES: usize = 64;
pub(crate) const ENTRY_BYTES: usize = 56;
pub(crate) const MAX_SECTIONS: u64 = 64;
pub(crate) const MAX_METADATA: u64 = 16 << 20;
pub(crate) const MAX_INDEX: u64 = 64 << 20;
pub(crate) const MAX_TENSORS: usize = 1_000_000;
pub(crate) const METADATA_PAD: u64 = 64 << 10;
pub(crate) const DATA_ALIGN: u64 = 4096;
pub(crate) const TENSOR_ALIGN: u64 = 256;
pub(crate) const FLAG_REQUIRED: u32 = 1;
/// Room for the header and up to 17 table entries (files written before format 1.1 put
/// METADATA at 256; readers accept any offset).
pub(crate) const METADATA_OFFSET: u64 = 1024;
/// Bits of `required_features` understood by this reader (none in v1).
pub(crate) const KNOWN_FEATURES: u64 = 0;

pub(crate) const fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn fmt_err(msg: impl Into<String>) -> Error {
    Error::new("format", msg)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Section {
    pub kind: u32,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub hash: [u8; 32],
}

impl Section {
    pub(crate) fn encode(&self) -> [u8; ENTRY_BYTES] {
        let mut e = [0u8; ENTRY_BYTES];
        e[0..4].copy_from_slice(&self.kind.to_le_bytes());
        e[4..8].copy_from_slice(&self.flags.to_le_bytes());
        e[8..16].copy_from_slice(&self.offset.to_le_bytes());
        e[16..24].copy_from_slice(&self.length.to_le_bytes());
        e[24..56].copy_from_slice(&self.hash);
        e
    }

    fn decode(e: &[u8]) -> Section {
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&e[24..56]);
        Section { kind: u32_at(e, 0), flags: u32_at(e, 4), offset: u64_at(e, 8), length: u64_at(e, 16), hash }
    }
}

pub(crate) fn encode_header(table_offset: u64, table_count: u64) -> [u8; HEADER_BYTES] {
    let mut h = [0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(&MAGIC);
    h[4..8].copy_from_slice(&VERSION.to_le_bytes());
    h[8..16].copy_from_slice(&0u64.to_le_bytes()); // required_features
    h[16..24].copy_from_slice(&table_offset.to_le_bytes());
    h[24..32].copy_from_slice(&table_count.to_le_bytes());
    h
}

/// Validates dtype/shape and returns the exact storage size, without panicking.
pub(crate) fn expected_nbytes(dtype: DType, shape: &[usize]) -> Result<u64> {
    let mut n: u64 = 1;
    for &d in shape {
        n = n.checked_mul(d as u64).ok_or_else(|| fmt_err(format!("shape {shape:?} overflows")))?;
    }
    if let Some(q) = dtype.quant_format() {
        let block = q.block_elems();
        let cols = shape.last().copied().unwrap_or(0);
        if !(shape.len() == 2 || shape.len() == 3) || cols == 0 || cols % block != 0 {
            bail!(
                "format",
                "quantized tensor ({}) must be 2-D (or a 3-D expert stack) with cols % {block} == 0, got {shape:?}",
                dtype.name()
            );
        }
        if n > (1u64 << 48) {
            bail!("format", "quantized tensor shape {shape:?} too large");
        }
        return Ok(dtype.storage_bytes(shape) as u64);
    }
    n.checked_mul(dtype.size() as u64).ok_or_else(|| fmt_err(format!("shape {shape:?} overflows")))
}

pub(crate) fn open(path: &Path) -> Result<Reader> {
    let data = read_image(path)?;
    from_bytes(data).map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))
}

/// Read-only mmap of the whole file.
#[cfg(not(target_arch = "wasm32"))]
fn read_image(path: &Path) -> Result<Bytes> {
    let file = File::open(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    let size = file.metadata()?.len();
    if size < HEADER_BYTES as u64 {
        bail!("format", "{}: file too small for a .lapid header ({size} B)", path.display());
    }
    // SAFETY: the file is opened read-only; like every mmap-based reader we rely on the
    // file not being truncated while mapped (documented in docs/spec/format.md).
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::new("io", format!("mmap {}: {e}", path.display())))?;
    Ok(Arc::new(mmap))
}

/// No mmap on wasm: the file is read into memory (where the target has a filesystem).
#[cfg(target_arch = "wasm32")]
fn read_image(path: &Path) -> Result<Bytes> {
    let bytes = std::fs::read(path).map_err(|e| Error::new("io", format!("{}: {e}", path.display())))?;
    Ok(Arc::new(bytes))
}

pub(crate) fn from_bytes(data: Bytes) -> Result<Reader> {
    let (metadata, tensors, parsed) = parse((*data).as_ref())?;
    Ok(Reader::from_parts(metadata, tensors, parsed, data))
}

/// Parses and validates a complete .lapid image (everything except tensor hashes).
pub(crate) fn parse(b: &[u8]) -> Result<(serde_json::Value, Vec<TensorInfo>, Parsed)> {
    let size = b.len() as u64;
    if b.len() < HEADER_BYTES || b[0..4] != MAGIC {
        bail!("format", "not a .lapid file (bad magic)");
    }
    let version = u32_at(b, 4);
    if version != VERSION {
        bail!("unsupported_version", "unsupported .lapid version {version} (this reader supports {VERSION})");
    }
    let features = u64_at(b, 8);
    if features & !KNOWN_FEATURES != 0 {
        bail!("unsupported_feature", "unknown required_features bits {:#x}", features & !KNOWN_FEATURES);
    }
    let table_offset = u64_at(b, 16);
    let table_count = u64_at(b, 24);
    if table_count == 0 || table_count > MAX_SECTIONS {
        bail!("format", "section count {table_count} out of range 1..={MAX_SECTIONS}");
    }
    let table_len = table_count * ENTRY_BYTES as u64;
    let table_end = table_offset.checked_add(table_len).ok_or_else(|| fmt_err("table range overflows"))?;
    if table_offset < HEADER_BYTES as u64 || table_end > size {
        bail!("format", "section table [{table_offset}, {table_end}) outside file of {size} B");
    }
    let mut ranges: Vec<(u64, u64, String)> = vec![(0, HEADER_BYTES as u64, "header".into()), (table_offset, table_end, "table".into())];
    let (mut meta, mut index, mut data, mut blob_sec) = (None, None, None, None);
    for i in 0..table_count as usize {
        let o = table_offset as usize + i * ENTRY_BYTES;
        let s = Section::decode(&b[o..o + ENTRY_BYTES]);
        let end = s.offset.checked_add(s.length).ok_or_else(|| fmt_err(format!("section {i} range overflows")))?;
        if end > size || s.offset < HEADER_BYTES as u64 {
            bail!("format", "section {i} (kind {}) [{}, {end}) outside file of {size} B", s.kind, s.offset);
        }
        ranges.push((s.offset, end, format!("section {i} (kind {})", s.kind)));
        let slot = match s.kind {
            SECTION_METADATA => &mut meta,
            SECTION_TENSOR_INDEX => &mut index,
            SECTION_TENSOR_DATA => &mut data,
            SECTION_BLOBS => &mut blob_sec,
            k => {
                if s.flags & FLAG_REQUIRED != 0 {
                    bail!("unsupported_section", "unknown required section kind {k}");
                }
                continue;
            }
        };
        if slot.is_some() {
            bail!("format", "duplicate section kind {}", s.kind);
        }
        *slot = Some(s);
    }
    check_disjoint(&mut ranges)?;
    let meta = meta.ok_or_else(|| fmt_err("missing METADATA section"))?;
    let index = index.ok_or_else(|| fmt_err("missing TENSOR_INDEX section"))?;
    let data = data.ok_or_else(|| fmt_err("missing TENSOR_DATA section"))?;
    if meta.length > MAX_METADATA {
        bail!("limit_exceeded", "metadata section is {} B (limit {MAX_METADATA})", meta.length);
    }
    if index.length > MAX_INDEX {
        bail!("limit_exceeded", "tensor index section is {} B (limit {MAX_INDEX})", index.length);
    }
    let meta_bytes = section_bytes(b, &meta);
    let index_bytes = section_bytes(b, &index);
    check_hash(meta_bytes, &meta.hash, "METADATA")?;
    check_hash(index_bytes, &index.hash, "TENSOR_INDEX")?;
    let metadata: serde_json::Value = serde_json::from_slice(meta_bytes).map_err(|e| fmt_err(format!("metadata JSON: {e}")))?;
    check_metadata(&metadata)?;
    let mut tensors: Vec<TensorInfo> = serde_json::from_slice(index_bytes).map_err(|e| fmt_err(format!("tensor index JSON: {e}")))?;
    check_tensors(&mut tensors, &data)?;
    let (kind, extends) = parse_kind(&metadata)?;
    let components = parse_components(&metadata, kind)?;
    let (blob_list, blobs_offset) = match blob_sec {
        None => (Vec::new(), 0),
        Some(s) => {
            if s.length > MAX_BLOBS_SECTION {
                bail!("limit_exceeded", "BLOBS section is {} B (limit {MAX_BLOBS_SECTION})", s.length);
            }
            let bytes = section_bytes(b, &s);
            if s.length <= blobs::VERIFY_ON_OPEN {
                check_hash(bytes, &s.hash, "BLOBS")?;
            }
            (blobs::parse(bytes)?, s.offset)
        }
    };
    Ok((metadata, tensors, Parsed { blobs: blob_list, blobs_offset, components, kind, extends }))
}

pub(crate) fn is_hex_hash(h: &str) -> bool {
    h.len() == 64 && h.bytes().all(|c| c.is_ascii_hexdigit())
}

fn section_bytes<'a>(b: &'a [u8], s: &Section) -> &'a [u8] {
    &b[s.offset as usize..(s.offset + s.length) as usize]
}

fn check_hash(bytes: &[u8], expect: &[u8; 32], what: &str) -> Result<()> {
    if blake3::hash(bytes).as_bytes() != expect {
        bail!("hash_mismatch", "{what} section hash mismatch (file corrupted)");
    }
    Ok(())
}

/// Non-empty ranges must not overlap. After sorting by start, any overlap implies an
/// overlap between neighbours, so checking adjacent pairs is sufficient.
pub(crate) fn check_disjoint(ranges: &mut Vec<(u64, u64, String)>) -> Result<()> {
    ranges.retain(|r| r.1 > r.0);
    ranges.sort_by_key(|r| (r.0, r.1));
    for w in ranges.windows(2) {
        if w[1].0 < w[0].1 {
            bail!("format", "{} overlaps {}", w[0].2, w[1].2);
        }
    }
    Ok(())
}

/// Block size of int8/int4 tensors written by this version (and the only one supported).
pub(crate) const QUANT_BLOCK_SIZE: u32 = 32;

/// `block_size` written for a quantized dtype: 32 for int8/int4, the super-block (256) for
/// the k-quants.
pub(crate) fn block_size_of(dtype: DType) -> Option<u32> {
    dtype.quant_format().map(|q| if q.is_aos() { q.block_elems() as u32 } else { QUANT_BLOCK_SIZE })
}

/// Quantized tensors carry `block_size` 32 (other sizes: "unsupported"; files written
/// before the attribute existed omit it and mean 32 — normalized here); non-quantized
/// tensors must not carry one. K-quant dtypes fix their block structure: any value (older
/// writers emitted 32) is read as 256.
fn check_block_size(t: &mut TensorInfo) -> Result<()> {
    if t.dtype.quant_format().is_some_and(|q| q.is_aos()) {
        t.block_size = block_size_of(t.dtype);
        return Ok(());
    }
    match (t.dtype.is_quant(), t.block_size) {
        (true, Some(QUANT_BLOCK_SIZE)) | (false, None) => Ok(()),
        (true, None) => {
            t.block_size = Some(QUANT_BLOCK_SIZE);
            Ok(())
        }
        (true, Some(b)) => bail!(
            "unsupported",
            "tensor {:?}: {} block_size {b} is not supported (only {QUANT_BLOCK_SIZE})",
            t.name,
            t.dtype.name()
        ),
        (false, Some(_)) => bail!("format", "tensor {:?}: block_size on non-quantized dtype {}", t.name, t.dtype.name()),
    }
}

fn check_tensors(tensors: &mut [TensorInfo], data: &Section) -> Result<()> {
    if tensors.len() > MAX_TENSORS {
        bail!("limit_exceeded", "{} tensors (limit {MAX_TENSORS})", tensors.len());
    }
    let data_end = data.offset + data.length;
    let mut names = HashSet::with_capacity(tensors.len());
    let mut spans = Vec::with_capacity(tensors.len());
    for t in tensors.iter_mut() {
        check_block_size(t)?;
    }
    for t in tensors.iter() {
        if t.name.is_empty() || t.name.len() > 1024 {
            bail!("format", "invalid tensor name {:?}", t.name);
        }
        if !names.insert(t.name.as_str()) {
            bail!("format", "duplicate tensor {:?}", t.name);
        }
        let want = expected_nbytes(t.dtype, &t.shape).map_err(|e| fmt_err(format!("tensor {:?}: {}", t.name, e.message)))?;
        if t.nbytes != want {
            bail!("format", "tensor {:?}: nbytes {} != {} for {} {:?}", t.name, t.nbytes, want, t.dtype.name(), t.shape);
        }
        if t.offset % TENSOR_ALIGN != 0 {
            bail!("format", "tensor {:?}: offset {} not {TENSOR_ALIGN}-aligned", t.name, t.offset);
        }
        let end = t.offset.checked_add(t.nbytes).ok_or_else(|| fmt_err(format!("tensor {:?} range overflows", t.name)))?;
        if t.offset < data.offset || end > data_end {
            bail!("format", "tensor {:?} [{}, {end}) outside TENSOR_DATA [{}, {data_end})", t.name, t.offset, data.offset);
        }
        if !is_hex_hash(&t.hash) {
            bail!("format", "tensor {:?}: hash is not 64 hex chars", t.name);
        }
        spans.push((t.offset, end, t.name.clone()));
    }
    // Zero-sized tensors may share an offset with anything (check_disjoint skips them).
    check_disjoint(&mut spans)
}

pub(crate) fn verify_tensors(r: &Reader) -> Result<()> {
    for t in &r.tensors {
        let got = crate::write::hash(r.tensor_bytes(t)).to_hex();
        if !got.as_str().eq_ignore_ascii_case(&t.hash) {
            bail!("hash_mismatch", "tensor {:?}: blake3 {} != stored {}", t.name, got.as_str(), t.hash);
        }
    }
    for b in r.blobs() {
        let got = crate::write::hash(r.blob_bytes(b)).to_hex();
        if !got.as_str().eq_ignore_ascii_case(&b.hash) {
            bail!("hash_mismatch", "blob {:?}: blake3 {} != stored {}", b.name, got.as_str(), b.hash);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "imp_tests.rs"]
mod tests;
