//! `.lapid` streaming writer (file order: see `imp`). Memory stays bounded whatever the file
//! size: tensors are written as they are added, blobs given as shared images
//! ([`WriterState::add_blob_shared`], e.g. a source file mmap) are copied in chunks by
//! `finish`; only the tensor / blob indexes are kept. Hashing runs on all cores (not wasm).

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lapid_tensor::{DType, Error, Result, bail};

use crate::blobs::{self, MAX_BLOBS, MAX_BLOBS_SECTION};
use crate::imp::{
    DATA_ALIGN, ENTRY_BYTES, FLAG_REQUIRED, HEADER_BYTES, MAX_INDEX, MAX_METADATA, MAX_TENSORS, METADATA_OFFSET,
    METADATA_PAD, Section, TENSOR_ALIGN, align_up, block_size_of, encode_header, expected_nbytes,
};
use crate::meta::{check_metadata, parse_components, parse_kind};
use crate::{
    Bytes, FORMAT_ID, FORMAT_IDS, SECTION_BLOBS, SECTION_METADATA, SECTION_TENSOR_DATA, SECTION_TENSOR_INDEX, TensorInfo,
};

/// Bytes per copy / hash step of a streamed blob.
const CHUNK: usize = 64 << 20;
/// Inputs at least this large are hashed on all cores.
#[cfg(not(target_arch = "wasm32"))]
const PAR_HASH: usize = 1 << 20;

/// `hasher.update(b)`, multithreaded for large inputs (same result).
pub(crate) fn update(h: &mut blake3::Hasher, b: &[u8]) {
    #[cfg(not(target_arch = "wasm32"))]
    if b.len() >= PAR_HASH {
        h.update_rayon(b);
        return;
    }
    h.update(b);
}

/// blake3 of `b` (multithreaded for large inputs).
pub(crate) fn hash(b: &[u8]) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    for c in b.chunks(CHUNK) {
        update(&mut h, c);
    }
    h.finalize()
}

/// A blob to write at `finish`: `data[range]`.
struct PendingBlob {
    name: String,
    media_type: String,
    data: Bytes,
    range: Range<usize>,
    hash: String,
}

impl PendingBlob {
    fn bytes(&self) -> &[u8] {
        &(*self.data).as_ref()[self.range.clone()]
    }
}

/// Streaming writer state (see module docs for the file order).
pub(crate) struct WriterState {
    path: PathBuf,
    tmp: PathBuf,
    out: Option<BufWriter<File>>,
    pos: u64,
    meta: Section,
    data_start: u64,
    data_hasher: blake3::Hasher,
    tensors: Vec<TensorInfo>,
    names: HashSet<String>,
    blobs: Vec<PendingBlob>,
    blob_names: HashSet<String>,
    blob_bytes: u64,
}

const TABLE_OFFSET: u64 = HEADER_BYTES as u64;

pub(crate) fn partial_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".partial");
    PathBuf::from(s)
}

impl WriterState {
    pub(crate) fn create(path: &Path, mut metadata: serde_json::Value) -> Result<WriterState> {
        // Absent or any accepted id: this writer produces FORMAT_ID files.
        if let Some(obj) = metadata.as_object_mut() {
            let f = obj.get("format").map(|f| f.as_str().filter(|f| FORMAT_IDS.contains(f)).is_some());
            if f != Some(false) {
                obj.insert("format".into(), serde_json::Value::String(FORMAT_ID.into()));
            }
        }
        check_metadata(&metadata)?;
        let (kind, _) = parse_kind(&metadata)?;
        parse_components(&metadata, kind)?;
        let meta_bytes = serde_json::to_vec(&metadata)?;
        if meta_bytes.len() as u64 > MAX_METADATA {
            bail!("limit_exceeded", "metadata is {} B (limit {MAX_METADATA})", meta_bytes.len());
        }
        let tmp = partial_path(path);
        let file = File::create(&tmp).map_err(|e| Error::new("io", format!("create {}: {e}", tmp.display())))?;
        let mut st = WriterState {
            path: path.to_path_buf(),
            tmp,
            out: Some(BufWriter::with_capacity(4 << 20, file)),
            pos: 0,
            meta: Section {
                kind: SECTION_METADATA,
                flags: FLAG_REQUIRED,
                offset: METADATA_OFFSET,
                length: meta_bytes.len() as u64,
                hash: *blake3::hash(&meta_bytes).as_bytes(),
            },
            data_start: 0,
            data_hasher: blake3::Hasher::new(),
            tensors: Vec::new(),
            names: HashSet::new(),
            blobs: Vec::new(),
            blob_names: HashSet::new(),
            blob_bytes: 0,
        };
        // Header + table placeholders (rewritten by finish), then metadata + pad.
        st.write_zeros(METADATA_OFFSET)?;
        st.write_all(&meta_bytes)?;
        let data_start = align_up(st.pos + METADATA_PAD, DATA_ALIGN);
        st.write_zeros(data_start - st.pos)?;
        st.data_start = data_start;
        Ok(st)
    }

    fn writer(&mut self) -> Result<&mut BufWriter<File>> {
        self.out.as_mut().ok_or_else(|| Error::new("io", "writer already finished"))
    }

    fn write_all(&mut self, b: &[u8]) -> Result<()> {
        self.writer()?.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }

    fn write_zeros(&mut self, mut n: u64) -> Result<()> {
        const Z: [u8; 4096] = [0; 4096];
        while n > 0 {
            let k = n.min(Z.len() as u64) as usize;
            self.write_all(&Z[..k])?;
            n -= k as u64;
        }
        Ok(())
    }

    /// Write `bytes`, updating `hashers`, chunk by chunk; on native targets the hashing of a
    /// chunk runs on other threads while it is written.
    fn hash_and_write(&mut self, bytes: &[u8], hashers: &mut [&mut blake3::Hasher]) -> Result<()> {
        for c in bytes.chunks(CHUNK) {
            #[cfg(not(target_arch = "wasm32"))]
            if c.len() >= PAR_HASH {
                let w = std::thread::scope(|s| {
                    for h in hashers.iter_mut() {
                        s.spawn(move || h.update_rayon(c));
                    }
                    self.write_all(c)
                });
                w?;
                continue;
            }
            for h in hashers.iter_mut() {
                h.update(c);
            }
            self.write_all(c)?;
        }
        Ok(())
    }

    pub(crate) fn add(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<()> {
        if name.is_empty() || name.len() > 1024 {
            bail!("invalid_argument", "invalid tensor name {name:?}");
        }
        if self.names.contains(name) {
            bail!("invalid_argument", "duplicate tensor {name:?}");
        }
        if self.tensors.len() >= MAX_TENSORS {
            bail!("limit_exceeded", "too many tensors (limit {MAX_TENSORS})");
        }
        let want = expected_nbytes(dtype, shape).map_err(|e| Error::new("invalid_argument", format!("tensor {name:?}: {}", e.message)))?;
        if bytes.len() as u64 != want {
            bail!("shape_mismatch", "tensor {name:?}: {} bytes given, {} {shape:?} needs {want}", bytes.len(), dtype.name());
        }
        let offset = align_up(self.pos, TENSOR_ALIGN);
        let pad = offset - self.pos;
        if pad > 0 {
            self.data_hasher.update(&[0u8; TENSOR_ALIGN as usize][..pad as usize]);
            self.write_zeros(pad)?;
        }
        let mut th = blake3::Hasher::new();
        let mut dh = std::mem::take(&mut self.data_hasher);
        let r = self.hash_and_write(bytes, &mut [&mut dh, &mut th]);
        self.data_hasher = dh;
        r?;
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            offset,
            nbytes: want,
            hash: th.finalize().to_hex().to_string(),
            block_size: block_size_of(dtype),
        });
        self.names.insert(name.to_string());
        Ok(())
    }

    pub(crate) fn add_blob(&mut self, name: &str, media_type: &str, bytes: &[u8]) -> Result<()> {
        let n = bytes.len();
        self.add_blob_shared(name, media_type, Arc::new(bytes.to_vec()), 0..n)
    }

    pub(crate) fn add_blob_shared(&mut self, name: &str, media_type: &str, data: Bytes, range: Range<usize>) -> Result<()> {
        blobs::check_name(name, "invalid_argument")?;
        blobs::check_media_type(media_type, "invalid_argument")?;
        if self.blob_names.contains(name) {
            bail!("invalid_argument", "duplicate blob {name:?}");
        }
        if self.blobs.len() >= MAX_BLOBS {
            bail!("limit_exceeded", "too many blobs (limit {MAX_BLOBS})");
        }
        let len = (*data).as_ref().len();
        if range.start > range.end || range.end > len {
            bail!("invalid_argument", "blob {name:?}: range {range:?} outside the {len} B image");
        }
        let total = self.blob_bytes + align_up(range.len() as u64, blobs::BLOB_ALIGN);
        if total > MAX_BLOBS_SECTION {
            bail!("limit_exceeded", "blobs exceed {MAX_BLOBS_SECTION} B");
        }
        let hash = hash(&(*data).as_ref()[range.clone()]).to_hex().to_string();
        self.blob_bytes = total;
        self.blob_names.insert(name.to_string());
        self.blobs.push(PendingBlob { name: name.into(), media_type: media_type.into(), data, range, hash });
        Ok(())
    }

    /// BLOBS section at the current (4096-aligned) position: index, then every blob
    /// streamed in chunks. Returns its section entry.
    fn write_blobs(&mut self) -> Result<Section> {
        let list: Vec<(&str, &str, u64, &str)> =
            self.blobs.iter().map(|b| (b.name.as_str(), b.media_type.as_str(), b.range.len() as u64, b.hash.as_str())).collect();
        let (infos, index, total) = blobs::layout(&list)?;
        let start = self.pos;
        let mut h = blake3::Hasher::new();
        self.hash_and_write(&(index.len() as u64).to_le_bytes(), &mut [&mut h])?;
        self.hash_and_write(&index, &mut [&mut h])?;
        let blobs = std::mem::take(&mut self.blobs);
        for (info, b) in infos.iter().zip(&blobs) {
            let pad = start + info.offset - self.pos;
            self.hash_and_write(&vec![0u8; pad as usize], &mut [&mut h])?;
            self.hash_and_write(b.bytes(), &mut [&mut h])?;
        }
        debug_assert_eq!(self.pos - start, total);
        Ok(Section { kind: SECTION_BLOBS, flags: FLAG_REQUIRED, offset: start, length: total, hash: *h.finalize().as_bytes() })
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        // Data section: [data_start, pos). Index follows, 256-aligned (pad not hashed).
        let data = Section {
            kind: SECTION_TENSOR_DATA,
            flags: FLAG_REQUIRED,
            offset: self.data_start,
            length: self.pos - self.data_start,
            hash: *self.data_hasher.finalize().as_bytes(),
        };
        let index_bytes = serde_json::to_vec(&self.tensors)?;
        if index_bytes.len() as u64 > MAX_INDEX {
            bail!("limit_exceeded", "tensor index is {} B (limit {MAX_INDEX})", index_bytes.len());
        }
        let index_offset = align_up(self.pos, TENSOR_ALIGN);
        self.write_zeros(index_offset - self.pos)?;
        self.write_all(&index_bytes)?;
        let index = Section {
            kind: SECTION_TENSOR_INDEX,
            flags: FLAG_REQUIRED,
            offset: index_offset,
            length: index_bytes.len() as u64,
            hash: *blake3::hash(&index_bytes).as_bytes(),
        };
        let mut sections = vec![self.meta.clone(), index, data];
        if !self.blobs.is_empty() {
            let offset = align_up(self.pos, DATA_ALIGN);
            self.write_zeros(offset - self.pos)?;
            sections.push(self.write_blobs()?);
        }
        let mut head = Vec::with_capacity(HEADER_BYTES + sections.len() * ENTRY_BYTES);
        head.extend_from_slice(&encode_header(TABLE_OFFSET, sections.len() as u64));
        for s in &sections {
            head.extend_from_slice(&s.encode());
        }
        debug_assert!(head.len() as u64 <= METADATA_OFFSET);
        let out = self.out.take().ok_or_else(|| Error::new("io", "writer already finished"))?;
        let mut file = out.into_inner().map_err(|e| Error::new("io", e.error().to_string()))?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&head)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&self.tmp, &self.path)
            .map_err(|e| Error::new("io", format!("rename {} -> {}: {e}", self.tmp.display(), self.path.display())))?;
        self.tmp = PathBuf::new();
        Ok(())
    }
}

impl Drop for WriterState {
    fn drop(&mut self) {
        // Unfinished writer: remove the partial file.
        if !self.tmp.as_os_str().is_empty() {
            self.out = None;
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}
