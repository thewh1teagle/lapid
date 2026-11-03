//! `.lapid` model file format v1 (metadata `"format"`: `"lapid/1"` or `"lapid/1.1"`).
//! CONTRACT: pub items/signatures are coordinator-owned. Full byte layout: docs/spec/format.md.
//!
//! ```text
//! [header 64 B]  magic "GELU" | version u32 = 1 | required_features u64 | table_offset u64
//!                | table_count u64 | reserved (zero) to 64 B           (all little endian)
//! [section table] table_count entries of 56 B:
//!                kind u32 | flags u32 (bit0 = required) | offset u64 | length u64 | blake3 [32]
//! [sections]     kind 1 = METADATA     : UTF-8 JSON object (free-form + required keys below)
//!                kind 2 = TENSOR_INDEX : UTF-8 JSON array of TensorInfo
//!                kind 3 = TENSOR_DATA  : tensor bytes, each tensor 256-B aligned relative to
//!                                        file start (section itself starts 4096-aligned)
//!                kind 4 = BLOBS (1.1)  : u64 index length | JSON array of BlobInfo | blob
//!                                        bytes, 256-B aligned relative to the section start
//!                METADATA is followed by 64 KiB of zero padding (in-place metadata edits).
//! ```
//! Required metadata keys: "format" ("lapid/1" or "lapid/1.1"), "name", "architecture",
//! "dtype" (main weight dtype, e.g. "int8"), "license" (SPDX id or "unknown"), "source"
//! (object: {"repo", "revision"?, "file_hash"?}), "converter" ("lapid-convert <version>").
//! Format 1.1 adds optional "components" (named sub-models with tensor prefixes),
//! "kind" ("model" | "extra") and "extends" (for extras).
//! Readers ignore unknown optional sections and refuse unknown required ones / unknown
//! required_features bits. Limits: metadata ≤ 16 MiB, index ≤ 64 MiB, ≤ 64 sections,
//! ≤ 1_000_000 tensors, BLOBS ≤ 4 TiB / 65 536 blobs (section hash checked on open up to
//! 256 MiB), ≤ 64 components, every range inside
//! the file and non-overlapping.
//!
//! A [`Reader`] is backed by one shared byte image ([`Bytes`]): a read-only mmap for
//! [`Reader::open`] (a heap copy on wasm, which has no mmap), or caller-provided memory for
//! [`Reader::from_bytes`] / [`Reader::from_static`] (`include_bytes!`). Tensor and blob
//! bytes are slices of that image (zero copy).

use std::path::Path;
use std::sync::Arc;

use lapid_tensor::{DType, Result};
use serde::{Deserialize, Serialize};

mod blobs;
pub mod cmd;
mod imp;
mod meta;
mod write;

pub const MAGIC: [u8; 4] = *b"LPID";
pub const VERSION: u32 = 1;
pub const SECTION_METADATA: u32 = 1;
pub const SECTION_TENSOR_INDEX: u32 = 2;
pub const SECTION_TENSOR_DATA: u32 = 3;
/// Named byte blobs (format 1.1).
pub const SECTION_BLOBS: u32 = 4;
/// Metadata `"format"` written by this version.
pub const FORMAT_ID: &str = "lapid/1.1";
/// Metadata `"format"` values this reader accepts.
pub const FORMAT_IDS: [&str; 2] = ["lapid/1", "lapid/1.1"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file (256-aligned).
    pub offset: u64,
    pub nbytes: u64,
    /// blake3 hex of the tensor bytes.
    pub hash: String,
    /// Quantization block (group) size: elements per scale along the last axis.
    /// Present for quantized dtypes only; v1 supports 32.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<u32>,
}

/// One entry of the BLOBS index: a named byte blob (non-tensor asset such as a tokenizer).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BlobInfo {
    pub name: String,
    /// IANA media type, e.g. `"application/json"`, `"application/octet-stream"`.
    pub media_type: String,
    /// Byte offset relative to the start of the BLOBS section (256-aligned).
    pub offset: u64,
    pub nbytes: u64,
    /// blake3 hex of the blob bytes.
    pub hash: String,
}

/// A named sub-model of a file (format 1.1 metadata `"components"`). Files without the
/// key have one implicit component named after `architecture` with prefix `""`.
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    pub name: String,
    pub architecture: String,
    /// The component's configuration object (inline, or resolved from the top-level
    /// metadata key the file names).
    pub config: serde_json::Value,
    /// Per-component license (SPDX id); `None` = the file's `license`.
    pub license: Option<String>,
    /// Prefix of this component's tensor names (`""` or ending in `.`, e.g. `"codec."`).
    pub tensor_prefix: String,
}

/// What a file is: a loadable model, or an extra that extends a component of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    Model,
    Extra,
}

/// `"extends"` of an extras file: the architecture and component it plugs into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extends {
    pub architecture: String,
    pub component: String,
}

/// A shared, immutable byte image of a whole `.lapid` file (an mmap, a `Vec<u8>`, a
/// `&'static [u8]`, ...). `as_ref()` must return the same slice on every call.
pub type Bytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// Read-only reader over one byte image (see [`Bytes`]). Cheap to share the image via
/// [`Reader::data`].
pub struct Reader {
    pub metadata: serde_json::Value,
    pub tensors: Vec<TensorInfo>,
    parsed: Parsed,
    data: Bytes,
}

/// Everything validated on open besides metadata JSON and the tensor index.
#[derive(Clone, Debug)]
pub(crate) struct Parsed {
    pub blobs: Vec<BlobInfo>,
    /// Absolute offset of the BLOBS section (0 when absent).
    pub blobs_offset: u64,
    pub components: Vec<Component>,
    pub kind: FileKind,
    pub extends: Option<Extends>,
}

impl Reader {
    /// Validates header, table, limits and section hashes of METADATA/TENSOR_INDEX
    /// (tensor data hashes are checked by `verify_tensors`, which reads everything).
    pub fn open(path: impl AsRef<Path>) -> Result<Reader> {
        imp::open(path.as_ref())
    }

    /// Parse a complete `.lapid` image held in memory (zero copy: tensors and blobs are
    /// slices of `bytes`). Same validation as [`Reader::open`].
    pub fn from_bytes(bytes: Bytes) -> Result<Reader> {
        imp::from_bytes(bytes)
    }

    /// [`Reader::from_bytes`] for a static image, e.g. `include_bytes!("model.lapid")`.
    pub fn from_static(bytes: &'static [u8]) -> Result<Reader> {
        imp::from_bytes(Arc::new(bytes))
    }

    /// The shared byte image the reader is backed by.
    pub fn data(&self) -> Bytes {
        self.data.clone()
    }

    /// The whole file image.
    pub fn as_bytes(&self) -> &[u8] {
        (*self.data).as_ref()
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.as_bytes()[info.offset as usize..(info.offset + info.nbytes) as usize]
    }

    /// Check every tensor's (and blob's) blake3. Error code "hash_mismatch" naming it.
    pub fn verify_tensors(&self) -> Result<()> {
        imp::verify_tensors(self)
    }

    /// Index of the BLOBS section (empty when the file has none).
    pub fn blobs(&self) -> &[BlobInfo] {
        &self.parsed.blobs
    }

    /// Bytes of the blob `name` (zero copy).
    pub fn blob(&self, name: &str) -> Option<&[u8]> {
        self.parsed.blobs.iter().find(|b| b.name == name).map(|b| self.blob_bytes(b))
    }

    pub fn blob_bytes(&self, info: &BlobInfo) -> &[u8] {
        let start = (self.parsed.blobs_offset + info.offset) as usize;
        &self.as_bytes()[start..start + info.nbytes as usize]
    }

    /// Declared components, or the single implicit one (model files without
    /// `"components"`). Extras files without `"components"` have none.
    pub fn components(&self) -> &[Component] {
        &self.parsed.components
    }

    pub fn kind(&self) -> FileKind {
        self.parsed.kind
    }

    /// `Some` for extras files (`"kind": "extra"`).
    pub fn extends(&self) -> Option<&Extends> {
        self.parsed.extends.as_ref()
    }

    pub(crate) fn from_parts(
        metadata: serde_json::Value,
        tensors: Vec<TensorInfo>,
        parsed: Parsed,
        data: Bytes,
    ) -> Reader {
        Reader { metadata, tensors, parsed, data }
    }
}

/// Streaming writer: tensors are written to disk as they are added (no full-model buffering).
pub struct Writer {
    inner: write::WriterState,
}

impl Writer {
    pub fn create(path: impl AsRef<Path>, metadata: serde_json::Value) -> Result<Writer> {
        Ok(Writer { inner: write::WriterState::create(path.as_ref(), metadata)? })
    }

    pub fn add(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<()> {
        self.inner.add(name, dtype, shape, bytes)
    }

    /// Add a named byte blob (BLOBS section), copied into memory until `finish` (small
    /// assets; see [`add_blob_shared`](Self::add_blob_shared) for large ones). Errors:
    /// "invalid_argument" (bad/duplicate name or media type), "limit_exceeded".
    pub fn add_blob(&mut self, name: &str, media_type: &str, bytes: &[u8]) -> Result<()> {
        self.inner.add_blob(name, media_type, bytes)
    }

    /// Add the blob `data[range]` of a shared image (e.g. a source file mmap) without
    /// copying it: hashed now, streamed into the file in chunks by `finish` (so `data`
    /// must stay unchanged until then). Errors as [`add_blob`](Self::add_blob), plus
    /// "invalid_argument" for a range outside the image.
    pub fn add_blob_shared(&mut self, name: &str, media_type: &str, data: Bytes, range: std::ops::Range<usize>) -> Result<()> {
        self.inner.add_blob_shared(name, media_type, data, range)
    }

    /// Writes index + blobs + table + header. Validates required metadata keys first
    /// (error "missing_metadata").
    pub fn finish(self) -> Result<()> {
        self.inner.finish()
    }
}
