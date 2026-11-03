//! Loading `.lapid` files as tensors. CONTRACT signatures — implementation by agent "runtime".

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};

use lapid_format::{Bytes, Reader};
use lapid_tensor::{Error, HostData, Op, Result, Tensor, bail};

use crate::runtime::lock;

/// An opened .lapid file. Tensors are `Op::Host` leaves backed by the file image (an mmap
/// for [`LapidFile::open`], caller memory for [`LapidFile::from_bytes`]): zero copy.
/// Blobs (format 1.1 non-tensor assets) are read through [`LapidFile::blob`].
pub struct LapidFile {
    pub metadata: serde_json::Value,
    pub tensors: BTreeMap<String, Tensor>,
    /// Keeps the blob index, components and kind (its `metadata` was moved out).
    reader: Reader,
}

/// A byte range of a shared file image; the bytes of one tensor. Zero copy.
struct ImageSlice {
    data: Bytes,
    start: usize,
    end: usize,
}

impl AsRef<[u8]> for ImageSlice {
    fn as_ref(&self) -> &[u8] {
        // In range: checked against the image length on construction.
        &(*self.data).as_ref()[self.start..self.end]
    }
}

type WeakImage = Weak<dyn AsRef<[u8]> + Send + Sync>;

/// Address ranges of live .lapid images. Host data inside them is model weights, which the
/// upload cache keeps pinned for the device's lifetime (see `runtime::cache`).
static FILE_RANGES: Mutex<Vec<(usize, usize, WeakImage)>> = Mutex::new(Vec::new());

fn register_image(data: &Bytes) {
    let bytes = (**data).as_ref();
    let start = bytes.as_ptr() as usize;
    let mut ranges = lock(&FILE_RANGES);
    ranges.retain(|(_, _, w)| w.strong_count() > 0);
    ranges.push((start, start + bytes.len(), Arc::downgrade(data)));
}

/// True when the `(ptr, len)` key of a `HostData` lies inside a live .lapid image.
pub(crate) fn is_file_backed(key: (usize, usize)) -> bool {
    let (ptr, len) = key;
    lock(&FILE_RANGES)
        .iter()
        .any(|(s, e, w)| ptr >= *s && ptr.saturating_add(len) <= *e && w.strong_count() > 0)
}

impl LapidFile {
    /// Open a file (read-only mmap; read into memory on wasm).
    pub fn open(path: impl AsRef<Path>) -> Result<LapidFile> {
        let path = path.as_ref();
        let reader = Reader::open(path)?;
        LapidFile::from_reader(reader)
            .map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))
    }

    /// A complete .lapid image in memory (e.g. `Arc::new(vec)`), zero copy: tensors are
    /// slices of `bytes`, which stay alive as long as any tensor does.
    pub fn from_bytes(bytes: Bytes) -> Result<LapidFile> {
        LapidFile::from_reader(Reader::from_bytes(bytes)?)
    }

    /// [`LapidFile::from_bytes`] for a static image, e.g. `include_bytes!("model.lapid")`.
    pub fn from_static(bytes: &'static [u8]) -> Result<LapidFile> {
        LapidFile::from_reader(Reader::from_static(bytes)?)
    }

    fn from_reader(mut reader: Reader) -> Result<LapidFile> {
        let data = reader.data();
        register_image(&data);
        let image_len = (*data).as_ref().len();

        let mut tensors = BTreeMap::new();
        for info in &reader.tensors {
            let bad_range =
                || Error::new("format", format!("tensor '{}' lies outside the file", info.name));
            let start = usize::try_from(info.offset).map_err(|_| bad_range())?;
            let len = usize::try_from(info.nbytes).map_err(|_| bad_range())?;
            let end = start.checked_add(len).ok_or_else(bad_range)?;
            if end > image_len {
                return Err(bad_range());
            }
            let host = HostData::new(Arc::new(ImageSlice { data: data.clone(), start, end }));
            let t = Tensor::from_host(host, &info.shape, info.dtype)
                .map_err(|e| Error::new("format", format!("tensor '{}': {}", info.name, e.message)))?;
            if tensors.insert(info.name.clone(), t).is_some() {
                bail!("format", "duplicate tensor name '{}'", info.name);
            }
        }
        let metadata = std::mem::take(&mut reader.metadata);
        Ok(LapidFile { metadata, tensors, reader })
    }

    /// Bytes of a named blob (e.g. `"tokenizer/tokens.json"`), zero copy. `None` if absent.
    pub fn blob(&self, name: &str) -> Option<&[u8]> {
        self.reader.blob(name)
    }

    /// A named blob as host data sharing the file image (zero copy; keeps the image alive),
    /// for blobs a model reads after loading, e.g. lookup tables gathered on the host.
    pub fn blob_data(&self, name: &str) -> Option<HostData> {
        let bytes = self.reader.blob(name)?;
        let data = self.reader.data();
        let start = bytes.as_ptr() as usize - (*data).as_ref().as_ptr() as usize;
        Some(HostData::new(Arc::new(ImageSlice { data, start, end: start + bytes.len() })))
    }

    /// The blob index (empty for files without a BLOBS section).
    pub fn blobs(&self) -> &[lapid_format::BlobInfo] {
        self.reader.blobs()
    }

    /// Declared components, or the single implicit one (see `lapid_format::Reader::components`).
    pub fn components(&self) -> &[lapid_format::Component] {
        self.reader.components()
    }

    /// `Some` for extras files (`"kind": "extra"`).
    pub fn extends(&self) -> Option<&lapid_format::Extends> {
        self.reader.extends()
    }

    /// Error code "missing_tensor" if absent.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::new("missing_tensor", format!("tensor '{name}' not found in model file")))
    }

    /// Raw bytes of a tensor that is stored but not meant to be a graph input
    /// (e.g. tokenizer blobs). Error "missing_tensor".
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        match self.tensors.get(name).map(Tensor::op) {
            Some(Op::Host(h)) => Ok(h.bytes()),
            Some(_) => bail!("internal", "tensor '{name}' is not a host leaf"),
            None => bail!("missing_tensor", "tensor '{name}' not found in model file"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_tensor::DType;

    fn sample_image(tag: &str) -> Vec<u8> {
        let path = std::env::temp_dir().join(format!("lapid-file-test-{}-{tag}.lapid", std::process::id()));
        let meta = serde_json::json!({
            "name": "t", "architecture": "t", "dtype": "fp32", "license": "unknown",
            "source": {"repo": "local/t"}, "converter": "test"
        });
        let mut w = lapid_format::Writer::create(&path, meta).expect("create");
        let vals: Vec<f32> = (0..6).map(|i| i as f32).collect();
        w.add("x", DType::F32, &[2, 3], bytemuck::cast_slice(&vals)).expect("add");
        w.add_blob("note.txt", "text/plain", b"hi").expect("blob");
        w.finish().expect("finish");
        let bytes = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        bytes
    }

    #[test]
    fn from_bytes_runs_on_cpu_zero_copy() {
        let image: Bytes = Arc::new(sample_image("bytes"));
        let f = LapidFile::from_bytes(image.clone()).expect("from_bytes");
        let x = f.get("x").expect("x");
        let bytes = f.bytes("x").expect("bytes");
        let base = (*image).as_ref().as_ptr() as usize;
        assert!(bytes.as_ptr() as usize >= base && (bytes.as_ptr() as usize) < base + (*image).as_ref().len());
        assert!(is_file_backed((bytes.as_ptr() as usize, bytes.len())));
        assert_eq!(f.blob("note.txt"), Some(&b"hi"[..]));
        let dev = crate::Device::cpu();
        let y = x.mul_scalar(2.0).expect("mul");
        assert_eq!(dev.to_vec_f32(&y).expect("eval"), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn from_static_image() {
        let image: &'static [u8] = Box::leak(sample_image("static").into_boxed_slice());
        let f = LapidFile::from_static(image).expect("from_static");
        assert_eq!(f.get("x").expect("x").shape(), &[2, 3]);
        assert!(LapidFile::from_bytes(Arc::new(vec![1u8; 8])).is_err());
    }
}
