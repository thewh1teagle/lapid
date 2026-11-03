//! Loading `.lapid` files as tensors. CONTRACT signatures — implementation by agent "runtime".

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};

use lapid_format::{Bytes, Reader};
use lapid_tensor::{Error, HostData, Op, Result, Tensor, bail};

use crate::residency::{self, Advice, Residency};
use crate::runtime::lock;

/// An opened .lapid file. Tensors are `Op::Host` leaves backed by the file image (an mmap
/// for [`LapidFile::open`], caller memory for [`LapidFile::from_bytes`]): zero copy.
/// Blobs (format 1.1 non-tensor assets) are read through [`LapidFile::blob`].
pub struct LapidFile {
    pub metadata: serde_json::Value,
    pub tensors: BTreeMap<String, Tensor>,
    /// Keeps the blob index, components and kind (its `metadata` was moved out).
    reader: Reader,
    /// The whole file image (for page-cache hints).
    image: Bytes,
    /// `Auto` decision made at open: stream the weights from the mapping.
    streams: bool,
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

/// A live .lapid image. Host data inside it is model weights, which the runtime either
/// keeps pinned on the device for its lifetime or reads in place (see `residency`).
struct Image {
    start: usize,
    end: usize,
    data: WeakImage,
    /// `Auto` chose to stream this file (decided at open).
    streams: bool,
    /// Per-file policy ([`LapidFile::set_residency`]); wins over the process-wide one.
    forced: Option<Residency>,
}

static FILE_RANGES: Mutex<Vec<Image>> = Mutex::new(Vec::new());

/// Register a file image; returns the `Auto` decision for it. Only real file mappings
/// (`mapped`) may stream: caller memory is resident anyway.
fn register_image(data: &Bytes, weights: u64, mapped: bool) -> bool {
    let bytes = (**data).as_ref();
    let start = bytes.as_ptr() as usize;
    let mut ranges = lock(&FILE_RANGES);
    ranges.retain(|i| i.data.strong_count() > 0);
    let streams = mapped && residency::residency() == Residency::Auto && {
        let resident: u64 = ranges.iter().filter(|i| !i.streams).map(|i| (i.end - i.start) as u64).sum();
        residency::auto_streams(weights, resident)
    };
    ranges.push(Image { start, end: start + bytes.len(), data: Arc::downgrade(data), streams, forced: None });
    streams
}

/// Effective residency (`Upload` or `Mmap`) of host data whose `(ptr, len)` key lies inside
/// a live .lapid image; `None` for other host data.
pub(crate) fn weight_residency(key: (usize, usize)) -> Option<Residency> {
    let (ptr, len) = key;
    let (streams, forced) = lock(&FILE_RANGES)
        .iter()
        .find(|i| ptr >= i.start && ptr.saturating_add(len) <= i.end && i.data.strong_count() > 0)
        .map(|i| (i.streams, i.forced))?;
    Some(effective(forced, streams))
}

/// `Upload` or `Mmap` for a file with this override and `Auto` decision.
fn effective(forced: Option<Residency>, streams: bool) -> Residency {
    match forced.unwrap_or_else(residency::residency) {
        Residency::Auto if streams => Residency::Mmap,
        Residency::Auto => Residency::Upload,
        r => r,
    }
}

impl LapidFile {
    /// Open a file (read-only mmap; read into memory on wasm).
    pub fn open(path: impl AsRef<Path>) -> Result<LapidFile> {
        let path = path.as_ref();
        let reader = Reader::open(path)?;
        let f = LapidFile::from_reader(reader, cfg!(not(target_arch = "wasm32")))
            .map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))?;
        if f.residency() == Residency::Mmap {
            // 2 MiB page-cache folios: ~10 GB/s cold reads instead of ~4, and mapping /
            // locking the pages around GPU evals gets far cheaper.
            f.advise(Advice::HugePage);
        }
        if f.streams {
            let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
            crate::log::emit(&format!(
                "[lapid] {}: {:.1} GiB of weights, {:.1} GiB available: weights do not fit, reading them from the file on demand (mmap)",
                path.display(),
                gb(f.weight_bytes()),
                gb(residency::available_memory().unwrap_or(0)),
            ));
        }
        Ok(f)
    }

    /// A complete .lapid image in memory (e.g. `Arc::new(vec)`), zero copy: tensors are
    /// slices of `bytes`, which stay alive as long as any tensor does.
    pub fn from_bytes(bytes: Bytes) -> Result<LapidFile> {
        LapidFile::from_reader(Reader::from_bytes(bytes)?, false)
    }

    /// [`LapidFile::from_bytes`] for a static image, e.g. `include_bytes!("model.lapid")`.
    pub fn from_static(bytes: &'static [u8]) -> Result<LapidFile> {
        LapidFile::from_reader(Reader::from_static(bytes)?, false)
    }

    fn from_reader(mut reader: Reader, mapped: bool) -> Result<LapidFile> {
        let data = reader.data();
        let image_len = (*data).as_ref().len();
        let weights: u64 = reader.tensors.iter().map(|t| t.nbytes).sum();

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
        let streams = register_image(&data, weights, mapped);
        let metadata = std::mem::take(&mut reader.metadata);
        Ok(LapidFile { metadata, tensors, reader, image: data, streams })
    }

    /// Total bytes of tensor data.
    pub fn weight_bytes(&self) -> u64 {
        self.reader.tensors.iter().map(|t| t.nbytes).sum()
    }

    /// Where this file's weights live at eval under the current policy: `Upload` (copied to
    /// the device once) or `Mmap` (read in place from the file mapping).
    pub fn residency(&self) -> Residency {
        effective(self.forced(), self.streams)
    }

    /// The image registry entry of this file.
    fn with_image<T>(&self, f: impl FnOnce(&mut Image) -> T) -> Option<T> {
        let start = (*self.image).as_ref().as_ptr() as usize;
        lock(&FILE_RANGES).iter_mut().find(|i| i.start == start && i.data.strong_count() > 0).map(f)
    }

    fn forced(&self) -> Option<Residency> {
        self.with_image(|i| i.forced).flatten()
    }

    /// Residency of this file's weights regardless of the process-wide policy (`Auto`
    /// returns to it), for models that manage part of their weights themselves: e.g. a
    /// model whose MoE experts stream through its own device cache keeps the dense
    /// weights uploaded (`Upload`) although the file is far larger than memory. Applies
    /// to weights first used after the call (call it before loading).
    pub fn set_residency(&self, r: Residency) {
        self.with_image(|i| i.forced = (r != Residency::Auto).then_some(r));
    }

    /// Page-cache hint for the whole file image (e.g. `Advice::Random` for weights read
    /// scattered, like MoE experts). See [`crate::advise`] for one tensor.
    pub fn advise(&self, advice: Advice) {
        let _ = residency::advise_host(&HostData::new(self.image.clone()), None, advice);
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

    pub(super) fn sample_image_pub(tag: &str) -> Vec<u8> {
        sample_image(tag)
    }

    #[test]
    fn from_bytes_runs_on_cpu_zero_copy() {
        let image: Bytes = Arc::new(sample_image("bytes"));
        let f = LapidFile::from_bytes(image.clone()).expect("from_bytes");
        let x = f.get("x").expect("x");
        let bytes = f.bytes("x").expect("bytes");
        let base = (*image).as_ref().as_ptr() as usize;
        assert!(bytes.as_ptr() as usize >= base && (bytes.as_ptr() as usize) < base + (*image).as_ref().len());
        assert_eq!(weight_residency((bytes.as_ptr() as usize, bytes.len())).map(|_| ()), Some(()));
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

#[cfg(test)]
mod residency_tests {
    use super::*;

    #[test]
    fn per_file_residency_and_device_write() {
        let image: Bytes = Arc::new(super::tests::sample_image_pub("res"));
        let f = LapidFile::from_bytes(image).expect("from_bytes");
        let key = |f: &LapidFile| {
            let b = f.bytes("x").expect("x");
            (b.as_ptr() as usize, b.len())
        };
        f.set_residency(Residency::Mmap);
        assert_eq!(f.residency(), Residency::Mmap);
        assert_eq!(weight_residency(key(&f)), Some(Residency::Mmap));
        f.set_residency(Residency::Upload);
        assert_eq!(weight_residency(key(&f)), Some(Residency::Upload));
        f.set_residency(Residency::Auto);
        assert_eq!(f.residency(), effective(None, false));

        let dev = crate::Device::cpu();
        let t = dev.alloc(&[2, 3], lapid_tensor::DType::F32).expect("alloc");
        let vals: Vec<f32> = (0..6).map(|i| i as f32).collect();
        dev.write(&t, 0, bytemuck::cast_slice(&vals)).expect("write");
        dev.write(&t, 4, &7f32.to_le_bytes()).expect("write one");
        assert_eq!(dev.to_vec_f32(&t).expect("read"), vec![0.0, 7.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(dev.write(&t, 21, &[0u8; 4]).is_err());
        assert!(dev.write(&f.get("x").expect("x"), 0, &[0u8; 4]).is_err());
    }
}
