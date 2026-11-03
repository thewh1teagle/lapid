//! Host-memory device buffers for the CPU backend.
//!
//! A `CpuBuffer` is a 64-byte aligned, zero-initialized heap allocation. Kernels read inputs
//! through shared slices and write their output through a raw pointer (`Dst`), i.e. the
//! buffer has interior mutability without a lock.
//!
//! Soundness invariants (upheld by `CpuBackend::execute`):
//! 1. Launches run strictly sequentially; a launch's writes are complete before the next
//!    launch starts (the pool joins all tasks before a kernel returns).
//! 2. Within a launch the output buffer is distinct from every input buffer (checked at
//!    dispatch, error `buffer_aliasing`), so shared input slices never observe writes.
//! 3. Parallel tasks of one kernel write disjoint element ranges of the output.
//! 4. Every view is bounds-checked against the buffer length before any access.
//! 5. `upload` / `download` / `alloc` never run concurrently with `execute` on the same
//!    buffer (the runtime owns that ordering).
//!
//! A *mapped* buffer ([`CpuBuffer::map`]) borrows host bytes instead (e.g. weights in a
//! read-only mmap of a `.lapid` file): it keeps the `HostData` alive, is never written
//! (`dst` refuses it, so no kernel can take it as an output) and is 64-byte aligned like
//! every other buffer (`map` declines anything else).

use std::alloc::{self, Layout};
use std::any::Any;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use lapid_tensor::{DType, DeviceBuffer, Error, HostData, Result};
use half::{bf16, f16};

use crate::typed::{Dst, Src};

pub(crate) const BUFFER_ALIGN: usize = 64;

pub struct CpuBuffer {
    ptr: NonNull<u8>,
    len: usize,
    store: Store,
}

/// Who owns the bytes behind `ptr`.
enum Store {
    /// Our own allocation with this layout; large ones return to the recycler on drop.
    Heap { layout: Layout, recycler: Option<Arc<Recycler>> },
    /// Borrowed, read-only host bytes kept alive by this handle.
    Mapped(#[allow(dead_code)] HostData),
}

/// Allocations from this size up are recycled instead of freed.
const RECYCLE_MIN: usize = 64 << 10;
/// Max bytes kept by a recycler.
const RECYCLE_MAX_BYTES: usize = 1 << 30;

/// Free list of large allocations, binned by capacity (1/8-octave classes).
///
/// Fresh large allocations are mmap'd by the system allocator and page-fault on first
/// touch (~1 ms per 4 MB output); outputs and temps of a decode loop or of repeated
/// transcriptions have recurring sizes, so recycling them keeps the pages mapped.
#[derive(Default)]
pub(crate) struct Recycler {
    inner: Mutex<RecInner>,
}

#[derive(Default)]
struct RecInner {
    bins: HashMap<usize, Vec<RawAlloc>>,
    bytes: usize,
}

struct RawAlloc(NonNull<u8>);
// SAFETY: an owned, unaliased heap allocation.
unsafe impl Send for RawAlloc {}

fn size_class(bytes: usize) -> usize {
    let msb = usize::BITS - 1 - bytes.max(1).leading_zeros();
    let step = 1usize << msb.saturating_sub(3);
    bytes.div_ceil(step) * step
}

impl Recycler {
    fn take(&self, cap: usize) -> Option<NonNull<u8>> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let p = g.bins.get_mut(&cap)?.pop()?;
        g.bytes -= cap;
        Some(p.0)
    }

    /// Keep `ptr` (capacity `cap`) for reuse; false if the recycler is full.
    fn give(&self, ptr: NonNull<u8>, cap: usize) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.bytes + cap > RECYCLE_MAX_BYTES {
            return false;
        }
        g.bytes += cap;
        g.bins.entry(cap).or_default().push(RawAlloc(ptr));
        true
    }
}

impl Drop for Recycler {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        for (cap, ptrs) in g.bins.drain() {
            for p in ptrs {
                // SAFETY: allocated by `CpuBuffer::recycled` with this capacity / alignment.
                unsafe { alloc::dealloc(p.0.as_ptr(), Layout::from_size_align_unchecked(cap, BUFFER_ALIGN)) };
            }
        }
    }
}

// SAFETY: the allocation is owned by the buffer; concurrent access follows the invariants in
// the module documentation (sequential launches, disjoint writes, no input/output aliasing).
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    /// Zero-initialized buffer of `len` bytes (64-byte aligned, capacity rounded up to 64).
    pub fn zeroed(len: usize) -> Result<CpuBuffer> {
        let cap = len.max(1).div_ceil(BUFFER_ALIGN) * BUFFER_ALIGN;
        let layout = Layout::from_size_align(cap, BUFFER_ALIGN)
            .map_err(|e| Error::new("out_of_memory", format!("invalid allocation of {len} bytes: {e}")))?;
        // SAFETY: layout has non-zero size.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error::new("out_of_memory", format!("failed to allocate {len} bytes")))?;
        Ok(CpuBuffer { ptr, len, store: Store::Heap { layout, recycler: None } })
    }

    /// Buffer of `len` bytes with UNSPECIFIED (but initialized) contents: large sizes are
    /// served from / returned to `rec`.
    pub(crate) fn recycled(len: usize, rec: &Arc<Recycler>) -> Result<CpuBuffer> {
        if len < RECYCLE_MIN {
            return CpuBuffer::zeroed(len);
        }
        let cap = size_class(len).div_ceil(BUFFER_ALIGN) * BUFFER_ALIGN;
        let layout = Layout::from_size_align(cap, BUFFER_ALIGN)
            .map_err(|e| Error::new("out_of_memory", format!("invalid allocation of {len} bytes: {e}")))?;
        let ptr = match rec.take(cap) {
            Some(p) => p,
            None => {
                // SAFETY: layout has non-zero size. Zeroed so every byte is initialized.
                let raw = unsafe { alloc::alloc_zeroed(layout) };
                NonNull::new(raw).ok_or_else(|| Error::new("out_of_memory", format!("failed to allocate {len} bytes")))?
            }
        };
        Ok(CpuBuffer { ptr, len, store: Store::Heap { layout, recycler: Some(rec.clone()) } })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<CpuBuffer> {
        let b = CpuBuffer::zeroed(bytes.len())?;
        // SAFETY: fresh allocation of at least bytes.len() bytes, not shared yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.ptr.as_ptr(), bytes.len()) };
        Ok(b)
    }

    /// Zero-copy, read-only view of `host` (e.g. an mmap'd weight). `None` when the bytes
    /// are empty or not 64-byte aligned (kernels assume that alignment): copy them instead.
    pub fn map(host: &HostData) -> Option<CpuBuffer> {
        let bytes = host.bytes();
        if bytes.is_empty() || !(bytes.as_ptr() as usize).is_multiple_of(BUFFER_ALIGN) {
            return None;
        }
        // Only ever read through shared slices (`dst` refuses mapped buffers).
        let ptr = NonNull::new(bytes.as_ptr().cast_mut())?;
        Some(CpuBuffer { ptr, len: bytes.len(), store: Store::Mapped(host.clone()) })
    }

    /// True for a [`CpuBuffer::map`] view (read-only).
    pub fn is_mapped(&self) -> bool {
        matches!(self.store, Store::Mapped(_))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Read-only view of the bytes.
    ///
    /// # Safety
    /// No kernel may write this buffer while the returned slice is alive (invariant 2).
    pub(crate) unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: allocation is valid for len bytes and initialized (zeroed at creation).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Typed read-only view (plain dtypes only).
    ///
    /// # Safety
    /// Same as [`CpuBuffer::bytes`].
    pub(crate) unsafe fn src(&self, dtype: DType) -> Result<Src<'_>> {
        let p = self.ptr.as_ptr();
        // SAFETY: the pointer is 64-byte aligned; element counts are floor(len / size).
        unsafe {
            Ok(match dtype {
                DType::F32 => Src::F32(std::slice::from_raw_parts(p as *const f32, self.len / 4)),
                DType::I32 => Src::I32(std::slice::from_raw_parts(p as *const i32, self.len / 4)),
                DType::F16 => Src::F16(std::slice::from_raw_parts(p as *const f16, self.len / 2)),
                DType::BF16 => Src::BF16(std::slice::from_raw_parts(p as *const bf16, self.len / 2)),
                DType::Int8 | DType::Int4 | DType::Q4K | DType::Q5K | DType::Q6K | DType::Mxfp4 | DType::Iq2Xxs | DType::Iq2S | DType::Iq3Xxs => {
                    return Err(Error::new(
                        "unsupported_dtype",
                        format!("cpu: quantized dtype {} cannot be read elementwise", dtype.name()),
                    ));
                }
            })
        }
    }

    /// Raw output handle (plain dtypes only).
    pub(crate) fn dst(&self, dtype: DType) -> Result<Dst> {
        if self.is_mapped() {
            return Err(Error::new("read_only_buffer", "cpu: a mapped (host-borrowed) buffer cannot be written"));
        }
        if dtype.is_quant() {
            return Err(Error::new(
                "unsupported_dtype",
                format!("cpu: cannot write quantized dtype {}", dtype.name()),
            ));
        }
        Ok(Dst::new(self.ptr.as_ptr(), self.len / dtype.size(), dtype))
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        let Store::Heap { layout, recycler } = &self.store else { return };
        if let Some(r) = recycler
            && r.give(self.ptr, layout.size()) {
                return;
            }
        // SAFETY: allocated in `zeroed` / `recycled` with exactly this layout.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), *layout) };
    }
}

impl DeviceBuffer for CpuBuffer {
    fn backend(&self) -> &'static str {
        "cpu"
    }
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl std::fmt::Debug for CpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CpuBuffer({} bytes)", self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_and_roundtrip() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let b = CpuBuffer::from_bytes(&data).unwrap();
        assert_eq!(b.as_ptr() as usize % BUFFER_ALIGN, 0);
        assert_eq!(unsafe { b.bytes() }, &data[..]);
        let z = CpuBuffer::zeroed(0).unwrap();
        assert!(z.is_empty());
        let z = CpuBuffer::zeroed(130).unwrap();
        assert!(unsafe { z.bytes() }.iter().all(|&x| x == 0));
    }

    #[test]
    fn typed_views() {
        let v: Vec<f32> = vec![1.0, 2.5, -3.0];
        let b = CpuBuffer::from_bytes(bytemuck::cast_slice(&v)).unwrap();
        let s = unsafe { b.src(DType::F32) }.unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(1), 2.5);
        assert!(unsafe { b.src(DType::Int8) }.is_err());
    }

    #[test]
    fn mapped_is_zero_copy_and_read_only() {
        // A 64-aligned backing store (a CpuBuffer), viewed as host data.
        let owner = CpuBuffer::from_bytes(&(0..256u32).map(|i| i as u8).collect::<Vec<_>>()).unwrap();
        struct View(CpuBuffer, usize, usize);
        impl AsRef<[u8]> for View {
            fn as_ref(&self) -> &[u8] {
                let b = unsafe { self.0.bytes() };
                &b[self.1..self.2]
            }
        }
        let host = HostData::new(Arc::new(View(owner, 64, 192)));
        let m = CpuBuffer::map(&host).expect("aligned");
        assert!(m.is_mapped());
        assert_eq!(m.as_ptr() as *const u8, host.bytes().as_ptr());
        assert_eq!(unsafe { m.bytes() }, host.bytes());
        assert_eq!(unsafe { m.src(DType::F32) }.unwrap().len(), 32);
        assert_eq!(m.dst(DType::F32).unwrap_err().code, "read_only_buffer");
        drop(m);
        assert_eq!(host.bytes()[0], 64);
        // Misaligned or empty: declined (the runtime copies instead).
        let owner = CpuBuffer::from_bytes(&[7u8; 128]).unwrap();
        let odd = HostData::new(Arc::new(View(owner, 1, 65)));
        assert!(CpuBuffer::map(&odd).is_none());
        assert!(CpuBuffer::map(&HostData::from_vec(Vec::new())).is_none());
    }
}
