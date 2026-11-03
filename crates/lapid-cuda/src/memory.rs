//! Device memory: `CudaBuffer` (a `DeviceBuffer`), a caching allocator per device, and the
//! pinned staging ring used for uploads and downloads.
//!
//! `cuMemAlloc` / `cuMemFree` are slow and `cuMemFree` synchronizes the device, so freed
//! buffers go back to a per-device free list keyed by size class and are handed out again
//! (every `execute` waits for its work, so a buffer that was dropped is idle).

use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use lapid_tensor::{DeviceBuffer, DeviceData, Error, HostData, Result};

use crate::context::Ctx;
use crate::driver::{self, CUdeviceptr, CUevent, CUstream, Driver};

/// Free bytes kept for reuse per device; beyond this, freed buffers are released.
const MAX_POOLED_BYTES: usize = 1 << 30;

/// Allocation size class: powers of two up to 1 MiB, then multiples of 256 KiB (bounded
/// waste for large weight tensors, exact reuse for recurring activation sizes).
pub(crate) fn size_class(bytes: usize) -> usize {
    if bytes <= 256 {
        256
    } else if bytes <= 1 << 20 {
        bytes.next_power_of_two()
    } else {
        bytes.div_ceil(256 << 10) * (256 << 10)
    }
}

#[derive(Default)]
pub(crate) struct Pool {
    free: HashMap<usize, Vec<CUdeviceptr>>,
    bytes: usize,
}

impl Pool {
    fn take(&mut self, class: usize) -> Option<CUdeviceptr> {
        let p = self.free.get_mut(&class)?.pop()?;
        self.bytes -= class;
        Some(p)
    }

    /// Release every cached allocation (context must be current).
    fn release_all(&mut self, drv: &Driver) {
        for (_, list) in self.free.drain() {
            for p in list {
                // SAFETY: `p` came from cuMemAlloc and is not referenced by any buffer.
                unsafe { (drv.cuMemFree_v2)(p) };
            }
        }
        self.bytes = 0;
    }
}

/// A device allocation. Dropping it returns the memory to the device's pool.
///
/// Or a *mapped* view of host bytes (`CudaBuffer::map`): the device address is the host
/// address (pageable memory access through the host page tables), the `HostData` is kept
/// alive, nothing is ever freed, and kernels never write it (`execute` refuses it as an
/// output).
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    len: usize,
    class: usize,
    ctx: Arc<Ctx>,
    host: Option<HostData>,
}

/// Alignment `cuMemAlloc` guarantees and kernels may assume (vector loads); `.lapid`
/// tensor data is 256-byte aligned in a page-aligned mmap.
const MAP_ALIGN: usize = 256;

impl CudaBuffer {
    pub(crate) fn alloc(ctx: &Arc<Ctx>, len: usize) -> Result<CudaBuffer> {
        if len == 0 {
            return Ok(CudaBuffer {
                ptr: 0,
                len: 0,
                class: 0,
                ctx: ctx.clone(),
                host: None,
            });
        }
        let class = size_class(len);
        let pooled = ctx
            .pool
            .lock()
            .map_err(|_| Error::new("cuda", "memory pool poisoned"))?
            .take(class);
        let ptr = match pooled {
            Some(p) => p,
            None => {
                ctx.bind()?;
                let drv = ctx.drv;
                let mut p: CUdeviceptr = 0;
                // SAFETY: writes one device pointer.
                let r = unsafe { (drv.cuMemAlloc_v2)(&mut p, class) };
                if r != driver::CUDA_SUCCESS {
                    // Possibly out of memory: give the cached blocks back and retry once.
                    if let Ok(mut pool) = ctx.pool.lock() {
                        pool.release_all(drv);
                    }
                    // SAFETY: as above.
                    let r = unsafe { (drv.cuMemAlloc_v2)(&mut p, class) };
                    if r != driver::CUDA_SUCCESS {
                        return Err(Error::new(
                            "out_of_memory",
                            format!("cuMemAlloc({class} bytes): {}", drv.error_text(r)),
                        ));
                    }
                }
                p
            }
        };
        Ok(CudaBuffer {
            ptr,
            len,
            class,
            ctx: ctx.clone(),
            host: None,
        })
    }

    /// Zero-copy view of `host` when the device reads pageable host memory in place and
    /// the bytes are non-empty and 256-byte aligned; `None` otherwise (upload instead).
    pub(crate) fn map(ctx: &Arc<Ctx>, host: &HostData) -> Option<CudaBuffer> {
        let b = host.bytes();
        let addr = b.as_ptr() as usize;
        if !ctx.host_access || b.is_empty() || !addr.is_multiple_of(MAP_ALIGN) {
            return None;
        }
        Some(CudaBuffer { ptr: addr as CUdeviceptr, len: b.len(), class: 0, ctx: ctx.clone(), host: Some(host.clone()) })
    }

    /// True for a zero-copy view of host memory (device address = host address).
    pub fn is_mapped(&self) -> bool {
        self.host.is_some()
    }

    /// The borrowed host bytes of a mapped buffer.
    pub(crate) fn mapped_bytes(&self) -> Option<&[u8]> {
        self.host.as_ref().map(HostData::bytes)
    }

    /// Device address.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn len_bytes(&self) -> usize {
        self.len
    }

    pub(crate) fn same_ctx(&self, ctx: &Arc<Ctx>) -> bool {
        Arc::ptr_eq(&self.ctx, ctx)
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // Zero-sized and mapped buffers own no device memory.
        if self.class == 0 || self.host.is_some() {
            return;
        }
        if let Ok(mut pool) = self.ctx.pool.lock()
            && pool.bytes + self.class <= MAX_POOLED_BYTES
        {
            pool.bytes += self.class;
            pool.free.entry(self.class).or_default().push(self.ptr);
            return;
        }
        if self.ctx.bind().is_ok() {
            // SAFETY: the allocation is idle (every execute waits for its work) and owned.
            unsafe { (self.ctx.drv.cuMemFree_v2)(self.ptr) };
        }
    }
}

impl DeviceBuffer for CudaBuffer {
    fn backend(&self) -> &'static str {
        "cuda"
    }
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Downcast a device buffer to a CUDA buffer of this context.
pub(crate) fn as_cuda<'a>(d: &'a DeviceData, ctx: &Arc<Ctx>) -> Result<&'a CudaBuffer> {
    let b = d.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| {
        Error::new(
            "wrong_backend",
            format!("buffer of backend '{}' passed to cuda", d.backend()),
        )
    })?;
    if !b.same_ctx(ctx) {
        return Err(Error::new(
            "wrong_backend",
            "buffer belongs to another CUDA device",
        ));
    }
    Ok(b)
}

/// Pinned host staging memory split in two halves, so the copy of one chunk overlaps the
/// host memcpy of the next.
pub(crate) struct Staging {
    host: *mut u8,
    half: usize,
    events: [CUevent; 2],
    pending: [bool; 2],
    /// Half the next chunk goes to (alternates across calls, so back-to-back uploads overlap).
    next: usize,
    /// Pinned ring for small uploads (`upload_small`), its bytes in use and the event that
    /// marks the end of its copies when it wraps.
    ring: *mut u8,
    ring_used: usize,
    ring_event: CUevent,
}

// SAFETY: the pinned host pointer and events are only used by the owner (behind the
// backend's execution mutex).
unsafe impl Send for Staging {}

/// Bytes per staging half.
const STAGING_HALF: usize = 16 << 20;
/// Bytes of the small-upload ring.
const RING: usize = 1 << 20;
/// Largest upload through the ring (never waited for: host leaves of a decode step).
pub(crate) const SMALL_UPLOAD: usize = 64 << 10;

/// Host threads of one large staging copy (one memcpy stream reaches ~20 GB/s).
const COPY_THREADS: usize = 4;

/// `src` to `dst` (`src.len()` writable bytes), split over threads when large.
///
/// # Safety
/// `dst` is valid for `src.len()` bytes and does not overlap `src`.
unsafe fn copy_parallel(src: &[u8], dst: *mut u8) {
    let part = src.len().div_ceil(COPY_THREADS).next_multiple_of(4096);
    if src.len() < (2 << 20) || part >= src.len() {
        // SAFETY: caller contract.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst = dst as usize;
    std::thread::scope(|s| {
        for (i, c) in src.chunks(part).enumerate() {
            s.spawn(move || {
                // SAFETY: disjoint sub-ranges of the caller's valid destination.
                unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (dst + i * part) as *mut u8, c.len()) };
            });
        }
    });
}

impl Staging {
    pub fn new(drv: &Driver) -> Result<Staging> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: allocates page-locked host memory owned by this struct.
        drv.check("cuMemAllocHost", unsafe {
            (drv.cuMemAllocHost_v2)(&mut host, 2 * STAGING_HALF + RING)
        })?;
        let mut events = [std::ptr::null_mut(); 3];
        for (i, e) in events.iter_mut().enumerate() {
            // SAFETY: creates an event owned by this struct.
            let r = unsafe { (drv.cuEventCreate)(e, driver::EVENT_DISABLE_TIMING) };
            if r != driver::CUDA_SUCCESS {
                // SAFETY: freeing what we just created / allocated.
                unsafe {
                    for e in &events[..i] {
                        (drv.cuEventDestroy_v2)(*e);
                    }
                    (drv.cuMemFreeHost)(host);
                }
                return Err(Error::new(
                    "cuda",
                    format!("cuEventCreate: {}", drv.error_text(r)),
                ));
            }
        }
        Ok(Staging {
            host: host as *mut u8,
            half: STAGING_HALF,
            events: [events[0], events[1]],
            pending: [false; 2],
            next: 0,
            // SAFETY: the ring is the tail of the allocation.
            ring: unsafe { (host as *mut u8).add(2 * STAGING_HALF) },
            ring_used: 0,
            ring_event: events[2],
        })
    }

    pub fn destroy(&mut self, drv: &Driver) {
        // SAFETY: called once with the stream idle.
        unsafe {
            for e in self.events {
                (drv.cuEventDestroy_v2)(e);
            }
            (drv.cuEventDestroy_v2)(self.ring_event);
            (drv.cuMemFreeHost)(self.host as *mut c_void);
        }
    }

    /// Copy `src` (at most `SMALL_UPLOAD` bytes) to device address `dst` through the ring:
    /// returns once `src` is in pinned memory, the device copy completes in stream order.
    /// Waits only when the ring wraps (for every copy issued from it so far).
    pub fn upload_small(&mut self, drv: &Driver, stream: CUstream, dst: CUdeviceptr, src: &[u8]) -> Result<()> {
        debug_assert!(src.len() <= SMALL_UPLOAD);
        let len = src.len().next_multiple_of(256);
        if self.ring_used + len > RING {
            // SAFETY: event + stream owned by us.
            unsafe {
                drv.check("cuEventRecord", (drv.cuEventRecord)(self.ring_event, stream))?;
                drv.check("cuEventSynchronize", (drv.cuEventSynchronize)(self.ring_event))?;
            }
            self.ring_used = 0;
        }
        // SAFETY: bytes ring_used.. of the ring are not read by any pending copy (the ring
        // was drained when it last wrapped) and `len <= RING - ring_used`.
        unsafe {
            let h = self.ring.add(self.ring_used);
            std::ptr::copy_nonoverlapping(src.as_ptr(), h, src.len());
            drv.check(
                "cuMemcpyHtoDAsync",
                (drv.cuMemcpyHtoDAsync_v2)(dst, h as *const c_void, src.len(), stream),
            )?;
        }
        self.ring_used += len;
        Ok(())
    }

    fn wait(&mut self, drv: &Driver, slot: usize) -> Result<()> {
        if self.pending[slot] {
            // SAFETY: event recorded on our stream.
            drv.check("cuEventSynchronize", unsafe {
                (drv.cuEventSynchronize)(self.events[slot])
            })?;
            self.pending[slot] = false;
        }
        Ok(())
    }

    /// Copy `src` to device address `dst` through the staging halves. With `wait`, returns
    /// when the copy is complete; else when `src` is in staging memory (the device side
    /// completes in stream order, before any later launch or copy on `stream`).
    pub fn upload(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        dst: CUdeviceptr,
        src: &[u8],
        wait: bool,
    ) -> Result<()> {
        for (i, chunk) in src.chunks(self.half).enumerate() {
            let slot = self.next;
            self.next ^= 1;
            self.wait(drv, slot)?;
            // SAFETY: the slot is idle (its last copy completed) and has `half` bytes.
            unsafe {
                let h = self.host.add(slot * self.half);
                copy_parallel(chunk, h);
                let d = dst + (i * self.half) as u64;
                drv.check(
                    "cuMemcpyHtoDAsync",
                    (drv.cuMemcpyHtoDAsync_v2)(d, h as *const c_void, chunk.len(), stream),
                )?;
                drv.check(
                    "cuEventRecord",
                    (drv.cuEventRecord)(self.events[slot], stream),
                )?;
            }
            self.pending[slot] = true;
        }
        if wait {
            self.wait(drv, 0)?;
            self.wait(drv, 1)?;
        }
        Ok(())
    }

    /// Copy `out.len()` bytes from device address `src` into `out`; waits for the end.
    pub fn download(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        src: CUdeviceptr,
        out: &mut [u8],
    ) -> Result<()> {
        let total = out.len();
        let n = total.div_ceil(self.half);
        if n == 0 {
            return Ok(());
        }
        self.issue_download(drv, stream, src, total, 0)?;
        for i in 0..n {
            if i + 1 < n {
                self.issue_download(drv, stream, src, total, i + 1)?;
            }
            let slot = i % 2;
            self.wait(drv, slot)?;
            let off = i * self.half;
            let len = (out.len() - off).min(self.half);
            // SAFETY: the copy into this slot completed (event waited).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.host.add(slot * self.half),
                    out.as_mut_ptr().add(off),
                    len,
                )
            };
        }
        Ok(())
    }

    fn issue_download(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        src: CUdeviceptr,
        total: usize,
        i: usize,
    ) -> Result<()> {
        let slot = i % 2;
        let off = i * self.half;
        let len = (total - off).min(self.half);
        // SAFETY: the slot was drained by the host before being reused.
        unsafe {
            let h = self.host.add(slot * self.half);
            drv.check(
                "cuMemcpyDtoHAsync",
                (drv.cuMemcpyDtoHAsync_v2)(h as *mut c_void, src + off as u64, len, stream),
            )?;
            drv.check(
                "cuEventRecord",
                (drv.cuEventRecord)(self.events[slot], stream),
            )?;
        }
        self.pending[slot] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::size_class;

    #[test]
    fn classes() {
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(300), 512);
        assert_eq!(size_class(1 << 20), 1 << 20);
        assert_eq!(size_class((1 << 20) + 1), (1 << 20) + (256 << 10));
        for n in [5usize, 999, 123_457, 18_432_000] {
            assert!(size_class(n) >= n);
        }
    }
}
