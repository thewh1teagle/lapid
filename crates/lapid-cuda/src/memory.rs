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
}

// SAFETY: the pinned host pointer and events are only used by the owner (behind the
// backend's execution mutex).
unsafe impl Send for Staging {}

/// Bytes per staging half.
const STAGING_HALF: usize = 4 << 20;

impl Staging {
    pub fn new(drv: &Driver) -> Result<Staging> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: allocates page-locked host memory owned by this struct.
        drv.check("cuMemAllocHost", unsafe {
            (drv.cuMemAllocHost_v2)(&mut host, 2 * STAGING_HALF)
        })?;
        let mut events = [std::ptr::null_mut(); 2];
        for e in &mut events {
            // SAFETY: creates an event owned by this struct.
            let r = unsafe { (drv.cuEventCreate)(e, driver::EVENT_DISABLE_TIMING) };
            if r != driver::CUDA_SUCCESS {
                // SAFETY: freeing what we just allocated.
                unsafe { (drv.cuMemFreeHost)(host) };
                return Err(Error::new(
                    "cuda",
                    format!("cuEventCreate: {}", drv.error_text(r)),
                ));
            }
        }
        Ok(Staging {
            host: host as *mut u8,
            half: STAGING_HALF,
            events,
            pending: [false; 2],
        })
    }

    pub fn destroy(&mut self, drv: &Driver) {
        // SAFETY: called once with the stream idle.
        unsafe {
            for e in self.events {
                (drv.cuEventDestroy_v2)(e);
            }
            (drv.cuMemFreeHost)(self.host as *mut c_void);
        }
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

    /// Copy `src` to device address `dst` through the staging halves; waits for the end.
    pub fn upload(
        &mut self,
        drv: &Driver,
        stream: CUstream,
        dst: CUdeviceptr,
        src: &[u8],
    ) -> Result<()> {
        for (i, chunk) in src.chunks(self.half).enumerate() {
            let slot = i % 2;
            self.wait(drv, slot)?;
            // SAFETY: the slot is idle (its last copy completed) and has `half` bytes.
            unsafe {
                let h = self.host.add(slot * self.half);
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), h, chunk.len());
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
        self.wait(drv, 0)?;
        self.wait(drv, 1)
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
