//! Devices, capabilities and the per-device context.
//!
//! Every device uses its primary context, retained once per process and never released
//! (contexts are shared by all backends and buffers of the device; tearing them down while
//! another thread works on the device is never worth the risk). Each public entry point
//! makes the context current on the calling thread first (`Ctx::bind`, a cheap
//! thread-local switch).

use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::sync::{Arc, Mutex, OnceLock};

use lapid_tensor::{Error, Result};

use crate::driver::{self, CUcontext, CUdevice, Driver};
use crate::memory::Pool;

/// A CUDA device as reported by the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub index: usize,
    pub name: String,
    /// Compute capability (major, minor), e.g. (12, 1) for GB10.
    pub compute_capability: (u32, u32),
    pub multiprocessors: u32,
    pub total_memory: u64,
    /// Integrated GPU sharing system memory (e.g. GB10, Jetson).
    pub integrated: bool,
}

/// What the code generator needs to know about the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Caps {
    /// Compute capability as major * 10 + minor (e.g. 121).
    pub sm: u32,
    pub multiprocessors: u32,
    pub max_shared: u32,
    pub max_threads: u32,
    pub max_grid: [u32; 3],
    /// Highest PTX ISA the driver accepts, as major * 10 + minor (e.g. 90).
    pub ptx_isa: u32,
}

impl Caps {
    /// Tensor-core `mma.sync` m16n8k16 (f16 in, f32 accumulate) and `ldmatrix`: sm_80+.
    pub fn mma(&self) -> bool {
        self.sm >= 80
            && std::env::var("LAPID_CUDA_NO_MMA")
                .map(|v| v.is_empty() || v == "0")
                .unwrap_or(true)
    }

    /// Caps for code generation tests (no device needed).
    #[cfg(test)]
    pub fn test(sm: u32) -> Caps {
        Caps {
            sm,
            multiprocessors: 48,
            max_shared: 49152,
            max_threads: 1024,
            max_grid: [i32::MAX as u32, 65535, 65535],
            ptx_isa: 90,
        }
    }
}

/// Highest PTX ISA version supported by a driver of CUDA version (major, minor).
pub(crate) fn ptx_isa_for_driver(major: u32, minor: u32) -> u32 {
    match (major, minor) {
        (0..=10, _) => 63,
        (11, m) => 70 + m.min(8),
        (12, 0..=5) => 80 + minor,
        (12, 6) => 85,
        (12, 7) => 86,
        (12, 8) => 87,
        (12, _) => 88,
        (13, m) => 90 + m.min(9),
        _ => 90,
    }
}

/// Lowest PTX ISA version that knows `.target sm_XY`.
pub(crate) fn min_isa_for_sm(sm: u32) -> u32 {
    match sm {
        0..=75 => 63,
        80 => 70,
        86 => 71,
        87 => 74,
        89 | 90 => 78,
        100 | 101 | 103 => 86,
        110 => 90,
        120 => 87,
        121 => 88,
        _ => 90,
    }
}

pub(crate) struct Ctx {
    pub drv: &'static Driver,
    pub ctx: CUcontext,
    pub name: String,
    pub caps: Caps,
    pub integrated: bool,
    /// Kernels can read pageable host memory through the host page tables (attributes 88
    /// and 100): host bytes (e.g. an mmap'd file) are used in place by `map_host`.
    pub host_access: bool,
    /// Freed device allocations, reused by size class.
    pub pool: Mutex<Pool>,
}

// SAFETY: a CUcontext handle may be used from any thread (it is made current per thread
// before use); the driver API is thread-safe.
unsafe impl Send for Ctx {}
// SAFETY: see above; mutable state lives behind mutexes.
unsafe impl Sync for Ctx {}

impl Ctx {
    /// Make this context current on the calling thread.
    pub fn bind(&self) -> Result<()> {
        // SAFETY: the primary context is retained for the whole process.
        self.drv.check("cuCtxSetCurrent", unsafe {
            (self.drv.cuCtxSetCurrent)(self.ctx)
        })
    }
}

fn device_handle(drv: &Driver, index: usize) -> Result<CUdevice> {
    let mut count = 0;
    // SAFETY: writes one int.
    drv.check("cuDeviceGetCount", unsafe {
        (drv.cuDeviceGetCount)(&mut count)
    })?;
    if index >= count.max(0) as usize {
        return Err(Error::new(
            "device_unavailable",
            format!("cuda:{index}: only {count} CUDA device(s) found"),
        ));
    }
    let mut dev: CUdevice = 0;
    // SAFETY: index checked against the device count.
    drv.check("cuDeviceGet", unsafe {
        (drv.cuDeviceGet)(&mut dev, index as i32)
    })?;
    Ok(dev)
}

fn device_info(drv: &Driver, index: usize) -> Result<CudaDeviceInfo> {
    let dev = device_handle(drv, index)?;
    let mut buf = [0 as c_char; 256];
    // SAFETY: the driver writes a NUL-terminated name of at most `len` bytes.
    drv.check("cuDeviceGetName", unsafe {
        (drv.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32 - 1, dev)
    })?;
    // SAFETY: NUL-terminated (buffer zeroed, one byte reserved).
    let name = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string();
    let mut mem = 0usize;
    // SAFETY: writes one size_t.
    drv.check("cuDeviceTotalMem", unsafe {
        (drv.cuDeviceTotalMem_v2)(&mut mem, dev)
    })?;
    let major = drv.attr(dev, driver::ATTR_COMPUTE_CAPABILITY_MAJOR)?.max(0) as u32;
    let minor = drv.attr(dev, driver::ATTR_COMPUTE_CAPABILITY_MINOR)?.max(0) as u32;
    Ok(CudaDeviceInfo {
        index,
        name,
        compute_capability: (major, minor),
        multiprocessors: drv.attr(dev, driver::ATTR_MULTIPROCESSOR_COUNT)?.max(0) as u32,
        total_memory: mem as u64,
        integrated: drv.attr(dev, driver::ATTR_INTEGRATED)? != 0,
    })
}

pub(crate) fn enumerate() -> Result<Vec<CudaDeviceInfo>> {
    let drv = driver::driver()?;
    let mut count = 0;
    // SAFETY: writes one int.
    drv.check("cuDeviceGetCount", unsafe {
        (drv.cuDeviceGetCount)(&mut count)
    })?;
    (0..count.max(0) as usize)
        .map(|i| device_info(drv, i))
        .collect()
}

static CTXS: OnceLock<Mutex<HashMap<usize, Arc<Ctx>>>> = OnceLock::new();

/// The process-wide context of device `index` (created on first use).
pub(crate) fn shared_ctx(index: usize) -> Result<Arc<Ctx>> {
    let map = CTXS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map
        .lock()
        .map_err(|_| Error::new("cuda", "context table poisoned"))?;
    if let Some(c) = g.get(&index) {
        return Ok(c.clone());
    }
    let drv = driver::driver()?;
    let info = device_info(drv, index)?;
    let dev = device_handle(drv, index)?;
    let mut ctx: CUcontext = std::ptr::null_mut();
    // SAFETY: retains the device's primary context (never released, see module docs).
    drv.check("cuDevicePrimaryCtxRetain", unsafe {
        (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev)
    })?;
    let (vmaj, vmin) = drv.version()?;
    let sm = info.compute_capability.0 * 10 + info.compute_capability.1;
    let caps = Caps {
        sm,
        multiprocessors: info.multiprocessors,
        max_shared: drv
            .attr(dev, driver::ATTR_MAX_SHARED_MEMORY_PER_BLOCK)?
            .max(0) as u32,
        max_threads: drv.attr(dev, driver::ATTR_MAX_THREADS_PER_BLOCK)?.max(0) as u32,
        max_grid: [
            drv.attr(dev, driver::ATTR_MAX_GRID_DIM_X)?.max(1) as u32,
            drv.attr(dev, driver::ATTR_MAX_GRID_DIM_Y)?.max(1) as u32,
            drv.attr(dev, driver::ATTR_MAX_GRID_DIM_Z)?.max(1) as u32,
        ],
        ptx_isa: ptx_isa_for_driver(vmaj, vmin),
    };
    if caps.sm < 70 {
        return Err(Error::new(
            "device_unavailable",
            format!(
                "cuda:{index}: {} has compute capability {}.{}; lapid needs 7.0 or newer",
                info.name,
                sm / 10,
                sm % 10
            ),
        ));
    }
    // Older drivers may not know the attributes: treat errors as "no".
    let host_access = drv.attr(dev, driver::ATTR_PAGEABLE_MEMORY_ACCESS).unwrap_or(0) == 1
        && drv.attr(dev, driver::ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES).unwrap_or(0) == 1
        && std::env::var("LAPID_CUDA_NO_HOST_MAP").map_or(true, |v| v.trim().is_empty() || v.trim() == "0");
    let c = Arc::new(Ctx {
        drv,
        ctx,
        name: info.name,
        caps,
        integrated: info.integrated,
        host_access,
        pool: Mutex::new(Pool::default()),
    });
    g.insert(index, c.clone());
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_tables() {
        assert_eq!(ptx_isa_for_driver(13, 0), 90);
        assert_eq!(ptx_isa_for_driver(12, 9), 88);
        assert_eq!(ptx_isa_for_driver(12, 4), 84);
        assert_eq!(ptx_isa_for_driver(11, 8), 78);
        assert!(min_isa_for_sm(121) <= ptx_isa_for_driver(13, 0));
        assert_eq!(min_isa_for_sm(80), 70);
    }
}
