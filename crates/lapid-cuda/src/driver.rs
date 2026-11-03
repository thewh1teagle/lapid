//! Minimal CUDA driver API bindings, resolved at runtime from `libcuda.so.1` /
//! `nvcuda.dll` (no CUDA toolkit, no link-time dependency). Only the entry points the
//! backend uses are declared; every call goes through `check` which turns a `CUresult` into
//! a `lapid_tensor::Error` with code "cuda".

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use lapid_tensor::{Error, Result};

pub(crate) type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUdeviceptr = u64;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUmodule = *mut c_void;
pub(crate) type CUfunction = *mut c_void;
pub(crate) type CUstream = *mut c_void;
pub(crate) type CUevent = *mut c_void;
pub(crate) type CUgraph = *mut c_void;
pub(crate) type CUgraphExec = *mut c_void;

pub(crate) const CUDA_SUCCESS: CUresult = 0;

// CUdevice_attribute
pub(crate) const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub(crate) const ATTR_MAX_GRID_DIM_X: c_int = 5;
pub(crate) const ATTR_MAX_GRID_DIM_Y: c_int = 6;
pub(crate) const ATTR_MAX_GRID_DIM_Z: c_int = 7;
pub(crate) const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub(crate) const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub(crate) const ATTR_INTEGRATED: c_int = 18;
pub(crate) const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// Kernels may dereference any pageable host pointer (HMM / ATS).
pub(crate) const ATTR_PAGEABLE_MEMORY_ACCESS: c_int = 88;
/// ... through the host's own page tables (coherent, no migration; e.g. GB10, Grace Hopper).
pub(crate) const ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: c_int = 100;
pub(crate) const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;

// CUjit_option
pub(crate) const JIT_INFO_LOG_BUFFER: c_int = 3;
pub(crate) const JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub(crate) const JIT_ERROR_LOG_BUFFER: c_int = 5;
pub(crate) const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
pub(crate) const JIT_OPTIMIZATION_LEVEL: c_int = 7;
pub(crate) const JIT_LOG_VERBOSE: c_int = 12;

pub(crate) const STREAM_NON_BLOCKING: c_uint = 1;
pub(crate) const EVENT_DEFAULT: c_uint = 0;
pub(crate) const EVENT_DISABLE_TIMING: c_uint = 2;
/// CU_STREAM_CAPTURE_MODE_RELAXED: other threads may keep using the driver.
pub(crate) const CAPTURE_MODE_RELAXED: c_int = 2;

macro_rules! driver_api {
    ($($name:ident : fn($($arg:ty),*);)*) => {
        /// Resolved driver entry points. The library handle is kept alive for the process.
        #[allow(non_snake_case)]
        pub(crate) struct Driver {
            _lib: libloading::Library,
            /// CUDA graphs (driver 12.0+); `None` on older drivers.
            pub graph: Option<GraphApi>,
            /// JIT linker (PTX → cubin bytes, for the on-disk kernel cache).
            pub link: Option<LinkApi>,
            $(pub $name: unsafe extern "C" fn($($arg),*) -> CUresult,)*
        }

        impl Driver {
            #[allow(non_snake_case)]
            fn load(lib: libloading::Library) -> Result<Driver> {
                $(
                    // SAFETY: the symbol is a C function with exactly this signature in every
                    // CUDA driver since 10.x (the `_v2` names are the ABI-stable variants).
                    let $name = unsafe {
                        *lib.get::<unsafe extern "C" fn($($arg),*) -> CUresult>(concat!(stringify!($name), "\0").as_bytes())
                            .map_err(|e| Error::new("cuda_unavailable", format!("CUDA driver lacks {}: {e}", stringify!($name))))?
                    };
                )*
                let graph = GraphApi::load(&lib);
                let link = LinkApi::load(&lib);
                Ok(Driver { _lib: lib, graph, link, $($name,)* })
            }
        }
    };
}

driver_api! {
    cuInit: fn(c_uint);
    cuDriverGetVersion: fn(*mut c_int);
    cuGetErrorName: fn(CUresult, *mut *const c_char);
    cuGetErrorString: fn(CUresult, *mut *const c_char);
    cuDeviceGetCount: fn(*mut c_int);
    cuDeviceGet: fn(*mut CUdevice, c_int);
    cuDeviceGetName: fn(*mut c_char, c_int, CUdevice);
    cuDeviceGetAttribute: fn(*mut c_int, c_int, CUdevice);
    cuDeviceTotalMem_v2: fn(*mut usize, CUdevice);
    cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice);
    cuCtxSetCurrent: fn(CUcontext);
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize);
    cuMemFree_v2: fn(CUdeviceptr);
    cuMemAllocHost_v2: fn(*mut *mut c_void, usize);
    cuMemFreeHost: fn(*mut c_void);
    cuMemHostGetDevicePointer_v2: fn(*mut CUdeviceptr, *mut c_void, c_uint);
    cuMemcpyHtoDAsync_v2: fn(CUdeviceptr, *const c_void, usize, CUstream);
    cuMemcpyDtoHAsync_v2: fn(*mut c_void, CUdeviceptr, usize, CUstream);
    cuStreamCreate: fn(*mut CUstream, c_uint);
    cuStreamDestroy_v2: fn(CUstream);
    cuStreamSynchronize: fn(CUstream);
    cuEventCreate: fn(*mut CUevent, c_uint);
    cuEventRecord: fn(CUevent, CUstream);
    cuEventSynchronize: fn(CUevent);
    cuEventElapsedTime: fn(*mut f32, CUevent, CUevent);
    cuEventDestroy_v2: fn(CUevent);
    cuModuleLoadDataEx: fn(*mut CUmodule, *const c_void, c_uint, *mut c_int, *mut *mut c_void);
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const c_char);
    cuModuleUnload: fn(CUmodule);
    cuLaunchKernel: fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void);
}

/// Stream capture + executable graphs, resolved only when the driver has them.
#[allow(non_snake_case)]
pub(crate) struct GraphApi {
    pub cuStreamBeginCapture_v2: unsafe extern "C" fn(CUstream, c_int) -> CUresult,
    pub cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    pub cuGraphExecUpdate_v2: unsafe extern "C" fn(CUgraphExec, CUgraph, *mut c_void) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
}

impl GraphApi {
    fn load(lib: &libloading::Library) -> Option<GraphApi> {
        macro_rules! sym {
            ($n:literal) => {
                // SAFETY: C functions with the declared signatures (CUDA 12.0+ ABI).
                unsafe { *lib.get(concat!($n, "\0").as_bytes()).ok()? }
            };
        }
        Some(GraphApi {
            cuStreamBeginCapture_v2: sym!("cuStreamBeginCapture_v2"),
            cuStreamEndCapture: sym!("cuStreamEndCapture"),
            cuGraphInstantiateWithFlags: sym!("cuGraphInstantiateWithFlags"),
            cuGraphExecUpdate_v2: sym!("cuGraphExecUpdate_v2"),
            cuGraphLaunch: sym!("cuGraphLaunch"),
            cuGraphDestroy: sym!("cuGraphDestroy"),
            cuGraphExecDestroy: sym!("cuGraphExecDestroy"),
        })
    }
}

pub(crate) type CUlinkState = *mut c_void;
/// CU_JIT_INPUT_PTX
pub(crate) const JIT_INPUT_PTX: c_int = 1;

/// `cuLink*`: JIT-compile PTX into a cubin image the backend can cache on disk.
#[allow(non_snake_case)]
pub(crate) struct LinkApi {
    pub cuLinkCreate_v2: unsafe extern "C" fn(c_uint, *mut c_int, *mut *mut c_void, *mut CUlinkState) -> CUresult,
    pub cuLinkAddData_v2: unsafe extern "C" fn(
        CUlinkState,
        c_int,
        *mut c_void,
        usize,
        *const c_char,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CUresult,
    pub cuLinkComplete: unsafe extern "C" fn(CUlinkState, *mut *mut c_void, *mut usize) -> CUresult,
    pub cuLinkDestroy: unsafe extern "C" fn(CUlinkState) -> CUresult,
}

impl LinkApi {
    fn load(lib: &libloading::Library) -> Option<LinkApi> {
        macro_rules! sym {
            ($n:literal) => {
                // SAFETY: C functions with the declared signatures (CUDA 6.5+ ABI).
                unsafe { *lib.get(concat!($n, "\0").as_bytes()).ok()? }
            };
        }
        Some(LinkApi {
            cuLinkCreate_v2: sym!("cuLinkCreate_v2"),
            cuLinkAddData_v2: sym!("cuLinkAddData_v2"),
            cuLinkComplete: sym!("cuLinkComplete"),
            cuLinkDestroy: sym!("cuLinkDestroy"),
        })
    }
}

#[cfg(windows)]
const LIB_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(target_os = "macos")]
const LIB_NAMES: &[&str] = &["libcuda.dylib"];
#[cfg(not(any(windows, target_os = "macos")))]
const LIB_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

static DRIVER: OnceLock<std::result::Result<Driver, Error>> = OnceLock::new();

/// The process-wide driver (loaded and `cuInit`ed once). `Err("cuda_unavailable")` when
/// no driver library is installed or initialization fails (e.g. no NVIDIA GPU).
pub(crate) fn driver() -> Result<&'static Driver> {
    DRIVER
        .get_or_init(load_driver)
        .as_ref()
        .map_err(Clone::clone)
}

fn load_driver() -> Result<Driver> {
    if std::env::var("LAPID_CUDA_DISABLE").is_ok_and(|v| !v.is_empty() && v != "0") {
        return Err(Error::new(
            "cuda_unavailable",
            "CUDA disabled by LAPID_CUDA_DISABLE",
        ));
    }
    let mut last = String::from("no candidate library");
    for name in LIB_NAMES {
        // SAFETY: loading the vendor driver runs its initializers, which is the documented
        // way to use the CUDA driver API without linking against it.
        match unsafe { libloading::Library::new(name) } {
            Ok(lib) => {
                let d = Driver::load(lib)?;
                // SAFETY: plain FFI call with flags 0.
                let r = unsafe { (d.cuInit)(0) };
                if r != CUDA_SUCCESS {
                    return Err(Error::new(
                        "cuda_unavailable",
                        format!("cuInit failed: {}", d.error_text(r)),
                    ));
                }
                return Ok(d);
            }
            Err(e) => last = format!("{name}: {e}"),
        }
    }
    Err(Error::new(
        "cuda_unavailable",
        format!("CUDA driver not found ({last})"),
    ))
}

impl Driver {
    /// "CUDA_ERROR_X (code): description".
    pub(crate) fn error_text(&self, r: CUresult) -> String {
        let mut name: *const c_char = std::ptr::null();
        let mut desc: *const c_char = std::ptr::null();
        // SAFETY: the driver writes pointers to static strings (or leaves them null).
        unsafe {
            (self.cuGetErrorName)(r, &mut name);
            (self.cuGetErrorString)(r, &mut desc);
        }
        let s = |p: *const c_char| {
            if p.is_null() {
                String::from("?")
            } else {
                // SAFETY: non-null pointers from the driver are NUL-terminated static strings.
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        format!("{} ({r}): {}", s(name), s(desc))
    }

    /// Map a `CUresult` to `Ok` or an `Error("cuda", "<call>: <text>")`.
    pub(crate) fn check(&self, call: &str, r: CUresult) -> Result<()> {
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::new(
                "cuda",
                format!("{call}: {}", self.error_text(r)),
            ))
        }
    }

    /// Driver version as (major, minor), e.g. (13, 0).
    pub(crate) fn version(&self) -> Result<(u32, u32)> {
        let mut v: c_int = 0;
        // SAFETY: writes one int.
        self.check("cuDriverGetVersion", unsafe {
            (self.cuDriverGetVersion)(&mut v)
        })?;
        let v = v.max(0) as u32;
        Ok((v / 1000, (v % 1000) / 10))
    }

    pub(crate) fn attr(&self, dev: CUdevice, attr: c_int) -> Result<i32> {
        let mut v: c_int = 0;
        // SAFETY: writes one int.
        self.check("cuDeviceGetAttribute", unsafe {
            (self.cuDeviceGetAttribute)(&mut v, attr, dev)
        })?;
        Ok(v)
    }
}
