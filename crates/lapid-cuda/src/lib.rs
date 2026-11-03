//! lapid-cuda: CUDA backend.
//!
//! Pure Rust, no CUDA toolkit: the driver (`libcuda.so.1` / `nvcuda.dll`) is loaded at
//! runtime and kernels are PTX templates (`src/kernels/*.ptx`) whose slots are filled per
//! kernel spec, JIT-compiled by the driver (`cuModuleLoadDataEx`) for the device's
//! compute capability and cached per kernel key. Sizes travel in a by-value params block,
//! so one compiled kernel serves every shape.
//!
//! Hot paths: tensor-core GEMM (`mma.sync` f16 x f16 + f32, quantized weights dequantized
//! into shared memory per 32-group), warp-per-column GEMV for the decoder, tensor-core
//! flash attention for many queries and split-K ("flash decoding") attention for few.
//!
//! Environment:
//! - `LAPID_DEBUG=1` writes every kernel to `$LAPID_KERNEL_DIR` (default `<tmp>/lapid-kernels`)
//!   and prints the JIT info log; JIT errors always carry the log and numbered source.
//! - `LAPID_PROFILE=1` times every launch with events and prints per-kernel totals when the
//!   backend is dropped (`CudaBackend::profile_report`).
//! - `LAPID_CUDA_NO_MMA=1` forces the scalar (non-tensor-core) kernels (testing).
//! - `LAPID_CUDA_FAST_ATTENTION=1` rounds f32 q/k/v to f16 in the tensor-core flash attention
//!   (default: hi/lo f16 splits, ~f32 accuracy, about 2.5x the attention time).
//! - Compiled kernels (cubins) are cached under `$LAPID_CACHE/kernels/cuda` (see `cache`);
//!   `LAPID_NO_KERNEL_CACHE=1` disables that cache.
//! - `LAPID_CUDA_DISABLE=1` hides the driver (`devices()` → `cuda_unavailable`).

#[cfg(test)]
mod bench_tests;
mod cache;
mod codegen;
mod context;
mod driver;
mod execute;
mod graph;
#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod kquant_tests;
#[cfg(test)]
mod matmul_attn_tests;
mod memory;
mod module;
#[cfg(test)]
mod r5_tests;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, MutexGuard};

use lapid_compiler::{Backend, Launch};
use lapid_tensor::{DeviceData, Error, Result};

pub use context::CudaDeviceInfo;
pub use memory::CudaBuffer;

use crate::context::Ctx;
use crate::execute::{Exec, Timing};
use crate::memory::as_cuda;
use crate::module::Modules;

/// List CUDA devices. Loads the driver at runtime; `Err("cuda_unavailable")` when no
/// driver is installed or it cannot initialize. Never panics.
pub fn devices() -> Result<Vec<CudaDeviceInfo>> {
    match std::panic::catch_unwind(context::enumerate) {
        Ok(r) => r,
        Err(_) => Err(Error::new("cuda_unavailable", "CUDA enumeration panicked")),
    }
}

/// The CUDA backend for one device.
pub struct CudaBackend {
    ctx: Arc<Ctx>,
    modules: Modules,
    exec: Mutex<Exec>,
}

impl CudaBackend {
    /// Open device `index` (index into `devices()`).
    pub fn new(index: usize) -> Result<CudaBackend> {
        let ctx = match std::panic::catch_unwind(|| context::shared_ctx(index)) {
            Ok(r) => r?,
            Err(_) => {
                return Err(Error::new(
                    "cuda_unavailable",
                    "CUDA initialization panicked",
                ));
            }
        };
        let exec = Exec::new(&ctx)?;
        Ok(CudaBackend {
            ctx,
            modules: Modules::default(),
            exec: Mutex::new(exec),
        })
    }

    /// Short capability summary, e.g. "sm_121, 48 SMs, tensor cores, unified memory".
    pub fn features(&self) -> String {
        let c = &self.ctx.caps;
        let mut v = vec![format!("sm_{}", c.sm), format!("{} SMs", c.multiprocessors)];
        if c.mma() {
            v.push("tensor cores".to_string());
        }
        if self.ctx.integrated {
            v.push("unified memory".to_string());
        }
        if self.ctx.host_access {
            v.push("host page tables".to_string());
        }
        v.push(format!("PTX {}.{}", c.ptx_isa / 10, c.ptx_isa % 10));
        v.join(", ")
    }

    /// Write the PTX of every kernel compiled so far to `dir`; returns the file count.
    pub fn dump_kernels(&self, dir: &std::path::Path) -> Result<usize> {
        self.modules.dump(dir)
    }

    /// Number of compiled kernels (diagnostics).
    pub fn kernel_count(&self) -> usize {
        self.modules.len()
    }

    /// Run `launches` once and return the GPU time (ms) of each non-empty launch,
    /// measured with events.
    pub fn time_launches(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        self.lock()?.run(
            &self.ctx,
            &self.modules,
            launches,
            buffers,
            Timing::Collect(&mut out),
        )?;
        Ok(out)
    }

    /// Per-kernel GPU totals collected so far (`LAPID_PROFILE=1`), or `None` when off.
    pub fn profile_report(&self) -> Option<String> {
        let e = self.exec.lock().ok()?;
        e.prof.enabled.then(|| e.prof.report())
    }

    /// Clear the profile totals.
    pub fn profile_reset(&self) {
        if let Ok(mut e) = self.exec.lock() {
            e.prof.reset();
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Exec>> {
        self.exec
            .lock()
            .map_err(|_| Error::new("cuda", "execution state poisoned"))
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        if self.ctx.bind().is_err() {
            return;
        }
        if let Some(r) = self.profile_report() {
            eprintln!("[lapid-cuda] GPU profile ({}):\n{r}", self.ctx.name);
        }
        if let Ok(mut e) = self.exec.lock() {
            let _ = e.sync(&self.ctx);
            e.destroy(&self.ctx);
        }
        self.modules.unload_all(&self.ctx);
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn device_name(&self) -> String {
        self.ctx.name.clone()
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceData> {
        Ok(Arc::new(CudaBuffer::alloc(&self.ctx, bytes)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        let buf = CudaBuffer::alloc(&self.ctx, bytes.len())?;
        if !bytes.is_empty() {
            self.lock()?.upload(&self.ctx, &buf, bytes)?;
        }
        Ok(Arc::new(buf))
    }

    fn map_host(&self, host: &lapid_tensor::HostData) -> Result<Option<DeviceData>> {
        Ok(CudaBuffer::map(&self.ctx, host).map(|b| Arc::new(b) as DeviceData))
    }

    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>> {
        let b = as_cuda(buf, &self.ctx)?;
        if let Some(bytes) = b.mapped_bytes() {
            return Ok(bytes.to_vec());
        }
        let mut out = vec![0u8; b.len_bytes()];
        if !out.is_empty() {
            self.lock()?.download(&self.ctx, b, &mut out)?;
        }
        Ok(out)
    }

    fn prepare(&self, launches: &[Launch]) -> Result<()> {
        for l in launches {
            let mut plan = Some(codegen::plan(&l.kernel, &self.ctx.caps)?);
            while let Some(p) = plan {
                self.modules.get(&self.ctx, &p.key)?;
                plan = p.next.map(|b| *b);
            }
        }
        Ok(())
    }

    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        if launches.is_empty() {
            return Ok(());
        }
        self.lock()?
            .run(&self.ctx, &self.modules, launches, buffers, Timing::None)
    }

    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        self.exec
            .lock()
            .map(|mut e| e.prof.take())
            .unwrap_or_default()
    }
}
