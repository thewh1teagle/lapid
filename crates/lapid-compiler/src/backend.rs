//! Backend trait. CONTRACT FILE — owned by the coordinator.

use lapid_tensor::{DeviceData, HostData, Result};

use crate::CompileOptions;
use crate::ir::Launch;

pub trait Backend: Send + Sync {
    /// "cpu" or "vulkan"; equals `DeviceBuffer::backend()` of its buffers.
    fn name(&self) -> &'static str;
    /// Human readable, e.g. "NVIDIA GB10" / "cpu (20 threads, neon+dotprod)".
    fn device_name(&self) -> String;
    /// Options the runtime compiles this backend's graphs with (opt-in kernels).
    fn compile_options(&self) -> CompileOptions {
        CompileOptions::default()
    }
    /// Allocate uninitialized device memory.
    fn alloc(&self, bytes: usize) -> Result<DeviceData>;
    /// Allocate and fill.
    fn upload(&self, bytes: &[u8]) -> Result<DeviceData>;
    /// Device view of host bytes without copying (file-backed weights that outlive the buffer).
    /// The returned buffer keeps `host` alive and is read-only: kernels never write it.
    /// `Ok(None)`: not supported here (or not for these bytes); the runtime uploads as before.
    fn map_host(&self, _host: &HostData) -> Result<Option<DeviceData>> {
        Ok(None)
    }
    /// Overwrite bytes `offset..offset + bytes.len()` of an existing (writable, not mapped)
    /// buffer, e.g. one slot of a device-resident expert cache. `bytes` may be reused when
    /// it returns and is read by the host, never by the device in place (file-backed memory
    /// is fine); the device side may still be in flight, ordered before every later
    /// `execute` / `download`. Must not be called while an `execute` reading the buffer runs.
    /// Default: `unsupported`.
    fn write(&self, _buf: &DeviceData, _offset: usize, _bytes: &[u8]) -> Result<()> {
        Err(lapid_tensor::Error::new("unsupported", format!("{}: writing into device buffers is not supported", self.name())))
    }
    /// Copy back to host (waits for pending work).
    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>>;
    /// Copy the first `out.len()` bytes back into `out` (waits for pending work), e.g. into
    /// a typed host vector without an intermediate copy. Default: through `download`.
    fn download_into(&self, buf: &DeviceData, out: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let v = self.download(buf)?;
        let Some(src) = v.get(..out.len()) else {
            return Err(lapid_tensor::Error::new("backend", format!("downloaded {} bytes, expected {}", v.len(), out.len())));
        };
        for (o, &b) in out.iter_mut().zip(src) {
            o.write(b);
        }
        Ok(())
    }
    /// Compile / cache everything the launches need. Optional (execute must also work cold).
    fn prepare(&self, _launches: &[Launch]) -> Result<()> {
        Ok(())
    }
    /// Run the launches in order and wait for completion. `buffers[id]` is the device buffer
    /// of BufId `id` (several ids may share one buffer — temp slots). Launch k+1 sees all
    /// writes of launch k.
    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()>;
    /// Per-kernel-kind time totals `(kind, launches, ms)` accumulated by `execute` since the
    /// last call, when the backend profiles (`LAPID_PROFILE=1`). Default: not supported.
    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        Vec::new()
    }
}
