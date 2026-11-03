//! Device: picks a backend, compiles graphs and runs them. CONTRACT signatures —
//! implementation by agent "runtime" (bodies may move into runtime.rs).

use std::sync::Arc;

use lapid_tensor::{Error, Result, Tensor};

use crate::runtime;

/// Cheap to clone; all clones share one backend, upload cache and memory pool.
#[derive(Clone)]
pub struct Device {
    pub(crate) rt: Arc<crate::runtime::Runtime>,
}

/// One usable device (see [`Device::list_info`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", self.description())
    }
}

impl Device {
    /// Best available device: env `LAPID_DEVICE` ("cpu" | "metal[:N]" | "cuda[:N]" |
    /// "vulkan[:N]") wins; otherwise Metal device 0 on Apple platforms, then CUDA device 0
    /// (NVIDIA), then the first discrete/integrated Vulkan GPU (never llvmpipe/software
    /// unless asked), else CPU. Never fails.
    pub fn best() -> Device {
        if let Ok(name) = std::env::var("LAPID_DEVICE") {
            let name = name.trim();
            if !name.is_empty() {
                match Device::from_name(name) {
                    Ok(d) => return d,
                    Err(e) => crate::log::emit(&format!(
                        "lapid: LAPID_DEVICE={name} is unavailable ({e}); selecting a device automatically"
                    )),
                }
            }
        }
        auto_gpu().unwrap_or_else(Device::cpu)
    }

    /// One line per usable device, e.g. ["cpu: 20 threads (neon, dotprod)", "vulkan:0: NVIDIA GB10 (integrated)",
    /// "metal:0: Apple M4 Pro (unified memory)"].
    pub fn list() -> Vec<String> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut out = vec![format!("cpu: {threads} threads ({})", lapid_cpu::CpuBackend::features())];
        out.extend(mtl::lines());
        out.extend(cu::lines());
        out.extend(vulkan_lines());
        out.extend(wg::lines());
        out
    }

    /// [`Device::list`] as structured entries: `name` is what [`Device::from_name`] accepts
    /// ("cpu", "vulkan:0", ...), `description` the rest of the line.
    pub fn list_info() -> Vec<DeviceInfo> {
        Device::list()
            .into_iter()
            .map(|line| match line.split_once(": ") {
                Some((name, desc)) => DeviceInfo { name: name.to_string(), description: desc.to_string() },
                None => DeviceInfo { name: line, description: String::new() },
            })
            .collect()
    }

    pub fn cpu() -> Device {
        let rt = runtime::shared_infallible("cpu", || {
            Arc::new(lapid_cpu::CpuBackend::new()) as Arc<dyn lapid_compiler::Backend>
        });
        Device { rt }
    }

    /// Vulkan device `index` among non-software adapters (or software if
    /// `LAPID_VULKAN_ALLOW_SOFTWARE=1`).
    pub fn vulkan(index: usize) -> Result<Device> {
        vk::open(index)
    }

    /// CUDA device `index` (NVIDIA driver loaded at runtime; `Err` when unavailable).
    pub fn cuda(index: usize) -> Result<Device> {
        cu::open(index)
    }

    /// WebGPU device `index` (wgpu: Vulkan/Metal/DX12 natively; feature "webgpu"). On wasm
    /// use [`Device::webgpu_async`] (adapter creation is asynchronous in browsers).
    pub fn webgpu(index: usize) -> Result<Device> {
        wg::open(index)
    }

    /// WebGPU device `index`, created asynchronously (works on every target; required on
    /// wasm, where index 0 is the browser's high-performance adapter).
    pub async fn webgpu_async(index: usize) -> Result<Device> {
        wg::open_async(index).await
    }

    /// Metal device `index` (Apple platforms; `Err("device_unavailable")` elsewhere).
    pub fn metal(index: usize) -> Result<Device> {
        mtl::open(index)
    }

    /// "cpu", "metal", "metal:1", "vulkan", "vulkan:1", "gpu" (best GPU)
    pub fn from_name(name: &str) -> Result<Device> {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "cpu" => Ok(Device::cpu()),
            "metal" => Device::metal(0),
            "cuda" => Device::cuda(0),
            "vulkan" => Device::vulkan(0),
            "webgpu" => Device::webgpu(0),
            "gpu" => auto_gpu().ok_or_else(|| Error::new("device_unavailable", "no usable GPU found")),
            other => {
                let parse = |prefix: &str| other.strip_prefix(prefix).and_then(|i| i.trim().parse::<usize>().ok());
                if let Some(index) = parse("metal:") {
                    return Device::metal(index);
                }
                if let Some(index) = parse("webgpu:") {
                    return Device::webgpu(index);
                }
                if let Some(index) = parse("cuda:") {
                    return Device::cuda(index);
                }
                let index = parse("vulkan:").ok_or_else(|| {
                    Error::new(
                        "invalid_argument",
                        format!("unknown device '{name}' (expected cpu, gpu, metal, metal:N, cuda, cuda:N, vulkan, vulkan:N, webgpu or webgpu:N)"),
                    )
                })?;
                Device::vulkan(index)
            }
        }
    }

    /// Backend name: "cpu" | "vulkan" | "metal".
    pub fn name(&self) -> &'static str {
        self.rt.backend.name()
    }

    /// e.g. "vulkan: NVIDIA GB10"
    pub fn description(&self) -> String {
        format!("{}: {}", self.name(), self.rt.backend.device_name())
    }

    /// Compile and run the graphs of `outputs` together. Returns realized tensors
    /// (`Op::Device` leaves, same shapes/dtypes) that can be fed into later graphs without
    /// copies (e.g. KV caches). Host leaves are uploaded once and cached for the lifetime of
    /// the Device (weights). Temps are pooled.
    pub fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.rt.eval(outputs)
    }

    /// [`eval`](Self::eval) taking ownership of the output handles: they are released as
    /// soon as the graph is planned, so a graph held by nothing else is freed before it runs
    /// (cheaper than tearing it down afterwards, when its nodes are out of cache).
    pub fn eval_owned(&self, outputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.rt.eval_owned(outputs)
    }

    /// Evaluate (if needed) and copy to host as f32 (casting f16/bf16/i32).
    pub fn to_vec_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        self.rt.to_vec_f32(t)
    }

    /// Evaluate (if needed) and copy the raw storage bytes to host, in the tensor's own dtype
    /// (f16/bf16 little endian, quantized tensors in their lapid-quant layout).
    pub fn to_vec_bytes(&self, t: &Tensor) -> Result<Vec<u8>> {
        self.rt.to_vec_bytes(t)
    }

    pub fn to_vec_i32(&self, t: &Tensor) -> Result<Vec<i32>> {
        self.rt.to_vec_i32(t)
    }

    /// Upload a host leaf now (e.g. all weights at load time). Returns a Device leaf.
    pub fn upload(&self, t: &Tensor) -> Result<Tensor> {
        self.rt.upload(t)
    }

    /// A new device tensor of `shape` / `dtype` with unspecified contents (fill it with
    /// [`Device::write`]), e.g. the slot pool of an expert cache.
    pub fn alloc(&self, shape: &[usize], dtype: lapid_tensor::DType) -> Result<Tensor> {
        self.rt.alloc(shape, dtype)
    }

    /// Overwrite bytes `offset..offset + bytes.len()` of the storage of a device tensor
    /// made by this device (e.g. [`Device::alloc`]), waiting for the copy. `bytes` is read
    /// by the host (file-backed memory is fine: the device never reads it in place).
    /// Graphs evaluated earlier have completed; later ones see the new bytes.
    /// Error `unsupported` on backends without in-place writes.
    pub fn write(&self, t: &Tensor, offset: usize, bytes: &[u8]) -> Result<()> {
        self.rt.write(t, offset, bytes)
    }

    /// When true, prints the compiled schedule (`lapid_compiler::dump`) and per-launch
    /// timings to stderr. Also enabled by env `LAPID_DEBUG=1`.
    pub fn set_debug(&self, on: bool) {
        self.rt.set_debug(on)
    }
}

/// Best GPU: Metal device 0 (Apple), else CUDA device 0 (NVIDIA: preferred over Vulkan on
/// the same GPU), else the first usable discrete/integrated Vulkan GPU.
fn auto_gpu() -> Option<Device> {
    mtl::open(0).ok().or_else(|| cu::open(0).ok()).or_else(auto_vulkan)
}

/// First usable discrete/integrated Vulkan GPU, if any opens successfully.
#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
fn auto_vulkan() -> Option<Device> {
    let adapters = vk::usable_adapters().ok()?;
    adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a.kind.as_str(), "discrete" | "integrated"))
        .find_map(|(i, _)| vk::open(i).ok())
}

#[cfg(not(all(feature = "vulkan", not(target_arch = "wasm32"))))]
fn auto_vulkan() -> Option<Device> {
    None
}

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
fn vulkan_lines() -> Vec<String> {
    vk::usable_adapters()
        .map(|list| list.iter().enumerate().map(|(i, a)| format!("vulkan:{i}: {} ({})", a.name, a.kind)).collect())
        .unwrap_or_default()
}

#[cfg(not(all(feature = "vulkan", not(target_arch = "wasm32"))))]
fn vulkan_lines() -> Vec<String> {
    Vec::new()
}

#[cfg(not(all(feature = "vulkan", not(target_arch = "wasm32"))))]
mod vk {
    use lapid_tensor::{Error, Result};

    use super::Device;

    pub(super) fn open(index: usize) -> Result<Device> {
        Err(Error::new(
            "device_unavailable",
            format!("vulkan:{index}: lapid was built without the \"vulkan\" feature (or for wasm)"),
        ))
    }
}

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
mod vk {
    use std::sync::Arc;

    use lapid_tensor::{Error, Result};
    use lapid_vulkan::{AdapterInfo, VulkanBackend};

    use super::Device;
    use crate::runtime;

    fn allow_software() -> bool {
        std::env::var("LAPID_VULKAN_ALLOW_SOFTWARE").is_ok_and(|v| v.trim() == "1")
    }

    /// Adapters addressable as `vulkan:N` (N indexes this list).
    pub(super) fn usable_adapters() -> Result<Vec<AdapterInfo>> {
        let allow_sw = allow_software();
        Ok(lapid_vulkan::adapters()?.into_iter().filter(|a| allow_sw || !a.is_software).collect())
    }

    pub(super) fn open(index: usize) -> Result<Device> {
        let adapters = usable_adapters()?;
        let Some(a) = adapters.get(index) else {
            return Err(Error::new(
                "device_unavailable",
                format!("vulkan:{index}: only {} usable Vulkan adapter(s) found", adapters.len()),
            ));
        };
        let adapter = a.index;
        let rt = runtime::shared(&format!("vulkan:{adapter}"), || {
            VulkanBackend::new(adapter).map(|b| Arc::new(b) as Arc<dyn lapid_compiler::Backend>)
        })?;
        Ok(Device { rt })
    }
}

#[cfg(not(all(feature = "metal", target_vendor = "apple")))]
mod mtl {
    use lapid_tensor::{Error, Result};

    use super::Device;

    pub(super) fn open(index: usize) -> Result<Device> {
        let why = if cfg!(target_vendor = "apple") {
            "lapid was built without the \"metal\" feature"
        } else {
            "Metal is only available on Apple platforms"
        };
        Err(Error::new("device_unavailable", format!("metal:{index}: {why}")))
    }

    pub(super) fn lines() -> Vec<String> {
        Vec::new()
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod mtl {
    use std::sync::Arc;

    use lapid_metal::MetalBackend;
    use lapid_tensor::{Error, Result};

    use super::Device;
    use crate::runtime;

    pub(super) fn open(index: usize) -> Result<Device> {
        let devices = lapid_metal::devices()?;
        if index >= devices.len() {
            return Err(Error::new(
                "device_unavailable",
                format!("metal:{index}: only {} Metal device(s) found", devices.len()),
            ));
        }
        let rt = runtime::shared(&format!("metal:{index}"), || {
            MetalBackend::new(index).map(|b| Arc::new(b) as Arc<dyn lapid_compiler::Backend>)
        })?;
        Ok(Device { rt })
    }

    pub(super) fn lines() -> Vec<String> {
        lapid_metal::devices()
            .map(|list| {
                list.iter()
                    .map(|d| {
                        let mem = if d.unified_memory { "unified memory" } else { "discrete memory" };
                        format!("metal:{}: {} ({mem}, {:.1} GB working set)", d.index, d.name, d.working_set_bytes as f64 / 1e9)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(not(all(feature = "cuda", not(target_arch = "wasm32"))))]
mod cu {
    use lapid_tensor::{Error, Result};

    use super::Device;

    pub(super) fn open(index: usize) -> Result<Device> {
        Err(Error::new("device_unavailable", format!("cuda:{index}: lapid was built without the \"cuda\" feature (or for wasm)")))
    }

    pub(super) fn lines() -> Vec<String> {
        Vec::new()
    }
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
mod cu {
    use std::sync::Arc;

    use lapid_cuda::CudaBackend;
    use lapid_tensor::{Error, Result};

    use super::Device;
    use crate::runtime;

    pub(super) fn open(index: usize) -> Result<Device> {
        let devices = lapid_cuda::devices()?;
        if index >= devices.len() {
            return Err(Error::new("device_unavailable", format!("cuda:{index}: only {} CUDA device(s) found", devices.len())));
        }
        let rt = runtime::shared(&format!("cuda:{index}"), || {
            CudaBackend::new(index).map(|b| Arc::new(b) as Arc<dyn lapid_compiler::Backend>)
        })?;
        Ok(Device { rt })
    }

    pub(super) fn lines() -> Vec<String> {
        lapid_cuda::devices()
            .map(|list| {
                list.iter()
                    .map(|d| {
                        let (ma, mi) = d.compute_capability;
                        let mem = if d.integrated { "unified memory" } else { "discrete memory" };
                        format!("cuda:{}: {} (sm_{ma}{mi}, {mem}, {:.1} GB)", d.index, d.name, d.total_memory as f64 / 1e9)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(not(feature = "webgpu"))]
mod wg {
    use lapid_tensor::{Error, Result};

    use super::Device;

    fn missing(index: usize) -> Error {
        Error::new("device_unavailable", format!("webgpu:{index}: lapid was built without the \"webgpu\" feature"))
    }

    pub(super) fn open(index: usize) -> Result<Device> {
        Err(missing(index))
    }

    pub(super) async fn open_async(index: usize) -> Result<Device> {
        Err(missing(index))
    }

    pub(super) fn lines() -> Vec<String> {
        Vec::new()
    }
}

#[cfg(feature = "webgpu")]
mod wg {
    use std::sync::Arc;

    use lapid_tensor::{Error, Result};
    use lapid_webgpu::WebGpuBackend;

    use super::Device;
    use crate::runtime;

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn open(index: usize) -> Result<Device> {
        let rt = runtime::shared(&format!("webgpu:{index}"), || {
            WebGpuBackend::new(index).map(|b| Arc::new(b) as Arc<dyn lapid_compiler::Backend>)
        })?;
        Ok(Device { rt })
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn open(index: usize) -> Result<Device> {
        Err(Error::new("device_unavailable", format!("webgpu:{index}: use Device::webgpu_async on wasm")))
    }

    pub(super) async fn open_async(index: usize) -> Result<Device> {
        let backend = WebGpuBackend::new_async(index).await?;
        let rt = runtime::shared(&format!("webgpu:{index}"), || {
            Ok::<_, Error>(Arc::new(backend) as Arc<dyn lapid_compiler::Backend>)
        })?;
        Ok(Device { rt })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn lines() -> Vec<String> {
        lapid_webgpu::adapters()
            .map(|list| list.iter().map(|a| format!("webgpu:{}: {} ({}, {})", a.index, a.name, a.backend, a.kind)).collect())
            .unwrap_or_default()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn lines() -> Vec<String> {
        Vec::new()
    }
}
