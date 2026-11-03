//! Launching: one stream per backend, launches issued back to back (stream order makes
//! launch k + 1 see every write of launch k, no barriers needed), one synchronize at the
//! end of `execute`. Also staged uploads / downloads and per-launch event timing.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use lapid_compiler::Launch;
use lapid_tensor::{DeviceData, Error, Result};

use crate::codegen::{self, PARAM_WORDS, Plan};
use crate::context::Ctx;
use crate::driver::{self, CUevent, CUstream};
use crate::graph::{self, Graphs};
use crate::memory::{CudaBuffer, Staging, as_cuda};
use crate::module::{CuKernel, Modules};

/// Per-backend execution state (behind the backend's mutex).
pub(crate) struct Exec {
    pub stream: CUstream,
    staging: Option<Staging>,
    /// Partials of multi-stage kernels (split attention); reused by every launch (stream
    /// order serializes them).
    scratch: Option<CudaBuffer>,
    events: Vec<CUevent>,
    /// Host-visible error flag (pinned, mapped): index kernels write 1 on an out-of-range
    /// index; checked and cleared after every execute that bound it.
    flag: Option<ErrorFlag>,
    graphs: Graphs,
    use_graphs: bool,
    pub prof: Profile,
}

// SAFETY: the stream / events are plain driver handles, used only under the mutex.
unsafe impl Send for Exec {}

/// What to do with per-launch GPU timings of one `run`.
pub(crate) enum Timing<'a> {
    None,
    /// Store the GPU ms of every (non-empty) launch, in order.
    Collect(&'a mut Vec<f64>),
}

/// Accumulated GPU times (`LAPID_PROFILE=1`).
#[derive(Default)]
pub(crate) struct Profile {
    pub enabled: bool,
    /// kernel name → (launches, ms)
    by_name: HashMap<String, (usize, f64)>,
    /// kernel kind → (launches, ms), drained by `take_profile`
    by_kind: HashMap<&'static str, (usize, f64)>,
    executes: usize,
    host_ms: f64,
}

impl Profile {
    fn add(&mut self, name: &str, kind: &'static str, ms: f64) {
        let e = self.by_name.entry(name.to_string()).or_default();
        e.0 += 1;
        e.1 += ms;
        let e = self.by_kind.entry(kind).or_default();
        e.0 += 1;
        e.1 += ms;
    }

    pub fn take(&mut self) -> Vec<(&'static str, usize, f64)> {
        let mut v: Vec<_> = self
            .by_kind
            .drain()
            .map(|(k, (n, ms))| (k, n, ms))
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    pub fn reset(&mut self) {
        self.by_name.clear();
        self.by_kind.clear();
        self.executes = 0;
        self.host_ms = 0.0;
    }

    pub fn report(&self) -> String {
        let mut v: Vec<_> = self.by_name.iter().collect();
        v.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let total: f64 = v.iter().map(|(_, (_, ms))| ms).sum();
        let mut s = format!(
            "{} executes, GPU {total:.3} ms, host issue {:.3} ms\n",
            self.executes, self.host_ms
        );
        for (name, (n, ms)) in v {
            s.push_str(&format!(
                "  {ms:10.3} ms {n:7} x  {:8.2} us  {name}\n",
                1e3 * ms / (*n).max(1) as f64
            ));
        }
        s
    }
}

/// The pinned u32 behind `Exec::flag` (host pointer + its device alias).
struct ErrorFlag {
    host: *mut u32,
    dev: u64,
}

impl ErrorFlag {
    /// Read and clear (stream idle).
    fn take(&self) -> u32 {
        // SAFETY: 4 live, aligned pinned bytes; no kernel is running (stream synchronized).
        unsafe {
            let v = std::ptr::read_volatile(self.host);
            std::ptr::write_volatile(self.host, 0);
            v
        }
    }
}

/// One kernel launch (a stage of a schedule launch).
struct Item<'a> {
    plan: Plan,
    launch: &'a Launch,
    li: usize,
}

/// Buffer addresses bound to the kernels of one `run`.
struct Bind<'a> {
    ptrs: &'a [u64],
    scratch: u64,
    flag: u64,
}

/// Launches in the first captured graph chunk of an execute, and in every later one.
const FIRST_CHUNK: usize = 16;
const CHUNK: usize = 64;

/// (first, later) chunk sizes; `LAPID_CUDA_CHUNK=first,later` overrides (experiments).
fn chunk_sizes() -> (usize, usize) {
    static V: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LAPID_CUDA_CHUNK")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or((FIRST_CHUNK, CHUNK))
    })
}

/// `LAPID_CUDA_TRACE=1`: one line per execute (host issue time, GPU wait time).
fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAPID_CUDA_TRACE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

pub(crate) fn profiling_enabled() -> bool {
    std::env::var("LAPID_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0")
}

impl Exec {
    pub fn new(ctx: &Ctx) -> Result<Exec> {
        ctx.bind()?;
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: creates a stream owned by this struct (destroyed in `destroy`).
        ctx.drv.check("cuStreamCreate", unsafe {
            (ctx.drv.cuStreamCreate)(&mut stream, driver::STREAM_NON_BLOCKING)
        })?;
        let prof = Profile {
            enabled: profiling_enabled(),
            ..Profile::default()
        };
        let use_graphs = graph::enabled(ctx.drv);
        Ok(Exec {
            stream,
            staging: None,
            scratch: None,
            events: Vec::new(),
            flag: None,
            graphs: Graphs::default(),
            use_graphs,
            prof,
        })
    }

    /// Release driver objects (context current, stream idle).
    pub fn destroy(&mut self, ctx: &Ctx) {
        let drv = ctx.drv;
        // SAFETY: called once at teardown after a stream synchronize.
        unsafe {
            self.graphs.clear(drv);
            for e in self.events.drain(..) {
                (drv.cuEventDestroy_v2)(e);
            }
            if let Some(mut s) = self.staging.take() {
                s.destroy(drv);
            }
            self.scratch = None;
            if let Some(f) = self.flag.take() {
                (drv.cuMemFreeHost)(f.host as *mut c_void);
            }
            (drv.cuStreamDestroy_v2)(self.stream);
        }
    }

    pub fn sync(&self, ctx: &Ctx) -> Result<()> {
        // SAFETY: our stream.
        ctx.drv.check("cuStreamSynchronize", unsafe {
            (ctx.drv.cuStreamSynchronize)(self.stream)
        })
    }

    fn staging(&mut self, ctx: &Ctx) -> Result<&mut Staging> {
        if self.staging.is_none() {
            self.staging = Some(Staging::new(ctx.drv)?);
        }
        self.staging
            .as_mut()
            .ok_or_else(|| Error::new("cuda", "staging buffer missing"))
    }

    pub fn upload(&mut self, ctx: &Ctx, dst: &CudaBuffer, bytes: &[u8]) -> Result<()> {
        ctx.bind()?;
        let stream = self.stream;
        self.staging(ctx)?.upload(ctx.drv, stream, dst.ptr(), bytes)
    }

    pub fn download(&mut self, ctx: &Ctx, src: &CudaBuffer, out: &mut [u8]) -> Result<()> {
        ctx.bind()?;
        let stream = self.stream;
        self.staging(ctx)?.download(ctx.drv, stream, src.ptr(), out)
    }

    fn ensure_scratch(&mut self, ctx: &Arc<Ctx>, bytes: usize) -> Result<u64> {
        if bytes == 0 {
            return Ok(0);
        }
        if self.scratch.as_ref().is_none_or(|s| s.len_bytes() < bytes) {
            self.scratch = None;
            self.scratch = Some(CudaBuffer::alloc(
                ctx,
                bytes.max(1 << 20).next_power_of_two(),
            )?);
        }
        Ok(self.scratch.as_ref().map(|s| s.ptr()).unwrap_or(0))
    }

    fn ensure_flag(&mut self, ctx: &Ctx) -> Result<u64> {
        if let Some(f) = &self.flag {
            return Ok(f.dev);
        }
        let drv = ctx.drv;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: allocates 4 bytes of pinned host memory owned by this struct (freed in
        // `destroy`).
        drv.check("cuMemAllocHost", unsafe { (drv.cuMemAllocHost_v2)(&mut host, 4) })?;
        let mut dev = 0u64;
        // SAFETY: `host` is a live pinned allocation of this context.
        let r = unsafe { (drv.cuMemHostGetDevicePointer_v2)(&mut dev, host, 0) };
        if let Err(e) = drv.check("cuMemHostGetDevicePointer", r) {
            // SAFETY: freeing the allocation made above.
            unsafe { (drv.cuMemFreeHost)(host) };
            return Err(e);
        }
        let f = ErrorFlag { host: host as *mut u32, dev };
        f.take();
        self.flag = Some(f);
        Ok(dev)
    }

    fn ensure_events(&mut self, ctx: &Ctx, n: usize) -> Result<()> {
        while self.events.len() < n {
            let mut e: CUevent = std::ptr::null_mut();
            // SAFETY: creates a timing event owned by this struct.
            ctx.drv.check("cuEventCreate", unsafe {
                (ctx.drv.cuEventCreate)(&mut e, driver::EVENT_DEFAULT)
            })?;
            self.events.push(e);
        }
        Ok(())
    }

    /// Run `launches` and wait for completion.
    pub fn run(
        &mut self,
        ctx: &Arc<Ctx>,
        modules: &Modules,
        launches: &[Launch],
        buffers: &[DeviceData],
        mut timing: Timing,
    ) -> Result<()> {
        let host_t0 = std::time::Instant::now();
        ctx.bind()?;
        let drv = ctx.drv;
        let mut ptrs: Vec<u64> = Vec::with_capacity(buffers.len());
        let mut mapped: Vec<bool> = Vec::with_capacity(buffers.len());
        for d in buffers {
            let b = as_cuda(d, ctx)?;
            ptrs.push(b.ptr());
            mapped.push(b.mapped_bytes().is_some());
        }

        // plan every stage first (validates the whole schedule before anything runs)
        let mut items: Vec<Item> = Vec::with_capacity(launches.len());
        let mut scratch_bytes = 0usize;
        for (li, l) in launches.iter().enumerate() {
            let mut plan = codegen::plan(&l.kernel, &ctx.caps).map_err(|e| {
                Error::new(
                    e.code,
                    format!("launch {li} ({}): {}", l.kernel.kind(), e.message),
                )
            })?;
            if plan.is_empty() {
                continue;
            }
            if l.inputs.len() != plan.key.num_inputs() {
                return Err(Error::new(
                    "bad_launch",
                    format!(
                        "launch {li} ({}): {} inputs, kernel expects {}",
                        l.kernel.kind(),
                        l.inputs.len(),
                        plan.key.num_inputs()
                    ),
                ));
            }
            if let Some(&id) = l
                .inputs
                .iter()
                .chain(std::iter::once(&l.output))
                .find(|&&id| id >= ptrs.len())
            {
                return Err(Error::new(
                    "bad_launch",
                    format!("launch {li}: buffer id {id} out of range"),
                ));
            }
            if mapped[l.output] {
                return Err(Error::new(
                    "read_only_buffer",
                    format!("launch {li}: output buffer {} is mapped host memory (read-only)", l.output),
                ));
            }
            scratch_bytes = scratch_bytes.max(plan.scratch);
            loop {
                let next = plan.next.take();
                items.push(Item {
                    plan,
                    launch: l,
                    li,
                });
                match next {
                    Some(n) if !n.is_empty() => plan = *n,
                    _ => break,
                }
            }
        }
        if items.is_empty() {
            return Ok(());
        }
        let plan_ms = host_t0.elapsed().as_secs_f64() * 1e3;
        let mut update_ms = 0.0;
        let scratch = self.ensure_scratch(ctx, scratch_bytes)?;
        let checks = items.iter().any(|it| it.plan.key.uses_flag());
        let flag = if checks { self.ensure_flag(ctx)? } else { 0 };
        let timed = self.prof.enabled || matches!(timing, Timing::Collect(_));
        if timed {
            self.ensure_events(ctx, items.len() + 1)?;
            // SAFETY: event + stream owned by us.
            drv.check("cuEventRecord", unsafe {
                (drv.cuEventRecord)(self.events[0], self.stream)
            })?;
        }

        // small multi-launch executes go through a captured graph (not when timing: events
        // between launches would be captured too)
        // Long executes are captured in chunks, each launched as soon as it is captured, so
        // the GPU runs chunk i while the host captures chunk i + 1 (the first chunk is short
        // to start the GPU early).
        let capture = self.use_graphs && !timed && items.len() >= 2;
        let mut kernels = Vec::with_capacity(items.len());
        let bind = Bind { ptrs: &ptrs, scratch, flag };
        if capture {
            let mut start = 0;
            while start < items.len() {
                let len = if start == 0 { chunk_sizes().0 } else { chunk_sizes().1 };
                let end = (start + len).min(items.len());
                let end = if items.len() - end < 2 { items.len() } else { end };
                let chunk = &items[start..end];
                let k0 = kernels.len();
                graph::begin(drv, self.stream)?;
                let issued = self.issue(ctx, modules, chunk, &bind, false, &mut kernels);
                let graph = graph::end(drv, self.stream);
                issued?;
                let graph = graph?;
                // the chunk position is part of the key: two identical chunks of one execute
                // must not share an executable (updating one that is still running waits)
                let key = graph::key(
                    std::iter::once((start, 0)).chain(
                        kernels[k0..]
                            .iter()
                            .zip(chunk)
                            .map(|(k, it)| (k.func as usize, it.plan.block)),
                    ),
                );
                let t = std::time::Instant::now();
                self.graphs.launch(drv, self.stream, key, graph)?;
                update_ms += t.elapsed().as_secs_f64() * 1e3;
                start = end;
            }
        } else {
            self.issue(ctx, modules, &items, &bind, timed, &mut kernels)?;
        }
        let host_ms = host_t0.elapsed().as_secs_f64() * 1e3;
        let sync_t0 = std::time::Instant::now();
        self.sync(ctx).map_err(|e| {
            Error::new(
                e.code,
                format!("execute ({} launches): {}", items.len(), e.message),
            )
        })?;
        if trace_enabled() {
            eprintln!(
                "[lapid-cuda] execute: {} launches, host {host_ms:.3} ms, wait {:.3} ms, graph {capture} (plan {plan_ms:.3}, graph update+launch {update_ms:.3}; graphs updated/instantiated/rejected {:?})",
                items.len(),
                sync_t0.elapsed().as_secs_f64() * 1e3,
                self.graphs.stats
            );
        }
        if checks && self.flag.as_ref().is_some_and(|f| f.take() != 0) {
            return Err(Error::new(
                "index_out_of_range",
                "gather index outside the indexed axis",
            ));
        }
        if timed {
            let mut ms = Vec::with_capacity(items.len());
            for i in 0..items.len() {
                let mut t = 0f32;
                // SAFETY: both events were recorded on our (now idle) stream.
                drv.check("cuEventElapsedTime", unsafe {
                    (drv.cuEventElapsedTime)(&mut t, self.events[i], self.events[i + 1])
                })?;
                ms.push(t as f64);
            }
            if self.prof.enabled {
                for ((it, k), &t) in items.iter().zip(&kernels).zip(&ms) {
                    self.prof.add(&k.name, it.plan.key.label(), t);
                }
                self.prof.host_ms += host_ms;
                self.prof.executes += 1;
            }
            if let Timing::Collect(out) = &mut timing {
                // one entry per launch: sum of its stages
                let mut last = usize::MAX;
                for (it, &t) in items.iter().zip(&ms) {
                    if it.li == last {
                        if let Some(v) = out.last_mut() {
                            *v += t;
                        }
                    } else {
                        out.push(t);
                        last = it.li;
                    }
                }
            }
        }
        Ok(())
    }

    /// Issue every launch on the stream (or into the capture). `kernels` receives the
    /// kernel of each item.
    fn issue(
        &mut self,
        ctx: &Ctx,
        modules: &Modules,
        items: &[Item],
        bind: &Bind,
        timed: bool,
        kernels: &mut Vec<Arc<CuKernel>>,
    ) -> Result<()> {
        let drv = ctx.drv;
        let mut args: Vec<u64> = Vec::with_capacity(16);
        let mut params = [0u32; PARAM_WORDS];
        let mut argp: Vec<*mut c_void> = Vec::with_capacity(17);
        for (ii, it) in items.iter().enumerate() {
            let k = modules.get(ctx, &it.plan.key)?;
            args.clear();
            args.extend(it.launch.inputs.iter().map(|&i| bind.ptrs[i]));
            args.push(bind.ptrs[it.launch.output]);
            if it.plan.key.uses_scratch() {
                args.push(bind.scratch);
            }
            if it.plan.key.uses_flag() {
                args.push(bind.flag);
            }
            params.fill(0);
            params[..it.plan.params.len()].copy_from_slice(&it.plan.params);
            argp.clear();
            for a in args.iter_mut() {
                argp.push(a as *mut u64 as *mut c_void);
            }
            argp.push(params.as_mut_ptr() as *mut c_void);
            let [gx, gy, gz] = it.plan.grid;
            // SAFETY: `argp` points at one u64 per pointer parameter and the params block,
            // matching the kernel signature generated for this key; the driver copies them
            // before returning. All buffers stay alive until the stream is synchronized.
            let r = unsafe {
                (drv.cuLaunchKernel)(
                    k.func,
                    gx,
                    gy,
                    gz,
                    it.plan.block,
                    1,
                    1,
                    0,
                    self.stream,
                    argp.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if r != driver::CUDA_SUCCESS {
                return Err(Error::new(
                    "cuda",
                    format!(
                        "launch {} ({}): cuLaunchKernel grid {:?} block {}: {}",
                        it.li,
                        k.name,
                        it.plan.grid,
                        it.plan.block,
                        drv.error_text(r)
                    ),
                ));
            }
            if timed {
                // SAFETY: as above.
                drv.check("cuEventRecord", unsafe {
                    (drv.cuEventRecord)(self.events[ii + 1], self.stream)
                })?;
            }
            kernels.push(k);
        }
        Ok(())
    }
}
