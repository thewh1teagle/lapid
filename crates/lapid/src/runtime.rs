//! Runtime behind a `Device`: one backend, an upload cache for host leaves, a pool of
//! temp buffers, and the `eval` pipeline (compile → resolve buffers → execute → wrap).
//!
//! Runtimes are shared process-wide per physical device (`"cpu"`, `"vulkan:<adapter>"`):
//! constructing `Device::cpu()` twice yields the same runtime, so weights are uploaded
//! once and device buffers from one handle are valid in the other.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use lapid_compiler::{Backend, BufferSource, CompileOptions};
use lapid_tensor::{DType, DeviceData, Error, Op, Result, Tensor, bail};

mod cache;
mod clock;
mod convert;
mod plan;
mod pool;
mod predict;

use cache::UploadCache;
use clock::Instant;
use plan::{GraphKey, LeafRef, Lookup, Plan, PlanCache, PlanSrc};
use pool::Pool;

/// Lock ignoring poisoning: every structure guarded here stays consistent across panics
/// (they only hold caches / free lists).
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `Runtime::host_leaf` resolved a host leaf to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostLeaf {
    /// A file-backed weight uploaded once and kept.
    Pinned,
    /// A weight read in place from its mapping by the CPU.
    Mapped,
    /// A weight read in place by a GPU: lock its pages in memory around every execute.
    MappedGpu,
    /// Other host data (transient upload cache).
    Transient,
}

/// A graph keyed and planned, ready to run.
struct Prepared {
    key: GraphKey,
    plan: Arc<Plan>,
    hit: Option<Lookup>,
    t0: Instant,
    t_compile: Duration,
}

/// Live runtimes, used to share backends between `Device` handles and to find the owner
/// of a foreign device buffer.
static REGISTRY: Mutex<Vec<Weak<Runtime>>> = Mutex::new(Vec::new());

pub(crate) struct Runtime {
    /// Registry key: "cpu" or "vulkan:<adapter index>".
    key: String,
    pub(crate) backend: Arc<dyn Backend>,
    cache: UploadCache,
    pool: Pool,
    plans: Mutex<PlanCache>,
    debug: AtomicBool,
    profile: AtomicBool,
}

/// Get the live runtime for `key` or create it with `make`.
pub(crate) fn shared<E>(
    key: &str,
    make: impl FnOnce() -> std::result::Result<Arc<dyn Backend>, E>,
) -> std::result::Result<Arc<Runtime>, E> {
    let mut reg = lock(&REGISTRY);
    reg.retain(|w| w.strong_count() > 0);
    if let Some(rt) = reg.iter().filter_map(Weak::upgrade).find(|rt| rt.key == key) {
        return Ok(rt);
    }
    let rt = Arc::new(Runtime::new(key.to_string(), make()?));
    reg.push(Arc::downgrade(&rt));
    Ok(rt)
}

/// `shared` for backends whose construction cannot fail.
pub(crate) fn shared_infallible(key: &str, make: impl FnOnce() -> Arc<dyn Backend>) -> Arc<Runtime> {
    match shared::<Infallible>(key, || Ok(make())) {
        Ok(rt) => rt,
        Err(never) => match never {},
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

impl Runtime {
    fn new(key: String, backend: Arc<dyn Backend>) -> Runtime {
        Runtime {
            key,
            backend,
            cache: UploadCache::new(),
            pool: Pool::new(),
            plans: Mutex::new(PlanCache::default()),
            debug: AtomicBool::new(env_flag("LAPID_DEBUG")),
            profile: AtomicBool::new(env_flag("LAPID_PROFILE")),
        }
    }

    pub(crate) fn set_debug(&self, on: bool) {
        self.debug.store(on, Ordering::Relaxed);
    }

    fn debug(&self) -> bool {
        self.debug.load(Ordering::Relaxed)
    }

    fn owns(&self, buf: &DeviceData) -> bool {
        buf.backend() == self.backend.name()
    }

    /// Compile and run `outputs` together; see `Device::eval`.
    pub(crate) fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let _eval = tracing::trace_span!("eval").entered();
        match self.prepare(outputs)? {
            None => Ok(outputs.iter().map(|t| (*t).clone()).collect()),
            Some(p) => self.execute(p),
        }
    }

    /// `eval` taking the outputs: once the schedule is known the handles are dropped, so a
    /// graph nobody else holds is torn down before execution (while its nodes are still in
    /// cache) instead of after it.
    pub(crate) fn eval_owned(&self, outputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let _eval = tracing::trace_span!("eval").entered();
        let prepared = {
            let refs: Vec<&Tensor> = outputs.iter().collect();
            self.prepare(&refs)?
        };
        match prepared {
            None => Ok(outputs),
            Some(p) => {
                tracing::trace_span!("eval.release").in_scope(|| drop(outputs));
                self.execute(p)
            }
        }
    }

    /// Key and plan of `outputs`; `None` when everything is already realized here.
    fn prepare(&self, outputs: &[&Tensor]) -> Result<Option<Prepared>> {
        // Fast path: everything requested is already realized on this device.
        if outputs.is_empty() || outputs.iter().all(|t| matches!(t.op(), Op::Device(d) if self.owns(d))) {
            return Ok(None);
        }
        let debug = self.debug();
        let t0 = Instant::now();
        let plan_span = tracing::trace_span!("eval.plan").entered();
        let key = GraphKey::new(outputs);
        let cached = lock(&self.plans).get(&key);
        let hit = cached.as_ref().map(|c| c.1);
        let plan = match cached {
            Some((p, _)) => p,
            None => {
                let owned: Vec<Tensor> = outputs.iter().map(|t| (*t).clone()).collect();
                let schedule = lapid_compiler::compile(&owned, &CompileOptions::default())?;
                if debug {
                    crate::log::emit(&format!(
                        "[lapid] {}: compiled {} launches, {} buffers, {} temp slots in {:.3} ms\n{}",
                        self.backend.name(),
                        schedule.launches.len(),
                        schedule.buffers.len(),
                        schedule.slot_bytes.len(),
                        ms(t0.elapsed()),
                        lapid_compiler::dump(&schedule)
                    ));
                }
                let plan = Arc::new(Plan::new(schedule, &key));
                lock(&self.plans).insert(&key, plan.clone());
                plan
            }
        };
        let t_compile = t0.elapsed();
        drop(plan_span);
        Ok(Some(Prepared { key, plan, hit, t0, t_compile }))
    }

    /// Run a prepared plan and wrap its outputs.
    fn execute(&self, p: Prepared) -> Result<Vec<Tensor>> {
        let Prepared { key, plan, hit, t0, t_compile } = p;
        let debug = self.debug();
        let profile = debug || self.profile.load(Ordering::Relaxed);

        let temps = self.pool.take_slots(&*self.backend, &plan.slot_bytes)?;
        let result = self.run(&plan, &key.leaves, &temps, debug);
        // `execute` waits for completion, so the temps are free again.
        self.pool.give_slots(&plan.slot_bytes, temps);
        let (buffers, t_resolve, t_exec) = result?;

        if profile {
            let kinds = self.backend.take_profile();
            let mut line = format!(
                "[lapid-profile] {} eval: {} nodes, {} launches, compile {:.3} ms ({}), resolve {:.3} ms, execute {:.3} ms, total {:.3} ms",
                self.backend.name(),
                key.nodes,
                plan.launches.len(),
                ms(t_compile),
                match hit {
                    Some(Lookup::Exact) => "cached",
                    Some(Lookup::Predicted) => "predicted",
                    None => "miss",
                },
                ms(t_resolve),
                ms(t_exec),
                ms(t0.elapsed())
            );
            if !kinds.is_empty() {
                line.push_str("\n[lapid-profile]   kernels:");
                for (kind, n, t) in kinds {
                    line.push_str(&format!(" {kind} {n}x {t:.3} ms;"));
                }
            }
            crate::log::emit(&line);
        }
        plan.outputs
            .iter()
            .map(|o| {
                let buf = buffers.get(o.buf).ok_or_else(|| {
                    Error::new("internal", format!("schedule output refers to missing buffer {}", o.buf))
                })?;
                Ok(Tensor::from_device(buf.clone(), &o.shape, o.dtype))
            })
            .collect()
    }

    fn resolve_source(&self, src: &BufferSource, bytes: usize) -> Result<DeviceData> {
        Ok(match src {
            BufferSource::Host(h) => self.resolve_host(h)?,
            BufferSource::Device(d) => self.resolve_device(d)?,
            BufferSource::Temp { .. } => bail!("internal", "unbound temp buffer"),
            // Never hand a zero-byte allocation to a backend.
            BufferSource::Output => self.backend.alloc(bytes.max(4))?,
        })
    }

    fn resolve_host(&self, h: &lapid_tensor::HostData) -> Result<DeviceData> {
        let (d, kind) = self.host_leaf(h)?;
        if kind == HostLeaf::MappedGpu {
            // Outside an eval nothing can keep the pages locked until the GPU reads them.
            crate::residency::populate_many(&crate::residency::read_slices(h));
        }
        Ok(d)
    }

    /// Device buffer of a (shared) host leaf and what it is. Weights are mapped in place
    /// under `Residency::Mmap` when the backend can, else uploaded once and pinned; other
    /// host data goes through the transient upload cache.
    fn host_leaf(&self, h: &lapid_tensor::HostData) -> Result<(DeviceData, HostLeaf)> {
        match crate::file::weight_residency(h.key()) {
            Some(r) => {
                if r == crate::Residency::Mmap
                    && !h.is_empty()
                    && let Some(d) = self.backend.map_host(h)?
                {
                    // A GPU must not page-fault on the mapping (slow, and wrong data on
                    // some drivers): the host reads in and locks the pages around execute.
                    let gpu = self.backend.name() != "cpu";
                    return Ok((d, if gpu { HostLeaf::MappedGpu } else { HostLeaf::Mapped }));
                }
                Ok((self.cache.get_or_upload(h, &*self.backend, true)?, HostLeaf::Pinned))
            }
            None => Ok((self.cache.get_or_upload(h, &*self.backend, false)?, HostLeaf::Transient)),
        }
    }

    fn resolve_device(&self, d: &DeviceData) -> Result<DeviceData> {
        if self.owns(d) { Ok(d.clone()) } else { self.backend.upload(&download_foreign(self, d)?) }
    }

    /// Resolve every plan buffer to a device buffer (leaves rebound to this graph's) and
    /// execute the launches. Returns the buffers and the resolve / execute times.
    fn run(
        &self,
        plan: &Plan,
        leaves: &[LeafRef],
        temps: &[DeviceData],
        debug: bool,
    ) -> Result<(Vec<DeviceData>, Duration, Duration)> {
        let t0 = Instant::now();
        let resolve_span = tracing::trace_span!("eval.resolve").entered();
        // Each distinct leaf is resolved once even if several buffers refer to it; weights
        // (file-backed host data) come from the plan's memo when it saw the same data.
        let mut leaf_bufs: Vec<Option<DeviceData>> = vec![None; leaves.len()];
        let mut memo = plan.memo.try_lock().ok();
        if let Some(m) = memo.as_deref_mut() {
            m.resize(leaves.len(), None);
            for ((slot, leaf), entry) in leaf_bufs.iter_mut().zip(leaves).zip(m.iter()) {
                if let (LeafRef::Host(h), Some((mh, md))) = (leaf, entry)
                    && mh.key() == h.key()
                {
                    *slot = Some(md.clone());
                }
            }
        }
        let mut buffers = Vec::with_capacity(plan.bufs.len());
        // Mapped weights to populate before a GPU executes (not memoized: every eval).
        let mut populate: Vec<&[u8]> = Vec::new();
        for (id, b) in plan.bufs.iter().enumerate() {
            let buf = match &b.src {
                PlanSrc::Leaf(i) => {
                    let Some(slot) = leaf_bufs.get_mut(*i) else {
                        bail!("internal", "buffer {id} refers to missing leaf {i}");
                    };
                    match slot {
                        Some(d) => d.clone(),
                        None => {
                            let d = match &leaves[*i] {
                                // Nothing outside this eval holds the data (the graph was
                                // released): it can never be passed again, so caching its
                                // upload would only cost a cache entry.
                                LeafRef::Host(h) if h.is_unique() => {
                                    let b = h.bytes();
                                    if b.is_empty() { self.backend.upload(&[0u8; 4])? } else { self.backend.upload(b)? }
                                }
                                LeafRef::Host(h) => {
                                    let (d, kind) = self.host_leaf(h)?;
                                    match kind {
                                        HostLeaf::Pinned | HostLeaf::Mapped => {
                                            if let Some(m) = memo.as_deref_mut() && let Some(e) = m.get_mut(*i) {
                                                *e = Some((h.clone(), d.clone()));
                                            }
                                        }
                                        HostLeaf::MappedGpu => populate.extend(crate::residency::read_slices(h)),
                                        HostLeaf::Transient => {}
                                    }
                                    d
                                }
                                LeafRef::Device(d) => self.resolve_device(d)?,
                            };
                            *slot = Some(d.clone());
                            d
                        }
                    }
                }
                PlanSrc::Temp(slot) => match temps.get(*slot) {
                    Some(b) => b.clone(),
                    None => bail!("internal", "buffer {id} refers to missing temp slot {slot}"),
                },
                PlanSrc::Output => self.backend.alloc(b.bytes.max(4))?,
                PlanSrc::Fixed(src) => self.resolve_source(src, b.bytes)?,
            };
            buffers.push(buf);
        }
        drop(memo);
        // Locked until the launches completed (dropped at the end of this function).
        let _pinned = (!populate.is_empty()).then(|| {
            let _p = tracing::trace_span!("eval.pin").entered();
            crate::residency::pin_many(&populate)
        });
        let t_resolve = t0.elapsed();
        drop(resolve_span);
        let _exec = tracing::trace_span!("eval.execute").entered();

        let t1 = Instant::now();
        if debug {
            // Run launches one by one to time them (same semantics: each `execute`
            // completes before the next starts).
            for (i, launch) in plan.launches.iter().enumerate() {
                let tl = Instant::now();
                self.backend.execute(std::slice::from_ref(launch), &buffers)?;
                crate::log::emit(&format!("[lapid]   #{i:<4} {:<12} {:>9.3} ms", launch.kernel.kind(), ms(tl.elapsed())));
            }
        } else {
            self.backend.execute(&plan.launches, &buffers)?;
        }
        let t_exec = t1.elapsed();
        if debug {
            crate::log::emit(&format!("[lapid] resolve buffers {:.3} ms, execute {:.3} ms", ms(t_resolve), ms(t_exec)));
        }
        Ok((buffers, t_resolve, t_exec))
    }

    /// Upload a host leaf now (cached), or realize anything else on this device.
    pub(crate) fn upload(&self, t: &Tensor) -> Result<Tensor> {
        match t.op() {
            Op::Host(h) => {
                let buf = self.resolve_host(h)?;
                Ok(Tensor::from_device(buf, t.shape(), t.dtype()))
            }
            Op::Device(d) if self.owns(d) => Ok(t.clone()),
            Op::Device(d) => {
                let bytes = download_foreign(self, d)?;
                Ok(Tensor::from_device(self.backend.upload(&bytes)?, t.shape(), t.dtype()))
            }
            _ => self.eval(&[t])?.pop().ok_or_else(|| Error::new("internal", "eval returned no output")),
        }
    }

    /// Bytes of `t` (contiguous row-major, exactly `storage_bytes`) on the host.
    fn host_bytes(&self, t: &Tensor) -> Result<Vec<u8>> {
        let want = expected_bytes(t)?;
        let mut bytes = match t.op() {
            Op::Host(h) => h.bytes().to_vec(),
            Op::Device(d) if self.owns(d) => self.backend.download(d)?,
            Op::Device(d) => download_foreign(self, d)?,
            _ => {
                let realized = self.eval(&[t])?;
                let Some(Op::Device(d)) = realized.first().map(Tensor::op) else {
                    bail!("internal", "eval did not return a device tensor");
                };
                self.backend.download(d)?
            }
        };
        if bytes.len() < want {
            bail!("backend", "downloaded {} bytes, expected {want}", bytes.len());
        }
        bytes.truncate(want);
        Ok(bytes)
    }

    pub(crate) fn to_vec_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        if let Op::Host(h) = t.op() {
            // No device round trip for host leaves.
            let want = expected_bytes(t)?;
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return convert::to_f32(bytes, t.dtype(), t.shape());
        }
        if let (Op::Device(d), DType::F32) = (t.op(), t.dtype())
            && self.owns(d)
        {
            // Straight into the result (no intermediate byte vector).
            let n = t.numel();
            let mut v: Vec<f32> = Vec::with_capacity(n);
            let out = v.spare_capacity_mut();
            // SAFETY: f32 has no invalid bit patterns; viewing the uninitialized slots as
            // uninitialized bytes is sound.
            let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(), n * 4) };
            self.backend.download_into(d, bytes)?;
            // SAFETY: `download_into` initialized all `n * 4` bytes.
            unsafe { v.set_len(n) };
            return Ok(v);
        }
        convert::to_f32(&self.host_bytes(t)?, t.dtype(), t.shape())
    }

    /// Raw storage bytes of `t` in its own dtype (quantized: the lapid-quant layout).
    pub(crate) fn to_vec_bytes(&self, t: &Tensor) -> Result<Vec<u8>> {
        let want = expected_bytes(t)?;
        if let Op::Host(h) = t.op() {
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return Ok(bytes.to_vec());
        }
        let mut bytes = self.host_bytes(t)?;
        bytes.truncate(want);
        Ok(bytes)
    }

    pub(crate) fn to_vec_i32(&self, t: &Tensor) -> Result<Vec<i32>> {
        if let Op::Host(h) = t.op() {
            let want = expected_bytes(t)?;
            let Some(bytes) = h.bytes().get(..want) else {
                bail!("shape_mismatch", "host leaf is smaller than its shape {:?}", t.shape());
            };
            return convert::to_i32(bytes, t.dtype(), t.shape());
        }
        convert::to_i32(&self.host_bytes(t)?, t.dtype(), t.shape())
    }
}

/// Storage size of `t`, validated so quantized shapes never panic.
fn expected_bytes(t: &Tensor) -> Result<usize> {
    let dt = t.dtype();
    let block = dt.quant_format().map_or(DType::QUANT_GROUP, |f| f.block_elems());
    if dt.is_quant() && (!(2..=3).contains(&t.rank()) || t.shape()[t.rank() - 1] % block != 0) {
        bail!("shape_mismatch", "quantized {} tensor has invalid shape {:?}", dt.name(), t.shape());
    }
    Ok(dt.storage_bytes(t.shape()))
}

/// Download a buffer owned by another backend through a live runtime of that backend.
///
/// Limitation: buffers are matched by backend name only; with several live Vulkan
/// devices the first one that accepts the buffer is used.
fn download_foreign(me: &Runtime, buf: &DeviceData) -> Result<Vec<u8>> {
    let name = buf.backend();
    let owners: Vec<Arc<Runtime>> = lock(&REGISTRY)
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|rt| rt.backend.name() == name && !std::ptr::eq(Arc::as_ptr(rt), me))
        .collect();
    let mut last_err = None;
    for rt in owners {
        match rt.backend.download(buf) {
            Ok(bytes) => return Ok(bytes),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Error::new(
            "backend",
            format!("cannot transfer a '{name}' buffer to '{}': no live '{name}' device", me.backend.name()),
        )
    }))
}
