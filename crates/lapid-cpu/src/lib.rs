//! lapid-cpu: the CPU backend.
//!
//! Executes `lapid_compiler` kernel launches on host memory with SIMD microkernels (NEON /
//! dotprod on aarch64, AVX2+FMA on x86_64, portable fallbacks) and a low-latency fork-join pool (`par`).
//!
//! Module map:
//! * `buffer` / `typed` / `view` — aligned buffers, dtype conversion, strides & bounds;
//! * `expr` / `math` — expression bytecode, vectorizable exp / erf;
//! * `elementwise`, `reduce`, `rows` (softmax, layer norm, gather), `attention`;
//! * `matmul` — GEMM / GEMV / quantized int8-dot paths and their microkernels.

mod attention;
mod attention_vec;
mod buffer;
mod delta;
mod elementwise;
mod expr;
mod expr_int;
mod fuse;
mod index;
mod kcache;
mod kquant;
mod math;
mod matmul;
mod moe;
mod mxfp4;
mod par;
mod qmat;
mod reduce;
mod rows;
mod typed;
mod view;

#[cfg(test)]
mod bench_tests;
#[cfg(test)]
mod bench_ops;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lapid_compiler::{Backend, BufId, Expr, ExprType, Kernel, Launch};
use lapid_tensor::{DType, DeviceData, Error, Result};

pub use buffer::CpuBuffer;

use attention::AttnInputs;
use expr::Program;
use matmul::{BInput, Workspace};
use qmat::QMat;
use rows::Table;

/// CPU features relevant to kernel selection (runtime detected).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Isa {
    pub neon: bool,
    pub dotprod: bool,
    pub fp16: bool,
    pub i8mm: bool,
    pub bf16: bool,
    pub sve: bool,
    pub sve2: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

impl Isa {
    /// Detect features; `LAPID_CPU_SCALAR=1` forces the portable kernels.
    pub(crate) fn detect() -> Isa {
        if std::env::var_os("LAPID_CPU_SCALAR").is_some_and(|v| v != "0") {
            return Isa::default();
        }
        #[allow(unused_mut)]
        let mut isa = Isa::default();
        #[cfg(target_arch = "aarch64")]
        {
            isa.neon = std::arch::is_aarch64_feature_detected!("neon");
            isa.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            isa.fp16 = std::arch::is_aarch64_feature_detected!("fp16");
            // `LAPID_CPU_NO_I8MM=1` falls back to the dotprod kernels (A/B comparisons).
            isa.i8mm = std::arch::is_aarch64_feature_detected!("i8mm")
                && !std::env::var_os("LAPID_CPU_NO_I8MM").is_some_and(|v| v != "0");
            // `LAPID_CPU_NO_BF16=1` keeps f32 activations in quantized GEMMs (A/B checks).
            isa.bf16 = std::arch::is_aarch64_feature_detected!("bf16")
                && !std::env::var_os("LAPID_CPU_NO_BF16").is_some_and(|v| v != "0");
            isa.sve = std::arch::is_aarch64_feature_detected!("sve");
            isa.sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        }
        #[cfg(target_arch = "x86_64")]
        {
            isa.avx2 = std::arch::is_x86_feature_detected!("avx2");
            isa.fma = std::arch::is_x86_feature_detected!("fma");
            isa.f16c = std::arch::is_x86_feature_detected!("f16c");
        }
        isa
    }

    fn names(&self) -> Vec<&'static str> {
        let all = [
            (self.neon, "neon"),
            (self.dotprod, "dotprod"),
            (self.fp16, "fp16"),
            (self.i8mm, "i8mm"),
            (self.bf16, "bf16"),
            (self.sve, "sve"),
            (self.sve2, "sve2"),
            (self.avx2, "avx2"),
            (self.fma, "fma"),
            (self.f16c, "f16c"),
        ];
        all.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect()
    }
}

/// Compiled-expression cache keyed by the expression and its raw-input mask.
#[derive(Default)]
pub(crate) struct ProgramCache {
    map: Mutex<HashMap<ProgramKey, Arc<Program>>>,
    /// Per-spec launch data (see `kcache`).
    specs: kcache::SpecCache,
}

/// Expression + raw-input mask.
type ProgramKey = (Expr, Vec<bool>);

const PROGRAM_CACHE_MAX: usize = 4096;

impl ProgramCache {
    pub(crate) fn get(&self, e: &Expr) -> Arc<Program> {
        self.get_raw(e, &[])
    }

    /// Program for `e` whose inputs flagged in `raw` hold raw i32 words.
    pub(crate) fn get_raw(&self, e: &Expr, raw: &[bool]) -> Arc<Program> {
        let raw = if raw.iter().any(|&r| r) { raw.to_vec() } else { Vec::new() };
        let key = (e.clone(), raw);
        let mut map = self.map.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = map.get(&key) {
            return p.clone();
        }
        if map.len() >= PROGRAM_CACHE_MAX {
            map.clear();
        }
        let p = Arc::new(Program::compile_raw(e, &key.1));
        map.insert(key, p.clone());
        p
    }
}

/// Per-execute kernel context.
pub(crate) struct Ctx<'a> {
    pub threads: usize,
    pub isa: Isa,
    programs: &'a ProgramCache,
}

impl Ctx<'_> {
    /// Program for a kernel expression over inputs of `in_dtypes`; `raw` marks the inputs
    /// that are bound as raw words (see `expr_int::raw_inputs`).
    pub(crate) fn program_typed(&self, e: &Expr, in_dtypes: &[DType], raw: &[bool]) -> Arc<Program> {
        if !e.has_int() {
            return self.programs.get(e);
        }
        self.programs.get_raw(&expr_int::normalize(e, in_dtypes), raw)
    }
}

pub struct CpuBackend {
    pool: par::Pool,
    threads: usize,
    isa: Isa,
    programs: ProgramCache,
    ws: Mutex<Workspace>,
    recycler: Arc<buffer::Recycler>,
    /// Per-kind launch time totals when `LAPID_PROFILE` is set.
    profile: Option<Mutex<HashMap<&'static str, (usize, f64)>>>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        CpuBackend::new()
    }
}

impl CpuBackend {
    /// All available cores (`LAPID_THREADS` overrides).
    pub fn new() -> CpuBackend {
        let n = std::env::var("LAPID_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        CpuBackend::with_threads(n)
    }

    pub fn with_threads(n: usize) -> CpuBackend {
        CpuBackend::build(n, Isa::detect())
    }

    pub(crate) fn build(n: usize, isa: Isa) -> CpuBackend {
        let n = n.max(1);
        let pool = par::Pool::new(n);
        // If threads could not be spawned everything runs on the caller's thread.
        let threads = pool.threads();
        let profile = std::env::var("LAPID_PROFILE")
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .then(|| Mutex::new(HashMap::new()));
        CpuBackend { pool, threads, isa, programs: ProgramCache::default(), ws: Mutex::new(Workspace::default()), recycler: Arc::default(), profile }
    }

    /// Detected SIMD features, e.g. "neon, dotprod, fp16".
    pub fn features() -> String {
        let names = Isa::detect().names();
        if names.is_empty() { "scalar".to_string() } else { names.join(", ") }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }
}

fn as_cpu(b: &DeviceData) -> Result<&CpuBuffer> {
    b.as_any().downcast_ref::<CpuBuffer>().ok_or_else(|| {
        Error::new("backend", format!("cpu backend got a buffer owned by backend '{}'", b.backend()))
    })
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_name(&self) -> String {
        let names = self.isa.names();
        let feats = if names.is_empty() { "scalar".to_string() } else { names.join(", ") };
        format!("cpu: {} threads ({feats})", self.threads)
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceData> {
        Ok(Arc::new(CpuBuffer::recycled(bytes, &self.recycler)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceData> {
        Ok(Arc::new(CpuBuffer::from_bytes(bytes)?))
    }

    fn download(&self, buf: &DeviceData) -> Result<Vec<u8>> {
        let b = as_cpu(buf)?;
        // SAFETY: execute is synchronous, no kernel is writing this buffer now.
        Ok(unsafe { b.bytes() }.to_vec())
    }

    fn download_into(&self, buf: &DeviceData, out: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let b = as_cpu(buf)?;
        // SAFETY: execute is synchronous, no kernel is writing this buffer now.
        let src = unsafe { b.bytes() };
        if src.len() < out.len() {
            return Err(lapid_tensor::Error::new("backend", format!("cpu: buffer has {} bytes, {} requested", src.len(), out.len())));
        }
        // SAFETY: `src` has at least `out.len()` initialized bytes; distinct allocations.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr().cast::<u8>(), out.len()) };
        Ok(())
    }

    fn prepare(&self, launches: &[Launch]) -> Result<()> {
        for l in launches {
            match &l.kernel {
                Kernel::Elementwise(s, _) => {
                    self.programs.get(&s.expr);
                }
                Kernel::Reduce(s, _) => {
                    self.programs.get(&s.expr);
                }
                Kernel::MatMul(s, _) => {
                    if let Some(e) = &s.epilogue {
                        self.programs.get(e);
                    }
                }
                _ => {}
            }
        }
        math::erf_table();
        Ok(())
    }

    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        let Some(p) = &self.profile else { return Vec::new() };
        let mut g = p.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = g.drain().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    fn execute(&self, launches: &[Launch], buffers: &[DeviceData]) -> Result<()> {
        let bufs = buffers.iter().map(as_cpu).collect::<Result<Vec<_>>>()?;
        let mut guard = self.ws.lock().unwrap_or_else(|p| p.into_inner());
        let ws: &mut Workspace = &mut guard;
        let ctx = Ctx { threads: self.threads, isa: self.isa, programs: &self.programs };
        let profile = self.profile.as_ref();
        let run = move || -> Result<()> {
            // Launches taken over by an earlier one (see `fuse`).
            let mut skip: Vec<bool> = Vec::new();
            for (li, l) in launches.iter().enumerate() {
                if skip.get(li).copied().unwrap_or(false) {
                    continue;
                }
                let t0 = profile.map(|_| std::time::Instant::now());
                let _k = tracing::trace_span!("cpu", kind = profile_kind(&l.kernel)).entered();
                let copies = match l.kernel {
                    Kernel::GatedDeltaNet(..) => {
                        let c = fuse::gdn_copies(launches, li, &bufs);
                        for (j, _) in c.state.iter().chain(c.o.iter()) {
                            if skip.is_empty() {
                                skip = vec![false; launches.len()];
                            }
                            skip[*j] = true;
                        }
                        c
                    }
                    _ => fuse::GdnCopies::default(),
                };
                run_launch(&ctx, ws, l, &bufs, &copies).map_err(|e| {
                    Error::new(e.code, format!("launch {li} ({}): {}", l.kernel.kind(), e.message))
                })?;
                if let (Some(p), Some(t0)) = (profile, t0) {
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    let mut g = p.lock().unwrap_or_else(|e| e.into_inner());
                    let e = g.entry(profile_kind(&l.kernel)).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += ms;
                }
            }
            Ok(())
        };
        self.pool.install(run)
    }
}

/// Profile bucket: kernel kind, refined by the code path for matmul / attention.
fn profile_kind(k: &Kernel) -> &'static str {
    match k {
        Kernel::MatMul(s, d) => match (qmat::format_of(s.b_dtype).is_some(), d.batch * d.m <= 8) {
            (true, true) => "matmul.q.gemv",
            (true, false) => "matmul.q.gemm",
            (false, true) => "matmul.f.gemv",
            (false, false) => "matmul.f.gemm",
        },
        Kernel::Attention(_, d) if d.sq == 1 => "attention.sq1",
        Kernel::Elementwise(_, d) => match d.shape.iter().product::<usize>() {
            n if n < 1 << 10 => "elementwise.<1k",
            n if n < 1 << 14 => "elementwise.<16k",
            n if n < 1 << 17 => "elementwise.<128k",
            _ => "elementwise.big",
        },
        k => k.kind(),
    }
}

fn run_launch(ctx: &Ctx, ws: &mut Workspace, l: &Launch, bufs: &[&CpuBuffer], copies: &fuse::GdnCopies) -> Result<()> {
    let get = |id: BufId| {
        bufs.get(id).copied().ok_or_else(|| {
            Error::new("invalid_launch", format!("buffer id {id} out of range ({} buffers)", bufs.len()))
        })
    };
    let out_buf = get(l.output)?;
    let mut ins = Vec::with_capacity(l.inputs.len());
    for &id in &l.inputs {
        let b = get(id)?;
        if b.as_ptr() == out_buf.as_ptr() {
            return Err(Error::new(
                "buffer_aliasing",
                format!("input buffer {id} is the output buffer {}", l.output),
            ));
        }
        ins.push(b);
    }
    let expect = |want: usize| -> Result<()> {
        if ins.len() != want {
            return Err(Error::new("invalid_launch", format!("expected {want} inputs, got {}", ins.len())));
        }
        Ok(())
    };
    // SAFETY: every input buffer differs from the output (checked above) and launches run
    // sequentially, so these shared views never observe concurrent writes.
    let src = |i: usize, dt: DType| unsafe { ins[i].src(dt) };
    let srcs = |dts: &[DType], first: usize| -> Result<Vec<_>> {
        dts.iter().enumerate().map(|(i, &dt)| src(first + i, dt)).collect()
    };

    match &l.kernel {
        Kernel::Elementwise(spec, dims) => {
            expect(spec.in_dtypes.len())?;
            // Raw-word inputs / outputs are bound as f32 lanes holding the bit patterns.
            let c = ctx.compiled_ew(spec);
            let inputs = srcs(&c.dts, 0)?;
            let int_root = spec.out_dtype == DType::I32 && spec.expr.ty() == Some(ExprType::Int);
            let out_dt = if int_root { DType::F32 } else { spec.out_dtype };
            elementwise::run(ctx, spec, dims, &inputs, &out_buf.dst(out_dt)?, &c)
        }
        Kernel::Reduce(spec, dims) => {
            expect(spec.in_dtypes.len())?;
            let c = ctx.compiled_reduce(spec);
            let inputs = srcs(&c.dts, 0)?;
            reduce::run(ctx, spec, dims, &inputs, &out_buf.dst(spec.out_dtype)?, &c)
        }
        Kernel::MatMul(spec, dims) => {
            expect(2 + spec.extra_dtypes.len())?;
            let a = src(0, spec.a_dtype)?;
            let b = match qmat::format_of(spec.b_dtype) {
                // SAFETY: as for `src`.
                Some(fmt) => BInput::Quant(QMat::new(fmt, unsafe { ins[1].bytes() }, dims.n, dims.k)?),
                None => BInput::Plain(src(1, spec.b_dtype)?),
            };
            // Extras read via `LoadBits` are bound as raw words (same mask as matmul::run).
            let raw = matmul::epilogue_raw(spec);
            let edts: Vec<DType> = spec
                .extra_dtypes
                .iter()
                .enumerate()
                .map(|(i, &d)| if raw.get(i + 1).copied().unwrap_or(false) { DType::F32 } else { d })
                .collect();
            let extras = srcs(&edts, 2)?;
            matmul::run(ctx, ws, spec, dims, &a, &b, &extras, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::Gather(spec, dims) => {
            expect(2)?;
            let table = match qmat::format_of(spec.table_dtype) {
                // SAFETY: as for `src`.
                Some(fmt) => Table::Quant(QMat::new(fmt, unsafe { ins[0].bytes() }, dims.rows, dims.cols)?),
                None => Table::Plain(src(0, spec.table_dtype)?),
            };
            let idx = src(1, DType::I32)?;
            rows::gather(ctx, spec, dims, &table, &idx, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::Softmax(spec, dims) => {
            expect(1)?;
            rows::softmax(ctx, spec, dims, &src(0, spec.in_dtype)?, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::LayerNorm(spec, dims) => {
            expect(3)?;
            let x = src(0, spec.in_dtype)?;
            let w = src(1, spec.param_dtype)?;
            let b = src(2, spec.param_dtype)?;
            rows::layer_norm(ctx, spec, dims, &x, &w, &b, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::RmsNorm(spec, dims) => {
            expect(2)?;
            let x = src(0, spec.in_dtype)?;
            let w = src(1, spec.param_dtype)?;
            rows::rms_norm(ctx, dims, &x, &w, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::GatherAxis(spec, dims) => {
            expect(2)?;
            index::gather_axis(ctx, dims, &src(0, spec.dtype)?, &src(1, DType::I32)?, &out_buf.dst(spec.dtype)?)
        }
        Kernel::TopK(spec, dims) => {
            expect(1)?;
            index::topk(ctx, dims, &src(0, spec.in_dtype)?, &out_buf.dst(DType::F32)?)
        }
        Kernel::MatMulId(spec, dims) => {
            expect(3)?;
            let x = src(0, spec.x_dtype)?;
            let ids = src(2, DType::I32)?;
            moe::matmul_id(ctx, ws, spec, dims, &x, ins[1], &ids, &out_buf.dst(spec.out_dtype)?)
        }
        Kernel::GatedDeltaNet(spec, dims) => {
            expect(6)?;
            let inputs = srcs(&[DType::F32; 6], 0)?;
            // The state destination may be the state input itself (the runtime recycles a
            // dropped state buffer as the new one): every element is read before the same
            // task overwrites it at the same position.
            let xdst = |c: Option<(usize, BufId)>, may_alias: Option<usize>| -> Result<Option<typed::Dst>> {
                let Some((_, id)) = c else { return Ok(None) };
                let b = get(id)?;
                let alias = ins.iter().position(|i| i.as_ptr() == b.as_ptr()).filter(|&k| Some(k) != may_alias);
                if b.as_ptr() == out_buf.as_ptr() || alias.is_some() {
                    return Err(Error::new(
                        "buffer_aliasing",
                        format!("cpu gated_delta_net: copy destination b{id} aliases the output or input {alias:?}"),
                    ));
                }
                b.dst(DType::F32).map(Some)
            };
            let extra = delta::Extra { state: xdst(copies.state, Some(5))?, o: xdst(copies.o, None)? };
            delta::gated_delta_net(ctx, dims, &inputs, &out_buf.dst(spec.out_dtype)?, &extra)
        }
        Kernel::Attention(spec, dims) => {
            expect(3 + spec.mask_dtype.is_some() as usize)?;
            let inputs = AttnInputs {
                q: src(0, spec.dtype)?,
                k: src(1, spec.dtype)?,
                v: src(2, spec.dtype)?,
                mask: match spec.mask_dtype {
                    Some(dt) => Some(src(3, dt)?),
                    None => None,
                },
            };
            attention::run(ctx, ws, spec, dims, &inputs, &out_buf.dst(spec.out_dtype)?)
        }
    }
}
