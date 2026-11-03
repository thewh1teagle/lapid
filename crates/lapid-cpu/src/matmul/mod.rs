//! MatMul: `out[bt, i, j] = epilogue(sum_k A[bt, i, k] * B[bt, k, j], extras...)`.
//!
//! Paths:
//! * quantized B (`int8` / `int4` / K-quants / MXFP4), m > 8 rows: activations x dequantized weights,
//!   per element (no int8 activation blocks): bf16 `bfmmla` with f32 accumulation where the
//!   CPU has BF16 (`qbf16.rs`, m >= 16), else f32 panels + the f32 GEMM microkernel
//!   (`qdeq.rs`). `LAPID_CPU_INT8_ACT=1` selects the int8 activation paths below instead;
//! * quantized B, few rows (decoder): A rows quantized to int8 per 32-block once per launch,
//!   int8 dot products (`sdot` on aarch64+dotprod, AVX2 `maddubs`, scalar fallback), 4x4
//!   register tiles, tasks over (row block, column block) — `quant.rs`; `qgemm.rs` (i8mm);
//! * float B with small m (decoder): GEMV-style, parallel over column blocks — `gemv.rs`;
//! * float B with larger m (encoder): packed panels + 6x16 microkernel — `gemm.rs`.
//!
//! The epilogue expression is applied on contiguous row segments of the accumulator tile.

mod gemm;
mod gemv;
mod hgemv;
#[cfg(target_arch = "aarch64")]
pub(crate) mod kdot;
pub(crate) mod kernels;
mod qbf16;
mod qdeq;
mod qgemm;
mod qkernels;
mod quant;

#[cfg(test)]
pub(crate) use quant::quantize_row;

use std::sync::Arc;

use lapid_compiler::{MatMulDims, MatMulSpec, View};
use lapid_tensor::{DType, Error, Result};

use crate::Ctx;
use crate::expr::{CHUNK, Program, Regs};
use crate::qmat::QMat;
use crate::typed::{Dst, Src};
use crate::view::{check_strided, check_view};

pub(crate) use kernels::{MR, MicroFn, NR};

/// m up to which float matmuls use the GEMV path.
const SMALL_M: usize = 8;

#[cfg(test)]
mod kbench_tests;
#[cfg(test)]
mod matmul_tests;

/// `LAPID_CPU_INT8_ACT=1`: keep int8-quantized activations for quantized matmuls with many
/// rows (faster with i8mm, less accurate) instead of f32 activations x dequantized weights.
fn int8_activations() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("LAPID_CPU_INT8_ACT").is_some_and(|v| v != "0"))
}

/// Reusable per-backend scratch (packed panels, quantized activations). Grow-only.
#[derive(Default)]
pub(crate) struct Workspace {
    pub f: Vec<f32>,
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    /// Packed operands of the i8mm GEMM (`qgemm`).
    pub pa: Vec<i8>,
    pub pb: Vec<i8>,
    pub psa: Vec<f32>,
    pub psb: Vec<f32>,
    /// Per-group activation sums of the K-quant paths.
    pub pa_f: Vec<f32>,
    /// bf16 activation pairs of `qbf16` (weight panels are per thread).
    pub ha: Vec<u16>,
    /// Quantized B with few rows still takes the float-activation path (set by MatMulId for
    /// prefill-sized launches).
    pub float_act: bool,
}

/// A operand: element (bt, i, p) at `offset + bt * bs + i * rs + p`.
pub(crate) struct AOp<'a> {
    pub src: Src<'a>,
    pub offset: usize,
    pub bs: usize,
    pub rs: usize,
}

impl AOp<'_> {
    #[inline]
    pub fn row(&self, bt: usize, i: usize) -> usize {
        self.offset + bt * self.bs + i * self.rs
    }
}

/// Float B operand. transposed: (bt, p, j) at `offset + bt * bs + j * rs + p`;
/// otherwise at `offset + bt * bs + p * rs + j`.
pub(crate) struct BOp<'a> {
    pub src: Src<'a>,
    pub offset: usize,
    pub bs: usize,
    pub rs: usize,
    pub transposed: bool,
}

pub(crate) enum BInput<'a> {
    Plain(Src<'a>),
    Quant(QMat<'a>),
}

/// Applies the epilogue expression and stores rows of the accumulator.
pub(crate) struct Epilogue<'a> {
    prog: Option<Arc<Program>>,
    extras: Vec<(Src<'a>, &'a View)>,
    out: Dst,
    /// Output element (bt, i, j) at `out_off + bt * out_bs + i * out_rs + j`.
    out_off: usize,
    out_bs: usize,
    out_rs: usize,
}

impl Epilogue<'_> {
    pub fn regs(&self) -> Regs {
        let mut r = Regs::default();
        if let Some(p) = &self.prog {
            p.prepare(&mut r);
        }
        r
    }

    /// Store `acc` as output row (bt, i), columns `j0 .. j0 + acc.len()`.
    ///
    /// # Safety
    /// The caller must be the only writer of that output row segment.
    pub unsafe fn apply(&self, regs: &mut Regs, bt: usize, i: usize, j0: usize, acc: &[f32]) {
        let base = self.out_off + bt * self.out_bs + i * self.out_rs + j0;
        let Some(p) = &self.prog else {
            // SAFETY: exclusive row segment (caller), bounds validated at dispatch.
            unsafe { self.out.store(base, 1, acc) };
            return;
        };
        let mut j = 0;
        while j < acc.len() {
            let len = CHUNK.min(acc.len() - j);
            if let Some(r) = p.input(regs, 0) {
                r[..len].copy_from_slice(&acc[j..j + len]);
            }
            for (e, (src, view)) in self.extras.iter().enumerate() {
                if let Some(r) = p.input(regs, 1 + e) {
                    let s = &view.strides;
                    let off = view.offset as isize
                        + bt as isize * s[0]
                        + i as isize * s[1]
                        + (j0 + j) as isize * s[2];
                    src.load(off as usize, s[2], &mut r[..len]);
                }
            }
            let res = p.run(regs, len);
            // SAFETY: as above.
            unsafe { self.out.store(base + j, 1, res) };
            j += len;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    ctx: &Ctx,
    ws: &mut Workspace,
    spec: &MatMulSpec,
    d: &MatMulDims,
    a: &Src,
    b: &BInput,
    extras: &[Src],
    out: &Dst,
) -> Result<()> {
    let (batch, m, n, k) = (d.batch, d.m, d.n, d.k);
    if d.extra_views.len() != extras.len() || extras.len() != spec.extra_dtypes.len() {
        return Err(Error::new(
            "invalid_launch",
            format!("cpu matmul: {} extra views, {} extras, {} extra dtypes", d.extra_views.len(), extras.len(), spec.extra_dtypes.len()),
        ));
    }
    let numel = batch * m * n;
    if numel == 0 {
        return Ok(());
    }
    let ov = &d.out;
    let last = ov.offset + (batch - 1) * ov.batch_stride + (m - 1) * ov.row_stride + n;
    if out.len() < last {
        return Err(Error::new("out_of_bounds", format!("cpu matmul: output holds {} < {last} elements", out.len())));
    }
    if k > 0 {
        check_strided("matmul a", &[batch, m, k], d.a.offset, &[d.a.batch_stride as isize, d.a.row_stride as isize, 1], a.len())?;
    }
    let space = [batch, m, n];
    for (e, (v, s)) in d.extra_views.iter().zip(extras).enumerate() {
        check_view(&format!("matmul extra {e}"), &space, v, s.len())?;
    }
    let prog = match &spec.epilogue {
        Some(e) => {
            // Load(0) is the f32 accumulator; i32 extras read by `LoadBits` are raw words.
            let mut dts = vec![DType::F32];
            dts.extend_from_slice(&spec.extra_dtypes);
            let p = ctx.program_typed(e, &dts, &epilogue_raw(spec));
            if p.n_inputs() > 1 + extras.len() {
                return Err(Error::new(
                    "invalid_launch",
                    format!("cpu matmul: epilogue loads {} inputs, have {}", p.n_inputs(), 1 + extras.len()),
                ));
            }
            if p.identity() == Some(0) { None } else { Some(p) }
        }
        None => None,
    };
    let epi = Epilogue {
        prog,
        extras: extras.iter().copied().zip(d.extra_views.iter()).collect(),
        out: *out,
        out_off: ov.offset,
        out_bs: ov.batch_stride,
        out_rs: ov.row_stride,
    };
    let aop = AOp { src: *a, offset: d.a.offset, bs: d.a.batch_stride, rs: d.a.row_stride };

    if k == 0 {
        let zeros = vec![0f32; n];
        let mut regs = epi.regs();
        for bt in 0..batch {
            for i in 0..m {
                // SAFETY: single-threaded.
                unsafe { epi.apply(&mut regs, bt, i, 0, &zeros) };
            }
        }
        return Ok(());
    }

    match b {
        BInput::Quant(q) => {
            // Per format / shape (only with `LAPID_TRACE`): `qmm[(Q4K, m, n, k)]`.
            let _s = tracing::trace_span!("qmm", kind = ?(q.fmt, batch * m, n, k)).entered();
            if !spec.b_transposed || d.b.offset != 0 || d.b.batch_stride != 0 {
                return Err(Error::new(
                    "invalid_launch",
                    "cpu matmul: quantized B must be transposed with zero offset and batch stride",
                ));
            }
            if q.rows != n || q.cols != k {
                return Err(Error::new(
                    "shape_mismatch",
                    format!("cpu matmul: quantized B is [{}, {}], expected [{n}, {k}]", q.rows, q.cols),
                ));
            }
            if (batch * m > SMALL_M || ws.float_act) && !int8_activations() {
                if qbf16::available(ctx) && batch * m >= qbf16::MIN_M {
                    qbf16::matmul(ctx, ws, &aop, q, d, &epi);
                } else {
                    qdeq::matmul(ctx, ws, &aop, q, d, &epi);
                }
            } else if qgemm::available(ctx) && batch * m >= qgemm::MIN_M && !q.is_k() {
                qgemm::matmul(ctx, ws, &aop, q, d, &epi);
            } else {
                quant::matmul(ctx, ws, &aop, q, d, &epi);
            }
        }
        BInput::Plain(bs) => {
            let (dims, strides) = if spec.b_transposed {
                ([batch, n, k], [d.b.batch_stride as isize, d.b.row_stride as isize, 1])
            } else {
                ([batch, k, n], [d.b.batch_stride as isize, d.b.row_stride as isize, 1])
            };
            check_strided("matmul b", &dims, d.b.offset, &strides, bs.len())?;
            let bop = BOp {
                src: *bs,
                offset: d.b.offset,
                bs: d.b.batch_stride,
                rs: d.b.row_stride,
                transposed: spec.b_transposed,
            };
            if m <= SMALL_M {
                gemv::matmul(ctx, ws, &aop, &bop, d, &epi);
            } else {
                gemm::matmul(ctx, ws, &aop, &bop, d, &epi);
            }
        }
    }
    Ok(())
}

/// Epilogue inputs (0 = accumulator, 1.. = extras) bound as raw i32 words.
pub(crate) fn epilogue_raw(spec: &MatMulSpec) -> Vec<bool> {
    let Some(e) = &spec.epilogue else { return Vec::new() };
    let mut dts = vec![DType::F32];
    dts.extend_from_slice(&spec.extra_dtypes);
    crate::expr_int::raw_inputs(e, &dts)
}
