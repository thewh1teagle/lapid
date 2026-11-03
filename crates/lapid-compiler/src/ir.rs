//! Kernel IR — the contract between the compiler and every backend.
//! CONTRACT FILE — owned by the coordinator. Do not change without updating
//! plans/lapid-v1/CONTRACT.md.
//!
//! Every launch is `Kernel` (spec + dims) reading `inputs` buffers and writing `output`.
//! *Spec* = structure (hashable, sizes excluded) → backends cache compiled pipelines by spec.
//! *Dims* = concrete sizes/strides/offsets → passed at launch (push constants / params),
//! so the same compiled kernel serves every shape (e.g. growing KV cache).
//! All strides and offsets are in ELEMENTS of the buffer's dtype. Arithmetic is f32 (loads
//! convert to f32, stores convert to the output dtype) except the explicitly typed integer
//! expressions (`LoadBits`, `ConstBits`, `IntUnary`, `IntBinary`, `Convert`), which work on
//! 32-bit words (see `Expr`).

use std::sync::Arc;

use lapid_tensor::{BinaryOp, DType, DeviceData, HostData, ReduceOp, UnaryOp};

pub type BufId = usize;

/// Type of an `Expr` value: an f32, or a 32-bit integer word (signedness is a property of
/// the operation reading it, like in assembly).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprType {
    F32,
    Int,
}

/// 32-bit integer binary ops (on `Int` operands, `Int` result). Add/Sub/Mul wrap. Shift
/// amounts are taken `& 31`. Division / remainder by zero (and `i32::MIN / -1`) give an
/// unspecified value but never trap. Comparisons return 1 or 0.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    /// Signed division, truncating toward zero.
    DivS,
    DivU,
    /// Signed remainder (sign of the dividend).
    RemS,
    RemU,
    MinS,
    MinU,
    MaxS,
    MaxU,
    And,
    Or,
    Xor,
    Shl,
    /// Arithmetic shift right.
    ShrS,
    /// Logical shift right.
    ShrU,
    CmpLtS,
    CmpLtU,
    CmpEq,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IntUnaryOp {
    /// Wrapping negation.
    Neg,
    /// Wrapping absolute value (signed).
    Abs,
    /// Bitwise not.
    Not,
}

/// Explicit conversions between f32 and integer words.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Conv {
    /// f32 → i32: truncate toward zero, saturating, NaN → 0 (Rust `as`).
    F32ToI32,
    /// f32 → u32: truncate toward zero, saturating (negative → 0), NaN → 0.
    F32ToU32,
    /// i32 → f32, round to nearest even.
    I32ToF32,
    /// u32 → f32, round to nearest even.
    U32ToF32,
    /// Bit pattern of an f32 as a word.
    F32Bits,
    /// Word reinterpreted as f32.
    BitsF32,
}

/// Scalar expression evaluated per element.
///
/// Typing: `Load`, `Const`, `Unary`, `Binary` are f32 and take f32 operands; the integer
/// nodes are `Int`; `Convert` changes type; `Where` takes a condition of either type
/// (true when non-zero; an f32 NaN counts as true) and two branches of the same type.
/// Expression roots are f32 everywhere except an `Elementwise` kernel whose `out_dtype` is
/// I32, whose root may be `Int` (stored as the raw word). Validated by `Expr::ty`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Value of input `i` (index into the kernel's expression inputs) at the current element,
    /// converted to f32 (I32 storage: numeric conversion).
    Load(usize),
    /// f32 constant stored as bits (so Expr is Hash/Eq).
    Const(u32),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// cond != 0 ? a : b
    Where(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Raw 32-bit word of input `i` (`Int`); the input dtype must be I32 or F32.
    LoadBits(usize),
    /// Integer constant (`Int`).
    ConstBits(u32),
    IntUnary(IntUnaryOp, Box<Expr>),
    IntBinary(IntOp, Box<Expr>, Box<Expr>),
    Convert(Conv, Box<Expr>),
}

impl Expr {
    pub fn konst(v: f32) -> Expr {
        Expr::Const(v.to_bits())
    }
    pub fn const_value(bits: u32) -> f32 {
        f32::from_bits(bits)
    }
    pub fn konst_int(v: i32) -> Expr {
        Expr::ConstBits(v as u32)
    }
    /// Highest Load / LoadBits index + 1.
    pub fn num_loads(&self) -> usize {
        match self {
            Expr::Load(i) | Expr::LoadBits(i) => i + 1,
            Expr::Const(_) | Expr::ConstBits(_) => 0,
            Expr::Unary(_, a) | Expr::IntUnary(_, a) | Expr::Convert(_, a) => a.num_loads(),
            Expr::Binary(_, a, b) | Expr::IntBinary(_, a, b) => a.num_loads().max(b.num_loads()),
            Expr::Where(c, a, b) => c.num_loads().max(a.num_loads()).max(b.num_loads()),
        }
    }
    /// True when the expression uses any integer node (backends without integer support
    /// can reject such kernels early).
    pub fn has_int(&self) -> bool {
        match self {
            Expr::Load(_) | Expr::Const(_) => false,
            Expr::LoadBits(_) | Expr::ConstBits(_) | Expr::IntUnary(..) | Expr::IntBinary(..) | Expr::Convert(..) => {
                true
            }
            Expr::Unary(_, a) => a.has_int(),
            Expr::Binary(_, a, b) => a.has_int() || b.has_int(),
            Expr::Where(c, a, b) => c.has_int() || a.has_int() || b.has_int(),
        }
    }
    /// Type of the expression, or `None` if it is ill-typed.
    pub fn ty(&self) -> Option<ExprType> {
        use ExprType::{F32, Int};
        let want = |e: &Expr, t: ExprType| e.ty().filter(|&x| x == t);
        match self {
            Expr::Load(_) | Expr::Const(_) => Some(F32),
            Expr::LoadBits(_) | Expr::ConstBits(_) => Some(Int),
            Expr::Unary(_, a) => want(a, F32),
            Expr::Binary(_, a, b) => want(a, F32).and(want(b, F32)),
            Expr::IntUnary(_, a) => want(a, Int),
            Expr::IntBinary(_, a, b) => want(a, Int).and(want(b, Int)),
            Expr::Convert(c, a) => match c {
                Conv::F32ToI32 | Conv::F32ToU32 | Conv::F32Bits => want(a, F32).map(|_| Int),
                Conv::I32ToF32 | Conv::U32ToF32 | Conv::BitsF32 => want(a, Int).map(|_| F32),
            },
            Expr::Where(c, a, b) => {
                c.ty()?;
                let t = a.ty()?;
                want(b, t)
            }
        }
    }
}

/// Strided view of a buffer over the kernel's iteration space.
/// element(idx) = buffer[offset + sum(idx[d] * strides[d])]; stride 0 = broadcast.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub offset: usize,
    pub strides: Vec<isize>,
}

impl View {
    pub fn contiguous(shape: &[usize]) -> View {
        let mut strides = vec![0isize; shape.len()];
        let mut s = 1isize;
        for d in (0..shape.len()).rev() {
            strides[d] = s;
            s *= shape[d] as isize;
        }
        View { offset: 0, strides }
    }
}

// ---------------------------------------------------------------- elementwise
/// out[out_view(idx)] = expr(inputs[i][in_views[i](idx)]) for every idx in `shape`.
/// Inputs: `inputs.len() == in_dtypes.len()`. Used for fused chains, copies
/// (Contiguous / Cat pieces / Pad interior) and fills (expr with no Loads).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EwSpec {
    pub rank: usize,
    pub in_dtypes: Vec<DType>,
    pub out_dtype: DType,
    pub expr: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EwDims {
    pub shape: Vec<usize>,
    pub in_views: Vec<View>,
    pub out_view: View,
}

// ---------------------------------------------------------------- reduce
/// Iteration space `shape` with the reduced axis LAST. For each outer index o,
/// out[o] = reduce_{j}( expr(inputs at (o, j)) ). Output contiguous, numel = prod(shape[..rank-1]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceSpec {
    pub op: ReduceOp,
    pub rank: usize,
    pub in_dtypes: Vec<DType>,
    pub out_dtype: DType,
    pub expr: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceDims {
    pub shape: Vec<usize>,
    pub in_views: Vec<View>,
}

// ---------------------------------------------------------------- matmul
/// out[bt, i, j] = epilogue( sum_k A[bt, i, k] * B[bt, k, j] , extras[bt, i, j]... )
/// A: element (bt,i,k) at a.offset + bt*a.batch_stride + i*a.row_stride + k.
/// B (b_transposed): element (bt,k,j) at b.offset + bt*b.batch_stride + j*b.row_stride + k.
/// B (not transposed): element (bt,k,j) at b.offset + bt*b.batch_stride + k*b.row_stride + j.
/// Quantized B: always b_transposed, batch_stride 0, offset 0, the whole buffer is one
/// lapid-quant SoA matrix of [n, k].
/// Epilogue: Load(0) = accumulator, Load(1 + e) = extras[e] at view extra_views[e] over
/// the rank-3 space [batch, m, n]. Output element (bt,i,j) at
/// out.offset + bt*out.batch_stride + i*out.row_stride + j (contiguous [batch, m, n] is
/// `MatView { offset: 0, batch_stride: m*n, row_stride: n }`, see `MatView::contiguous`);
/// a non-contiguous `out` writes straight into a slice of a larger buffer (KV caches).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatMulSpec {
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub b_transposed: bool,
    pub extra_dtypes: Vec<DType>,
    pub out_dtype: DType,
    pub epilogue: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatView {
    pub offset: usize,
    pub batch_stride: usize,
    pub row_stride: usize,
}

impl MatView {
    /// Row-major `[batch, rows, cols]` starting at element 0.
    pub fn contiguous(rows: usize, cols: usize) -> MatView {
        MatView { offset: 0, batch_stride: rows * cols, row_stride: cols }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatMulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: MatView,
    pub b: MatView,
    pub extra_views: Vec<View>,
    /// Output layout (see `MatMulSpec`).
    pub out: MatView,
}

impl MatMulDims {
    /// True when the output is plain row-major `[batch, m, n]` at offset 0.
    pub fn out_is_contiguous(&self) -> bool {
        let o = &self.out;
        o.offset == 0 && (self.m <= 1 || o.row_stride == self.n) && (self.batch <= 1 || o.batch_stride == self.m * self.n)
    }
}

// ---------------------------------------------------------------- gather
/// inputs = [table [rows, cols], indices I32 contiguous [count]]; out [count, cols]
/// contiguous, out[i, :] = dequant(table[indices[i], :]). An index outside [0, rows) makes
/// `execute` fail with code "index_out_of_range" (kernels must not read out of bounds;
/// GPUs check in-kernel and raise a flag that the host reads after the launch sequence).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherSpec {
    pub table_dtype: DType,
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherDims {
    pub rows: usize,
    pub cols: usize,
    pub count: usize,
}

/// take_along_axis over the canonical rank-3 space [outer, n, inner]:
/// `out[o, j, i] = x[x.offset + o*x.strides[0] + idx*x.strides[1] + i*x.strides[2]]` with
/// `idx = indices[ix.offset + o*ix.strides[0] + j*ix.strides[1] + i*ix.strides[2]]`.
/// inputs = [x, indices (I32)]; out contiguous [outer, n, inner] of `dtype` (exact element
/// copy, no conversion). `idx` outside [0, axis_len) → "index_out_of_range" (as Gather).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherAxisSpec {
    pub dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherAxisDims {
    pub outer: usize,
    pub n: usize,
    pub inner: usize,
    pub axis_len: usize,
    /// rank-3 views (3 strides each).
    pub x: View,
    pub idx: View,
}

// ---------------------------------------------------------------- top-k
/// inputs = [x contiguous [rows, cols]] (float or I32 storage, compared as f32) → out I32
/// contiguous [rows, k]: column indices of the k largest values of each row, by value
/// descending, ties → lower index first. NaN ordering is unspecified. 1 <= k <= cols.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TopKSpec {
    pub in_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TopKDims {
    pub rows: usize,
    pub cols: usize,
    pub k: usize,
}

// ---------------------------------------------------------------- composites
/// Row softmax. inputs=[x contiguous [rows, cols]] → out contiguous [rows, cols].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SoftmaxSpec {
    pub in_dtype: DType,
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowDims {
    pub rows: usize,
    pub cols: usize,
}

/// inputs=[x [rows, cols], weight [cols], bias [cols]] all contiguous → out [rows, cols].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayerNormSpec {
    pub in_dtype: DType,
    pub param_dtype: DType,
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayerNormDims {
    pub rows: usize,
    pub cols: usize,
    /// f32 bits
    pub eps: u32,
}

/// inputs=[q, k, v] (+ mask). Views are rank-4 over [B, H, S, D] (mask: [B, H, Sq, Sk],
/// broadcast via 0 strides). out contiguous [B, H, Sq, D].
/// causal: key j visible to query i iff j <= i + (sk - sq).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttentionSpec {
    pub dtype: DType,
    pub mask_dtype: Option<DType>,
    pub causal: bool,
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttentionDims {
    pub batch: usize,
    pub heads: usize,
    pub sq: usize,
    pub sk: usize,
    pub d: usize,
    /// f32 bits
    pub scale: u32,
    pub q: View,
    pub k: View,
    pub v: View,
    pub mask: Option<View>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Elementwise(Arc<EwSpec>, EwDims),
    Reduce(Arc<ReduceSpec>, ReduceDims),
    MatMul(Arc<MatMulSpec>, MatMulDims),
    Gather(Arc<GatherSpec>, GatherDims),
    Softmax(Arc<SoftmaxSpec>, RowDims),
    LayerNorm(Arc<LayerNormSpec>, LayerNormDims),
    Attention(Arc<AttentionSpec>, AttentionDims),
    GatherAxis(Arc<GatherAxisSpec>, GatherAxisDims),
    TopK(Arc<TopKSpec>, TopKDims),
    MatMulId(Arc<MatMulIdSpec>, MatMulIdDims),
    GatedDeltaNet(Arc<GatedDeltaNetSpec>, GatedDeltaNetDims),
    /// Round 5. inputs=[x [rows, cols], weight [cols]] contiguous → out [rows, cols] =
    /// x * rsqrt(mean(x^2) + eps) * weight (f32 math). Spec/dims as LayerNorm.
    RmsNorm(Arc<LayerNormSpec>, LayerNormDims),
    /// Round 6 (opt-in, `CompileOptions::native_sinkhorn`). inputs=[x f32 contiguous
    /// [count, n, n]] -> out f32 [count, n, n]: `Op::Sinkhorn` of every matrix.
    Sinkhorn(SinkhornDims),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SinkhornDims {
    pub count: usize,
    pub n: usize,
    pub iters: usize,
    /// f32 bits
    pub eps: u32,
}

// ---------------------------------------------------------------- round 3
/// MoE matmul. inputs = [x (contiguous [t, k]), w (expert stack [e, n, k]), ids (I32
/// contiguous [t, r])]; out contiguous [t, r, n] = w[ids[t, r]] @ x[t]. Quantized w: each
/// expert is one lapid-quant matrix of [n, k] at byte offset `expert * layout(n, k).total`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatMulIdSpec {
    pub x_dtype: DType,
    pub w_dtype: DType,
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatMulIdDims {
    pub t: usize,
    pub r: usize,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

/// Gated delta rule (see `Op::GatedDeltaNet`). inputs = [q, k, v, g, beta, state], all f32
/// contiguous in the op's shapes; out contiguous [b, h, t + dk, dv].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatedDeltaNetSpec {
    pub out_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatedDeltaNetDims {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
    /// f32 bits
    pub scale: u32,
}

impl Kernel {
    pub fn kind(&self) -> &'static str {
        match self {
            Kernel::Elementwise(..) => "elementwise",
            Kernel::Reduce(..) => "reduce",
            Kernel::MatMul(..) => "matmul",
            Kernel::Gather(..) => "gather",
            Kernel::Softmax(..) => "softmax",
            Kernel::LayerNorm(..) => "layer_norm",
            Kernel::RmsNorm(..) => "rms_norm",
            Kernel::MatMulId(..) => "matmul_id",
            Kernel::GatedDeltaNet(..) => "gated_delta_net",
            Kernel::Attention(..) => "attention",
            Kernel::GatherAxis(..) => "gather_axis",
            Kernel::TopK(..) => "topk",
            Kernel::Sinkhorn(..) => "sinkhorn",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Launch {
    pub kernel: Kernel,
    pub inputs: Vec<BufId>,
    pub output: BufId,
}

#[derive(Clone, Debug)]
pub enum BufferSource {
    /// Upload these host bytes (runtime caches uploads by `HostData::key`).
    Host(HostData),
    /// Already resident on the device.
    Device(DeviceData),
    /// Intermediate; buffers with the same slot never live at the same time and share memory.
    Temp { slot: usize },
    /// Result returned to the caller.
    Output,
}

#[derive(Clone, Debug)]
pub struct BufferSpec {
    pub bytes: usize,
    pub source: BufferSource,
}

#[derive(Clone, Debug)]
pub struct OutputSpec {
    pub buf: BufId,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Fully lowered program. `buffers[id]` describes BufId `id`. `slot_bytes[s]` is the size
/// of temp slot `s` (max over the temps sharing it).
#[derive(Clone, Debug, Default)]
pub struct Schedule {
    pub buffers: Vec<BufferSpec>,
    pub slot_bytes: Vec<usize>,
    pub launches: Vec<Launch>,
    pub outputs: Vec<OutputSpec>,
}
