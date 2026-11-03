# lapid ops — semantics

Normative description of every `lapid_tensor::Op`, the kernel IR conventions derived from
them, and the numeric tolerances backends must meet. Owner: agent `compiler`.

## Conventions

- **Shapes** are row-major (`[d0, …, dn-1]`, last dim fastest). Rank 0 (`[]`) is a scalar
  with one element. Zero-sized dims are allowed; kernels over zero elements are skipped.
- **Compute type.** Float arithmetic is f32. Loads convert storage → f32, stores convert
  f32 → the output dtype. f16/bf16 are storage formats only. Elementwise ops on `i32`
  tensors are exact 32-bit integer math (wrapping add/sub/mul, truncating division; see
  *Integer ops*); `i32` values feeding a float op are converted to f32.
- **Conversions** (store / `Cast`):
  - f32 → f16, f32 → bf16: round to nearest, ties to even; overflow → ±inf; NaN stays NaN.
  - f32 → i32: truncate toward zero, saturating at `i32::MIN/MAX`, NaN → 0 (Rust `as`).
  - i32 → f32: round to nearest even (exact for |x| ≤ 2^24). f16/bf16 → f32 exact.
- **Broadcasting** (numpy): shapes are right-aligned, a dim of size 1 (or a missing
  leading dim) stretches. The *builder* applies it: elementwise nodes always receive
  operands of exactly the output shape (via `Reshape` + `Expand`).
- **dtype rules.** Elementwise outputs take the dtype of the first float operand (builder
  rule); comparisons return the operand dtype holding 1.0 / 0.0. Quantized dtypes
  (`int8`, `int4`, and the ggml K-quants `q4_k`, `q5_k`, `q6_k`) are only legal as the `b`
  operand of `MatMul` (transposed, 2-D), the expert stack of `MatMulId` (3-D) and the table
  of `Gather`, optionally through a same-shape `Reshape`; anything else is
  `Error{code:"unsupported"}` at compile time. The last dim must be a multiple of the
  format's block (32 for int8/int4, 256 for K-quants).
- **NaN** propagation through `Max`/`Min` is unspecified (backends may return either
  operand); all other ops follow IEEE-754 f32.

## Leaves

| op | meaning |
|---|---|
| `Host(bytes)` | contiguous row-major data in the node dtype (quantized: lapid-quant SoA layout) |
| `Device(buf)` | already-realized contiguous buffer |
| `Const(v)` | scalar `v` (as f32, then stored in the node dtype) broadcast to the node shape; never allocated |

## Elementwise (srcs have the output shape)

| op | definition |
|---|---|
| `Cast(dt)` | value converted to `dt` (see conversions). Float→float casts are exact-or-rounded storage changes; when fused, intermediate rounding to f16/bf16 may be skipped (more precise, allowed). Float→i32 always rounds (truncation) at a store. |
| `Unary(Neg)` | `-x` |
| `Unary(Exp)` | `e^x` |
| `Unary(Log)` | natural log; `log(0) = -inf`, `log(x<0) = NaN` |
| `Unary(Sqrt)` | `√x`, NaN for x < 0 |
| `Unary(Rsqrt)` | `1/√x` |
| `Unary(Recip)` | `1/x` |
| `Unary(Sin/Cos/Tanh)` | as named (radians) |
| `Unary(Erf)` | Gauss error function |
| `Unary(Abs)` | `|x|` |
| `Binary(Add/Sub/Mul/Div)` | IEEE f32 |
| `Binary(Max/Min)` | larger / smaller operand |
| `Binary(Pow)` | `a^b` (`powf`); negative base with non-integer exponent → NaN |
| `Binary(CmpLt)` | `1.0` if `a < b` else `0.0` |
| `Binary(CmpEq)` | `1.0` if `a == b` else `0.0` |
| `Where` | srcs `[cond, a, b]`: `cond != 0 ? a : b` (NaN cond counts as true) |

### Integer ops (`i32` operands and result)

`Add/Sub/Mul` wrap; `Div` truncates toward zero; `Max/Min/CmpLt/CmpEq` are signed (compare
results are `1`/`0`); `Neg/Abs` wrap. Integer-only binary ops (builders reject other dtypes):

| op | definition |
|---|---|
| `Rem` | truncated remainder (sign of `a`) |
| `BitAnd/BitOr/BitXor` | bitwise |
| `Shl` | `a << (b & 31)` |
| `Shr` | arithmetic `a >> (b & 31)` |

Division or remainder by zero (and `i32::MIN / -1`) yields an unspecified value, never a
trap. `Pow` on `i32` is computed in f32 and truncated. `Cast` f32 → i32 / i32 → f32 follow the
conversions above.
## Reduction

`Reduce { op, axis, keepdim }`: reduce `axis` of the input. Output shape removes the axis
(`keepdim=false`) or keeps it as 1. `Sum` of an empty axis is `0`, `Max` is `-inf`.
Summation order is unspecified (pairwise/tree/vectorized all fine). `mean` is built as
`sum * (1/n)`.

## Movement (views — never compute)

| op | definition |
|---|---|
| `Reshape` | same elements in row-major order, new shape (same numel) |
| `Permute(p)` | output dim `i` is input dim `p[i]` |
| `Expand` | ranks equal; dims of size 1 repeat to the output size |
| `Slice{axis,start,end}` | `[start, end)` along `axis`, `start ≤ end ≤ dim` |

## Buffer-producing structure ops

| op | definition |
|---|---|
| `Pad{axis,before,after,value}` | output dim `axis` = `before + n + after`; border filled with `value`, interior = input |
| `Cat{axis}` | srcs (same rank, same dims except `axis`, same dtype) concatenated in order |
| `Contiguous` | identity on values; forces a row-major materialization (also a fusion barrier) |

## Linear algebra / indexing

**`MatMul { b_transposed }`**, srcs `[a, b]`: `a: [..., m, k]`; `b: [..., k, n]`, or
`[..., n, k]` when `b_transposed`. Batch dims broadcast numpy-style (b is often 2-D).
`out[..., i, j] = Σ_k a[..., i, k] · b[..., k, j]`, accumulated in f32 (order unspecified).
Quantized `b` (`int8`/`int4`) must be 2-D and transposed; dequantization:
`int8: x = q·d`, `int4: x = (q − 8)·d`, one f16 scale `d` per 32 consecutive
elements of a row of `[n, k]` (k % 32 == 0). Backends may quantize `a` on the fly to int8
per 32-block (`s = max|a|/127`, round to nearest) and use integer dot products.
K-quant `b` (`q4_k` / `q5_k` / `q6_k`): rows of 256-element ggml super-blocks stored exactly as
in GGUF; dequantization as ggml `dequantize_row_q*_K` (`q4_k/q5_k: x = d·sc·q − dmin·m` per
32-element sub-block with 6-bit `sc`/`m`; `q6_k: x = d·sc·(q − 32)` per 16-element sub-block
with i8 `sc`). The same on-the-fly int8 `a` quantization is allowed.

**`MatMulId`** (mixture of experts, ggml `MUL_MAT_ID`), srcs `[x [T, K] (float), w [E, N, K]
(float or quantized; each expert in `matmul_t` form; a quantized stack is E matrices of
`layout(N, K).total` bytes back to back), ids (i32 [T, R])]` → `[T, R, N]`, x's dtype:
`out[t, r, :] = w[ids[t, r]] @ x[t, :]`. An id outside `[0, E)` fails like `Gather`.
Decomposition (float experts): `gather` of `w` viewed `[E, N·K]`, reshape, batched
`matmul_t`. Tolerance: as `MatMul` with the same `b` dtype.

**`GatedDeltaNet { scale }`** (Gated DeltaNet linear attention, ggml `GATED_DELTA_NET`),
srcs `[q [B,T,H,Dk], k [B,T,H,Dk], v [B,T,H,Dv], g [B,T,H], beta [B,T,H], state [B,H,Dk,Dv]]`
(all f32) → `[B, H, T + Dk, Dv]` f32. Per (b, h), with `S = state[b, h]` (`S[i][j]`, i over
Dk, j over Dv), sequentially for t = 0..T: `S ← exp(g_t)·S`; `u = Sᵀk_t`;
`S ← S + k_t ⊗ (beta_t·(v_t − u))`; `o_t = Sᵀ(q_t·scale)`. Rows `[0, T)` of the output are
the `o_t`, rows `[T, T + Dk)` the final `S`. (ggml stores `S` transposed and fixes
`scale = 1/√Dv`; values are the same.) Decomposition: the recurrence unrolled over T with
elementwise / reduce primitives. Tolerance: abs ≤ 1e-4 · (1 + |y|) per element in f32.

**`Sinkhorn { iters, eps }`** (DeepSeek mHC `comb`), srcs `[x [..., n, n]]` (f32 logits) →
f32, same shape. Per matrix: `c = softmax(x) + eps` over the last axis, `c = c / (Σ_rows c +
eps)` (each column), then `iters − 1` times `c = c / (Σ_cols c + eps)` (each row) and the
column step again. Decomposition: softmax and one reduce + divide per step (~2 launches per
step); backends opt in to `Kernel::Sinkhorn` (n ≤ 8) through `Backend::compile_options`.
Tolerance: abs ≤ 1e-5 per element.

**`Gather`**, srcs `[table [V, D], indices (i32, any shape S)]` → `[S..., D]` f32:
`out[s, :] = dequant(table[indices[s], :])`. An index outside `[0, V)` makes the evaluation
fail with `Error{code:"index_out_of_range"}`; kernels never read out of bounds (GPUs check
in-kernel, write a clamped row and raise a device flag that the host checks after the launch
sequence; the output is then unspecified).

**`GatherAxis { axis }`** (take_along_axis), srcs `[x, indices (i32, rank of x, same dims
except axis)]` → indices' shape, x's dtype: `out[.., j, ..] = x[.., indices[.., j, ..], ..]`
along `axis`; an exact element copy (no conversion). Out-of-range indices: as `Gather`.

**`TopK { k, axis }`**, srcs `[x]` (float or i32, compared as f32) → i32 indices, x's shape
with `axis` = `k` (`1 ≤ k ≤ dim`): the positions of the `k` largest values, by value
descending, ties → lower index first. NaN ordering is unspecified. `topk(k)` returns
`(gather_axis(x, idx), idx)`; `argmax(axis)` is `topk(1)` with the axis removed.

## In-place update

**`WriteSlice { axis, start }`**, srcs `[dest, src]` (same dtype and rank, same dims except
`axis`, `start + src.dim(axis) ≤ dest.dim(axis)`) → dest's shape: dest with
`[start, start + n)` along `axis` replaced by `src`. Values are always as if functional.
*Aliasing:* when `dest` is a realized device tensor (`Op::Device`, or a chain of
`WriteSlice` on one) that nothing else in the evaluated graph reads and that is not itself a
requested output, the write happens in place: only the slice is written and the result
aliases dest's buffer, so the old handle observes the new contents after eval. Never call
it in place on a tensor obtained from `Device::upload` of host data (the upload cache shares
that buffer). Otherwise a fresh buffer is produced (copy of dest + slice).

## Composites

Each has a reference decomposition (`lapid_tensor::decompose`); native kernels must match it
within tolerance.

**`Softmax`** over the last axis: `y = exp(x − max(x)) / Σ exp(x − max(x))`.
Decomposition: `m = max(x, -1, keepdim)`, `e = exp(x − m)`, `y = e / sum(e, -1, keepdim)`.
A row of all `-inf` yields NaN in the decomposition; native kernels return 0 for it.

**`LayerNorm { eps }`**, srcs `[x, weight [C], bias [C]]`, normalized over the last axis
(C = last dim): `μ = mean(x)`, `σ² = mean((x − μ)²)` (biased), `y = (x − μ) / √(σ² + eps) ·
weight + bias`. Decomposition: exactly those primitives (`Rsqrt` for the division).
Native kernels may use two-pass or Welford; single-pass `E[x²] − E[x]²` is not allowed.

**`Attention { scale, causal, has_mask }`**, srcs `[q [B,H,Sq,D], k [B,H,Sk,D],
v [B,H,Sk,D]]` + `mask` (additive, f32/f16, broadcastable to `[B,H,Sq,Sk]`):
`s = scale · q kᵀ + mask`; if `causal`, `s[i, j] = −inf` unless `j ≤ i + (Sk − Sq)`
(0-based; i.e. the queries are the *last* Sq positions of the key sequence — this is the
KV-cache rule; with Sq == Sk it is the usual lower triangle); `out = softmax(s) · v`.
A query row with no visible key (only possible when Sk < Sq, or a mask of all `-inf`)
yields 0 from native kernels; the decomposition may produce NaN — callers must not rely on
it. Decomposition: `matmul_t(q, k)`, `* scale`, `+ mask`, causal mask built from `CmpLt`
on index ramps and `Where(…, −inf)`, `Softmax`, `matmul(·, v)`.

## Numerics / tolerances (vs. an f64 reference of the same op)

| class | tolerance |
|---|---|
| Add/Sub/Mul/Div/Neg/Abs/Max/Min/Cmp/Where/Cast, copies | exact (correctly rounded f32, then the store conversion) |
| Sqrt, Rsqrt, Recip | ≤ 2 ulp |
| Exp, Log, Sin, Cos, Tanh, Pow | ≤ 4 ulp, or abs error ≤ 1e-6 near 0 / for \|x\| > 100 trig |
| Erf | abs error ≤ 1e-6 |
| Reduce Sum (n terms) | \|err\| ≤ 1e-6 · n · max\|x\| (any summation order) |
| Reduce Max | exact |
| MatMul f32/f16/bf16 `b` | \|err\| ≤ 1e-5 · k · max\|a\|·max\|b\| |
| MatMul quantized `b` (incl. on-the-fly int8 `a`) | per-output relative ‖err‖/‖ref‖ ≤ 1e-2 (int8), ≤ 2e-2 (int4) vs. the dequantized-f32 product |
| Softmax | abs ≤ 1e-6 per element, row sum within 1e-5 of 1 |
| LayerNorm | abs ≤ 1e-5 · (1 + \|y\|) |
| Attention | abs ≤ 1e-5 · max\|v\| (f32), ≤ 2e-3 · max\|v\| (f16 storage) |
| Integer ops, GatherAxis, TopK, WriteSlice | exact |

## Kernel IR mapping (what the compiler emits)

- Movement ops never emit kernels: each value is a strided `View` (offset + per-dim
  element strides, stride 0 = broadcast) of a buffer. `Reshape` of a view that strides
  cannot express is materialized row-major first.
- Elementwise chains are fused into one `Elementwise` kernel whose `Expr` loads its inputs
  through composed views; single-user matmuls take the following elementwise chain as
  `epilogue` (`Load(0)` = accumulator); reductions take the input chain as prologue with
  the reduced axis last; `Cast` to the stored dtype is the kernel `out_dtype`.
- `Pad` = border fills + interior copy, `Cat` = one copy per piece, both writing through
  offset `out_view`s of one buffer.
- `WriteSlice` in place writes `src` through a strided `out_view` into dest's buffer; a
  pending matmul producing `src` writes there directly through `MatMulDims.out` (offset,
  batch stride, row stride; unit column stride) — e.g. a K/V projection straight into a
  `[B, cap, H, D]` cache.
- Integer expressions: i32 nodes lower to typed `Expr` nodes (`LoadBits`, `ConstBits`,
  `IntUnary`, `IntBinary`, `Convert`) on 32-bit words; the IR also has the unsigned forms
  (`DivU`, `RemU`, `MinU`, `MaxU`, `ShrU`, `CmpLtU`, `F32ToU32`, `U32ToF32`) and bit casts.
  Roots are f32 except an elementwise kernel storing to i32 (raw word store).
- `GatherAxis` runs over a canonical `[outer, n, inner]` space with rank-3 views of x and
  the indices; `TopK` reads a row-major `[rows, cols]` (axis moved last, copied if needed).
- `MatMulId` reads contiguous x / ids and the expert stack as stored; `GatedDeltaNet` and
  `Sinkhorn` read contiguous inputs (copies when they are views).
- `Softmax`/`LayerNorm` kernels read row-major inputs; `Attention` reads q/k/v/mask through
  rank-4 views (so `reshape → permute` to heads is free).
- Iteration spaces are coalesced (size-1 dims dropped, mutually contiguous dims merged)
  before emission, so kernel ranks are as small as possible.
