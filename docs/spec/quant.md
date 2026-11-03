# lapid quantization formats: `int8`, `int4`, `q4_k`, `q5_k`, `q6_k`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`

Reference implementation: `crates/lapid-quant` (`quantize`, `dequantize`, `dequantize_row`,
`from_ggml_blocks`, `pack_int4`, module `kquant`). All integers and f16 values are little
endian. `int8` / `int4` are lapid's SoA formats (below); `q4_k` / `q5_k` / `q6_k` are ggml's
k-quant super-blocks stored exactly as in GGUF (section "k-quants").

## Scope

Dtype names follow ONNX style: `int8` and `int4` name the element type only. The block
(group) size is a per-tensor attribute, `block_size` in the `.lapid` tensor index
(docs/spec/format.md); v1 defines and supports `block_size = 32` only.

A quantized tensor is a 2-D row-major matrix `[rows, cols]` with `cols % block_size == 0`.
Each row is split into `cols / 32` **blocks** of 32 consecutive elements. Every block has
one f16 scale `d`. Block `g` of row `r` has the global block index `gi = r * (cols/32) + g`.

## Byte layout (SoA, 256-aligned planes)

```
offset 0                        quants plane
    int8: rows*cols bytes; element (r, c) is the i8 at byte r*cols + c
    int4: rows*cols/2 bytes; block gi occupies bytes [gi*16, gi*16 + 16);
            byte j (0..16) of the block: low nibble = element j, high nibble = element j+16
            (same nibble order as ggml q4_0)
scales_offset = align_up(qs_bytes, 256)
                                scales plane: f16 per block, scale gi at scales_offset + 2*gi
total = align_up(scales_offset + rows*(cols/32)*2, 256)
```

Padding bytes (between the planes and at the end) are zero.

Examples: `[1, 32]` i8 → quants 32 B, scales at 256, total 512. `[3, 64]` i4 → quants 96 B,
scales at 256 (12 B), total 512. `[768, 768]` i8 → quants 589 824 B, scales at 589 824,
total 626 688.

## Dequantization

```
int8: x = q * d            q = signed byte in [-127, 127] (-128 never produced)
int4: x = (q - 8) * d      q = nibble in [0, 15]
```
`d` is the f16 scale converted to f32; the product is computed in f32.

## Quantization (bit-exact with ggml's q8_0 / q4_0 reference quantizers)

Per block of 32 f32 values `x[0..32]`:

**int8** (= ggml `quantize_row_q8_0_ref`)
```
amax = max_j |x[j]|
d    = amax / 127                  (f32)
id   = d != 0 ? 1 / d : 0          (f32)
q[j] = round_half_away_from_zero(x[j] * id), clamped to [-127, 127]
scale stored = f16_round_nearest_even(d)
```

**int4** (= ggml `quantize_row_q4_0_ref`)
```
max  = the first x[j] with the largest |x[j]| (sign kept)
d    = max / -8                    (f32)
id   = d != 0 ? 1 / d : 0
q[j] = min(15, trunc(x[j] * id + 8.5))   -- equals clamp(floor(x/d + 8.5), 0, 15)
scale stored = f16(d)
```
The extreme value maps to q = 0 and dequantizes to exactly `(0 - 8) * d = max` (up to the
f16 rounding of `d`). Values on the opposite side of zero can reach x/d = +8 and saturate at
q = 15, so the worst-case error is one step `|d|`; on the extreme's side it is `|d|/2`.

All-zero blocks: `d = 0`, `id = 0`, i8 quants are 0, i4 quants are 8; dequantizes to 0.
NaN inputs quantize to 0 (i8) / 8 (i4) unless they are the block's only non-zero value.
Scales above f16 range (65504) are not supported (become inf).

Error bounds (used by the tests): per element `|x - x̂| ≤ k·amax/L + 1e-3·amax`, with
`L = 127, k = 0.5` for i8 and `L = 8, k = 1` for i4.

## Test vectors

Input: one block `x[j] = j - 16` for j = 0..32 (values -16..15), shape `[1, 32]`.

int8:
```
d = 16/127 = 0.12598425 → f16 bits 0x3008 (bytes 08 30 at offset 256)
q = -127 -119 -111 -103 -95 -87 -79 -71 -64 -56 -48 -40 -32 -24 -16 -8
      0    8   16   24  32  40  48  56  64  71  79  87  95 103 111 119
```
(`x = -8` → `-8 * 7.9375 = -63.5` → -64: ties round away from zero; `id` is exactly 7.9375.)

int4:
```
max = -16, d = 2.0 → f16 bits 0x4000 (bytes 00 40 at offset 256)
q[j]  = 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 | 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 15
bytes = 80 91 91 a2 a2 b3 b3 c4 c4 d5 d5 e6 e6 f7 f7 f8
dequant: x̂[0] = -16, x̂[16] = 0, x̂[31] = 14 (saturated)
```

## ggml import (`from_ggml_blocks`)

ggml stores the same blocks, as AoS structs in row-major block order:
`block_q8_0 = { f16 d; i8 qs[32] }` (34 B), `block_q4_0 = { f16 d; u8 qs[16] }` (18 B).
Conversion copies `d` to scale `gi` and `qs` to the quants plane at `gi*32` (i8) or `gi*16`
(i4). The ggml row length `ne[0]` equals `cols`. No re-quantization happens, so values are
bit-identical to whisper.cpp's.

Other ggml types (q4_1, q5_0, q5_1) are dequantized to f32 by lapid-convert and quantized
again with the rules above.

## k-quants: `q4_k`, `q5_k`, `q6_k`

ggml's k-quant super-blocks, byte-identical to GGUF (`ggml-common.h`): a `[rows, cols]`
matrix with `cols % 256 == 0` is `rows * cols/256` blocks in row-major order (AoS, block
`(r, b)` at byte `(r * cols/256 + b) * block_bytes`), the total padded with zeros to a
multiple of 256 (`layout`: `qs_offset = 0`, `qs_bytes` = blocks, no scales plane).

```
q4_k (144 B / 256): f16 d | f16 dmin | scales[12] | qs[128]
q5_k (176 B / 256): f16 d | f16 dmin | scales[12] | qh[32] | qs[128]
q6_k (210 B / 256): ql[128] | qh[64] | i8 scales[16] | f16 d
```
q4_k / q5_k: 8 sub-blocks of 32 with 6-bit scale `sc` and min `m` packed in `scales`
(`get_scale_min_k4`); `x = d*sc * q - dmin*m`, q 4-bit (q4_k) or 5-bit with the high bit in
`qh` (q5_k). Elements: sub-block pair `j` (0..4) covers 64 elements; byte `l` of
`qs[32j..32j+32]` holds element `64j + l` (low nibble) and `64j + 32 + l` (high nibble); q5_k
bit `2j` / `2j+1` of `qh[l]` is the 5th bit of those two. q6_k: 16 sub-blocks of 16 with i8
scales, `x = d*sc * (q - 32)`, q 6-bit from `ql` (low 4 bits) and `qh` (high 2 bits) in
ggml's 128-element interleave (see `kquant::dequantize_q6k`).

Quantization (`quantize`) is a port of `quantize_row_q{4,5,6}_K_ref` (weighted min/scale
search `make_qkx2_quants` for q4_k/q5_k, `make_qx_quants` rmse search for q6_k, ggml's
`nearest_int` rounding) and dequantization of `dequantize_row_q*_K`, **bit-exact** with ggml
compiled without floating-point contraction (`-ffp-contract=off`; GCC's default contraction
on aarch64 fuses some products into FMAs, which changes last bits). Checked by
`LAPID_GGML_KQ_REF=<file> cargo test -p lapid-quant matches_ggml`, with the file written by
a local harness (`build.sh`; links ggml's `ggml-quants.c`, 24
records of 4096 values over 8 distributions incl. zero blocks, outliers, mixed scales).

GGUF import keeps q4_K / q5_K / q6_K tensors as they are (`from_ggml_blocks` copies and
pads) whenever the dtype policy quantizes the tensor at all.

## i-quants: `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`

ggml's grid-codebook super-blocks, byte-identical to GGUF types 16, 17, 22 and 18, with the same
row-major AoS layout and padding as the k-quants (256 elements per block):

```
iq2_xxs (66 B, 2.0625 bpw): f16 d | 8 x { u8 idx[4] | u32 (4 x 7 sign bits, 4-bit scale s) }
    x = d·(0.5 + s)·0.25 · iq2xxs_grid[idx][j] · ±1          (scale per 32)
iq2_xs  (74 B, 2.3125 bpw): f16 d | qs u16[32] (9-bit index | 7 sign bits << 9) | scales[8]
    x = d·(0.5 + s)·0.25 · iq2xs_grid[idx][j] · ±1            (scale per 16, 512-entry grid)
iq2_s   (82 B, 2.5625 bpw): f16 d | qs[32] (index low bytes) | signs[32] | qh[8] (2 index
    bits per 8 values) | scales[8] (4 bits per 16)
    x = d·(0.5 + s)·0.25 · iq2s_grid[idx][j] · ±1             (scale per 16, 1024-entry grid)
iq3_xxs (98 B, 3.0625 bpw): f16 d | idx[64] (one per 4 values) | 8 x u32 (4 x 7 sign bits,
    4-bit scale s)
    x = d·(0.5 + s)·0.5 · iq3xxs_grid[idx][j] · ±1            (scale per 32)
```
The 7 stored sign bits of an 8-value group get an 8th bit making the count of negatives
even (`ksigns_iq2xs`). Grid bytes: iq2 {8, 25, 43}, iq3 {4, 12, …, 52, 62}; the grids are
derived from ggml's lattice point lists (`kgrid_*`) and checked against `ggml-common.h`.

Quantization (`lapid_quant::{iq2_xxs, iq2_xs, iq2_s, iq3_xxs}::quantize(rows, n_per_row,
imatrix)`) ports `quantize_iq2_xxs` / `quantize_iq2_xs` / `quantize_iq2_s` / `quantize_iq3_xxs` with their
lattice maps and neighbour lists (`iq2xs_init_impl` / `iq3xs_init_impl`): optional
importance weights per column (an imatrix); without them ggml's default weighting (iq2_s:
`0.25·σ² + x²`, iq3_xxs: `x²`; ggml requires an imatrix for iq2_xxs and iq2_xs, lapid then
uses unit importance). Multithreaded over blocks, deterministic. Bit-exactness (bytes and dequantized
values, with and without an imatrix) is checked against fixtures generated once by
`tools/iquant_golden/gen.sh` (links ggml's `ggml-quants.c`, `-ffp-contract=off`).
`quantize(format, ..)` uses the default weighting.

GGUF import keeps these tensors as they are.

## Calibrated int4 (GPTQ)

`lapid convert --dtype int4 --calibrate builtin|<text file>` stores the same `int4` format,
but chooses quants and scales with GPTQ on activations captured from the model's own
forward pass (`lapid_convert::calibrate`, `gptq.rs`), instead of round-to-nearest:

- per linear weight `W [out, in]`: Gram matrix `H = Σ x xᵀ` of its input over all
  calibration tokens (computed on the device), damped `H += 0.01·mean(diag H)·I`;
- LDLQ form of GPTQ: with `H = R Rᵀ` (R upper, from one Cholesky of the index-reversed
  `H`), columns are quantized in order, `ŵ_k = Q(w_k + Σ_{i<k} (w_i − ŵ_i) R[i,k]/R[k,k])`;
- each 32-group's scale is chosen when the group starts, from the error-corrected values:
  the best of `f16(m·max/−8)`, `m ∈ {1, 0.80, 0.82, …, 1.10}`, under the `diag(H)`-weighted
  squared error (`max` = signed value of largest magnitude, as q4_0);
- codes `q = clamp(floor(v/d + 8.5), 0, 15)`, packed with `pack_int4`.

Tensors the int4 policy stores otherwise (int8 embeddings, f16 convolutions, f32 norms) are
unchanged. The file's metadata gets `"quantization": {"method": "gptq", "group_size": 32,
"damp", "scale_search", "calibration", "calibration_tokens", "weights", "rtn_fallbacks",
"passes", "seconds"}`. A weight whose Gram matrix stays indefinite after raising the damping
falls back to round-to-nearest (listed in `rtn_fallbacks`).

