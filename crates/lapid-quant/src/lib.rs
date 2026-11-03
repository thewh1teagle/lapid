//! Quantization formats. The `Format`, `Layout` and `layout()` items are a CONTRACT —
//! owned by the coordinator. Implementers fill in the functions marked `todo!()` and may
//! add modules (reference kernels, test vectors) but must not change these signatures.
//!
//! SoA layout of a quantized `[rows, cols]` matrix (cols % 32 == 0), row-major groups:
//!
//! ```text
//! offset 0                      : quants plane
//!     Int8: rows*cols bytes, i8, element (r, c) at r*cols + c
//!     Int4: rows*cols/2 bytes; byte (r*cols + c)/2 holds two elements of the SAME
//!            group: for group g (32 elements, 16 bytes), byte j (0..16) low nibble =
//!            element j, high nibble = element j+16  (same as ggml q4_0 nibble order)
//! offset align_up(qbytes, 256)  : scales plane, f16 little endian, one per group,
//!                                 scale (r, g) at index r*(cols/32) + g
//! total = scales_offset + rows*cols/32*2, rounded up to 256
//! ```
//! Dequant: Int8 x = q * d ; Int4 x = (q - 8) * d with q in 0..16.
//!
//! The k-quant formats (Q4K, Q5K, Q6K) are ggml super-blocks stored exactly as in GGUF
//! (see `layout` and `kquant`); `quantize` / `dequantize` are bit-exact with ggml's
//! `quantize_row_q*_K_ref` / `dequantize_row_q*_K`.

pub const GROUP: usize = 32;
pub const ALIGN: usize = 256;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Int8,
    Int4,
    /// ggml Q4_K: 256-element super-blocks of 144 bytes, stored exactly as in GGUF (AoS).
    Q4K,
    /// ggml Q5_K: 256-element super-blocks of 176 bytes (AoS, as in GGUF).
    Q5K,
    /// ggml Q6_K: 256-element super-blocks of 210 bytes (AoS, as in GGUF).
    Q6K,
    /// MXFP4: 32-element blocks of 17 bytes (e8m0 scale + 16 e2m1 nibble pairs), AoS as
    /// ggml `block_mxfp4` (see [`mxfp4`]).
    Mxfp4,
}

impl Format {
    /// Elements per quantization block along a row (32 for Int8/Int4, 256 for K formats).
    pub const fn block_elems(self) -> usize {
        match self {
            Format::Int8 | Format::Int4 | Format::Mxfp4 => GROUP,
            Format::Q4K | Format::Q5K | Format::Q6K => 256,
        }
    }
    /// Bytes per block for the AoS K formats (0 for the SoA Int8/Int4 formats).
    pub const fn block_bytes(self) -> usize {
        match self {
            Format::Int8 | Format::Int4 => 0,
            Format::Q4K => 144,
            Format::Q5K => 176,
            Format::Q6K => 210,
            Format::Mxfp4 => mxfp4::BYTES,
        }
    }
    pub const fn is_aos(self) -> bool {
        self.block_bytes() != 0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub rows: usize,
    pub cols: usize,
    pub qs_offset: usize,
    pub qs_bytes: usize,
    pub scales_offset: usize,
    pub scales_bytes: usize,
    pub total: usize,
}

pub const fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Layout of a quantized `[rows, cols]` matrix. Int8/Int4: SoA planes (see module docs).
/// K formats: AoS ggml super-blocks, row-major — `qs_offset = 0`, `qs_bytes = total`, no
/// scales plane (`scales_bytes = 0`); block (r, b) at byte `(r * cols/256 + b) * block_bytes`.
pub fn layout(format: Format, rows: usize, cols: usize) -> Layout {
    let be = format.block_elems();
    assert!(cols % be == 0, "quantized cols must be a multiple of {be}, got {cols}");
    if format.is_aos() {
        let n = rows * (cols / be) * format.block_bytes();
        return Layout { rows, cols, qs_offset: 0, qs_bytes: n, scales_offset: n, scales_bytes: 0, total: align_up(n, ALIGN) };
    }
    let qs_bytes = match format {
        Format::Int8 => rows * cols,
        _ => rows * cols / 2,
    };
    let scales_offset = align_up(qs_bytes, ALIGN);
    let scales_bytes = rows * (cols / GROUP) * 2;
    Layout {
        rows,
        cols,
        qs_offset: 0,
        qs_bytes,
        scales_offset,
        scales_bytes,
        total: align_up(scales_offset + scales_bytes, ALIGN),
    }
}

/// Quantize a row-major f32 matrix. Returns `layout(..).total` bytes.
/// Scale per group: Int8 d = max|x| / 127 ; Int4 d = max|x| / 8 with sign chosen as
/// ggml q4_0 does (d = max_signed / -8) so that the extreme value maps exactly.
/// K formats: ggml's reference quantizers, block by block.
pub fn quantize(format: Format, data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    imp::quantize(format, data, rows, cols)
}

/// Dequantize to row-major f32 (reference implementation, used by tests and fallbacks).
pub fn dequantize(format: Format, bytes: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    imp::dequantize(format, bytes, rows, cols)
}

/// Dequantize one row into `out` (len == cols).
pub fn dequantize_row(format: Format, bytes: &[u8], rows: usize, cols: usize, row: usize, out: &mut [f32]) {
    imp::dequantize_row(format, bytes, rows, cols, row, out)
}

/// Convert ggml-style AoS blocks (q8_0: f16 d + 32 i8 ; q4_0: f16 d + 16 nibble bytes)
/// for a `[rows, cols]` matrix into our SoA layout. Used by the whisper.cpp importer.
/// For the k-quant formats the GGUF blocks are our layout already: copied and padded.
pub fn from_ggml_blocks(format: Format, blocks: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    imp::from_ggml_blocks(format, blocks, rows, cols)
}

/// Pack int4 quants `q` (0..16, row-major `[rows, cols]`, element x = (q - 8) * d) and
/// one scale per 32-group (rounded to f16) into the Int4 layout. Used by calibrated
/// quantizers that choose their own quants and scales.
pub fn pack_int4(q: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    imp::pack_int4(q, scales, rows, cols)
}

mod imp;
pub mod kquant;
pub mod mxfp4;
