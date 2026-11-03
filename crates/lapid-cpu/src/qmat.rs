//! Read access to lapid-quant matrices: SoA `int8` / `int4`, AoS ggml K-quants (`kquant`).

use lapid_quant::{Format, GROUP, layout};
use lapid_tensor::{DType, Error, Result};

use crate::kquant::{self, KMat};
use crate::typed::f16_bits_to_f32;

/// Borrowed quantized `[rows, cols]` matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QMat<'a> {
    pub fmt: Format,
    pub rows: usize,
    pub cols: usize,
    /// Quants plane (i8 values, or packed nibbles for Int4); for K formats all blocks.
    pub qs: &'a [u8],
    /// f16 scale bits, `rows * cols / 32` entries.
    pub scales: &'a [u16],
}

pub(crate) fn format_of(dtype: DType) -> Option<Format> {
    match dtype {
        DType::Int8 => Some(Format::Int8),
        DType::Int4 => Some(Format::Int4),
        DType::Q4K => Some(Format::Q4K),
        DType::Q5K => Some(Format::Q5K),
        DType::Q6K => Some(Format::Q6K),
        DType::Mxfp4 => Some(Format::Mxfp4),
        _ => None,
    }
}

impl<'a> QMat<'a> {
    pub fn new(fmt: Format, bytes: &'a [u8], rows: usize, cols: usize) -> Result<QMat<'a>> {
        let be = fmt.block_elems();
        if !cols.is_multiple_of(be) {
            return Err(Error::new(
                "invalid_launch",
                format!("cpu: quantized matrix cols {cols} not a multiple of {be}"),
            ));
        }
        let l = layout(fmt, rows, cols);
        if bytes.len() < l.scales_offset + l.scales_bytes {
            return Err(Error::new(
                "out_of_bounds",
                format!("cpu: quantized [{rows}, {cols}] needs {} bytes, buffer has {}", l.total, bytes.len()),
            ));
        }
        let qs = &bytes[l.qs_offset..l.qs_offset + l.qs_bytes];
        if fmt.is_aos() {
            return Ok(QMat { fmt, rows, cols, qs, scales: &[] });
        }
        let scales: &[u16] = bytemuck::try_cast_slice(&bytes[l.scales_offset..l.scales_offset + l.scales_bytes])
            .map_err(|e| Error::new("backend", format!("cpu: misaligned quant scales: {e}")))?;
        Ok(QMat { fmt, rows, cols, qs, scales })
    }

    /// AoS ggml block matrix: K-quants or MXFP4 (no SoA scales plane).
    #[inline(always)]
    pub fn is_k(&self) -> bool {
        self.fmt.is_aos()
    }

    pub fn kmat(&self) -> KMat<'a> {
        KMat { fmt: self.fmt, bytes: self.qs, cols: self.cols }
    }

    #[inline(always)]
    pub fn groups(&self) -> usize {
        self.cols / GROUP
    }

    #[inline(always)]
    pub fn scale(&self, row: usize, g: usize) -> f32 {
        f16_bits_to_f32(self.scales[row * self.groups() + g])
    }

    /// Bytes of group `g` of `row` in the quants plane (32 for Int8, 16 for Int4).
    #[inline(always)]
    pub fn group_bytes(&self, row: usize, g: usize) -> &'a [u8] {
        match self.fmt {
            Format::Int8 => {
                let o = row * self.cols + g * GROUP;
                &self.qs[o..o + GROUP]
            }
            _ => {
                let o = (row * self.cols + g * GROUP) / 2;
                &self.qs[o..o + GROUP / 2]
            }
        }
    }

    /// Group `g` of `row` as 32 signed integers (I4 values re-centered to -8..=7).
    #[inline(always)]
    pub fn group_i8(&self, row: usize, g: usize, out: &mut [i8; GROUP]) {
        let b = self.group_bytes(row, g);
        match self.fmt {
            Format::Int8 => {
                for (o, &v) in out.iter_mut().zip(b) {
                    *o = v as i8;
                }
            }
            _ => {
                for j in 0..GROUP / 2 {
                    out[j] = (b[j] & 0x0f) as i8 - 8;
                    out[j + GROUP / 2] = (b[j] >> 4) as i8 - 8;
                }
            }
        }
    }

    /// Dequantize one row into `out` (len == cols).
    pub fn dequant_row(&self, row: usize, out: &mut [f32]) {
        if self.is_k() {
            let km = self.kmat();
            for (b, o) in out.chunks_exact_mut(self.fmt.block_elems()).enumerate() {
                kquant::dequant_block(self.fmt, km.block(row, b), o);
            }
            return;
        }
        let mut q = [0i8; GROUP];
        for g in 0..self.groups() {
            let d = self.scale(row, g);
            self.group_i8(row, g, &mut q);
            for (o, &v) in out[g * GROUP..(g + 1) * GROUP].iter_mut().zip(&q) {
                *o = v as f32 * d;
            }
        }
    }
}
