//! Append-only rows `[len, width]` (compressed KV entries, index keys, ...) kept in one growing
//! `[capacity, width]` device buffer.
//!
//! One step is *append → eval → commit*, like [`KvCache`](super::KvCache):
//!
//! ```text
//! let (next, all) = buf.append(&new_rows)?;   // `all` = [len + n, width] view, attend to it
//! // evaluate `next.buffer()` with the step's outputs, then
//! *next.buffer_mut() = Some(realized);        // and keep `next`
//! ```
//!
//! New rows are written with [`Tensor::write_slice`], in place on the realized buffer (no copy
//! of the old rows per step). Capacity doubles (one copy) when full and starts at
//! [`MIN_ROWS`], so memory follows the sequence length instead of the maximum context.

use lapid::{Result, Tensor, bail};

/// First capacity (rows) of a buffer.
pub const MIN_ROWS: usize = 256;

#[derive(Clone, Default)]
pub struct RowBuffer {
    /// `[capacity, width]`.
    buf: Option<Tensor>,
    len: usize,
}

impl RowBuffer {
    /// Rows held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `[capacity, width]` storage (to realize with a step; `None` while empty).
    pub fn buffer(&self) -> Option<&Tensor> {
        self.buf.as_ref()
    }

    /// Slot for the realized storage after evaluation (same shape).
    pub fn buffer_mut(&mut self) -> &mut Option<Tensor> {
        &mut self.buf
    }

    /// The rows held, `[len, width]` (a view of the storage).
    pub fn rows(&self) -> Result<Option<Tensor>> {
        match &self.buf {
            Some(b) if self.len > 0 => Ok(Some(b.slice(0, 0, self.len)?)),
            _ => Ok(None),
        }
    }

    /// Keep the first `len` rows (a rewind: later appends overwrite the rest).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// `rows [n, width]` appended: the next buffer (stage its storage) and all its rows
    /// `[len + n, width]`.
    pub fn append(&self, rows: &Tensor) -> Result<(RowBuffer, Tensor)> {
        if rows.rank() != 2 {
            bail!("shape_mismatch", "row buffer: rows must be [n, width], got {:?}", rows.shape());
        }
        let (n, w) = (rows.shape()[0], rows.shape()[1]);
        let old = self.buf.as_ref().filter(|b| b.shape()[1] == w && b.dtype() == rows.dtype());
        if self.len > 0 && old.is_none() {
            bail!("shape_mismatch", "row buffer: rows {:?} do not match the stored width / dtype", rows.shape());
        }
        let need = self.len + n;
        let buf = match old {
            Some(b) if need <= b.shape()[0] => b.clone(),
            _ => {
                let cap = need.next_power_of_two().max(MIN_ROWS);
                let fresh = Tensor::zeros(&[cap, w], rows.dtype());
                match self.rows()? {
                    Some(kept) => fresh.write_slice(0, 0, &kept)?,
                    None => fresh,
                }
            }
        };
        let buf = buf.write_slice(0, self.len, rows)?;
        let all = buf.slice(0, 0, need)?;
        Ok((RowBuffer { buf: Some(buf), len: need }, all))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::eval;

    #[test]
    fn append_grows_and_keeps_rows() {
        let mut b = RowBuffer::default();
        let mut want = Vec::new();
        for (step, n) in [3usize, 1, 300, 2].into_iter().enumerate() {
            let v: Vec<f32> = (0..n * 2).map(|i| (step * 1000 + i) as f32).collect();
            want.extend_from_slice(&v);
            let (mut next, all) = b.append(&Tensor::from_f32(&v, &[n, 2]).expect("rows")).expect("append");
            assert_eq!(eval(&all), want);
            let realized = lapid::Device::cpu().eval(&[next.buffer().expect("buffer")]).expect("eval").remove(0);
            *next.buffer_mut() = Some(realized);
            b = next;
        }
        assert_eq!(b.len(), 306);
        assert_eq!(b.buffer().map(|t| t.shape()[0]), Some(512));
        assert_eq!(eval(&b.rows().expect("rows").expect("some")), want);
    }
}
