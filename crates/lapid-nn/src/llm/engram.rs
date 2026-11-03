//! Engram (DeepSeek V4.1, upstream `inference/engram.py`; llama.cpp `engram_hash` /
//! `build_engram`): n-gram hashed lookup tables whose rows are written into the
//! hyper-connection streams, gated by how well they match each stream.
//!
//! ```text
//! c_t   = token_map[t]                         compressed id (tokens that normalize alike)
//! tok_s = c at position p - s (s = 0..N-1)     pad once the look-back leaves the sequence
//! h_o   = xor_{s <= o} tok_s * mult[s]         (o + 1)-gram hash, wrapping u64
//! row   = h_o % prime[o, h] + offset[o, h]     one table row per (n-gram order o, head h)
//! kv    = Wkv concat(rows)                     -> hc keys + one value       [n, (hc + 1) D]
//! dot_s = sum_d rms(x_s) rms(key_s) w / sqrt(D)       per stream s, w = q_weight * k_weight
//! x_s  += sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))) value
//! ```
//!
//! The tables are huge (hundreds of millions of FP8 rows), so hashing ([`NgramHash`]) and the
//! row gather ([`Fp8Rows`], e4m3 with one e8m0 scale per 32 values, straight from the file
//! image) run on the host; [`EngramGate`] is the graph part.

use lapid::{BinaryOp, DType, HostData, Result, Tensor, UnaryOp, bail};

use crate::layers::Linear;
use crate::ops;

/// A token that takes part in no n-gram (outside the token map).
pub const DEAD: i64 = -1;

/// Hash layout of the Engram modules: one row of multipliers / primes per module (layer).
#[derive(Clone, Debug)]
pub struct NgramHash {
    /// N: hashes of the 2-gram .. N-gram ending at each token.
    pub max_ngram: usize,
    /// Heads per n-gram order.
    pub heads: usize,
    /// Token id -> compressed id.
    pub token_map: Vec<i32>,
    /// Compressed id filling look-back slots with no history.
    pub pad: i64,
    /// Per module: `N` odd multipliers.
    pub multipliers: Vec<Vec<u64>>,
    /// Per module: `(N - 1) * H` bucket counts (primes), order-major.
    pub primes: Vec<Vec<u64>>,
}

impl NgramHash {
    pub fn new(max_ngram: usize, heads: usize, token_map: Vec<i32>, pad: i64, multipliers: Vec<Vec<u64>>, primes: Vec<Vec<u64>>) -> Result<NgramHash> {
        let cols = max_ngram.saturating_sub(1) * heads;
        if max_ngram < 2 || heads == 0 || multipliers.len() != primes.len() {
            bail!("metadata", "engram: invalid layout (ngram {max_ngram}, {heads} heads, {} / {} modules)", multipliers.len(), primes.len());
        }
        if multipliers.iter().any(|m| m.len() != max_ngram) || primes.iter().any(|p| p.len() != cols || p.contains(&0)) {
            bail!("metadata", "engram: expected {max_ngram} multipliers and {cols} non-zero primes per module");
        }
        Ok(NgramHash { max_ngram, heads, token_map, pad, multipliers, primes })
    }

    /// Hash columns (table rows) per token and module: `(N - 1) * H`.
    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.heads
    }

    pub fn modules(&self) -> usize {
        self.primes.len()
    }

    /// Rows of the module's table: the sum of its primes.
    pub fn table_rows(&self, module: usize) -> u64 {
        self.primes[module].iter().sum()
    }

    /// Compressed id of `token` ([`DEAD`] outside the map).
    pub fn compress(&self, token: u32) -> i64 {
        self.token_map.get(token as usize).map_or(DEAD, |&c| i64::from(c))
    }

    /// The last `N - 1` compressed ids of `history ++ compress(ids)` (the history the next
    /// forward needs; oldest first).
    pub fn next_history(&self, history: &[i64], ids: &[u32]) -> Vec<i64> {
        let seq: Vec<i64> = history.iter().copied().chain(ids.iter().map(|&t| self.compress(t))).collect();
        seq[seq.len().saturating_sub(self.max_ngram - 1)..].to_vec()
    }

    /// Table rows `[ids.len() * cols]` of `module` for `ids`, preceded by `history` (the
    /// compressed ids of the previous tokens, oldest first; fewer than `N - 1` at the start of
    /// the sequence).
    pub fn rows(&self, module: usize, history: &[i64], ids: &[u32]) -> Vec<u64> {
        let (n, h) = (self.max_ngram, self.heads);
        let mult = &self.multipliers[module];
        let primes = &self.primes[module];
        let mut offsets = Vec::with_capacity(primes.len());
        let mut acc = 0u64;
        for &p in primes {
            offsets.push(acc);
            acc += p;
        }
        let seq: Vec<i64> = history.iter().copied().chain(ids.iter().map(|&t| self.compress(t))).collect();
        let base = history.len();
        let mut out = Vec::with_capacity(ids.len() * self.cols());
        let mut toks = vec![0u64; n];
        for i in 0..ids.len() {
            let mut blocked = false;
            for (s, tok) in toks.iter_mut().enumerate() {
                let c = (base + i).checked_sub(s).map_or(DEAD, |j| seq[j]);
                blocked |= c == DEAD;
                *tok = if blocked { self.pad as u64 } else { c as u64 };
            }
            let mut rolling = toks[0].wrapping_mul(mult[0]);
            for o in 1..n {
                rolling ^= toks[o].wrapping_mul(mult[o]);
                for hd in 0..h {
                    let col = (o - 1) * h + hd;
                    out.push(rolling % primes[col] + offsets[col]);
                }
            }
        }
        out
    }
}

/// e4m3fn (no infinities; 0x7f / 0xff are NaN) -> f32.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0x0f), f32::from(b & 0x07));
    let mag = match (e, m) {
        (15, 7.0) => f32::NAN,
        (0, _) => m / 8.0 * 2f32.powi(-6),
        _ => (1.0 + m / 8.0) * 2f32.powi(e - 7),
    };
    sign * mag
}

/// A table of FP8 e4m3 rows with one e8m0 scale (`2^(e - 127)`) per 32 values, read on the
/// host (zero copy from the file image).
pub struct Fp8Rows {
    values: HostData,
    scales: HostData,
    rows: usize,
    dim: usize,
    lut: [f32; 256],
}

impl Fp8Rows {
    /// `values` `[rows, dim]` e4m3 bytes, `scales` `[rows, dim / 32]` e8m0 bytes.
    pub fn new(values: HostData, scales: HostData, dim: usize) -> Result<Fp8Rows> {
        if dim == 0 || !dim.is_multiple_of(32) || !values.bytes().len().is_multiple_of(dim) {
            bail!("format", "fp8 rows: {} bytes are not rows of {dim} (a multiple of 32)", values.bytes().len());
        }
        let rows = values.bytes().len() / dim;
        if scales.bytes().len() != rows * dim / 32 {
            bail!("format", "fp8 rows: {} scale bytes for {rows} rows of {dim}", scales.bytes().len());
        }
        let lut = std::array::from_fn(|b| e4m3(b as u8));
        Ok(Fp8Rows { values, scales, rows, dim, lut })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Start reading rows `ids` into the page cache (asynchronous), ahead of [`gather`](Self::gather).
    pub fn prefetch(&self, ids: &[u64]) {
        let d = self.dim;
        for &id in ids {
            let Some(r) = usize::try_from(id).ok().filter(|&r| r < self.rows) else { continue };
            let _ = lapid::residency::advise_host(&self.values, Some(r * d..(r + 1) * d), lapid::Advice::WillNeed);
            let _ = lapid::residency::advise_host(&self.scales, Some(r * d / 32..(r + 1) * d / 32), lapid::Advice::WillNeed);
        }
    }

    /// Append the dequantized rows `ids` (`dim` values each) to `out`.
    pub fn gather(&self, ids: &[u64], out: &mut Vec<f32>) -> Result<()> {
        let (v, s, d) = (self.values.bytes(), self.scales.bytes(), self.dim);
        out.reserve(ids.len() * d);
        for &id in ids {
            let r = usize::try_from(id).ok().filter(|&r| r < self.rows);
            let Some(r) = r else { bail!("invalid_argument", "fp8 rows: row {id} outside the table ({} rows)", self.rows) };
            let (row, sc) = (&v[r * d..(r + 1) * d], &s[r * d / 32..(r + 1) * d / 32]);
            for (block, &e) in row.as_chunks::<32>().0.iter().zip(sc) {
                let scale = f32::from_bits(u32::from(e) << 23); // 2^(e - 127); e = 0 -> 0 (2^-127 is subnormal)
                let scale = if e == 0 { 2f32.powi(-127) } else if e == 255 { f32::NAN } else { scale };
                out.extend(block.iter().map(|&q| self.lut[q as usize] * scale));
            }
        }
        Ok(())
    }
}

/// The graph part: keys / value from the gathered rows, a gate per stream.
pub struct EngramGate {
    /// `[(hc + 1) D, cols * head_dim]`.
    pub wkv: Linear,
    /// `q_weight * k_weight` `[hc, D]` f32.
    pub weight: Tensor,
    pub hc: usize,
    pub eps: f32,
}

impl EngramGate {
    /// `x [n, hc, D]` (f32 streams), `rows [n, cols * head_dim]` (the gathered table rows) ->
    /// the updated streams.
    pub fn forward(&self, x: &Tensor, rows: &Tensor) -> Result<Tensor> {
        let (n, hc, d) = (x.shape()[0], self.hc, x.shape()[2]);
        if x.shape()[1] != hc {
            bail!("shape_mismatch", "engram: expected {hc} streams, got {:?}", x.shape());
        }
        let kv = self.wkv.forward(rows)?.cast(DType::F32)?; // [n, (hc + 1) D]
        let key = kv.slice(1, 0, hc * d)?.reshape(&[n, hc, d])?;
        let value = kv.slice(1, hc * d, (hc + 1) * d)?.reshape(&[n, 1, d])?;
        let rstd = |t: &Tensor| -> Result<Tensor> { t.mul(t)?.mean(2, false)?.add_scalar(f64::from(self.eps))?.unary(UnaryOp::Rsqrt) };
        let dot = x.mul(&key)?.mul(&self.weight)?.sum(2, false)?; // [n, hc]
        let dot = dot.mul(&rstd(x)?)?.mul(&rstd(&key)?)?.mul_scalar(1.0 / (d as f64).sqrt())?;
        let mag = dot.unary(UnaryOp::Abs)?.binary_scalar(BinaryOp::Max, 1e-6)?.sqrt()?;
        // copysign: -mag where dot < 0
        let neg = dot.binary_scalar(BinaryOp::CmpLt, 0.0)?;
        let signed = mag.sub(&mag.mul(&neg)?.mul_scalar(2.0)?)?;
        let gate = ops::sigmoid(&signed)?.unsqueeze(2)?; // [n, hc, 1]
        x.add(&gate.mul(&value)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_values() {
        assert_eq!([0x00, 0x08, 0x38, 0x40, 0x7e, 0xb8, 0x01].map(e4m3), [0.0, 2f32.powi(-6), 1.0, 2.0, 448.0, -1.0, 2f32.powi(-9)]);
        assert!(e4m3(0x7f).is_nan());
    }

    /// Hash of a hand-computed case: pads before the sequence start, rolling xor per order.
    #[test]
    fn hash_rows() {
        let h = NgramHash::new(3, 2, vec![5, 6, 7, 8], 1, vec![vec![3, 5, 7]], vec![vec![11, 13, 17, 19]]).expect("layout");
        let rows = h.rows(0, &[], &[2, 3]);
        let reference = |t: [u64; 3]| {
            let r1 = (t[0] * 3) ^ (t[1] * 5);
            let r2 = r1 ^ (t[2] * 7);
            vec![r1 % 11, r1 % 13 + 11, r2 % 17 + 24, r2 % 19 + 41]
        };
        let want: Vec<u64> = [reference([7, 1, 1]), reference([8, 7, 1])].concat();
        assert_eq!(rows, want);
        // the same tokens fed one by one with the carried history
        let hist = h.next_history(&[], &[2]);
        assert_eq!(hist, vec![7]);
        assert_eq!(h.rows(0, &hist, &[3]), reference([8, 7, 1]));
        assert_eq!(h.next_history(&[7], &[3, 0]), vec![8, 5]);
    }
}
