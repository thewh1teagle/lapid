//! Data parallelism inside one tensor (dequantize, quantize, repack): work is split into
//! contiguous row ranges on scoped std threads, so memory stays that of one tensor.

use std::cell::Cell;
use std::ops::Range;

thread_local! {
    /// Set on worker threads: nested parallel calls run inline there.
    static WORKER: Cell<bool> = const { Cell::new(false) };
}

fn on_worker<T>(f: impl FnOnce() -> T) -> T {
    WORKER.with(|w| w.set(true));
    f()
}

/// Worker threads: `$LAPID_CONVERT_THREADS` (>= 1), else the available parallelism.
pub fn threads() -> usize {
    std::env::var("LAPID_CONVERT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Split `0..n` into at most `threads()` contiguous ranges of at least `min` items.
fn ranges(n: usize, min: usize) -> Vec<Range<usize>> {
    let t = if WORKER.with(Cell::get) { 1 } else { threads() };
    let k = t.min(n / min.max(1)).max(1);
    let per = n.div_ceil(k);
    (0..k).map(|i| i * per..((i + 1) * per).min(n)).filter(|r| !r.is_empty()).collect()
}

/// `f` over contiguous ranges covering `0..n` (each >= `min` items, except a lone short
/// one), in parallel; results in range order.
pub fn map_ranges<T: Send>(n: usize, min: usize, f: impl Fn(Range<usize>) -> T + Sync) -> Vec<T> {
    let rs = ranges(n, min);
    if rs.len() <= 1 {
        return rs.into_iter().map(&f).collect();
    }
    std::thread::scope(|s| {
        let hs: Vec<_> = rs.into_iter().map(|r| s.spawn(|| on_worker(|| f(r)))).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    })
}

/// Fill `out` (`n` rows of `width` items) in parallel: `f(rows, out_rows)` for contiguous row
/// ranges of at least `min` rows.
pub fn fill_rows<T: Send>(out: &mut [T], width: usize, min: usize, f: impl Fn(Range<usize>, &mut [T]) + Sync) {
    let n = if width == 0 { 0 } else { out.len() / width };
    let rs = ranges(n, min);
    if rs.len() <= 1 {
        for r in rs {
            let o = &mut out[r.start * width..r.end * width];
            f(r, o);
        }
        return;
    }
    std::thread::scope(|s| {
        let mut rest = out;
        for r in rs {
            let (head, tail) = rest.split_at_mut(r.len() * width);
            rest = tail;
            let f = &f;
            s.spawn(move || on_worker(|| f(r, head)));
        }
    });
}

/// Elementwise `src -> dst` conversion in parallel (`dst.len() == src.len() * k`, `k`
/// outputs per input via `f(input, outputs)`).
pub fn convert<A: Sync, B: Send>(src: &[A], dst: &mut [B], k: usize, f: impl Fn(&A, &mut [B]) + Sync) {
    debug_assert_eq!(dst.len(), src.len() * k);
    fill_rows(dst, k, 1 << 16, |r, o| {
        for (a, b) in src[r].iter().zip(o.chunks_exact_mut(k)) {
            f(a, b);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_in_order() {
        let parts = map_ranges(1000, 7, |r| r);
        assert_eq!(parts.first().map(|r| r.start), Some(0));
        assert_eq!(parts.last().map(|r| r.end), Some(1000));
        assert!(parts.windows(2).all(|w| w[0].end == w[1].start));
        let mut out = vec![0usize; 3 * 500];
        fill_rows(&mut out, 3, 1, |r, o| {
            for (i, row) in r.zip(o.chunks_exact_mut(3)) {
                row.fill(i);
            }
        });
        assert!(out.chunks(3).enumerate().all(|(i, c)| c.iter().all(|&v| v == i)));
        let src: Vec<u16> = (0..200_000).map(|i| i as u16).collect();
        let mut dst = vec![0u8; src.len() * 2];
        convert(&src, &mut dst, 2, |a, b| b.copy_from_slice(&a.to_le_bytes()));
        assert_eq!(&dst[2 * 1234..2 * 1234 + 2], &1234u16.to_le_bytes());
        assert!(map_ranges(0, 4, |r| r).len() <= 1);
    }
}
