//! `Backend::write` (expert-cache slot fills): placement and bandwidth.

use lapid_compiler::Backend;

use crate::tests::cuda;

/// Host -> device write bandwidth through `Backend::write` (pageable source, hot), by size.
#[test]
#[ignore]
fn write_bandwidth() {
    let Some(b) = cuda() else { return };
    let src = vec![7u8; 1 << 30];
    let dst = b.alloc(256 << 20).expect("alloc");
    let tiny = b.alloc(256).expect("alloc");
    for size in [1usize << 20, 6 << 20, 32 << 20, 256 << 20] {
        let reps = (1usize << 30) / size;
        b.download(&tiny).expect("sync");
        let t = std::time::Instant::now();
        for r in 0..reps {
            let off = (r * size) % ((256 << 20) - size + 1);
            b.write(&dst, off, &src[r * size..(r + 1) * size]).expect("write");
        }
        b.download(&tiny).expect("sync");
        let s = t.elapsed().as_secs_f64();
        eprintln!("write {} MiB x{reps}: {:.1} GB/s", size >> 20, (reps * size) as f64 / s / 1e9);
    }
}

/// `Backend::write` lands at the offset (several staging chunks, back to back without
/// waiting), leaves the rest alone and refuses out-of-range writes.
#[test]
fn write_at_offset() {
    let Some(b) = cuda() else { return };
    let n = 40 << 20;
    let buf = b.upload(&vec![1u8; n]).expect("upload");
    let a: Vec<u8> = (0..(33 << 20) + 5).map(|i| (i % 251) as u8).collect();
    b.write(&buf, 3, &a).expect("write a");
    b.write(&buf, n - 7, &[9u8; 7]).expect("write b");
    let got = b.download(&buf).expect("download");
    assert_eq!(got[..3], [1, 1, 1]);
    assert!(got[3..3 + a.len()] == a[..]);
    assert!(got[3 + a.len()..n - 7].iter().all(|&v| v == 1));
    assert_eq!(got[n - 7..], [9u8; 7]);
    assert!(b.write(&buf, n - 6, &[0u8; 7]).is_err());
}
