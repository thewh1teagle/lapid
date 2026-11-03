//! Weight residency end to end: under `Residency::Mmap` the weights of an opened `.lapid`
//! file are read in place from the read-only mapping (CPU: borrowed buffer; CUDA with
//! pageable memory access: device pointer = host pointer), including pages that are not in
//! the page cache (evicted with `posix_fadvise(DONTNEED)` before the eval).
//!
//! Ignored (run explicitly, `--release`, under a cgroup cap; files go to
//! `LAPID_MMAP_BENCH_DIR`, default the temp dir; size `LAPID_MMAP_BENCH_GB`):
//! * `mmap_bandwidth`: read bandwidth of uploaded vs mapped weights, hot and cold
//!   (`LAPID_MMAP_BENCH_DEVICE`, default cuda).
//! * `mmap_streams_larger_than_memory`: `LAPID_MMAP_STREAM_GB` (default 24) of mapped
//!   weights read in 1 GiB evals, `LAPID_MMAP_STREAM_PASSES` times; run it under a smaller
//!   `MemoryMax`: streaming must stay under the cap and give exact results every pass
//!   (this caught stale GPU translations after reclaim before evals re-mapped the pages).
//! * `mmap_cold_stress`, `mmap_fresh_mapping_repro`: repeated GPU reads of cold / freshly
//!   mapped pages (these caught wrong data from GPU page faults before evals populated).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use lapid::{Advice, DType, Device, LapidFile, Op, Residency, Tensor};

/// Residency is process-wide: tests that set it run one at a time.
static LOCK: Mutex<()> = Mutex::new(());

const COLS: usize = 4096;

/// Value at (row, col) of tensor `t`: small integers, so f32 row sums are exact.
fn row_values(row: usize, t: usize) -> Vec<f32> {
    (0..COLS).map(|c| (c % 16) as f32 + ((row % 5) + t) as f32).collect()
}

fn row_sum(row: usize, t: usize) -> f64 {
    (COLS / 16 * 120 + COLS * ((row % 5) + t)) as f64
}

fn dir() -> PathBuf {
    std::env::var_os("LAPID_MMAP_BENCH_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}

/// Write `n` f32 tensors `w{t}` of `[rows, COLS]`, one at a time.
fn write_file(path: &Path, n: usize, rows: usize) {
    let meta = serde_json::json!({
        "name": "mmap-test", "architecture": "test", "dtype": "fp32", "license": "unknown",
        "source": {"repo": "local/mmap-test"}, "converter": "test"
    });
    let mut w = lapid::format::Writer::create(path, meta).expect("create");
    for t in 0..n {
        let mut v = Vec::with_capacity(rows * COLS);
        let pattern: Vec<Vec<f32>> = (0..5).map(|r| row_values(r, t)).collect();
        for r in 0..rows {
            v.extend_from_slice(&pattern[r % 5]);
        }
        w.add(&format!("w{t}"), DType::F32, &[rows, COLS], bytemuck::cast_slice(&v)).expect("add");
    }
    w.finish().expect("finish");
}

#[cfg(target_os = "linux")]
mod os {
    use std::ffi::{c_int, c_void};
    use std::os::fd::AsRawFd;
    use std::path::Path;

    unsafe extern "C" {
        fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
        fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        fn getpagesize() -> c_int;
    }


    /// Drop the file's pages from the page cache (they must not be mapped / dirty).
    pub fn evict(path: &Path) {
        let f = std::fs::File::open(path).expect("open");
        f.sync_all().expect("sync");
        // SAFETY: plain syscall on an open fd. 4 = POSIX_FADV_DONTNEED.
        let r = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4) };
        assert_eq!(r, 0, "posix_fadvise");
    }

    /// Fraction of the pages of `bytes` resident in memory.
    pub fn resident(bytes: &[u8]) -> f64 {
        // SAFETY: no arguments.
        let page = unsafe { getpagesize() } as usize;
        let lo = bytes.as_ptr() as usize / page * page;
        let hi = (bytes.as_ptr() as usize + bytes.len()).div_ceil(page) * page;
        let mut v = vec![0u8; (hi - lo) / page];
        // SAFETY: the range covers pages of a live mapping; `v` has one byte per page.
        let r = unsafe { mincore(lo as *mut c_void, hi - lo, v.as_mut_ptr()) };
        assert_eq!(r, 0, "mincore");
        v.iter().filter(|&&b| b & 1 != 0).count() as f64 / v.len() as f64
    }
}

fn host_bytes(t: &Tensor) -> &[u8] {
    match t.op() {
        Op::Host(h) => h.bytes(),
        _ => panic!("not a host leaf"),
    }
}

/// (device, mapped?, device address) of `t` uploaded under the current residency.
fn placement(dev: &Device, t: &Tensor) -> (bool, Option<u64>) {
    let up = dev.upload(t).expect("upload");
    let Op::Device(d) = up.op() else { panic!("upload gave no device buffer") };
    if let Some(b) = d.as_any().downcast_ref::<lapid_cpu::CpuBuffer>() {
        return (b.is_mapped(), None);
    }
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    if let Some(b) = d.as_any().downcast_ref::<lapid_cuda::CudaBuffer>() {
        return (b.is_mapped(), Some(b.ptr()));
    }
    (false, None)
}

fn devices() -> Vec<Device> {
    let mut v = vec![Device::cpu()];
    match Device::cuda(0) {
        Ok(d) => v.push(d),
        Err(e) => eprintln!("cuda skipped: {}", e.message),
    }
    v
}

fn check_row_sums(dev: &Device, t: &Tensor, rows: usize, ti: usize) {
    let s = dev.to_vec_f32(&t.sum(-1, false).expect("sum")).expect("eval");
    assert_eq!(s.len(), rows);
    for (r, &v) in s.iter().enumerate() {
        assert_eq!(v as f64, row_sum(r, ti), "{} row {r}", dev.name());
    }
}

#[test]
fn mmap_reads_cold_pages_in_place() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = dir().join(format!("lapid-mmap-test-{}.lapid", std::process::id()));
    let rows = 1024; // 16 MiB per tensor
    write_file(&path, 2, rows);
    lapid::set_residency(Residency::Mmap);
    for dev in devices() {
        #[cfg(target_os = "linux")]
        os::evict(&path);
        let f = LapidFile::open(&path).expect("open");
        assert_eq!(f.residency(), Residency::Mmap);
        let w = f.get("w1").expect("w1");
        #[cfg(target_os = "linux")]
        let cold = os::resident(host_bytes(&w));
        let (mapped, ptr) = placement(&dev, &w);
        // CUDA maps only with pageable memory access through host page tables.
        if dev.name() == "cuda" && !mapped {
            eprintln!("cuda: device cannot read host memory in place; uploads instead");
        } else {
            assert!(mapped, "{}: weight not mapped", dev.name());
        }
        if let Some(p) = ptr.filter(|_| mapped) {
            assert_eq!(p as usize, host_bytes(&w).as_ptr() as usize);
        }
        check_row_sums(&dev, &w, rows, 1);
        // Elementwise over the mapping, full readback.
        let y = dev.to_vec_f32(&w.mul_scalar(2.0).expect("mul")).expect("eval");
        assert_eq!(&y[..COLS], &row_values(0, 1).iter().map(|v| v * 2.0).collect::<Vec<_>>()[..]);
        #[cfg(target_os = "linux")]
        {
            let warm = os::resident(host_bytes(&w));
            eprintln!("{}: resident before eval {:.1}%, after {:.1}%", dev.name(), cold * 100.0, warm * 100.0);
            // The data was read through the file mapping (page cache filled by the reads).
            if mapped {
                assert!(cold < 0.5 && warm > 0.99, "cold {cold}, warm {warm}");
            }
        }
        // Hints are harmless on mapped weights.
        lapid::prefetch(&w, 0..4096).expect("prefetch");
        f.advise(Advice::Random);
        check_row_sums(&dev, &f.get("w0").expect("w0"), rows, 0);
        f.advise(Advice::Normal);
    }

    // Upload: copies (no mapping), same results.
    lapid::set_residency(Residency::Upload);
    let f = LapidFile::open(&path).expect("open");
    assert_eq!(f.residency(), Residency::Upload);
    for dev in devices() {
        let w = f.get("w0").expect("w0");
        assert!(!placement(&dev, &w).0);
        check_row_sums(&dev, &w, rows, 0);
    }
    // Auto: a small file fits, so it is uploaded.
    lapid::set_residency(Residency::Auto);
    assert_eq!(LapidFile::open(&path).expect("open").residency(), Residency::Upload);
    drop(f);
    let _ = std::fs::remove_file(&path);
}

/// GB/s of reading `bytes` in `secs`.
fn gbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

/// Sum every row of every tensor in one eval; returns (seconds, row sums per tensor).
fn sum_all(dev: &Device, ws: &[Tensor]) -> (f64, Vec<Vec<f32>>) {
    let outs: Vec<Tensor> = ws.iter().map(|w| w.sum(-1, false).expect("sum")).collect();
    let refs: Vec<&Tensor> = outs.iter().collect();
    let t0 = Instant::now();
    let done = dev.eval(&refs).expect("eval");
    let secs = t0.elapsed().as_secs_f64();
    (secs, done.iter().map(|o| dev.to_vec_f32(o).expect("read")).collect())
}

fn check_sums(sums: &[Vec<f32>]) {
    let mut bad = 0usize;
    for (ti, s) in sums.iter().enumerate() {
        for (r, &v) in s.iter().enumerate() {
            if v as f64 != row_sum(r, ti) {
                if bad < 8 {
                    eprintln!("    mismatch: tensor {ti} row {r}: got {v}, want {}", row_sum(r, ti));
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} wrong row sums");
}

/// `x[1, rows] @ w[rows, COLS]` for every tensor, one eval (a GEMV over each weight).
fn gemv_all(dev: &Device, ws: &[Tensor]) -> f64 {
    let outs: Vec<Tensor> = ws
        .iter()
        .map(|w| {
            let x = Tensor::from_f32(&vec![1.0; w.shape()[0]], &[1, w.shape()[0]]).expect("x");
            x.matmul(w).expect("matmul")
        })
        .collect();
    let refs: Vec<&Tensor> = outs.iter().collect();
    let t0 = Instant::now();
    dev.eval(&refs).expect("eval");
    t0.elapsed().as_secs_f64()
}

#[test]
#[ignore]
fn mmap_bandwidth() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gb: usize = std::env::var("LAPID_MMAP_BENCH_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let per = 256usize << 20; // bytes per tensor
    let n = (gb << 30) / per;
    let rows = per / (COLS * 4);
    let bytes = n * per;
    let path = dir().join(format!("lapid-mmap-bench-{}.lapid", std::process::id()));
    write_file(&path, n, rows);
    let want_device = std::env::var("LAPID_MMAP_BENCH_DEVICE").unwrap_or_else(|_| "cuda".into());
    let dev = Device::from_name(&want_device).expect("device");
    eprintln!("{} GiB in {n} tensors of [{rows}, {COLS}] f32 on {}", gb, dev.description());

    let report = |what: &str, secs: f64| eprintln!("  {what:<46} {:>8.1} ms  {:>6.2} GB/s", secs * 1e3, gbs(bytes, secs));
    let open = || {
        let f = LapidFile::open(&path).expect("open");
        let ws: Vec<Tensor> = (0..n).map(|t| f.get(&format!("w{t}")).expect("w")).collect();
        (f, ws)
    };
    #[cfg(target_os = "linux")]
    let evict = || os::evict(&path);
    #[cfg(not(target_os = "linux"))]
    let evict = || {};

    // Mmap first (Upload pins its mapping, whose pages then cannot be evicted).
    lapid::set_residency(Residency::Mmap);
    {
        evict();
        let (_f, ws) = open();
        let (cold, sums) = sum_all(&dev, &ws);
        check_sums(&sums);
        report("mmap cold: row sums (lock + read from NVMe)", cold);
        let hot = (0..3).map(|_| sum_all(&dev, &ws).0).fold(f64::MAX, f64::min);
        report("mmap hot: row sums (page cache, lock per eval)", hot);
        report("mmap hot: gemv", (0..3).map(|_| gemv_all(&dev, &ws)).fold(f64::MAX, f64::min));
    }
    {
        evict();
        let (_f, ws) = open();
        let t0 = Instant::now();
        let refs: Vec<(&Tensor, std::ops::Range<usize>)> = ws.iter().map(|w| (w, 0..per)).collect();
        lapid::populate_all(&refs).expect("populate");
        report("populate_all, cold (32 threads, huge pages)", t0.elapsed().as_secs_f64());
        report("  then row sums", sum_all(&dev, &ws).0);
    }
    #[cfg(target_os = "linux")]
    {
        // CPU read of cold pages for comparison (one thread touching every page).
        evict();
        let (_f, ws) = open();
        let t0 = Instant::now();
        let acc: u64 = ws.iter().map(|w| host_bytes(w).iter().step_by(4096).map(|&b| b as u64).sum::<u64>()).sum();
        std::hint::black_box(acc);
        report("host page-touch, 1 thread (cold)", t0.elapsed().as_secs_f64());
    }

    // Upload: first eval copies, later evals read device memory.
    lapid::set_residency(Residency::Upload);
    {
        evict();
        let (_f, ws) = open();
        let (first, _) = sum_all(&dev, &ws);
        report("upload: first eval (read file + copy)", first);
        let (t, sums) = sum_all(&dev, &ws);
        check_sums(&sums);
        report("upload: row sums, device memory", t.min(sum_all(&dev, &ws).0));
        report("upload: gemv", gemv_all(&dev, &ws).min(gemv_all(&dev, &ws)));
    }
    lapid::set_residency(Residency::Auto);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore]
fn mmap_streams_larger_than_memory() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gb: usize = std::env::var("LAPID_MMAP_STREAM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let per = 1usize << 30;
    let rows = per / (COLS * 4);
    let path = dir().join(format!("lapid-mmap-stream-{}.lapid", std::process::id()));
    write_file(&path, gb, rows);
    #[cfg(target_os = "linux")]
    os::evict(&path);
    let want_device = std::env::var("LAPID_MMAP_BENCH_DEVICE").unwrap_or_else(|_| "cuda".into());
    let dev = Device::from_name(&want_device).expect("device");
    // Auto must pick Mmap by itself when the file exceeds the (cgroup) memory.
    lapid::set_residency(Residency::Auto);
    let f = LapidFile::open(&path).expect("open");
    eprintln!("{gb} GiB file, available {:?} GiB, residency {:?}", lapid::residency::available_memory().map(|b| b >> 30), f.residency());
    let ws: Vec<Tensor> = (0..gb).map(|t| f.get(&format!("w{t}")).expect("w")).collect();
    let passes: usize = std::env::var("LAPID_MMAP_STREAM_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let populate = std::env::var("LAPID_MMAP_STREAM_POPULATE").is_ok();
    let mut bad = 0;
    for pass in 0..passes {
        let t0 = Instant::now();
        if populate {
            for w in &ws {
                lapid::advise(w, None, Advice::Populate).expect("populate");
            }
        }
        let tp = t0.elapsed().as_secs_f64();
        // One eval per tensor (a model's working set per eval fits in memory; the file
        // does not).
        let mut secs = 0.0;
        let mut sums = Vec::new();
        for w in &ws {
            let (t, mut s) = sum_all(&dev, std::slice::from_ref(w));
            secs += t;
            sums.append(&mut s);
        }
        let b = std::panic::catch_unwind(|| check_sums(&sums)).is_err();
        bad += b as usize;
        eprintln!("pass {pass}: {gb} evals of 1 GiB on {}: populate {tp:.2} s + evals {secs:.2} s, {:.2} GB/s{}", dev.name(), gbs(gb << 30, secs + tp), if b { "  WRONG" } else { "" });
    }
    assert_eq!(bad, 0, "{bad} passes with wrong sums");
    drop(ws);
    drop(f);
    let _ = std::fs::remove_file(&path);
}

/// Repeated truly cold GPU reads of a mapping (two files used alternately, so the previous
/// trial's mapping, still memoized by the plan, never pins the pages being evicted).
/// `LAPID_STRESS_MODE`: `cold` (default), `touch` (CPU reads a second mapping of the file
/// first), `populate` (MADV_WILLNEED + CPU touch of the mapping itself first).
#[test]
#[ignore]
fn mmap_cold_stress() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gb: usize = std::env::var("LAPID_MMAP_BENCH_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let trials: usize = std::env::var("LAPID_STRESS_TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mode = std::env::var("LAPID_STRESS_MODE").unwrap_or_else(|_| "cold".into());
    let per = 256usize << 20;
    let n = (gb << 30) / per;
    let rows = per / (COLS * 4);
    let paths: Vec<PathBuf> = (0..2).map(|i| dir().join(format!("lapid-mmap-stress{i}-{}.lapid", std::process::id()))).collect();
    for p in &paths {
        write_file(p, n, rows);
    }
    let dev = Device::from_name("cuda").expect("device");
    lapid::set_residency(Residency::Mmap);
    let touch = |ws: &[Tensor]| ws.iter().map(|w| host_bytes(w).iter().step_by(4096).map(|&b| b as u64).sum::<u64>()).sum::<u64>();
    let mut total_bad = 0;
    for i in 0..trials {
        let path = &paths[i % 2];
        #[cfg(target_os = "linux")]
        os::evict(path);
        let f = LapidFile::open(path).expect("open");
        let ws: Vec<Tensor> = (0..n).map(|t| f.get(&format!("w{t}")).expect("w")).collect();
        #[cfg(target_os = "linux")]
        let res0 = os::resident(host_bytes(&ws[0]));
        match mode.as_str() {
            "cold" => {}
            "touch" => {
                let g = LapidFile::open(path).expect("open");
                let wg: Vec<Tensor> = (0..n).map(|t| g.get(&format!("w{t}")).expect("w")).collect();
                { std::hint::black_box(touch(&wg)); };
            }
            "touchevict" | "touchevictpop" => {
                let g = LapidFile::open(path).expect("open");
                let wg: Vec<Tensor> = (0..n).map(|t| g.get(&format!("w{t}")).expect("w")).collect();
                { std::hint::black_box(touch(&wg)); }
                #[cfg(target_os = "linux")]
                os::evict(path);
                #[cfg(target_os = "linux")]
                eprintln!("  after touch + evict: resident {:.0}%", os::resident(host_bytes(&ws[0])) * 100.0);
                if mode == "touchevictpop" {
                    for w in &ws {
                        lapid::advise(w, None, Advice::Populate).expect("populate");
                    }
                }
                drop(wg);
                drop(g);
            }
            "populate" => {
                for w in &ws {
                    lapid::prefetch(w, 0..per).expect("prefetch");
                }
                { std::hint::black_box(touch(&ws)); };
            }
            m => panic!("mode {m}"),
        }
        let (t, sums) = sum_all(&dev, &ws);
        let bad: usize = sums.iter().enumerate().map(|(ti, s)| s.iter().enumerate().filter(|(r, v)| **v as f64 != row_sum(*r, ti)).count()).sum();
        total_bad += bad;
        #[cfg(target_os = "linux")]
        eprintln!("trial {i}: resident {:.0}% -> {:>8.1} ms, {bad} wrong rows", res0 * 100.0, t * 1e3);
    }
    lapid::set_residency(Residency::Auto);
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
    assert_eq!(total_bad, 0);
}

/// Minimal reproduction attempts of wrong GPU reads of a fresh mapping whose pages are
/// resident (`LAPID_REPRO`: `upload` = Upload-mode evals of another mapping first, `copy` =
/// CPU memcpy of another mapping first, `none`).
#[test]
#[ignore]
fn mmap_fresh_mapping_repro() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gb: usize = std::env::var("LAPID_MMAP_BENCH_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let per = 256usize << 20;
    let n = (gb << 30) / per;
    let rows = per / (COLS * 4);
    let path = dir().join(format!("lapid-mmap-repro-{}.lapid", std::process::id()));
    write_file(&path, n, rows);
    let dev = Device::from_name("cuda").expect("device");
    let what = std::env::var("LAPID_REPRO").unwrap_or_else(|_| "upload".into());
    let populate = std::env::var("LAPID_REPRO_POPULATE").is_ok();
    let open = || {
        let f = LapidFile::open(&path).expect("open");
        let ws: Vec<Tensor> = (0..n).map(|t| f.get(&format!("w{t}")).expect("w")).collect();
        (f, ws)
    };
    let mut keep = Vec::new();
    for round in 0..3 {
        match what.as_str() {
            "upload" => {
                lapid::set_residency(Residency::Upload);
                let (_f, ws) = open();
                let _ = sum_all(&dev, &ws);
            }
            "copy" => {
                let (f, ws) = open();
                let total: usize = ws.iter().map(|w| host_bytes(w).to_vec().len()).sum();
                std::hint::black_box(total);
                keep.push((f, ws));
            }
            _ => {}
        }
        lapid::set_residency(Residency::Mmap);
        let (_f, ws) = open();
        #[cfg(target_os = "linux")]
        let res = os::resident(host_bytes(&ws[0]));
        #[cfg(not(target_os = "linux"))]
        let res = 0.0;
        if populate {
            for w in &ws {
                lapid::advise(w, None, Advice::Populate).expect("populate");
            }
        }
        let (t, sums) = sum_all(&dev, &ws);
        let bad: usize = sums.iter().enumerate().map(|(ti, s)| s.iter().enumerate().filter(|(r, v)| **v as f64 != row_sum(*r, ti)).count()).sum();
        eprintln!("{what} round {round}: resident {:.0}%, {:.1} ms, {bad} wrong rows", res * 100.0, t * 1e3);
        keep.push((_f, ws));
    }
    lapid::set_residency(Residency::Auto);
    let _ = std::fs::remove_file(&path);
}
