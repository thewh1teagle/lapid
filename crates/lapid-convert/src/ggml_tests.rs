//! Dequantizer tests. Each block is built by a test encoder from chosen per-element quant
//! values and scales, using an element → (byte, bits, scale) mapping written out
//! independently from the dequantizer loops (derived from ggml-common.h /
//! ggml-quants.c); the expected value is ggml's formula applied per element.

use super::*;

/// Deterministic pseudo-random bytes.
pub(crate) struct Lcg(pub u64);

impl Lcg {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub(crate) fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

fn h(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_le_bytes()
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "element {i}: got {g}, want {w}");
    }
}

/// 6-bit scale/min packing of q4_K / q5_K (inverse of `get_scale_min_k4`).
fn pack_k4(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = sc[j] & 63;
        q[j + 4] = mn[j] & 63;
    }
    for j in 4..8 {
        q[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
        q[j - 4] |= (sc[j] >> 4) << 6;
        q[j] |= (mn[j] >> 4) << 6;
    }
    q
}

/// One block of `t` with random content plus the expected dequantized values.
pub(crate) fn sample_block(t: GgmlType, rng: &mut Lcg) -> (Vec<u8>, Vec<f32>) {
    let (d, dmin) = (0.5f32, 0.25f32);
    match t {
        GgmlType::Q4_1 => {
            let q: Vec<u8> = (0..32).map(|_| rng.below(16) as u8).collect();
            let mut b = [h(d), h(-1.5)].concat();
            b.extend((0..16).map(|j| q[j] | (q[j + 16] << 4)));
            (b, q.iter().map(|&x| x as f32 * d - 1.5).collect())
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let q: Vec<u32> = (0..32).map(|_| rng.below(32)).collect();
            let has_min = t == GgmlType::Q5_1;
            let mut b = h(d).to_vec();
            if has_min {
                b.extend(h(2.0));
            }
            let qh: u32 = (0..32).map(|i| (q[i] >> 4) << i).sum();
            b.extend(qh.to_le_bytes());
            b.extend((0..16).map(|j| (q[j] & 0xF) as u8 | (((q[j + 16] & 0xF) as u8) << 4)));
            let want = q.iter().map(|&x| if has_min { x as f32 * d + 2.0 } else { (x as f32 - 16.0) * d }).collect();
            (b, want)
        }
        GgmlType::Q8_1 => {
            let q: Vec<i8> = (0..32).map(|_| rng.next() as i8).collect();
            let mut b = [h(d), h(0.0)].concat();
            b.extend(q.iter().map(|&x| x as u8));
            (b, q.iter().map(|&x| x as f32 * d).collect())
        }
        GgmlType::Q2K => {
            let (mut scales, mut qs) = ([0u8; 16], [0u8; 64]);
            let s: Vec<(u8, u8)> = (0..16).map(|_| (rng.below(16) as u8, rng.below(16) as u8)).collect();
            for (i, (sc, m)) in s.iter().enumerate() {
                scales[i] = sc | (m << 4);
            }
            let mut want = vec![0f32; QK_K];
            for (i, w) in want.iter_mut().enumerate() {
                let (n, r) = (i / 128, i % 128);
                let (j, half, l) = (r / 32, (r % 32) / 16, r % 16);
                let q = rng.below(4) as u8;
                qs[n * 32 + half * 16 + l] |= q << (2 * j);
                let (sc, m) = s[n * 8 + j * 2 + half];
                *w = d * sc as f32 * q as f32 - dmin * m as f32;
            }
            ([&scales[..], &qs[..], &h(d), &h(dmin)].concat(), want)
        }
        GgmlType::Q3K => {
            let (mut hm, mut qs, mut sc) = ([0u8; 32], [0u8; 64], [0u8; 12]);
            let s: Vec<u8> = (0..16).map(|_| rng.below(64) as u8).collect();
            for (i, &v) in s.iter().enumerate() {
                sc[i % 8] |= (v & 0xF) << (4 * (i / 8));
                sc[8 + i % 4] |= (v >> 4) << (2 * (i / 4));
            }
            let mut want = vec![0f32; QK_K];
            for (i, w) in want.iter_mut().enumerate() {
                let (n, r) = (i / 128, i % 128);
                let (j, half, l) = (r / 32, (r % 32) / 16, r % 16);
                let v = rng.below(8) as i32 - 4; // -4..=3
                let (low, hbit) = if v >= 0 { (v as u8, 1u8) } else { ((v + 4) as u8, 0) };
                qs[n * 32 + half * 16 + l] |= low << (2 * j);
                hm[half * 16 + l] |= hbit << (n * 4 + j);
                *w = d * (s[n * 8 + j * 2 + half] as i32 - 32) as f32 * v as f32;
            }
            ([&hm[..], &qs[..], &sc[..], &h(d)].concat(), want)
        }
        GgmlType::Q4K | GgmlType::Q5K => {
            let five = t == GgmlType::Q5K;
            let sc: [u8; 8] = std::array::from_fn(|_| rng.below(64) as u8);
            let mn: [u8; 8] = std::array::from_fn(|_| rng.below(64) as u8);
            let (mut qh, mut qs) = ([0u8; 32], [0u8; 128]);
            let mut want = vec![0f32; QK_K];
            for (i, w) in want.iter_mut().enumerate() {
                let (j, r) = (i / 64, i % 64);
                let (sub, l) = (2 * j + r / 32, r % 32);
                let q = rng.below(if five { 32 } else { 16 }) as u8;
                qs[j * 32 + l] |= (q & 0xF) << (4 * (r / 32));
                qh[l] |= (q >> 4) << sub;
                *w = d * sc[sub] as f32 * q as f32 - dmin * mn[sub] as f32;
            }
            let mut b = [&h(d)[..], &h(dmin)[..], &pack_k4(&sc, &mn)[..]].concat();
            if five {
                b.extend(qh);
            }
            b.extend(qs);
            (b, want)
        }
        GgmlType::Q6K => {
            let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
            let sc: Vec<i8> = (0..16).map(|_| rng.below(128) as i8 - 64).collect();
            let mut want = vec![0f32; QK_K];
            for (i, w) in want.iter_mut().enumerate() {
                let (n, r) = (i / 128, i % 128);
                let (k, l) = (r / 32, r % 32);
                let q = rng.below(64) as u8;
                let byte = n * 64 + l + if k % 2 == 1 { 32 } else { 0 };
                ql[byte] |= (q & 0xF) << (4 * (k / 2));
                qh[n * 32 + l] |= (q >> 4) << (2 * k);
                *w = d * sc[n * 8 + l / 16 + 2 * k] as f32 * (q as i32 - 32) as f32;
            }
            let scb: Vec<u8> = sc.iter().map(|&x| x as u8).collect();
            ([&ql[..], &qh[..], &scb[..], &h(d)].concat(), want)
        }
        GgmlType::Q8K => {
            let q: Vec<i8> = (0..QK_K).map(|_| rng.next() as i8).collect();
            let mut b = 0.125f32.to_le_bytes().to_vec();
            b.extend(q.iter().map(|&x| x as u8));
            b.extend([0u8; 32]); // bsums (unused when dequantizing)
            (b, q.iter().map(|&x| 0.125 * x as f32).collect())
        }
        GgmlType::Q8_0 => {
            let q: Vec<i8> = (0..32).map(|_| rng.next() as i8).collect();
            let mut b = h(d).to_vec();
            b.extend(q.iter().map(|&x| x as u8));
            (b, q.iter().map(|&x| x as f32 * d).collect())
        }
        GgmlType::Q4_0 => {
            let q: Vec<u8> = (0..32).map(|_| rng.below(16) as u8).collect();
            let mut b = h(d).to_vec();
            b.extend((0..16).map(|j| q[j] | (q[j + 16] << 4)));
            (b, q.iter().map(|&x| (x as f32 - 8.0) * d).collect())
        }
        GgmlType::F16 => {
            let v: Vec<f32> = (0..4).map(|i| i as f32 * 0.5 - 1.0).collect();
            (v.iter().flat_map(|&x| h(x)).collect(), v)
        }
        GgmlType::BF16 => {
            let v: Vec<f32> = (0..4).map(|i| i as f32 * 0.25 - 0.5).collect();
            (v.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect(), v)
        }
        GgmlType::F32 => {
            let v: Vec<f32> = (0..4).map(|i| i as f32 * 0.1).collect();
            (v.iter().flat_map(|x| x.to_le_bytes()).collect(), v)
        }
        GgmlType::Other(_) => unreachable!("no sample for unsupported types"),
    }
}

pub(crate) const SUPPORTED: [GgmlType; 15] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q8_1,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q8K,
];

#[test]
fn every_type_dequantizes() {
    let mut rng = Lcg(7);
    for t in SUPPORTED {
        // Two blocks, to catch per-block offset errors.
        let (b1, w1) = sample_block(t, &mut rng);
        let (b2, w2) = sample_block(t, &mut rng);
        assert_eq!(Some(b1.len()), t.block().map(|(e, b)| b * w1.len() / e), "{}: block size", t.name());
        let got = dequantize(t, &[b1, b2].concat(), w1.len() * 2).expect("dequantize");
        close(&got, &[w1, w2].concat());
        assert_eq!(GgmlType::from_id(t.id()), t);
    }
}

#[test]
fn errors() {
    let e = dequantize(GgmlType::from_id(19), &[], 0).err().expect("iq1_s");
    assert_eq!(e.code, "unsupported_gguf_type");
    assert!(e.message.contains("iq1_s"), "{}", e.message);
    assert_eq!(dequantize(GgmlType::Q4K, &[0; 144], 128).err().map(|e| e.code), Some("format"));
    assert_eq!(dequantize(GgmlType::Q8_0, &[0; 33], 32).err().map(|e| e.code), Some("format"));
}

#[test]
fn spot_values_from_ggml() {
    // q4_K: d = 1, dmin = 1, scale0 = 3, min0 = 2, first nibble 5 -> 1*3*5 - 1*2 = 13.
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    sc[0] = 3;
    mn[0] = 2;
    let mut b = [&h(1.0)[..], &h(1.0)[..], &pack_k4(&sc, &mn)[..]].concat();
    b.extend([0u8; 128]);
    b[16] = 5;
    let x = dequantize(GgmlType::Q4K, &b, 256).expect("q4_K");
    assert_eq!((x[0], x[1], x[32]), (13.0, -2.0, 0.0));
    // q6_K: all ql/qh zero -> q = -32; scale[0] = 1, d = 0.5 -> -16.
    let mut b = vec![0u8; 210];
    b[192] = 1;
    b[208..210].copy_from_slice(&h(0.5));
    let x = dequantize(GgmlType::Q6K, &b, 256).expect("q6_K");
    assert_eq!((x[0], x[16]), (-16.0, 0.0));
}

/// Cross-check against ggml's own C code: `LAPID_GGML_REF=<file>` written by a tiny C
/// program linking `ggml-quants.c` (records of `u32 type, u32 nbytes, blocks, 512 f32` =
/// `quantize_row_*_ref` + `dequantize_row_*` of the same 512 floats). Skipped when unset.
#[test]
fn matches_ggml_c_reference() {
    let Some(path) = std::env::var_os("LAPID_GGML_REF") else { return };
    let b = std::fs::read(path).expect("read reference");
    let mut o = 0;
    let mut n = 0;
    while o < b.len() {
        let id = u32::from_le_bytes(b[o..o + 4].try_into().expect("4"));
        let nb = u32::from_le_bytes(b[o + 4..o + 8].try_into().expect("4")) as usize;
        let blocks = &b[o + 8..o + 8 + nb];
        let want: Vec<f32> =
            b[o + 8 + nb..o + 8 + nb + 2048].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4"))).collect();
        let t = GgmlType::from_id(id);
        assert_eq!(dequantize(t, blocks, 512).expect("dequantize"), want, "{}", t.name());
        o += 8 + nb + 2048;
        n += 1;
    }
    assert!(n > 0);
}

#[test]
fn mxfp4_and_integer_types() {
    // e = 129 → d = 2^(129 - 128) = 2; nibbles j and j + 16 per byte.
    let mut b = vec![129u8];
    b.extend((0..16u8).map(|j| (15 - j) << 4 | j));
    let t = GgmlType::from_id(39);
    assert_eq!(t.nbytes(64).expect("mxfp4 size"), 34);
    let got = dequantize(t, &b, 32).expect("mxfp4");
    let kv = [0f32, 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.];
    for j in 0..16 {
        assert_eq!(got[j], kv[j] * 2.0);
        assert_eq!(got[j + 16], kv[15 - j] * 2.0);
    }
    let i32s: Vec<u8> = [7i32, -3].iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = GgmlType::from_id(26);
    assert_eq!(t.nbytes(2).expect("i32 size"), 8);
    assert_eq!(dequantize(t, &i32s, 2).expect("i32"), vec![7.0, -3.0]);
}
