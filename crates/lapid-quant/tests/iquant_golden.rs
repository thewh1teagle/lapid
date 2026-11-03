//! Bit-exactness of the i-quants against ggml: fixtures from `tools/iquant_golden/gen.sh`
//! (ggml's `quantize_iq*` / `dequantize_row_iq*` and grid tables).

const INPUT: &[u8] = include_bytes!("fixtures/iquant/input.bin");
const GRIDS: &[u8] = include_bytes!("fixtures/iquant/grids.bin");
const IQ2_XXS: &[u8] = include_bytes!("fixtures/iquant/iq2_xxs.bin");
const IQ2_S: &[u8] = include_bytes!("fixtures/iquant/iq2_s.bin");
const IQ3_XXS: &[u8] = include_bytes!("fixtures/iquant/iq3_xxs.bin");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

struct Input {
    rows: usize,
    cols: usize,
    x: Vec<f32>,
    w: Vec<f32>,
}

fn input() -> Input {
    let rows = u32::from_le_bytes(INPUT[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(INPUT[4..8].try_into().unwrap()) as usize;
    let x = f32s(&INPUT[8..8 + 4 * rows * cols]);
    let w = f32s(&INPUT[8 + 4 * rows * cols..]);
    assert_eq!(w.len(), cols);
    Input { rows, cols, x, w }
}

type Q = fn(&[f32], usize, Option<&[f32]>) -> Vec<u8>;
type D = fn(&[u8]) -> Vec<f32>;

/// Check `cases` consecutive (quant bytes, dequantized f32) sections of `fixture`.
fn check(name: &str, fixture: &[u8], bytes: usize, q: Q, d: D, imatrix: &[bool]) {
    let inp = input();
    let qn = inp.rows * inp.cols / 256 * bytes;
    let sec = qn + 4 * inp.rows * inp.cols;
    assert_eq!(fixture.len(), sec * imatrix.len(), "{name} fixture size");
    for (i, &with) in imatrix.iter().enumerate() {
        let s = &fixture[i * sec..(i + 1) * sec];
        let (want_q, want_y) = (&s[..qn], f32s(&s[qn..]));
        let got = q(&inp.x, inp.cols, with.then_some(&inp.w[..]));
        for (b, (g, w)) in got.chunks(bytes).zip(want_q.chunks(bytes)).enumerate() {
            assert_eq!(g, w, "{name} (imatrix {with}): block {b} differs from ggml");
        }
        let y = d(want_q);
        for (j, (a, b)) in y.iter().zip(&want_y).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{name}: dequantized element {j}: {a} vs ggml {b}");
        }
    }
}

#[test]
fn grids_match_ggml() {
    let u64s = |b: &[u8]| -> Vec<u64> { b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect() };
    let u32s = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() };
    assert_eq!(u64s(&GRIDS[..2048]), lapid_quant::IQ2XXS_GRID);
    assert_eq!(u64s(&GRIDS[2048..10240]), lapid_quant::IQ2S_GRID);
    assert_eq!(u32s(&GRIDS[10240..11264]), lapid_quant::IQ3XXS_GRID);
    assert_eq!(&GRIDS[11264..], &lapid_quant::KSIGNS[..]);
}

#[test]
fn iq2_xxs_matches_ggml() {
    use lapid_quant::iq2_xxs as m;
    check("iq2_xxs", IQ2_XXS, m::BYTES, m::quantize, m::dequantize, &[true]);
}

#[test]
fn iq2_s_matches_ggml() {
    use lapid_quant::iq2_s as m;
    check("iq2_s", IQ2_S, m::BYTES, m::quantize, m::dequantize, &[true, false]);
}

#[test]
fn iq3_xxs_matches_ggml() {
    use lapid_quant::iq3_xxs as m;
    check("iq3_xxs", IQ3_XXS, m::BYTES, m::quantize, m::dequantize, &[true, false]);
}

#[test]
fn layout_and_generic_quantize() {
    use lapid_quant::{Format, dequantize, layout, quantize};
    let inp = input();
    for (f, bytes, bpw) in [(Format::Iq2Xxs, 66, 2.0625), (Format::Iq2S, 82, 2.5625), (Format::Iq3Xxs, 98, 3.0625)] {
        assert_eq!(f.block_bytes(), bytes);
        assert_eq!(f.block_elems(), 256);
        assert_eq!(bytes as f64 * 8.0 / 256.0, bpw);
        let l = layout(f, inp.rows, inp.cols);
        assert_eq!(l.qs_bytes, inp.rows * inp.cols / 256 * bytes);
        let q = quantize(f, &inp.x, inp.rows, inp.cols);
        assert_eq!(q.len(), l.total);
        let y = dequantize(f, &q, inp.rows, inp.cols);
        // relative RMS error of a smooth row
        let r = 7 * inp.cols;
        let (mut e, mut s) = (0f64, 0f64);
        for i in r..r + inp.cols {
            e += f64::from(inp.x[i] - y[i]).powi(2);
            s += f64::from(inp.x[i]).powi(2);
        }
        let rel = (e / s).sqrt();
        assert!(rel < 0.5, "{f:?}: relative rms error {rel}");
    }
}

/// Quantization throughput on one DeepSeek V4.1 expert matrix (ignored; release):
/// `cargo test -p lapid-quant --release --test iquant_golden bench_quantize -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_quantize() {
    let (rows, cols) = (2304, 5120);
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let x: Vec<f32> = (0..rows * cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02
        })
        .collect();
    let w: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 7) as f32).collect();
    type Q = fn(&[f32], usize, Option<&[f32]>) -> Vec<u8>;
    let fmts: [(&str, Q); 3] = [("iq2_xxs", lapid_quant::iq2_xxs::quantize), ("iq2_s", lapid_quant::iq2_s::quantize), ("iq3_xxs", lapid_quant::iq3_xxs::quantize)];
    for (name, q) in fmts {
        let _ = q(&x[..cols * 8], cols, Some(&w)); // lattice init
        let t = std::time::Instant::now();
        let b = q(&x, cols, Some(&w));
        let dt = t.elapsed().as_secs_f64();
        println!("{name}: {rows}x{cols} in {:.3} s ({:.1} M elements/s, {} bytes)", dt, (rows * cols) as f64 / dt / 1e6, b.len());
    }
}
