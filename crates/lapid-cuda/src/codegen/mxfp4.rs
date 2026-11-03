//! MXFP4 (ggml `block_mxfp4`, AoS as in GGUF): element dequantization (gather, scalar GEMM),
//! the GEMV lane dot product (half a block per lane) and the tensor-core GEMM B-tile loader.
//!
//! Block (17 bytes, 32 elements): `e u8 (e8m0) | qs[16]`; byte j holds element j (low
//! nibble) and j + 16 (high nibble); `x = KV[q] * 2^(e - 128)`, `KV` the doubled e2m1 values
//! `{0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}`. Rows are cols / 32 consecutive blocks, so
//! element `e` (row-major linear index) lives in block `e / 32`.
//!
//! Blocks are byte aligned only: multi-byte reads are aligned words combined with funnel
//! shifts, and never touch an aligned word that holds no byte of the block (the buffer is a
//! whole number of 256-byte units or ends exactly at a block, so no read leaves it).
//! Nibbles become biased bytes `KV[q] + 128` with `prmt` lookups into a table held in four
//! registers, and floats by splicing the byte into the mantissa of `2^23`.

use lapid_compiler::MatMulSpec;

use super::kquant::a_run;
use super::ptx::Gen;

/// Bytes per block.
const BB: usize = lapid_quant::mxfp4::BYTES;
/// `KV` magnitudes `KV[0..8]` as little-endian bytes.
const T: [u32; 2] = [0x0302_0100, 0x0C08_0604];
/// `KV + 128` as little-endian bytes: [0..4], [4..8], [8..12], [12..16].
const TB: [u32; 4] = [0x8382_8180, 0x8C88_8684, 0x7D7E_7F80, 0x7478_7A7C];
/// f32 `2^23` bits: `MAGIC | b` is the float `2^23 + b` for a byte b.
const MAGIC: u32 = 0x4B00_0000;

/// `ptr + blk * 17` (64-bit).
fn block_addr(g: &mut Gen, ptr: &str, blk: &str) -> String {
    let a = g.reg("b64");
    g.op(format!("mul.wide.u32 {a}, {blk}, {BB}"));
    g.op(format!("add.s64 {a}, {a}, {ptr}"));
    a
}

/// `d = 2^(e - 128)` (f32 bits; e < 2 denormal as ggml) from the scale byte `e` (u32).
fn scale(g: &mut Gen, e: &str) -> String {
    let (n, dn, p, f) = (g.reg("u32"), g.reg("u32"), g.reg("pred"), g.reg("f32"));
    g.op(format!("sub.u32 {n}, {e}, 1"));
    g.op(format!("shl.b32 {n}, {n}, 23"));
    g.op(format!("mov.b32 {dn}, 0x00200000"));
    g.op(format!("shl.b32 {dn}, {dn}, {e}"));
    g.op(format!("setp.lt.u32 {p}, {e}, 2"));
    g.op(format!("selp.b32 {n}, {dn}, {n}, {p}"));
    g.op(format!("mov.b32 {f}, {n}"));
    f
}

/// `n` (1..=2) 32-bit little-endian words starting at byte address `a` (any alignment) via
/// aligned loads; the last aligned load is the word holding byte `a + 4n - 1`.
fn load_words(g: &mut Gen, a: &str, dst: &[String]) {
    let n = dst.len();
    let (al, last, sh, lo) = (g.reg("b64"), g.reg("b64"), g.reg("u32"), g.reg("u32"));
    g.op(format!("and.b64 {al}, {a}, -4"));
    g.op(format!("add.s64 {last}, {a}, {}", 4 * n - 1));
    g.op(format!("and.b64 {last}, {last}, -4"));
    g.op(format!("cvt.u32.u64 {lo}, {a}"));
    g.op(format!("and.b32 {sh}, {lo}, 3"));
    g.op(format!("shl.b32 {sh}, {sh}, 3"));
    let x: Vec<String> = (0..=n).map(|_| g.reg("b32")).collect();
    for (i, xi) in x.iter().take(n).enumerate() {
        g.op(format!("ld.global.nc.u32 {xi}, [{al}+{}]", 4 * i));
    }
    g.op(format!("ld.global.nc.u32 {}, [{last}]", x[n]));
    for i in 0..n {
        g.op(format!("shf.r.clamp.b32 {}, {}, {}, {sh}", dst[i], x[i], x[i + 1]));
    }
}

/// Constant words in fresh registers.
fn consts<const N: usize>(g: &mut Gen, v: [u32; N]) -> [String; N] {
    std::array::from_fn(|i| {
        let r = g.reg("b32");
        g.op(format!("mov.b32 {r}, {}", v[i]));
        r
    })
}

/// Table registers `KV + 128` (4 x b32) and the `MAGIC` word.
fn table(g: &mut Gen) -> ([String; 4], String) {
    let [m] = consts(g, [MAGIC]);
    (consts(g, TB), m)
}

/// Decode the 8 nibbles of `w` (bytes b0..b3) into two words of biased bytes `KV[q] + 128`:
/// (lo nibbles of b0..b3, hi nibbles of b0..b3).
fn decode(g: &mut Gen, t: &[String; 4], w: &str) -> (String, String) {
    let (sel, hs) = (g.reg("b32"), g.reg("b32"));
    g.op(format!("and.b32 {sel}, {w}, 0x77777777"));
    g.op(format!("and.b32 {hs}, {w}, 0x88888888"));
    g.op(format!("shr.b32 {hs}, {hs}, 1"));
    g.op(format!("or.b32 {hs}, {hs}, 0x32103210"));
    let mut v = Vec::new();
    for i in 0..2 {
        let (s, h, lo, hi, r) = (g.reg("b32"), g.reg("b32"), g.reg("b32"), g.reg("b32"), g.reg("b32"));
        g.op(format!("shr.b32 {s}, {sel}, {}", 16 * i));
        g.op(format!("shr.b32 {h}, {hs}, {}", 16 * i));
        g.op(format!("prmt.b32 {lo}, {}, {}, {s}", t[0], t[1]));
        g.op(format!("prmt.b32 {hi}, {}, {}, {s}", t[2], t[3]));
        g.op(format!("prmt.b32 {r}, {lo}, {hi}, {h}"));
        v.push(r);
    }
    // v[0] = nibbles 0..4 (b0 lo, b0 hi, b1 lo, b1 hi), v[1] = nibbles 4..8
    let (even, odd) = (g.reg("b32"), g.reg("b32"));
    g.op(format!("prmt.b32 {even}, {}, {}, 0x6420", v[0], v[1]));
    g.op(format!("prmt.b32 {odd}, {}, {}, 0x7531", v[0], v[1]));
    (even, odd)
}

/// Biased byte `i` of `w` as the exact f32 `KV[q]`: the byte spliced into the mantissa of
/// `2^23` (`prmt`), minus `2^23 + 128` (no int -> float conversion).
fn byte_f32(g: &mut Gen, w: &str, magic: &str, i: usize) -> String {
    let f = g.reg("f32");
    g.op(format!("prmt.b32 {f}, {w}, {magic}, {}", 0x7650 | i));
    g.op(format!("sub.f32 {f}, {f}, 0f4B000080"));
    f
}

/// Dequantize element `e` (row-major linear index, u32 register) into `dst`.
pub(crate) fn load_elem(g: &mut Gen, ptr: &str, e: &str, dst: &str) {
    let (blk, pos, o, q, es) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {blk}, {e}, 5"));
    g.op(format!("and.b32 {pos}, {e}, 31"));
    let base = block_addr(g, ptr, &blk);
    g.op(format!("ld.global.nc.u8 {es}, [{base}]"));
    g.op(format!("and.b32 {o}, {pos}, 15"));
    let a = g.reg("b64");
    g.op(format!("cvt.u64.u32 {a}, {o}"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    g.op(format!("ld.global.nc.u8 {q}, [{a}+1]"));
    g.op(format!("shr.u32 {o}, {pos}, 4"));
    g.op(format!("shl.b32 {o}, {o}, 2"));
    g.op(format!("bfe.u32 {q}, {q}, {o}, 4"));
    // KV[q]: magnitude byte (q & 7) of table words 0/1, sign from bit 3
    let (m, s) = (g.reg("u32"), g.reg("pred"));
    let tb = consts(g, T);
    g.op(format!("and.b32 {m}, {q}, 7"));
    g.op(format!("prmt.b32 {m}, {}, {}, {m}", tb[0], tb[1]));
    let (f, d) = (g.reg("f32"), scale(g, &es));
    g.op(format!("cvt.rn.f32.u32 {f}, {m}"));
    g.op(format!("setp.gt.u32 {s}, {q}, 8")); // KV[8] = +0
    g.op(format!("@{s} neg.f32 {f}, {f}"));
    g.op(format!("mul.f32 {dst}, {f}, {d}"));
}

/// GEMV lane dot product over half a block: `out = sum A * B[gj, ..]` over the 16
/// elements of block k / 32 held by quant bytes 8h..8h+8 (h = (k / 16) % 2: elements
/// 8h + 0..8 (low nibbles) and 16 + 8h + 0..8 (high nibbles)); two lanes share a block, a
/// warp step reads 16 consecutive blocks. Expects `%gj`, `%K`, `%arow`, `%g_a`, `%g_b`.
pub(crate) fn gemv_dot_half(g: &mut Gen, spec: &MatMulSpec, vec_a: bool, k: &str, out: &str) {
    let (blk, h, t, es) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {t}, %K, 5"));
    g.op(format!("shr.u32 {blk}, {k}, 5"));
    g.op(format!("mad.lo.u32 {blk}, %gj, {t}, {blk}"));
    g.op(format!("bfe.u32 {h}, {k}, 4, 1"));
    let base = block_addr(g, "%g_b", &blk);
    g.op(format!("ld.global.nc.u8 {es}, [{base}]"));
    let a = g.reg("b64");
    g.op(format!("mul.wide.u32 {a}, {h}, 8"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    g.op(format!("add.s64 {a}, {a}, 1"));
    let w = [g.reg("b32"), g.reg("b32")];
    load_words(g, &a, &w);
    let (klo, khi) = (g.reg("u32"), g.reg("u32"));
    g.op(format!("and.b32 {klo}, {k}, -32"));
    g.op(format!("mad.lo.u32 {klo}, {h}, 8, {klo}"));
    g.op(format!("add.u32 {khi}, {klo}, 16"));
    let alo: Vec<String> = (0..2).flat_map(|c| a_run(g, spec, vec_a, &klo, 4 * c, 4)).collect();
    let ahi: Vec<String> = (0..2).flat_map(|c| a_run(g, spec, vec_a, &khi, 4 * c, 4)).collect();
    let (tb, magic) = table(g);
    let s = g.reg("f32");
    g.op(format!("mov.f32 {s}, 0f00000000"));
    for (i, wi) in w.iter().enumerate() {
        let (even, odd) = decode(g, &tb, wi);
        for (word, av) in [(&even, &alo), (&odd, &ahi)] {
            for j in 0..4 {
                let f = byte_f32(g, word, &magic, j);
                g.op(format!("fma.rn.f32 {s}, {f}, {}, {s}", av[4 * i + j]));
            }
        }
    }
    let d = scale(g, &es);
    g.op(format!("mul.f32 {out}, {s}, {d}"));
}

/// Tensor-core GEMM B tile (128 columns x 32 k = one block per column): thread t loads
/// column t / 2, quant bytes 8 * (t % 2) .. +8 (elements 8h + 0..8 and 16 + 8h + 0..8),
/// dequantizes them into f16 shared memory (row stride `lds` halves) like Int4.
/// Returns (REGS, B_LOAD, B_STORE).
pub(crate) fn mma_b_parts(lds: usize) -> (String, String, String) {
    let regs = "    .reg .b32 %bw<2>;\n    .reg .b32 %be;\n    .reg .pred %bv;\n".to_string();
    // ---- load
    let mut g = Gen::new();
    let (row, half, gj, blk, t) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("and.b32 {half}, %tz, 1"));
    g.op(format!("add.u32 {gj}, %col0, {row}"));
    g.op(format!("setp.lt.u32 %bv, {gj}, %N"));
    g.op(format!("selp.b32 {gj}, {gj}, 0, %bv"));
    g.op(format!("shr.u32 {t}, %K, 5"));
    g.op(format!("shr.u32 {blk}, %k0, 5"));
    g.op(format!("mad.lo.u32 {blk}, {gj}, {t}, {blk}"));
    let base = block_addr(&mut g, "%g_b", &blk);
    g.op(format!("ld.global.nc.u8 %be, [{base}]"));
    let a = g.reg("b64");
    g.op(format!("mul.wide.u32 {a}, {half}, 8"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    g.op(format!("add.s64 {a}, {a}, 1"));
    load_words(&mut g, &a, &["%bw0".to_string(), "%bw1".to_string()]);
    let load = g.finish();

    // ---- store
    let mut g = Gen::new();
    let d = scale(&mut g, "%be");
    g.op(format!("selp.f32 {d}, {d}, 0f00000000, %bv"));
    let (row, half, dst) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("and.b32 {half}, %tz, 1"));
    g.op(format!("mad.lo.u32 {dst}, {row}, {}, %sb_st", 2 * lds));
    g.op(format!("mad.lo.u32 {dst}, {half}, 16, {dst}"));
    let (tb, magic) = table(&mut g);
    let (l0, h0) = decode(&mut g, &tb, "%bw0");
    let (l1, h1) = decode(&mut g, &tb, "%bw1");
    let halves = |g: &mut Gen, words: [&String; 2]| -> Vec<String> {
        let f: Vec<String> = (0..8)
            .map(|i| {
                let v = byte_f32(g, words[i / 4], &magic, i % 4);
                g.op(format!("mul.f32 {v}, {v}, {d}"));
                v
            })
            .collect();
        f.chunks(2)
            .map(|p| {
                let r = g.reg("b32");
                g.op(format!("cvt.rn.f16x2.f32 {r}, {}, {}", p[1], p[0]));
                r
            })
            .collect()
    };
    let lo = halves(&mut g, [&l0, &l1]);
    let hi = halves(&mut g, [&h0, &h1]);
    g.op(format!("st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}}", lo[0], lo[1], lo[2], lo[3]));
    g.op(format!("st.shared.v4.b32 [{dst}+32], {{{}, {}, {}, {}}}", hi[0], hi[1], hi[2], hi[3]));
    let store = g.finish();
    (regs, load, store)
}
