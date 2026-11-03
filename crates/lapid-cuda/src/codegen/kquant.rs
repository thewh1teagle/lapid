//! ggml K-quant super-blocks (Q4_K / Q5_K / Q6_K, AoS exactly as in GGUF): element
//! dequantization (gather, scalar GEMM), the GEMV lane dot product and the tensor-core GEMM
//! B-tile loader.
//!
//! Block layouts (bytes, 256 elements each):
//! - Q4_K (144): `d f16 | dmin f16 | scales[12] | qs[128]`; sub-block j (32 elements) uses
//!   the low (j even) / high (j odd) nibbles of `qs[32*(j/2)..+32]`, 6-bit scale / min
//!   packed in `scales` (`get_scale_min_k4`): x = d*sc*q - dmin*m.
//! - Q5_K (176): `d | dmin | scales[12] | qh[32] | qs[128]`; as Q4_K plus bit j of `qh[l]`
//!   as the 5th bit (16).
//! - Q6_K (210): `ql[128] | qh[64] | scales i8[16] | d f16`; two halves of 128 elements,
//!   element (half n, quarter qq, l) = (nibble qq/2 of `ql[64n + 32*(qq%2) + l]`) |
//!   (bits 2qq.. of `qh[32n + l]`) << 4, minus 32, times d * scales[8n + l/16 + 2qq].
//!
//! A row of a quantized `[rows, cols]` matrix is cols / 256 consecutive blocks, so element
//! `e` (row-major linear index) lives in block `e / 256` at position `e % 256`.

use lapid_compiler::MatMulSpec;
use lapid_tensor::DType;

use super::ptx::{self, Gen};

pub(crate) fn is_kquant(d: DType) -> bool {
    matches!(d, DType::Q4K | DType::Q5K | DType::Q6K)
}

/// Bytes per 256-element block.
pub(crate) fn block_bytes(d: DType) -> usize {
    d.quant_format().map(|f| f.block_bytes()).unwrap_or(0)
}

/// `ptr + blk * block_bytes` (64-bit).
fn block_addr(g: &mut Gen, d: DType, ptr: &str, blk: &str) -> String {
    let a = g.reg("b64");
    g.op(format!("mul.wide.u32 {a}, {blk}, {}", block_bytes(d)));
    g.op(format!("add.s64 {a}, {a}, {ptr}"));
    a
}

/// (d, dmin) as f32 from the first header word of a Q4_K / Q5_K block.
fn d_dmin(g: &mut Gen, w0: &str) -> (String, String) {
    let (h0, h1) = (g.reg("b16"), g.reg("b16"));
    let (d, m) = (g.reg("f32"), g.reg("f32"));
    g.op(format!("mov.b32 {{{h0}, {h1}}}, {w0}"));
    g.op(format!("cvt.f32.f16 {d}, {h0}"));
    g.op(format!("cvt.f32.f16 {m}, {h1}"));
    (d, m)
}

/// 6-bit (scale, min) of sub-block `j` (u32 register, 0..8) as f32, from the scale words
/// `w1, w2, w3` (= scale bytes 0..4, 4..8, 8..12), ggml `get_scale_min_k4`.
fn scale_min(g: &mut Gen, w: [&str; 3], j: &str) -> (String, String) {
    let (bp, a, b, c) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("and.b32 {bp}, {j}, 3"));
    g.op(format!("shl.b32 {bp}, {bp}, 3"));
    g.op(format!("bfe.u32 {a}, {}, {bp}, 8", w[0]));
    g.op(format!("bfe.u32 {b}, {}, {bp}, 8", w[1]));
    g.op(format!("bfe.u32 {c}, {}, {bp}, 8", w[2]));
    let lo = g.reg("pred");
    g.op(format!("setp.lt.u32 {lo}, {j}, 4"));
    let (sl, ml, sh, mh, t) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    // j < 4: sc = s[j] & 63, m = s[j+4] & 63
    g.op(format!("and.b32 {sl}, {a}, 63"));
    g.op(format!("and.b32 {ml}, {b}, 63"));
    // j >= 4: sc = (s[j+4] & 15) | (s[j-4] >> 6) << 4 ; m = (s[j+4] >> 4) | (s[j] >> 6) << 4
    g.op(format!("and.b32 {sh}, {c}, 15"));
    g.op(format!("shr.u32 {t}, {a}, 6"));
    g.op(format!("shl.b32 {t}, {t}, 4"));
    g.op(format!("or.b32 {sh}, {sh}, {t}"));
    g.op(format!("shr.u32 {mh}, {c}, 4"));
    g.op(format!("shr.u32 {t}, {b}, 6"));
    g.op(format!("shl.b32 {t}, {t}, 4"));
    g.op(format!("or.b32 {mh}, {mh}, {t}"));
    g.op(format!("selp.b32 {sl}, {sl}, {sh}, {lo}"));
    g.op(format!("selp.b32 {ml}, {ml}, {mh}, {lo}"));
    let (sf, mf) = (g.reg("f32"), g.reg("f32"));
    g.op(format!("cvt.rn.f32.u32 {sf}, {sl}"));
    g.op(format!("cvt.rn.f32.u32 {mf}, {ml}"));
    (sf, mf)
}

/// Dequantize element `e` (row-major linear index, u32 register) of a K-quant buffer.
pub(crate) fn load_elem(g: &mut Gen, d: DType, ptr: &str, e: &str, dst: &str) {
    let (blk, pos) = (g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {blk}, {e}, 8"));
    g.op(format!("and.b32 {pos}, {e}, 255"));
    let base = block_addr(g, d, ptr, &blk);
    if d == DType::Q6K {
        let (n, r, qq, l, t, o) = (
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
            g.reg("u32"),
        );
        g.op(format!("shr.u32 {n}, {pos}, 7"));
        g.op(format!("and.b32 {r}, {pos}, 127"));
        g.op(format!("shr.u32 {qq}, {r}, 5"));
        g.op(format!("and.b32 {l}, {r}, 31"));
        // ql byte 64n + 32*(qq & 1) + l, nibble qq >> 1
        g.op(format!("and.b32 {t}, {qq}, 1"));
        g.op(format!("shl.b32 {o}, {t}, 5"));
        g.op(format!("mad.lo.u32 {o}, {n}, 64, {o}"));
        g.op(format!("add.u32 {o}, {o}, {l}"));
        let (lw, hw, sh) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
        let a = g.reg("b64");
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u8 {lw}, [{a}]"));
        g.op(format!("shr.u32 {sh}, {qq}, 1"));
        g.op(format!("shl.b32 {sh}, {sh}, 2"));
        g.op(format!("bfe.u32 {lw}, {lw}, {sh}, 4"));
        // qh byte 128 + 32n + l, bits 2qq..2qq+2
        g.op(format!("mad.lo.u32 {o}, {n}, 32, {l}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u8 {hw}, [{a}+128]"));
        g.op(format!("shl.b32 {sh}, {qq}, 1"));
        g.op(format!("bfe.u32 {hw}, {hw}, {sh}, 2"));
        g.op(format!("shl.b32 {hw}, {hw}, 4"));
        g.op(format!("or.b32 {lw}, {lw}, {hw}"));
        g.op(format!("sub.s32 {lw}, {lw}, 32"));
        // scale i8 at 192 + 8n + l/16 + 2qq
        g.op(format!("shr.u32 {o}, {l}, 4"));
        g.op(format!("mad.lo.u32 {o}, {qq}, 2, {o}"));
        g.op(format!("mad.lo.u32 {o}, {n}, 8, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        let sc = g.reg("s32");
        g.op(format!("ld.global.nc.s8 {sc}, [{a}+192]"));
        let hd = g.reg("b16");
        g.op(format!("ld.global.nc.b16 {hd}, [{base}+208]"));
        let (df, sf, qf) = (g.reg("f32"), g.reg("f32"), g.reg("f32"));
        g.op(format!("cvt.f32.f16 {df}, {hd}"));
        g.op(format!("cvt.rn.f32.s32 {sf}, {sc}"));
        g.op(format!("cvt.rn.f32.s32 {qf}, {lw}"));
        g.op(format!("mul.f32 {df}, {df}, {sf}"));
        g.op(format!("mul.f32 {dst}, {df}, {qf}"));
        return;
    }
    let w: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
    g.op(format!(
        "ld.global.nc.v4.u32 {{{}, {}, {}, {}}}, [{base}]",
        w[0], w[1], w[2], w[3]
    ));
    let (j, l, o, q, t) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("shr.u32 {j}, {pos}, 5"));
    g.op(format!("and.b32 {l}, {pos}, 31"));
    let qs0 = if d == DType::Q4K { 16 } else { 48 };
    g.op(format!("shr.u32 {o}, {j}, 1"));
    g.op(format!("mad.lo.u32 {o}, {o}, 32, {l}"));
    let a = g.reg("b64");
    g.op(format!("cvt.u64.u32 {a}, {o}"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    g.op(format!("ld.global.nc.u8 {q}, [{a}+{qs0}]"));
    g.op(format!("and.b32 {t}, {j}, 1"));
    g.op(format!("shl.b32 {t}, {t}, 2"));
    g.op(format!("bfe.u32 {q}, {q}, {t}, 4"));
    if d == DType::Q5K {
        let h = g.reg("u32");
        g.op(format!("cvt.u64.u32 {a}, {l}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u8 {h}, [{a}+16]"));
        g.op(format!("bfe.u32 {h}, {h}, {j}, 1"));
        g.op(format!("shl.b32 {h}, {h}, 4"));
        g.op(format!("or.b32 {q}, {q}, {h}"));
    }
    let (df, mf) = d_dmin(g, &w[0]);
    let (sf, nf) = scale_min(g, [&w[1], &w[2], &w[3]], &j);
    let qf = g.reg("f32");
    g.op(format!("cvt.rn.f32.u32 {qf}, {q}"));
    g.op(format!("mul.f32 {df}, {df}, {sf}"));
    g.op(format!("mul.f32 {mf}, {mf}, {nf}"));
    g.op(format!("mul.f32 {qf}, {df}, {qf}"));
    g.op(format!("sub.f32 {dst}, {qf}, {mf}"));
}

/// A[%arow + kx + off + t] for t in 0..len into fresh f32 registers (`vec`: f32 A whose
/// row start and `kx + off` are aligned to `len` elements → one vector load).
pub(crate) fn a_run(g: &mut Gen, spec: &MatMulSpec, vec: bool, kx: &str, off: usize, len: usize) -> Vec<String> {
    let v: Vec<String> = (0..len).map(|_| g.reg("f32")).collect();
    let idx = g.reg("u32");
    g.op(format!("add.u32 {idx}, %arow, {kx}"));
    if vec {
        let a = g.addr("%g_a", &idx, 4);
        let regs = v.join(", ");
        g.op(format!("ld.global.nc.v{len}.f32 {{{regs}}}, [{a}+{}]", 4 * off));
        return v;
    }
    for (t, dst) in v.iter().enumerate() {
        let i = g.reg("u32");
        g.op(format!("add.u32 {i}, {idx}, {}", off + t));
        ptx::load(g, spec.a_dtype, "%g_a", &i, dst, true);
    }
    v
}

/// GEMV lane dot product over one 256-element block: `out = sum A * B[gj, ..]` over the 8
/// elements of lane `(k >> 3) & 31` of the block containing k-offset `k` (see
/// `kernels/gemv.ptx`; every lane of the warp works on the same block). Expects `%gj`,
/// `%K`, `%arow`, `%g_a`, `%g_b`.
pub(crate) fn gemv_dot(g: &mut Gen, spec: &MatMulSpec, vec_a: bool, k: &str, out: &str) {
    let d = spec.b_dtype;
    let (lane, kblk, blk, t) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    g.op(format!("bfe.u32 {lane}, {k}, 3, 5"));
    g.op(format!("and.b32 {kblk}, {k}, -256"));
    g.op(format!("shr.u32 {t}, %K, 8"));
    g.op(format!("shr.u32 {blk}, {k}, 8"));
    g.op(format!("mad.lo.u32 {blk}, %gj, {t}, {blk}"));
    let base = block_addr(g, d, "%g_b", &blk);
    let a = g.reg("b64");
    if d == DType::Q6K {
        // half n = lane / 16, l0 = 2 * (lane % 16): elements l0, l0+1 of each quarter
        let (n, l0, o, kx) = (g.reg("u32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
        g.op(format!("shr.u32 {n}, {lane}, 4"));
        g.op(format!("and.b32 {l0}, {lane}, 15"));
        g.op(format!("shl.b32 {l0}, {l0}, 1"));
        let (ql0, ql1, qh) = (g.reg("u32"), g.reg("u32"), g.reg("u32"));
        g.op(format!("mad.lo.u32 {o}, {n}, 64, {l0}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u16 {ql0}, [{a}]"));
        g.op(format!("ld.global.nc.u16 {ql1}, [{a}+32]"));
        g.op(format!("mad.lo.u32 {o}, {n}, 32, {l0}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u16 {qh}, [{a}+128]"));
        // scales 8n + l0/16 + 2g
        g.op(format!("shr.u32 {o}, {l0}, 4"));
        g.op(format!("mad.lo.u32 {o}, {n}, 8, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        let sc: Vec<String> = (0..4).map(|_| g.reg("s32")).collect();
        for (q, s) in sc.iter().enumerate() {
            g.op(format!("ld.global.nc.s8 {s}, [{a}+{}]", 192 + 2 * q));
        }
        let hd = g.reg("b16");
        g.op(format!("ld.global.nc.b16 {hd}, [{base}+208]"));
        g.op(format!("mad.lo.u32 {kx}, {n}, 128, {kblk}"));
        g.op(format!("add.u32 {kx}, {kx}, {l0}"));
        let (qi, hb, f, s, sf) = (g.reg("u32"), g.reg("u32"), g.reg("f32"), g.reg("f32"), g.reg("f32"));
        g.op(format!("mov.f32 {out}, 0f00000000"));
        for (q, scq) in sc.iter().enumerate() {
            let av = a_run(g, spec, vec_a, &kx, 32 * q, 2);
            let w = if q % 2 == 0 { &ql0 } else { &ql1 };
            g.op(format!("mov.f32 {s}, 0f00000000"));
            for (tt, at) in av.iter().enumerate() {
                g.op(format!("bfe.u32 {qi}, {w}, {}, 4", 8 * tt + 4 * (q / 2)));
                g.op(format!("bfe.u32 {hb}, {qh}, {}, 2", 8 * tt + 2 * q));
                g.op(format!("shl.b32 {hb}, {hb}, 4"));
                g.op(format!("or.b32 {qi}, {qi}, {hb}"));
                g.op(format!("sub.s32 {qi}, {qi}, 32"));
                g.op(format!("cvt.rn.f32.s32 {f}, {qi}"));
                g.op(format!("fma.rn.f32 {s}, {f}, {at}, {s}"));
            }
            g.op(format!("cvt.rn.f32.s32 {sf}, {scq}"));
            g.op(format!("fma.rn.f32 {out}, {sf}, {s}, {out}"));
        }
        let df = g.reg("f32");
        g.op(format!("cvt.f32.f16 {df}, {hd}"));
        g.op(format!("mul.f32 {out}, {out}, {df}"));
        return;
    }
    // Q4_K / Q5_K: pair p = lane / 8 (sub-blocks 2p, 2p+1), c = lane % 8: qs bytes 4*lane..+4
    // (low nibbles → k = 64p + 4c + i, high → +32)
    let w: Vec<String> = (0..4).map(|_| g.reg("b32")).collect();
    g.op(format!(
        "ld.global.nc.v4.u32 {{{}, {}, {}, {}}}, [{base}]",
        w[0], w[1], w[2], w[3]
    ));
    let (o, qs, p, c, kx) = (g.reg("u32"), g.reg("b32"), g.reg("u32"), g.reg("u32"), g.reg("u32"));
    let qs0 = if d == DType::Q4K { 16 } else { 48 };
    g.op(format!("shl.b32 {o}, {lane}, 2"));
    g.op(format!("cvt.u64.u32 {a}, {o}"));
    g.op(format!("add.s64 {a}, {a}, {base}"));
    g.op(format!("ld.global.nc.u32 {qs}, [{a}+{qs0}]"));
    g.op(format!("shr.u32 {p}, {lane}, 3"));
    g.op(format!("and.b32 {c}, {lane}, 7"));
    let qh = if d == DType::Q5K {
        // qh bytes 4c..4c+4: bit 2p (low nibbles) / 2p+1 (high nibbles)
        let qh = g.reg("b32");
        g.op(format!("shl.b32 {o}, {c}, 2"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.u32 {qh}, [{a}+16]"));
        g.op(format!("shl.b32 {o}, {p}, 1"));
        g.op(format!("shr.b32 {qh}, {qh}, {o}"));
        Some(qh)
    } else {
        None
    };
    g.op(format!("shl.b32 {kx}, {p}, 6"));
    g.op(format!("mad.lo.u32 {kx}, {c}, 4, {kx}"));
    g.op(format!("add.u32 {kx}, {kx}, {kblk}"));
    let alo = a_run(g, spec, vec_a, &kx, 0, 4);
    let ahi = a_run(g, spec, vec_a, &kx, 32, 4);
    let (qi, hb, f) = (g.reg("u32"), g.reg("u32"), g.reg("f32"));
    let sums: Vec<String> = (0..4).map(|_| g.reg("f32")).collect();
    for s in &sums {
        g.op(format!("mov.f32 {s}, 0f00000000"));
    }
    for (hi, av) in [&alo, &ahi].iter().enumerate() {
        for (i, at) in av.iter().enumerate() {
            g.op(format!("bfe.u32 {qi}, {qs}, {}, 4", 8 * i + 4 * hi));
            if let Some(qh) = &qh {
                g.op(format!("bfe.u32 {hb}, {qh}, {}, 1", 8 * i + hi));
                g.op(format!("shl.b32 {hb}, {hb}, 4"));
                g.op(format!("or.b32 {qi}, {qi}, {hb}"));
            }
            g.op(format!("cvt.rn.f32.u32 {f}, {qi}"));
            g.op(format!("fma.rn.f32 {}, {f}, {at}, {}", sums[2 * hi], sums[2 * hi]));
            g.op(format!("add.f32 {}, {}, {at}", sums[2 * hi + 1], sums[2 * hi + 1]));
        }
    }
    let (dd, dm) = d_dmin(g, &w[0]);
    let (j0, j1) = (g.reg("u32"), g.reg("u32"));
    g.op(format!("shl.b32 {j0}, {p}, 1"));
    g.op(format!("add.u32 {j1}, {j0}, 1"));
    let (s0, m0) = scale_min(g, [&w[1], &w[2], &w[3]], &j0);
    let (s1, m1) = scale_min(g, [&w[1], &w[2], &w[3]], &j1);
    let (x, y) = (g.reg("f32"), g.reg("f32"));
    g.op(format!("mul.f32 {x}, {s0}, {}", sums[0]));
    g.op(format!("fma.rn.f32 {x}, {s1}, {}, {x}", sums[2]));
    g.op(format!("mul.f32 {y}, {m0}, {}", sums[1]));
    g.op(format!("fma.rn.f32 {y}, {m1}, {}, {y}", sums[3]));
    g.op(format!("mul.f32 {y}, {y}, {dm}"));
    g.op(format!("mul.f32 {x}, {x}, {dd}"));
    g.op(format!("sub.f32 {out}, {x}, {y}"));
}

/// Tensor-core GEMM B tile (128 columns x 32 k = one sub-block per column): thread t loads
/// column t / 2, elements 16 * (t % 2) .. +16 of sub-block (k0 / 32) % 8, dequantizes them
/// into f16 shared memory (row stride `lds` halves). Returns (REGS, B_LOAD, B_STORE).
pub(crate) fn mma_b_parts(d: DType, lds: usize) -> (String, String, String) {
    let six = d == DType::Q6K;
    let mut regs = String::from("    .reg .pred %bv;\n");
    if six {
        regs.push_str("    .reg .b32 %kq<4>;\n    .reg .b32 %kh<4>;\n    .reg .s32 %ks;\n    .reg .b16 %kd;\n");
    } else {
        regs.push_str("    .reg .b32 %kw<4>;\n    .reg .b32 %kq<4>;\n");
        if d == DType::Q5K {
            regs.push_str("    .reg .b32 %kh<4>;\n");
        }
    }
    // ---- load
    let mut g = Gen::new();
    let (row, half, gj, blk, t, sb, o) = (
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
    );
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("and.b32 {half}, %tz, 1"));
    g.op(format!("add.u32 {gj}, %col0, {row}"));
    g.op(format!("setp.lt.u32 %bv, {gj}, %N"));
    g.op(format!("selp.b32 {gj}, {gj}, 0, %bv"));
    g.op(format!("shr.u32 {t}, %K, 8"));
    g.op(format!("shr.u32 {blk}, %k0, 8"));
    g.op(format!("mad.lo.u32 {blk}, {gj}, {t}, {blk}"));
    g.op(format!("bfe.u32 {sb}, %k0, 5, 3"));
    let base = block_addr(&mut g, d, "%g_b", &blk);
    let a = g.reg("b64");
    if six {
        let (n, qq) = (g.reg("u32"), g.reg("u32"));
        g.op(format!("shr.u32 {n}, {sb}, 2"));
        g.op(format!("and.b32 {qq}, {sb}, 3"));
        // ql bytes 64n + 32*(qq & 1) + 16*half .. +16
        g.op(format!("and.b32 {o}, {qq}, 1"));
        g.op(format!("shl.b32 {o}, {o}, 5"));
        g.op(format!("mad.lo.u32 {o}, {n}, 64, {o}"));
        g.op(format!("mad.lo.u32 {o}, {half}, 16, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        // blocks are 210 bytes (2-byte aligned): 16 bytes = 5 aligned words + funnel shifts
        let load16 = |g: &mut Gen, a: &str, dst: &str| {
            let (al, lo32, sh) = (g.reg("b64"), g.reg("u32"), g.reg("u32"));
            let x: Vec<String> = (0..5).map(|_| g.reg("b32")).collect();
            g.op(format!("and.b64 {al}, {a}, -4"));
            g.op(format!("cvt.u32.u64 {lo32}, {a}"));
            g.op(format!("and.b32 {sh}, {lo32}, 3"));
            g.op(format!("shl.b32 {sh}, {sh}, 3"));
            for (i, xi) in x.iter().enumerate() {
                g.op(format!("ld.global.nc.u32 {xi}, [{al}+{}]", 4 * i));
            }
            for i in 0..4 {
                g.op(format!("shf.r.clamp.b32 %{dst}{i}, {}, {}, {sh}", x[i], x[i + 1]));
            }
        };
        load16(&mut g, &a, "kq");
        // qh bytes 128 + 32n + 16*half .. +16
        g.op(format!("mad.lo.u32 {o}, {n}, 32, 128"));
        g.op(format!("mad.lo.u32 {o}, {half}, 16, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        load16(&mut g, &a, "kh");
        // scale 192 + 8n + half + 2qq
        g.op(format!("mad.lo.u32 {o}, {n}, 8, {half}"));
        g.op(format!("mad.lo.u32 {o}, {qq}, 2, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.s8 %ks, [{a}+192]"));
        g.op(format!("ld.global.nc.b16 %kd, [{base}+208]"));
    } else {
        g.op(format!("ld.global.nc.v4.u32 {{%kw0, %kw1, %kw2, %kw3}}, [{base}]"));
        let qs0 = if d == DType::Q4K { 16 } else { 48 };
        g.op(format!("shr.u32 {o}, {sb}, 1"));
        g.op(format!("shl.b32 {o}, {o}, 5"));
        g.op(format!("mad.lo.u32 {o}, {half}, 16, {o}"));
        g.op(format!("cvt.u64.u32 {a}, {o}"));
        g.op(format!("add.s64 {a}, {a}, {base}"));
        g.op(format!("ld.global.nc.v4.u32 {{%kq0, %kq1, %kq2, %kq3}}, [{a}+{qs0}]"));
        if d == DType::Q5K {
            g.op(format!("mul.wide.u32 {a}, {half}, 16"));
            g.op(format!("add.s64 {a}, {a}, {base}"));
            g.op(format!("ld.global.nc.v4.u32 {{%kh0, %kh1, %kh2, %kh3}}, [{a}+16]"));
        }
    }
    let load = g.finish();

    // ---- store
    let mut g = Gen::new();
    let (row, half, sb, dst, q, h, sh, hs) = (
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
        g.reg("u32"),
    );
    g.op(format!("shr.u32 {row}, %tz, 1"));
    g.op(format!("and.b32 {half}, %tz, 1"));
    g.op(format!("bfe.u32 {sb}, %k0, 5, 3"));
    g.op(format!("mul.lo.u32 {dst}, {row}, {}", 2 * lds));
    g.op(format!("mad.lo.u32 {dst}, {half}, 32, {dst}"));
    g.op(format!("add.u32 {dst}, {dst}, %sb_st"));
    let f = g.reg("f32");
    let mut vals = Vec::with_capacity(16);
    if six {
        let (qq, ds, sf) = (g.reg("u32"), g.reg("f32"), g.reg("f32"));
        g.op(format!("and.b32 {qq}, {sb}, 3"));
        g.op(format!("cvt.f32.f16 {ds}, %kd"));
        g.op(format!("cvt.rn.f32.s32 {sf}, %ks"));
        g.op(format!("mul.f32 {ds}, {ds}, {sf}"));
        g.op(format!("selp.f32 {ds}, {ds}, 0f00000000, %bv"));
        // nibble shift 4 * (qq >> 1), high-bits shift 2 * qq
        g.op(format!("shr.u32 {sh}, {qq}, 1"));
        g.op(format!("shl.b32 {sh}, {sh}, 2"));
        g.op(format!("shl.b32 {hs}, {qq}, 1"));
        // 4 values per word: (ql >> sh) & 0xF | ((qh >> hs) & 3) << 4, minus 32
        let (b, m32) = (g.reg("u32"), g.reg("f32"));
        g.op(format!("mul.f32 {m32}, {ds}, 0fC2000000"));
        for w in 0..4 {
            g.op(format!("shr.b32 {q}, %kq{w}, {sh}"));
            g.op(format!("and.b32 {q}, {q}, 0x0F0F0F0F"));
            g.op(format!("shr.b32 {h}, %kh{w}, {hs}"));
            g.op(format!("and.b32 {h}, {h}, 0x03030303"));
            g.op(format!("shl.b32 {h}, {h}, 4"));
            g.op(format!("or.b32 {q}, {q}, {h}"));
            for j in 0..4 {
                g.op(format!("bfe.u32 {b}, {q}, {}, 8", 8 * j));
                g.op(format!("cvt.rn.f32.u32 {f}, {b}"));
                let v = g.reg("f32");
                g.op(format!("fma.rn.f32 {v}, {ds}, {f}, {m32}"));
                vals.push(v);
            }
        }
    } else {
        let (dd, dm) = d_dmin(&mut g, "%kw0");
        let (sf, mf) = scale_min(&mut g, ["%kw1", "%kw2", "%kw3"], &sb);
        g.op(format!("mul.f32 {dd}, {dd}, {sf}"));
        g.op(format!("mul.f32 {dm}, {dm}, {mf}"));
        g.op(format!("selp.f32 {dd}, {dd}, 0f00000000, %bv"));
        g.op(format!("selp.f32 {dm}, {dm}, 0f00000000, %bv"));
        g.op(format!("neg.f32 {dm}, {dm}"));
        g.op(format!("and.b32 {sh}, {sb}, 1"));
        g.op(format!("shl.b32 {sh}, {sh}, 2"));
        let pos = g.reg("u32");
        for t in 0..16 {
            let byte = 8 * (t % 4);
            g.op(format!("add.u32 {pos}, {sh}, {byte}"));
            g.op(format!("bfe.u32 {q}, %kq{}, {pos}, 4", t / 4));
            if d == DType::Q5K {
                g.op(format!("add.u32 {pos}, {sb}, {byte}"));
                g.op(format!("bfe.u32 {h}, %kh{}, {pos}, 1", t / 4));
                g.op(format!("shl.b32 {h}, {h}, 4"));
                g.op(format!("or.b32 {q}, {q}, {h}"));
            }
            g.op(format!("cvt.rn.f32.u32 {f}, {q}"));
            let v = g.reg("f32");
            g.op(format!("fma.rn.f32 {v}, {dd}, {f}, {dm}"));
            vals.push(v);
        }
    }
    let hw: Vec<String> = vals
        .chunks(2)
        .map(|p| {
            let r = g.reg("b32");
            g.op(format!("cvt.rn.f16x2.f32 {r}, {}, {}", p[1], p[0]));
            r
        })
        .collect();
    g.op(format!(
        "st.shared.v4.b32 [{dst}], {{{}, {}, {}, {}}}",
        hw[0], hw[1], hw[2], hw[3]
    ));
    g.op(format!(
        "st.shared.v4.b32 [{dst}+16], {{{}, {}, {}, {}}}",
        hw[4], hw[5], hw[6], hw[7]
    ));
    let store = g.finish();
    (regs, load, store)
}
