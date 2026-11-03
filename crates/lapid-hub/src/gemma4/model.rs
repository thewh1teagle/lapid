//! Gemma 4 text decoder, dense or MoE (HF `Gemma4TextModel`, `Gemma4Model.forward` for the
//! image rows).
//!
//! Input `embed(ids) * sqrt(D)` with image rows replacing their placeholders; per-layer
//! embeddings (PLE, E2B / E4B) `(norm(proj(x) / sqrt(D)) + embed_pl(ids) * sqrt(P)) / sqrt(2)`
//! look up the pad id at image rows. Layer (sandwich norms):
//! `x += post_attn_norm(attn(input_norm(x)))`; `f = geglu(pre_ff_norm(x))`, MoE layers
//! `f = post_ff_norm_1(f) + moe(x)`; `x += post_ff_norm(f)`; PLE gate block;
//! `x *= layer_scalar`. Attention: q/k RMSNorm, weight-free v norm, scale 1, GQA, zero-tail
//! RoPE, sliding (window) or full layers (head dim 512: unfused), KV sharing for the last
//! `num_kv_shared_layers`, `attention_k_eq_v`. With `use_bidirectional_attention = "vision"`
//! the rows of one image see each other on sliding layers. Final norm, tied logits,
//! `tanh(logits / c) * c`. MoE (26B-A4B): `softmax((rms(r) * scale * D^-0.5) Wr^T)`, top-k,
//! renormalized, times `per_expert_scale`; fused `gate_up` [E, 2I, D] and `down` [E, D, I].

use lapid::{BinaryOp, DType, Error, HostData, Result, Tensor, UnaryOp, bail};
use lapid_nn::attention::{merge_heads, repeat_kv, scaled_dot_product_attention};
use lapid_nn::vision::image_block_mask;
use lapid_nn::weights::fake_quant_act;
use lapid_nn::{Embedding, GatedMlp, KvCache, Linear, Loader, Mask, Precision, RmsNorm, ZeroTailRope, ops};

use super::config::{Gemma4Config, LayerType};
use crate::runtime::{Hook, Media, Taps};

pub const EMBED: &str = "model.embed_tokens.weight";
pub const PLE_EMBED: &str = "model.embed_tokens_per_layer.weight";
/// Largest head dim of the fused attention kernels on every backend.
const FUSED_MAX_HEAD_DIM: usize = 256;
/// Float gather tables above this size are split into column chunks (Vulkan
/// `maxStorageBufferRange` is 4 GiB; the f16 PLE table of E2B is 4.7 GB).
const MAX_TABLE_BYTES: usize = 1 << 31;

/// Weight-free RMSNorm over the last axis, in f32.
pub fn rms_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let xf = x.cast(DType::F32)?;
    xf.mul(&xf.mul(&xf)?.mean(-1, true)?.add_scalar(f64::from(eps))?.unary(UnaryOp::Rsqrt)?)
}

struct Attn {
    q: Linear,
    /// `None` on kv-shared layers; `v` also `None` on k_eq_v layers (V = K before k_norm).
    k: Option<(Linear, Option<Linear>, RmsNorm)>,
    o: Linear,
    q_norm: RmsNorm,
    rope: ZeroTailRope,
    window: Option<usize>,
    heads: (usize, usize, usize),
    source: usize,
    eps: f32,
}

type SharedKv = Vec<Option<(Tensor, Tensor)>>;

impl Attn {
    fn forward(&self, h: &Tensor, cache: &mut KvCache, layer: usize, shared: &mut SharedKv, blocks: Option<&[i32]>) -> Result<Tensor> {
        let (n, (nh, nkv, hd), off) = (h.shape()[1], self.heads, cache.offset());
        let q = self.q_norm.forward(&self.q.forward(h)?.reshape(&[1, n, nh, hd])?)?.permute(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, off)?;
        let (k, v) = match &self.k {
            Some((kp, vp, kn)) => {
                let kraw = kp.forward(h)?.reshape(&[1, n, nkv, hd])?;
                let vraw = match vp {
                    Some(vp) => vp.forward(h)?.reshape(&[1, n, nkv, hd])?,
                    None => kraw.clone(),
                };
                let k = self.rope.apply(&kn.forward(&kraw)?.permute(&[0, 2, 1, 3])?, off)?;
                let kv = cache.append(layer, &k, &rms_plain(&vraw, self.eps)?.permute(&[0, 2, 1, 3])?)?;
                shared[layer] = Some(kv.clone());
                kv
            }
            None => shared[self.source].clone().ok_or_else(|| Error::new("internal", format!("gemma4: layer {layer} reads K/V of layer {} before it ran", self.source)))?,
        };
        let sk = k.shape()[2];
        let blocks = blocks.filter(|_| self.window.is_some());
        let a = if hd > FUSED_MAX_HEAD_DIM && (self.window.is_none() || blocks.is_some()) {
            // Matmul + softmax for head dims the fused kernels do not take.
            let rep = nh / nkv;
            let mut s = q.cast(DType::F32)?.matmul_t(&repeat_kv(&k, rep)?.cast(DType::F32)?)?;
            if n > 1 || blocks.is_some() {
                s = s.add(&image_block_mask(blocks, n, sk, self.window)?)?;
            }
            s.softmax()?.matmul(&repeat_kv(&v, rep)?.cast(DType::F32)?)?.cast(q.dtype())?
        } else {
            let m = match (blocks, self.window) {
                (Some(b), w) => Mask::Custom { additive: image_block_mask(Some(b), n, sk, w)?, causal: false },
                (None, Some(w)) => Mask::Sliding(w),
                (None, None) => Mask::Causal,
            };
            scaled_dot_product_attention(&q, &k, &v, 1.0, &m)?
        };
        self.o.forward(&merge_heads(&a)?)
    }
}

/// Sparse expert block of one layer.
struct Moe {
    router: Tensor,
    router_scale: Tensor,
    per_expert_scale: Tensor,
    gate_up: Tensor,
    down: Tensor,
    top_k: usize,
    act_quant: bool,
    pre_norm_2: RmsNorm,
    post_norm_2: RmsNorm,
    post_norm_1: RmsNorm,
    eps: f32,
}

impl Moe {
    /// residual [1, n, D] -> post_norm_2(experts(pre_norm_2(residual))).
    fn forward(&self, residual: &Tensor) -> Result<Tensor> {
        let (n, d, k, e) = (residual.shape()[1], residual.shape()[2], self.top_k, self.router.shape()[0]);
        let r = residual.reshape(&[n, d])?;
        let rin = rms_plain(&r, self.eps)?.mul(&self.router_scale)?.mul_scalar(1.0 / (d as f64).sqrt())?;
        let (w, ids) = rin.matmul_t(&self.router)?.cast(DType::F32)?.softmax()?.topk(k, 1)?;
        let w = w.div(&w.sum(1, true)?)?.mul(&self.per_expert_scale.reshape(&[1, e])?.expand(&[n, e])?.gather_axis(1, &ids)?)?;
        let quant = |t: Tensor| if self.act_quant { fake_quant_act(&t) } else { Ok(t) };
        let gu = quant(self.pre_norm_2.forward(&r)?.cast(DType::F32)?)?.matmul_id(&self.gate_up, &ids)?;
        let inter = gu.shape()[2] / 2;
        let act = ops::gelu_tanh(&gu.slice(2, 0, inter)?)?.mul(&gu.slice(2, inter, 2 * inter)?)?.reshape(&[n * k, inter])?;
        let down = quant(act)?.matmul_id(&self.down, &ids.reshape(&[n * k, 1])?)?;
        let y = down.reshape(&[n, k, d])?.mul(&w.unsqueeze(2)?)?.sum(1, false)?;
        self.post_norm_2.forward(&y.reshape(&[1, n, d])?)
    }
}

struct Layer {
    attn: Attn,
    norms: [RmsNorm; 4],
    mlp: GatedMlp,
    moe: Option<Moe>,
    /// PLE gate, projection, post norm.
    ple: Option<(Linear, Linear, RmsNorm)>,
    scalar: f32,
}

pub struct Decoder {
    pub cfg: Gemma4Config,
    embed: Tensor,
    logits_act_quant: bool,
    /// PLE table as column chunks [V, cols_i], projection and norm.
    ple: Option<(Vec<Tensor>, Linear, RmsNorm)>,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

/// Host value of a 1-element weight, `default` when absent.
pub fn scalar(l: &Loader<'_>, name: &str, default: f32) -> Result<f32> {
    if !l.contains(name) {
        return Ok(default);
    }
    match l.device.to_vec_f32(&l.tensor_f32(name)?)?.as_slice() {
        [x] => Ok(*x),
        v => bail!("shape_mismatch", "{name}: expected one value, got {}", v.len()),
    }
}

/// The PLE table, split into column chunks of whole layers when it is a float table larger
/// than `MAX_TABLE_BYTES` (copied on the host once at load).
fn ple_table(l: &Loader<'_>, cfg: &Gemma4Config) -> Result<Vec<Tensor>> {
    let (v, lp, p) = (cfg.ple_vocab, cfg.n_layers * cfg.ple_dim, cfg.ple_dim);
    let host = l.file.get(PLE_EMBED)?;
    l.expect_shape(&host, PLE_EMBED, &[v, lp])?;
    let dt = host.dtype();
    if dt.is_quant() || l.precision != Precision::Stored || v * lp * dt.size() <= MAX_TABLE_BYTES {
        return Ok(vec![l.tensor(PLE_EMBED)?]);
    }
    let (bytes, es) = (l.file.bytes(PLE_EMBED)?, dt.size());
    let per = cfg.n_layers.div_ceil((v * lp * es).div_ceil(MAX_TABLE_BYTES)) * p;
    (0..lp)
        .step_by(per)
        .map(|c0| {
            let c1 = (c0 + per).min(lp);
            let mut t = Vec::with_capacity(v * (c1 - c0) * es);
            for r in 0..v {
                t.extend_from_slice(&bytes[(r * lp + c0) * es..(r * lp + c1) * es]);
            }
            l.device.upload(&Tensor::from_host(HostData::from_vec(t), &[v, c1 - c0], dt)?)
        })
        .collect()
}

impl Decoder {
    pub fn load(l: &Loader<'_>, cfg: &Gemma4Config) -> Result<Decoder> {
        let (d, eps) = (cfg.hidden_size, cfg.norm_eps);
        let lin = |n: &str, o: usize, i: usize| l.linear_checked(n, o, i, false);
        let norm = |n: &str, dim: usize| l.rms_norm_checked(n, dim, eps);
        let ple = if cfg.ple_dim > 0 {
            let p = cfg.ple_dim;
            Some((ple_table(l, cfg)?, lin("model.per_layer_model_projection", cfg.n_layers * p, d)?, norm("model.per_layer_projection_norm", p)?))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let (p, (hd, kvh)) = (format!("model.layers.{i}"), cfg.attn_dims(i));
            let a = format!("{p}.self_attn");
            let k = if i >= cfg.first_shared() {
                None
            } else {
                let v = if cfg.k_eq_v(i) { None } else { Some(lin(&format!("{a}.v_proj"), kvh * hd, d)?) };
                Some((lin(&format!("{a}.k_proj"), kvh * hd, d)?, v, norm(&format!("{a}.k_norm"), hd)?))
            };
            let (rp, window) = match cfg.layer_types[i] {
                LayerType::SlidingAttention => (cfg.rope_sliding, Some(cfg.sliding_window)),
                LayerType::FullAttention => (cfg.rope_full, None),
            };
            let attn = Attn {
                q: lin(&format!("{a}.q_proj"), cfg.n_head * hd, d)?,
                k,
                o: lin(&format!("{a}.o_proj"), d, cfg.n_head * hd)?,
                q_norm: norm(&format!("{a}.q_norm"), hd)?,
                rope: ZeroTailRope::new(hd, rp.theta, rp.partial),
                window,
                heads: (cfg.n_head, kvh, hd),
                source: cfg.kv_source(i).ok_or_else(|| Error::new("metadata", format!("gemma4: layer {i} has no kv source")))?,
                eps,
            };
            let moe = match cfg.moe {
                Some(m) => {
                    let (e, mi, gu) = (m.num_experts, m.intermediate, format!("{p}.experts.gate_up_proj"));
                    let f32p = |n: &str, s: &[usize]| -> Result<Tensor> {
                        let t = l.tensor_f32(n)?;
                        l.expect_shape(&t, n, s)?;
                        Ok(t)
                    };
                    Some(Moe {
                        router: l.tensor_checked(&format!("{p}.router.proj.weight"), &[e, d])?,
                        router_scale: f32p(&format!("{p}.router.scale"), &[d])?,
                        per_expert_scale: f32p(&format!("{p}.router.per_expert_scale"), &[e])?,
                        gate_up: l.tensor_checked(&gu, &[e, 2 * mi, d])?,
                        down: l.tensor_checked(&format!("{p}.experts.down_proj"), &[e, d, mi])?,
                        top_k: m.top_k,
                        act_quant: l.act_quant(&gu)?,
                        pre_norm_2: norm(&format!("{p}.pre_feedforward_layernorm_2"), d)?,
                        post_norm_2: norm(&format!("{p}.post_feedforward_layernorm_2"), d)?,
                        post_norm_1: norm(&format!("{p}.post_feedforward_layernorm_1"), d)?,
                        eps,
                    })
                }
                None => None,
            };
            let (ff, m) = (cfg.layer_intermediate(i), format!("{p}.mlp"));
            let ple = if cfg.ple_dim > 0 {
                let pd = cfg.ple_dim;
                Some((lin(&format!("{p}.per_layer_input_gate"), pd, d)?, lin(&format!("{p}.per_layer_projection"), d, pd)?, norm(&format!("{p}.post_per_layer_input_norm"), d)?))
            } else {
                None
            };
            let ln = |n: &str| norm(&format!("{p}.{n}"), d);
            layers.push(Layer {
                attn,
                norms: [ln("input_layernorm")?, ln("post_attention_layernorm")?, ln("pre_feedforward_layernorm")?, ln("post_feedforward_layernorm")?],
                mlp: GatedMlp::geglu(lin(&format!("{m}.gate_proj"), ff, d)?, lin(&format!("{m}.up_proj"), ff, d)?, lin(&format!("{m}.down_proj"), d, ff)?),
                moe,
                ple,
                scalar: scalar(l, &format!("{p}.layer_scalar"), 1.0)?,
            });
        }
        Ok(Decoder {
            embed: l.tensor_checked(EMBED, &[cfg.vocab_size, d])?,
            logits_act_quant: l.act_quant(EMBED)?,
            ple,
            layers,
            norm: norm("model.norm", d)?,
            cfg: cfg.clone(),
        })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::with_windows(self.cfg.kv_windows())
    }

    /// Input embeddings [1, n, D] with the image rows merged, and per-row image block ids.
    fn embeddings(&self, ids: &[u32], media: &[Media]) -> Result<(Tensor, Vec<u32>, Vec<i32>)> {
        let (n, d) = (ids.len(), self.cfg.hidden_size);
        let (mut eff, mut blocks) = (ids.to_vec(), vec![-1i32; n]);
        for (b, m) in media.iter().enumerate() {
            eff[m.start..m.start + m.len].iter_mut().for_each(|x| *x = self.cfg.pad.unwrap_or(0));
            blocks[m.start..m.start + m.len].iter_mut().for_each(|x| *x = b as i32);
        }
        let x = Embedding::new(self.embed.clone()).lookup(&eff)?.cast(DType::F32)?.mul_scalar((d as f64).sqrt())?;
        if media.is_empty() {
            return Ok((x, eff, blocks));
        }
        let (mut parts, mut at) = (Vec::new(), 0);
        for m in media {
            if m.embeds[0].shape() != [m.len, d] {
                bail!("shape_mismatch", "gemma4: image rows must be [{}, {d}], got {:?}", m.len, m.embeds[0].shape());
            }
            if m.start > at {
                parts.push(x.slice(1, at, m.start)?);
            }
            parts.push(m.embeds[0].cast(DType::F32)?.reshape(&[1, m.len, d])?);
            at = m.start + m.len;
        }
        if at < n {
            parts.push(x.slice(1, at, n)?);
        }
        Ok((Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?, eff, blocks))
    }

    /// Graph for `ids` (image rows `media`) at `kv.offset()`; stages K/V. Taps: `embed`,
    /// `layers.{i}.attn`, `layers.{i}`, `norm` (all rows).
    pub fn forward(&self, ids: &[u32], media: &[Media], kv: &mut KvCache, rows: crate::runtime::Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        use crate::runtime::Rows;
        let (n, c) = (ids.len(), &self.cfg);
        if n == 0 {
            bail!("invalid_argument", "gemma4: forward of zero tokens");
        }
        let (mut x, eff, blocks) = self.embeddings(ids, media)?;
        let blocks = (c.bidirectional_vision && !media.is_empty()).then_some(blocks.as_slice());
        taps.tap("embed", &x);
        let ple = match &self.ple {
            Some((tables, proj, pnorm)) => {
                let (nl, p) = (c.n_layers, c.ple_dim);
                let parts = tables.iter().map(|t| Embedding::new(t.clone()).lookup(&eff)?.cast(DType::F32)).collect::<Result<Vec<_>>>()?;
                let tok = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)?.mul_scalar((p as f64).sqrt())?.reshape(&[1, n, nl, p])?;
                let pr = pnorm.forward(&proj.forward(&x)?.mul_scalar(1.0 / (c.hidden_size as f64).sqrt())?.reshape(&[1, n, nl, p])?)?;
                Some(pr.add(&tok)?.mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?)
            }
            None => None,
        };
        let mut shared: SharedKv = vec![None; self.layers.len()];
        for (i, ly) in self.layers.iter().enumerate() {
            let a = ly.attn.forward(&ly.norms[0].forward(&x)?, kv, i, &mut shared, blocks)?;
            taps.tap(format_args!("layers.{i}.attn"), &a);
            x = x.add(&taps.steer(Hook::AttnOut, i, &ly.norms[1].forward(&a)?)?)?;
            let mut f = ly.mlp.forward(&ly.norms[2].forward(&x)?)?;
            if let Some(moe) = &ly.moe {
                f = moe.post_norm_1.forward(&f)?.add(&moe.forward(&x)?)?;
            }
            x = x.add(&taps.steer(Hook::FfnOut, i, &ly.norms[3].forward(&f)?)?)?;
            if let (Some((gate, proj, post)), Some(pi)) = (&ly.ple, &ple) {
                let g = ops::gelu_tanh(&gate.forward(&x)?)?.mul(&pi.slice(2, i, i + 1)?.reshape(&[1, n, c.ple_dim])?)?;
                x = x.add(&post.forward(&proj.forward(&g)?)?)?;
            }
            if ly.scalar != 1.0 {
                x = x.mul_scalar(f64::from(ly.scalar))?;
            }
            x = taps.steer(Hook::Resid, i, &x)?;
            taps.tap(format_args!("layers.{i}"), &x);
        }
        let x = match rows {
            Rows::None => return Ok(None),
            Rows::Last if n > 1 => x.slice(1, n - 1, n)?,
            _ => x,
        };
        let mut h = self.norm.forward(&x)?;
        if rows == Rows::All {
            taps.tap("norm", &h);
        }
        if self.logits_act_quant {
            h = fake_quant_act(&h)?;
        }
        let mut logits = h.matmul_t(&self.embed)?.cast(DType::F32)?;
        if let Some(cap) = c.final_logit_softcapping.filter(|c| *c > 0.0) {
            logits = logits.mul_scalar(1.0 / f64::from(cap))?.tanh()?.mul_scalar(f64::from(cap))?;
        }
        Ok(Some(logits))
    }
}

/// `clamp(x, lo, hi)` for finite bounds.
pub fn clamp(x: &Tensor, (lo, hi): (f32, f32)) -> Result<Tensor> {
    let x = if lo.is_finite() { x.binary_scalar(BinaryOp::Max, f64::from(lo))? } else { x.clone() };
    if hi.is_finite() { x.binary_scalar(BinaryOp::Min, f64::from(hi)) } else { Ok(x) }
}
