//! Weights of [`Deepseek41Model`](super::Deepseek41Model) from a `.lapid` (names: `convert.rs`).

use lapid::{Advice, DType, Error, Residency, Result, Tensor};
use lapid_nn::llm::engram::{EngramGate, Fp8Rows};
use lapid_nn::llm::expert_cache::{self, ExpertCache};
use lapid_nn::llm::hyper::SinkhornHc;
use lapid_nn::llm::moe::{Experts, GateUp};
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Embedding, Loader};
use serde_json::Value;

use super::attention::Attention;
use super::compress::Compressor;
use super::index::{Candidates, Indexer};
use super::moe::{Moe, Routed};
use super::{Deepseek41Model, EMBED, Engram, Layer};
use crate::deepseek41::config::Deepseek41Config;
use crate::deepseek41::engram;

fn f32_param(l: &Loader<'_>, name: &str, shape: &[usize]) -> Result<Tensor> {
    l.tensor_checked(name, shape)?.cast(DType::F32)
}

fn compressor(l: &Loader<'_>, cfg: &Deepseek41Config, p: &str, ratio: usize) -> Result<Compressor> {
    let (d, hd, id, eps) = (cfg.hidden_size, cfg.head_dim, cfg.index_head_dim, cfg.norm_eps);
    Ok(Compressor {
        wkv: l.linear_checked(&format!("{p}.compressor.wkv"), hd, d, false)?,
        wgate: (ratio > 1).then(|| l.linear_checked(&format!("{p}.compressor.wgate"), hd, d, false)).transpose()?,
        norm: l.rms_norm_checked(&format!("{p}.compressor.norm"), hd, eps)?,
        wk: l.linear_checked(&format!("{p}.indexer.wk"), id, hd, false)?,
        k_norm: l.rms_norm_checked(&format!("{p}.indexer.k_norm"), id, eps)?,
        ratio,
    })
}

fn attention(l: &Loader<'_>, cfg: &Deepseek41Config, i: usize) -> Result<Attention> {
    let p = format!("layers.{i}.attn");
    let (d, h, hd, eps) = (cfg.hidden_size, cfg.n_head, cfg.head_dim, cfg.norm_eps);
    let (ql, g, ol) = (cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank);
    let lin = |n: &str, o: usize, inp: usize| l.linear_checked(&format!("{p}.{n}"), o, inp, false);
    let ratio = cfg.compress_ratios[i];
    let rope = if ratio == 0 { TailRope::new(cfg.rope_dim, cfg.rope_theta, None) } else { TailRope::new(cfg.rope_dim, cfg.compress_rope_theta, cfg.yarn.as_ref()) };
    let kv_source = cfg.kv_source_layers.contains(&i);
    let index_source = cfg.index_source_layers.contains(&i);
    let indexer = if index_source {
        let (ih, id) = (cfg.index_n_heads, cfg.index_head_dim);
        Some(Indexer { wq_b: lin("indexer.wq_b", ih * id, ql)?, weights_proj: lin("indexer.weights_proj", ih, d)?, heads: ih, dim: id, topk: cfg.index_topk })
    } else {
        None
    };
    let wo_a = format!("{p}.wo_a.weight");
    Ok(Attention {
        wq_a: lin("wq_a", ql, d)?,
        q_norm: l.rms_norm_checked(&format!("{p}.q_norm"), ql, eps)?,
        wq_b: lin("wq_b", h * hd, ql)?,
        wkv: lin("wkv", hd, d)?,
        kv_norm: l.rms_norm_checked(&format!("{p}.kv_norm"), hd, eps)?,
        wo_a: l.tensor_checked(&wo_a, &[g, ol, h / g * hd])?,
        wo_a_act_quant: l.act_quant(&wo_a)?,
        wo_b: lin("wo_b", d, g * ol)?,
        sinks: f32_param(l, &format!("{p}.attn_sink"), &[h])?.reshape(&[1, h, 1])?,
        comp: if kv_source { Some(compressor(l, cfg, &p, ratio)?) } else { None },
        indexer,
        candidates: (cfg.candidate_source == Some(i)).then_some(Candidates { topk_blocks: cfg.candidate_topk_blocks, block_size: cfg.candidate_block_size }),
        uses_candidates: index_source && cfg.candidate_source.is_some_and(|c| c < i),
        rope,
        n_head: h,
        head_dim: hd,
        groups: g,
        window: cfg.sliding_window,
        ratio,
    })
}

/// The MoE of layer `i` with its routed experts still unplaced, and their host stacks
/// `[w1, w3, w2]` (see [`place_experts`]).
fn moe(l: &Loader<'_>, cfg: &Deepseek41Config, i: usize) -> Result<(Moe, Vec<Tensor>)> {
    let p = format!("layers.{i}.ffn");
    let (d, e, inter) = (cfg.hidden_size, cfg.n_experts, cfg.moe_intermediate);
    let stack = |w: &str, shape: &[usize]| l.host_tensor_checked(&format!("{p}.experts.{w}"), shape);
    let stacks = vec![stack("w1", &[e, inter, d])?, stack("w3", &[e, inter, d])?, stack("w2", &[e, d, inter])?];
    let shared = |w: &str, o: usize, inp: usize| l.linear_checked(&format!("{p}.shared_experts.{w}"), o, inp, false);
    let moe = Moe {
        router: f32_param(l, &format!("{p}.gate.weight"), &[e, d])?,
        bias: f32_param(l, &format!("{p}.gate.bias"), &[e])?,
        routed: Routed::Cached { layer: i, act_quant: l.act_quant(&format!("{p}.experts.w1"))? },
        shared: [shared("w1", inter, d)?, shared("w3", inter, d)?, shared("w2", d, inter)?],
        top_k: cfg.top_k,
        norm_topk: cfg.norm_topk_prob,
        route_scale: cfg.route_scale,
        limit: (cfg.swiglu_limit > 0.0).then_some(cfg.swiglu_limit),
    };
    Ok((moe, stacks))
}

/// Device memory reserved besides the expert cache for activations and KV state.
const SCRATCH: u64 = 4 << 30;

/// Where the routed experts live: graph leaves when they fit the expert budget (and always
/// on the CPU, which reads the file mapping in place), else a device [`ExpertCache`] sized
/// by the budget (`LAPID_EXPERT_CACHE_SLOTS` forces the cache with that many slots on any
/// device). Call after every other weight is loaded (the budget is what is left).
fn place_experts(l: &Loader<'_>, cfg: &Deepseek41Config, moes: &mut [&mut Moe], stacks: Vec<Vec<Tensor>>) -> Result<Option<ExpertCache>> {
    let gpu = l.device.name() != "cpu";
    let bytes: u64 = stacks.iter().flatten().map(|t| t.dtype().storage_bytes(t.shape()) as u64).sum();
    let total = (stacks.len() * cfg.n_experts).max(1);
    let per = bytes / total as u64;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let slots = match expert_cache::forced_slots() {
        Some(n) => Some(n.min(total)),
        None if gpu => {
            let (budget, why) = expert_cache::budget_bytes(SCRATCH);
            // Fits as graph leaves (the usual rule for uploading weights): no cache.
            let fits = lapid::residency::available_memory().is_some_and(|a| bytes + SCRATCH <= a.saturating_sub((4u64 << 30).max(a / 10)));
            if fits || bytes <= budget {
                None
            } else {
                let n = (budget / per.max(1)) as usize;
                if n < cfg.top_k {
                    return Err(Error::new(
                        "out_of_memory",
                        format!("deepseek41: {:.1} GiB of experts do not fit and the expert cache budget is too small for {} experts: {why}", gib(bytes), cfg.top_k),
                    ));
                }
                lapid::log_line(&format!("[lapid] deepseek41: {:.1} GiB of routed experts do not fit: device expert cache of {:.1} GiB ({}); {} of {total} experts", gib(bytes), gib(n as u64 * per), why, n.min(total)));
                Some(n.min(total))
            }
        }
        None => None,
    };
    match slots {
        None => {
            for (m, s) in moes.iter_mut().zip(stacks) {
                let Routed::Cached { act_quant, .. } = m.routed else { continue };
                let [w1, w3, w2] = <[Tensor; 3]>::try_from(s).map_err(|_| Error::new("internal", "deepseek41: expert stacks"))?;
                let up = |t: &Tensor| l.device.upload(t);
                m.routed = Routed::Direct(Experts { gate_up: GateUp::Split { gate: up(&w1)?, up: up(&w3)? }, down: up(&w2)?, act: lapid_nn::Activation::Silu, limit: m.limit, act_quant });
            }
            Ok(None)
        }
        Some(n) => {
            for t in stacks.iter().flatten() {
                // 2 MiB page-cache folios: faster cold reads of whole experts.
                lapid::advise(t, None, Advice::HugePage)?;
            }
            Ok(Some(ExpertCache::new(l.device, stacks.into_iter().map(Some).collect(), n)?))
        }
    }
}

fn engram_layer(l: &Loader<'_>, cfg: &Deepseek41Config, i: usize, module: usize) -> Result<Engram> {
    let ec = cfg.engram.as_ref().ok_or_else(|| Error::new("internal", "deepseek41: no engram config"))?;
    let (d, hc) = (cfg.hidden_size, cfg.hc_mult);
    let p = format!("layers.{i}.engram");
    let blob = |k: &str| l.file.blob_data(&format!("{p}.embed.{k}")).ok_or_else(|| Error::new("missing_tensor", format!("{p}.embed.{k}: blob not found")));
    let (values, scales) = (blob("fp8")?, blob("e8m0")?);
    for b in [&values, &scales] {
        // Rows are read scattered (a few per token): no readahead around them.
        lapid::residency::advise_host(b, None, Advice::Random)?;
    }
    let table = Fp8Rows::new(values, scales, ec.head_dim)?;
    if table.rows() as u64 != ec.num_embeddings[module] {
        return Err(Error::new("format", format!("{p}: {} table rows, config says {}", table.rows(), ec.num_embeddings[module])));
    }
    // the gate weight is only ever used as the product q_weight * k_weight
    let host = |n: &str| -> Result<Vec<f32>> { l.device.to_vec_f32(&f32_param(l, &format!("{p}.{n}"), &[hc, d])?) };
    let w: Vec<f32> = host("q_weight")?.iter().zip(host("k_weight")?).map(|(q, k)| q * k).collect();
    Ok(Engram {
        module,
        table,
        gate: EngramGate {
            wkv: l.linear_checked(&format!("{p}.wkv"), (hc + 1) * d, ec.cols() * ec.head_dim, false)?,
            weight: l.device.upload(&Tensor::from_f32(&w, &[hc, d])?)?,
            hc,
            eps: cfg.norm_eps,
        },
    })
}

pub(super) fn model(l: &Loader<'_>, meta: &Value) -> Result<Deepseek41Model> {
    let cfg = Deepseek41Config::from_json(meta)?;
    if l.device.name() != "cpu" {
        // Dense weights live on the device even when the file (experts, Engram tables) is
        // far larger than memory: a GPU must not read file-backed pages in place.
        l.file.set_residency(Residency::Upload);
    }
    let (d, eps, hc) = (cfg.hidden_size, cfg.norm_eps, cfg.hc_mult);
    let hash = match &cfg.engram {
        Some(ec) => {
            let m = meta.get("engram_hash").ok_or_else(|| Error::new("metadata", "deepseek41: engram_hash missing (convert again)"))?;
            let map = l.file.blob(engram::TOKEN_MAP_BLOB).ok_or_else(|| Error::new("missing_tensor", "deepseek41: engram token map blob missing"))?;
            Some(engram::load(ec, m, map)?)
        }
        None => None,
    };
    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut stacks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let (moe, s) = moe(l, &cfg, i)?;
        stacks.push(s);
        let p = format!("layers.{i}");
        let conn = |kind: &str| SinkhornHc::load(l, &format!("{p}.hc_{kind}"), [hc, d, cfg.hc_sinkhorn_iters], cfg.hc_eps, eps);
        layers.push(Layer {
            engram: cfg.engram_module(i).map(|e| engram_layer(l, &cfg, i, e)).transpose()?,
            attn_hc: conn("attn")?,
            ffn_hc: conn("ffn")?,
            attn_norm: l.rms_norm_checked(&format!("{p}.attn_norm"), d, eps)?,
            ffn_norm: l.rms_norm_checked(&format!("{p}.ffn_norm"), d, eps)?,
            attn: attention(l, &cfg, i)?,
            moe,
        });
    }
    let embed = Embedding::new(l.tensor_checked(EMBED, &[cfg.vocab_size, d])?);
    let head = l.linear_checked("head", cfg.vocab_size, d, false)?;
    let norm = l.rms_norm_checked("norm", d, eps)?;
    let cache = place_experts(l, &cfg, &mut layers.iter_mut().map(|l| &mut l.moe).collect::<Vec<_>>(), stacks)?;
    Ok(Deepseek41Model {
        embed,
        head,
        norm,
        cache,
        device: l.device.clone(),
        lock: std::sync::Mutex::new(()),
        hash,
        layers,
        cfg,
    })
}
