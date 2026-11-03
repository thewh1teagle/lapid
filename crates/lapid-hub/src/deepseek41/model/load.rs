//! Weights of [`Deepseek41Model`](super::Deepseek41Model) from a `.lapid` (names: `convert.rs`).

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::llm::engram::{EngramGate, Fp8Rows};
use lapid_nn::llm::hyper::SinkhornHc;
use lapid_nn::llm::moe::Experts;
use lapid_nn::llm::rope::TailRope;
use lapid_nn::{Embedding, Loader};
use serde_json::Value;

use super::attention::Attention;
use super::compress::Compressor;
use super::index::{Candidates, Indexer};
use super::moe::Moe;
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

fn moe(l: &Loader<'_>, cfg: &Deepseek41Config, i: usize) -> Result<Moe> {
    let p = format!("layers.{i}.ffn");
    let (d, e, inter) = (cfg.hidden_size, cfg.n_experts, cfg.moe_intermediate);
    let stacks = ["w1", "w3", "w2"].map(|w| format!("{p}.experts.{w}"));
    let shared = |w: &str, o: usize, inp: usize| l.linear_checked(&format!("{p}.shared_experts.{w}"), o, inp, false);
    Ok(Moe {
        router: f32_param(l, &format!("{p}.gate.weight"), &[e, d])?,
        bias: f32_param(l, &format!("{p}.gate.bias"), &[e])?,
        experts: Experts::load(l, stacks.each_ref().map(String::as_str), e, inter, d)?.with_limit(cfg.swiglu_limit),
        shared: [shared("w1", inter, d)?, shared("w3", inter, d)?, shared("w2", d, inter)?],
        top_k: cfg.top_k,
        norm_topk: cfg.norm_topk_prob,
        route_scale: cfg.route_scale,
    })
}

fn engram_layer(l: &Loader<'_>, cfg: &Deepseek41Config, i: usize, module: usize) -> Result<Engram> {
    let ec = cfg.engram.as_ref().ok_or_else(|| Error::new("internal", "deepseek41: no engram config"))?;
    let (d, hc) = (cfg.hidden_size, cfg.hc_mult);
    let p = format!("layers.{i}.engram");
    let blob = |k: &str| l.file.blob_data(&format!("{p}.embed.{k}")).ok_or_else(|| Error::new("missing_tensor", format!("{p}.embed.{k}: blob not found")));
    let table = Fp8Rows::new(blob("fp8")?, blob("e8m0")?, ec.head_dim)?;
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
    for i in 0..cfg.n_layers {
        let p = format!("layers.{i}");
        let conn = |kind: &str| SinkhornHc::load(l, &format!("{p}.hc_{kind}"), [hc, d, cfg.hc_sinkhorn_iters], cfg.hc_eps, eps);
        layers.push(Layer {
            engram: cfg.engram_module(i).map(|e| engram_layer(l, &cfg, i, e)).transpose()?,
            attn_hc: conn("attn")?,
            ffn_hc: conn("ffn")?,
            attn_norm: l.rms_norm_checked(&format!("{p}.attn_norm"), d, eps)?,
            ffn_norm: l.rms_norm_checked(&format!("{p}.ffn_norm"), d, eps)?,
            attn: attention(l, &cfg, i)?,
            moe: moe(l, &cfg, i)?,
        });
    }
    Ok(Deepseek41Model {
        embed: Embedding::new(l.tensor_checked(EMBED, &[cfg.vocab_size, d])?),
        head: l.linear_checked("head", cfg.vocab_size, d, false)?,
        norm: l.rms_norm_checked("norm", d, eps)?,
        hash,
        layers,
        cfg,
    })
}
