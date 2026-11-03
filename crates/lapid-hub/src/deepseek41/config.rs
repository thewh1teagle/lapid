//! DeepSeek V4.1 hyper-parameters, from the HF `config.json` (`model_type` `deepseek_v41`,
//! the text model under `text_config`) or the `.lapid` metadata object `"deepseek41"` (the
//! flattened text config as converted, plus the Engram hash constants).

use lapid::{Error, Result, bail};
use serde_json::Value;

use crate::deepseek4::config::yarn;
pub use lapid_nn::llm::rope::Yarn;

/// Engram modules (`engram_*` keys).
#[derive(Clone, Debug, PartialEq)]
pub struct EngramConfig {
    /// Layers with an Engram, strictly increasing; module `e` is layer `layer_ids[e]`.
    pub layer_ids: Vec<usize>,
    /// Table rows per module.
    pub num_embeddings: Vec<u64>,
    /// N: hashes of the 2-gram .. N-gram.
    pub max_ngram: usize,
    /// Primes are drawn from this bucket size up.
    pub vocab_size: usize,
    pub n_heads: usize,
    /// Width of a table row.
    pub head_dim: usize,
    pub pad_token_id: usize,
    pub compressed_vocab_size: usize,
}

impl EngramConfig {
    /// Table rows gathered per token: `(N - 1) * H`.
    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }
}

/// The DSpark draft head (`dspark_*` keys; checkpoint `mtp.{0..stages}`): block-diffusion
/// stages over the target's features, with a Markov and a confidence head on the last.
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkConfig {
    pub stages: usize,
    /// Tokens per draft block (anchor + noise), = draft tokens at most.
    pub block_size: usize,
    pub noise_token: u32,
    /// Target layers whose input streams (mean over hc) feed the draft.
    pub targets: Vec<usize>,
    pub markov_rank: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Deepseek41Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    /// Shared K = V head width (512).
    pub head_dim: usize,
    /// Rotated dims at the tail of each head (64).
    pub rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub route_scale: f32,
    pub norm_topk_prob: bool,
    /// SwiGLU clamp (`gate <= l`, `-l <= up <= l`); 0 disables.
    pub swiglu_limit: f32,
    pub sliding_window: usize,
    /// Per layer: 0 = sliding window only, 1 / 2 = compressed KV (one entry per 1 / 2 tokens).
    pub compress_ratios: Vec<usize>,
    /// Layers that compress their own KV (and make the index keys); the compressed layers after
    /// one read its rows.
    pub kv_source_layers: Vec<usize>,
    /// Layers that run their own indexer top-k; the compressed layers after one reuse it.
    pub index_source_layers: Vec<usize>,
    /// Layer that picks candidate blocks for the index sources after it.
    pub candidate_source: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub yarn: Option<Yarn>,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub norm_eps: f32,
    pub max_positions: usize,
    pub engram: Option<EngramConfig>,
    pub dspark: Option<DsparkConfig>,
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("metadata", format!("deepseek41 config: missing integer {key:?}")))
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn opt_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

fn usize_list(v: &Value, key: &str) -> Vec<usize> {
    v.get(key).and_then(Value::as_array).map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
}

/// The text model's keys: `text_config` merged over the top level (HF multimodal configs).
pub fn text_config(v: &Value) -> Value {
    let mut out = v.clone();
    if let (Some(o), Some(t)) = (out.as_object_mut(), v.get("text_config").and_then(Value::as_object)) {
        o.remove("text_config");
        o.remove("vision_config");
        for (k, x) in t {
            o.insert(k.clone(), x.clone());
        }
        if t.get("model_type").is_some() {
            o.insert("model_type".into(), v.get("model_type").cloned().unwrap_or(Value::Null));
        }
    }
    out
}

fn engram(v: &Value) -> Result<Option<EngramConfig>> {
    let layer_ids = usize_list(v, "engram_layer_ids");
    if layer_ids.is_empty() {
        return Ok(None);
    }
    let num_embeddings = v
        .get("engram_num_embeddings")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_u64).collect());
    Ok(Some(EngramConfig {
        layer_ids,
        num_embeddings,
        max_ngram: get_usize(v, "engram_max_ngram_size")?,
        vocab_size: get_usize(v, "engram_vocab_size")?,
        n_heads: get_usize(v, "engram_n_heads")?,
        head_dim: get_usize(v, "engram_head_dim")?,
        pad_token_id: opt_usize(v, "engram_pad_token_id").or_else(|| opt_usize(v, "engram_pad_id")).unwrap_or(2),
        compressed_vocab_size: get_usize(v, "engram_compressed_vocab_size")?,
    }))
}

fn dspark(v: &Value, n_layers: usize, ratios: &[usize]) -> Result<Option<DsparkConfig>> {
    let stages = opt_usize(v, "num_nextn_predict_layers").unwrap_or(0);
    let block_size = opt_usize(v, "dspark_block_size").unwrap_or(0);
    if stages == 0 || block_size == 0 {
        return Ok(None);
    }
    let c = DsparkConfig {
        stages,
        block_size,
        noise_token: get_usize(v, "dspark_noise_token_id")? as u32,
        targets: usize_list(v, "dspark_target_layer_ids"),
        markov_rank: opt_usize(v, "dspark_markov_rank").unwrap_or(256),
        n_experts: opt_usize(v, "dspark_n_routed_experts").unwrap_or(get_usize(v, "n_routed_experts")?),
        top_k: opt_usize(v, "dspark_num_experts_per_tok").unwrap_or(get_usize(v, "num_experts_per_tok")?),
    };
    let stage_ratios = ratios.get(n_layers..).unwrap_or_default();
    if c.targets.is_empty() || c.targets.iter().any(|&t| t >= n_layers) || stage_ratios.iter().any(|&r| r != 0) || c.top_k == 0 || c.top_k > c.n_experts {
        bail!("metadata", "deepseek41: invalid dspark config {c:?} (stage compress ratios {stage_ratios:?})");
    }
    Ok(Some(c))
}

impl Deepseek41Config {
    pub fn from_json(v: &Value) -> Result<Deepseek41Config> {
        let v = &text_config(v);
        if let Some(mt) = v.get("model_type").and_then(Value::as_str)
            && mt != "deepseek_v41"
        {
            bail!("unsupported_model", "deepseek41: model_type {mt:?} is not deepseek_v41");
        }
        if let Some(sf) = v.get("scoring_func").and_then(Value::as_str)
            && sf != "sqrtsoftplus"
        {
            bail!("unsupported_model", "deepseek41: scoring_func {sf:?} (only sqrtsoftplus)");
        }
        if opt_f32(v, "gate_temp").is_some_and(|t| t != 1.0) {
            bail!("unsupported_model", "deepseek41: gate_temp != 1");
        }
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let ratios = usize_list(v, "compress_ratios");
        if ratios.len() < n_layers {
            bail!("metadata", "deepseek41: {} compress ratios for {n_layers} layers", ratios.len());
        }
        let cand = v.get("candidate_source_layer_id").and_then(Value::as_i64).filter(|&c| c >= 0).map(|c| c as usize);
        let cfg = Deepseek41Config {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size: get_usize(v, "hidden_size")?,
            n_layers,
            n_head: get_usize(v, "num_attention_heads")?,
            head_dim: get_usize(v, "head_dim")?,
            rope_dim: get_usize(v, "qk_rope_head_dim")?,
            q_lora_rank: get_usize(v, "q_lora_rank")?,
            o_groups: get_usize(v, "o_groups")?,
            o_lora_rank: get_usize(v, "o_lora_rank")?,
            n_experts: get_usize(v, "n_routed_experts")?,
            top_k: get_usize(v, "num_experts_per_tok")?,
            moe_intermediate: get_usize(v, "moe_intermediate_size")?,
            route_scale: opt_f32(v, "routed_scaling_factor").unwrap_or(1.0),
            norm_topk_prob: v.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
            swiglu_limit: opt_f32(v, "swiglu_limit").unwrap_or(0.0),
            sliding_window: get_usize(v, "sliding_window")?,
            compress_ratios: ratios[..n_layers].to_vec(),
            kv_source_layers: usize_list(v, "kv_source_layer_ids"),
            index_source_layers: usize_list(v, "index_source_layer_ids"),
            candidate_source: cand,
            candidate_topk_blocks: opt_usize(v, "candidate_topk_blocks").unwrap_or(0),
            candidate_block_size: opt_usize(v, "candidate_block_size").unwrap_or(0),
            rope_theta: opt_f32(v, "rope_theta").unwrap_or(10000.0),
            compress_rope_theta: opt_f32(v, "compress_rope_theta").unwrap_or(160000.0),
            yarn: yarn(v),
            hc_mult: opt_usize(v, "hc_mult").unwrap_or(4),
            hc_sinkhorn_iters: opt_usize(v, "hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: opt_f32(v, "hc_eps").unwrap_or(1e-6),
            index_n_heads: get_usize(v, "index_n_heads")?,
            index_head_dim: get_usize(v, "index_head_dim")?,
            index_topk: get_usize(v, "index_topk")?,
            norm_eps: opt_f32(v, "rms_norm_eps").unwrap_or(1e-6),
            max_positions: opt_usize(v, "max_position_embeddings").unwrap_or(1 << 20),
            engram: engram(v)?,
            dspark: dspark(v, n_layers, &ratios)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::new("metadata", format!("deepseek41: {m}")));
        if self.n_head == 0 || self.o_groups == 0 || !self.n_head.is_multiple_of(self.o_groups) {
            return bad(format!("{} heads not divisible into {} output groups", self.n_head, self.o_groups));
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim.min(self.index_head_dim) {
            return bad(format!("rope dim {} invalid for head dims {} / {}", self.rope_dim, self.head_dim, self.index_head_dim));
        }
        if let Some(r) = self.compress_ratios.iter().find(|&&r| r > 2) {
            bail!("unsupported_model", "deepseek41: compress ratio {r} (only 0, 1 and 2)");
        }
        if self.hc_mult == 0 || self.sliding_window == 0 || self.top_k == 0 || self.top_k > self.n_experts || self.index_topk == 0 {
            return bad("invalid hc_mult / window / top_k / index_topk".into());
        }
        for (name, list) in [("kv", &self.kv_source_layers), ("index", &self.index_source_layers)] {
            if list.iter().any(|&l| l >= self.n_layers || self.compress_ratios[l] == 0) || !list.windows(2).all(|w| w[0] < w[1]) {
                return bad(format!("{name} source layers {list:?} must be increasing compressed layers"));
            }
        }
        for il in (0..self.n_layers).filter(|&l| self.compress_ratios[l] > 0) {
            let r = self.compress_ratios[il];
            let (kv, ix) = (self.kv_source(il), self.index_source(il));
            if kv.is_none_or(|s| self.compress_ratios[s] != r) || ix.is_none_or(|s| self.compress_ratios[s] != r) {
                return bad(format!("layer {il}: no kv / index source of ratio {r} at or before it"));
            }
        }
        if let Some(c) = self.candidate_source {
            if !self.index_source_layers.contains(&c) || self.candidate_topk_blocks == 0 || self.candidate_block_size == 0 {
                return bad(format!("candidate source {c} must be an index source with blocks"));
            }
            if self.index_source_layers.iter().any(|&l| l > c && self.compress_ratios[l] != self.compress_ratios[c]) {
                return bad("index sources after the candidate source must share its ratio".into());
            }
        }
        if let Some(e) = &self.engram {
            let ok = e.layer_ids.windows(2).all(|w| w[0] < w[1])
                && e.layer_ids.last().is_some_and(|&l| l < self.n_layers)
                && e.num_embeddings.len() == e.layer_ids.len()
                && e.max_ngram >= 2
                && e.n_heads > 0
                && e.head_dim > 0
                && e.head_dim.is_multiple_of(32);
            if !ok {
                return bad(format!("invalid engram config {e:?}"));
            }
        }
        Ok(())
    }

    /// Width of the hyper-connection mixing projection: pre (n) + post (n) + comb (n * n).
    pub fn hc_mix(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// The latest kv source at or before compressed layer `il`.
    pub fn kv_source(&self, il: usize) -> Option<usize> {
        self.kv_source_layers.iter().copied().rfind(|&s| s <= il)
    }

    /// The latest index source at or before compressed layer `il`.
    pub fn index_source(&self, il: usize) -> Option<usize> {
        self.index_source_layers.iter().copied().rfind(|&s| s <= il)
    }

    /// Engram module of layer `il`.
    pub fn engram_module(&self, il: usize) -> Option<usize> {
        self.engram.as_ref()?.layer_ids.iter().position(|&l| l == il)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The V4.1-Flash `config.json` text part (shortened to 6 layers).
    pub(crate) fn flash() -> Value {
        serde_json::from_str(
            r#"{"model_type": "deepseek_v41", "text_config": {"model_type": "deepseek_v41_text",
            "vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304, "num_hidden_layers": 6,
            "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280, "o_lora_rank": 1024,
            "o_groups": 8, "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "max_position_embeddings": 1048576, "rope_theta": 10000,
            "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536},
            "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 128, "compress_ratios": [0, 2, 2, 1, 1, 1, 0, 0],
            "compress_rope_theta": 160000, "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 5], "index_n_heads": 32,
            "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 3, "candidate_topk_blocks": 2048,
            "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 4],
            "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4, "engram_vocab_size": 16000000,
            "engram_n_heads": 8, "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092},
            "vision_config": {"num_hidden_layers": 32}}"#,
        )
        .expect("json")
    }

    #[test]
    fn parse() {
        let c = Deepseek41Config::from_json(&flash()).expect("cfg");
        assert_eq!(c.compress_ratios, vec![0, 2, 2, 1, 1, 1]);
        assert_eq!((c.kv_source(2), c.index_source(5), c.kv_source(5)), (Some(1), Some(5), Some(3)));
        assert_eq!(c.candidate_source, Some(3));
        assert_eq!(c.engram.as_ref().map(EngramConfig::cols), Some(24));
        assert_eq!(c.engram_module(4), Some(1));
        assert_eq!(c.norm_eps, 1e-20);
        // the stored (flattened) form parses the same
        assert_eq!(Deepseek41Config::from_json(&text_config(&flash())).expect("flat"), c);
    }

    #[test]
    fn rejects_bad_sources() {
        let mut v = flash();
        v["text_config"]["kv_source_layer_ids"] = json!([2, 3]);
        assert!(Deepseek41Config::from_json(&v).is_err());
    }
}
