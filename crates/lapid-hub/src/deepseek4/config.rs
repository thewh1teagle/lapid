//! DeepSeek V4 hyper-parameters. Read from the upstream `config.json` (V4-Flash / V4-Pro keys:
//! `compress_ratios`, `num_hash_layers`, `rope_scaling`, ...), from a transformers 5 style config
//! (`layer_types` + `compress_rates`, `mlp_layer_types`, nested `rope_parameters`) or from the
//! `.lapid` metadata object `"deepseek4"` (the source config as converted; round-5 files hold an
//! upstream-style copy), so one parser reads all of them.

use lapid::{Error, Result, bail};
pub use lapid_nn::llm::rope::Yarn;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Deepseek4Config {
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
    pub n_shared_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub route_scale: f32,
    /// SwiGLU clamp (`gate <= l`, `-l <= up <= l`); 0 disables.
    pub swiglu_limit: f32,
    pub sliding_window: usize,
    /// Per layer: 0 = sliding window only, 4 = CSA (overlapping compressor + indexer), 128 = HCA.
    pub compress_ratios: Vec<usize>,
    /// The first `n_hash_layers` MoE layers route by a token-id table.
    pub n_hash_layers: usize,
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
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::new("metadata", format!("deepseek4 config: missing integer {key:?}")))
}

fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn opt_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

/// Per-layer compression ratios: upstream `compress_ratios`, else transformers `layer_types`.
fn ratios(v: &Value, n_layers: usize) -> Result<Vec<usize>> {
    if let Some(a) = v.get("compress_ratios").and_then(Value::as_array) {
        let r: Vec<usize> = a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect();
        if r.len() < n_layers {
            bail!("metadata", "deepseek4: {} compress ratios for {n_layers} layers", r.len());
        }
        return Ok(r[..n_layers].to_vec());
    }
    let types = v
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("metadata", "deepseek4 config: neither compress_ratios nor layer_types"))?;
    let rates = v.get("compress_rates");
    let rate = |k: &str, d: usize| rates.and_then(|r| r.get(k)).and_then(Value::as_u64).map_or(d, |x| x as usize);
    types
        .iter()
        .take(n_layers)
        .map(|t| match t.as_str() {
            Some("sliding_attention") => Ok(0),
            Some("compressed_sparse_attention") => Ok(rate("compressed_sparse_attention", 4)),
            Some("heavily_compressed_attention") => Ok(rate("heavily_compressed_attention", 128)),
            other => Err(Error::new("unsupported_model", format!("deepseek4: unknown layer type {other:?}"))),
        })
        .collect()
}

/// YaRN from `rope_scaling` (`type` / `rope_type` "yarn") or transformers `rope_parameters`.
pub(crate) fn yarn(v: &Value) -> Option<Yarn> {
    let r = v
        .get("rope_scaling")
        .filter(|r| r.is_object())
        .or_else(|| v.get("rope_parameters").and_then(|p| p.get("compress").or(Some(p))))?;
    let kind = r.get("rope_type").or_else(|| r.get("type")).and_then(Value::as_str).unwrap_or("default");
    if kind != "yarn" {
        return None;
    }
    Some(Yarn {
        factor: opt_f32(r, "factor").unwrap_or(1.0),
        original_max_positions: opt_usize(r, "original_max_position_embeddings").unwrap_or(65536),
        beta_fast: opt_f32(r, "beta_fast").unwrap_or(32.0),
        beta_slow: opt_f32(r, "beta_slow").unwrap_or(1.0),
    })
}

impl Deepseek4Config {
    pub fn from_json(v: &Value) -> Result<Deepseek4Config> {
        if let Some(mt) = v.get("model_type").and_then(Value::as_str)
            && mt != "deepseek_v4"
        {
            bail!("unsupported_model", "deepseek4: model_type {mt:?} is not deepseek_v4");
        }
        let n_layers = get_usize(v, "num_hidden_layers")?;
        let head_dim = get_usize(v, "head_dim")?;
        let rope_dim = match opt_usize(v, "qk_rope_head_dim") {
            Some(r) => r,
            None => {
                let f = v.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(0.125);
                (head_dim as f64 * f) as usize
            }
        };
        let n_hash_layers = match opt_usize(v, "num_hash_layers") {
            Some(n) => n,
            None => v
                .get("mlp_layer_types")
                .and_then(Value::as_array)
                .map_or(3, |a| a.iter().filter(|t| t.as_str() == Some("hash_moe")).count()),
        };
        let main_theta = v
            .get("rope_parameters")
            .and_then(|p| p.get("main"))
            .and_then(|m| opt_f32(m, "rope_theta"))
            .or_else(|| opt_f32(v, "rope_theta"))
            .unwrap_or(10000.0);
        let cfg = Deepseek4Config {
            vocab_size: get_usize(v, "vocab_size")?,
            hidden_size: get_usize(v, "hidden_size")?,
            n_layers,
            n_head: get_usize(v, "num_attention_heads")?,
            head_dim,
            rope_dim,
            q_lora_rank: get_usize(v, "q_lora_rank")?,
            o_groups: get_usize(v, "o_groups")?,
            o_lora_rank: get_usize(v, "o_lora_rank")?,
            n_experts: get_usize(v, "n_routed_experts")?,
            n_shared_experts: opt_usize(v, "n_shared_experts").unwrap_or(1),
            top_k: get_usize(v, "num_experts_per_tok")?,
            moe_intermediate: get_usize(v, "moe_intermediate_size")?,
            route_scale: opt_f32(v, "routed_scaling_factor").unwrap_or(1.0),
            swiglu_limit: opt_f32(v, "swiglu_limit").unwrap_or(0.0),
            sliding_window: get_usize(v, "sliding_window")?,
            compress_ratios: ratios(v, n_layers)?,
            n_hash_layers,
            rope_theta: main_theta,
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
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.n_head == 0 || self.o_groups == 0 || !self.n_head.is_multiple_of(self.o_groups) {
            bail!("metadata", "deepseek4: {} heads not divisible into {} output groups", self.n_head, self.o_groups);
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim.min(self.index_head_dim) {
            bail!("metadata", "deepseek4: rope dim {} invalid for head dims {} / {}", self.rope_dim, self.head_dim, self.index_head_dim);
        }
        if let Some(r) = self.compress_ratios.iter().find(|&&r| !matches!(r, 0 | 4 | 128)) {
            bail!("unsupported_model", "deepseek4: compress ratio {r} (only 0, 4 and 128 exist)");
        }
        if self.n_shared_experts != 1 {
            bail!("unsupported_model", "deepseek4: {} shared experts (expected 1)", self.n_shared_experts);
        }
        if self.hc_mult == 0 || self.sliding_window == 0 || self.top_k == 0 || self.top_k > self.n_experts {
            bail!("metadata", "deepseek4: invalid hc_mult / window / top_k");
        }
        Ok(())
    }

    /// Width of the hyper-connection mixing projection: pre (n) + post (n) + comb (n * n).
    pub fn hc_mix(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// Compressor output width per token: CSA (ratio 4) carries two overlapping series.
    pub fn compress_coff(ratio: usize) -> usize {
        if ratio == 4 { 2 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flash() -> Value {
        json!({"model_type": "deepseek_v4", "vocab_size": 129280, "hidden_size": 4096, "num_hidden_layers": 4,
            "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_groups": 8,
            "o_lora_rank": 1024, "n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 6,
            "moe_intermediate_size": 2048, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "sliding_window": 128,
            "compress_ratios": [0, 4, 128, 4, 0], "num_hash_layers": 3, "rope_theta": 10000, "compress_rope_theta": 160000,
            "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"},
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": 64, "index_head_dim": 128,
            "index_topk": 512, "eos_token_id": 1, "bos_token_id": 0})
    }

    #[test]
    fn parse() {
        let c = Deepseek4Config::from_json(&flash()).expect("cfg");
        assert_eq!(c.compress_ratios, vec![0, 4, 128, 4]);
        assert_eq!(c.hc_mix(), 24);
        assert_eq!(c.yarn.as_ref().map(|y| y.factor), Some(16.0));
    }

    #[test]
    fn transformers_style() {
        let mut v = flash();
        let o = v.as_object_mut().expect("obj");
        o.remove("compress_ratios");
        o.remove("num_hash_layers");
        o.insert("layer_types".into(), json!(["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"]));
        o.insert("mlp_layer_types".into(), json!(["hash_moe", "moe", "moe", "moe"]));
        let c = Deepseek4Config::from_json(&v).expect("cfg");
        assert_eq!(c.compress_ratios, vec![0, 4, 128, 0]);
        assert_eq!(c.n_hash_layers, 1);
    }
}
