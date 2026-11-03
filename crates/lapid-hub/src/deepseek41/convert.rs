//! DeepSeek V4.1 checkpoints -> `.lapid`: the official safetensors release
//! (`deepseek-ai/DeepSeek-V4.1-*`: FP8 e4m3 linears with 32x32 e8m0 block scales, FP4 e2m1
//! routed experts with e8m0 / 32, FP8 Engram tables), read through lapid-convert's
//! FP8 / FP4 scale pairing.
//!
//! Our names = upstream names, except: routed experts stacked as `layers.{i}.ffn.experts.{w}`
//! `[E, N, K]` (official FP4 -> `mxfp4`, a lossless repack, whatever the output dtype),
//! `attn.wo_a.weight` as `[G, o_lora, H / G * head_dim]`, and the Engram tables written
//! unchanged as blobs `layers.{i}.engram.embed.fp8` (e4m3 `[rows, head_dim]`) and `.e8m0`
//! (`[rows, head_dim / 32]`), gathered on the host at run time. The DSpark draft stages
//! `mtp.{s}.*` are kept (their 128 experts stacked like the layers'); [`CONVERT_DSPARK`]
//! writes them alone (`lapid convert --dspark`: a draft file for a model converted without
//! them). Dropped: vision tower, aligner, image embeddings, the vision routing bias. The Engram hash
//! constants are derived here from the config and the tokenizer ([`engram`](super::engram)).
//!
//! Precision: FP8 linears dequantized exactly, then the dtype policy (Linear: int8 by
//! default); router, compressors, indexer weights projection, norms, sinks, hyper-connections
//! and the Engram gate weights f32.

use lapid::{Error, Result};
use lapid_convert::assets::{HfFiles, TOKENIZER_BLOB, special_tokens};
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, SourceKind, TensorSource};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::writer::read_json;
use serde_json::{Value, json};

use super::config::{Deepseek41Config, text_config};
use super::engram;
use super::model::{EMBED, HEAD};

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "deepseek41",
    license: "mit",
    hf_model_types: &["deepseek_v41"],
    gguf_architectures: &[],
    hf_names: HF_NAMES,
    gguf_names: &[],
    roles: ROLES,
    text_assets: false,
    hooks: &Hooks { draft_only: false },
};

/// The DSpark draft stages alone (`mtp.*`), for `--spec-draft-model`.
pub static CONVERT_DSPARK: ConvertSpec = ConvertSpec {
    architecture: "deepseek41",
    license: "mit",
    hf_model_types: &[],
    gguf_architectures: &[],
    hf_names: HF_NAMES,
    gguf_names: &[],
    roles: ROLES,
    text_assets: false,
    hooks: &Hooks { draft_only: true },
};

const HF_NAMES: &[Rule] = &[
    Rule::drop("vision.*"),
    Rule::drop("aligner.*"),
    Rule::drop("image_*"),
    Rule::drop("layers.{i}.ffn.gate.bias_vl"),
    Rule::drop("mtp.{i}.ffn.gate.bias_vl"),
];

const ROLES: &[RoleRule] = &[
        RoleRule::new("*.compressor.*", Role::Param),
        RoleRule::new("*weights_proj.weight", Role::Param),
        RoleRule::new("*.ffn.gate.*", Role::Param),
        RoleRule::new("*hc_*", Role::Param),
        RoleRule::new("*.engram.q_weight", Role::Param),
        RoleRule::new("*.engram.k_weight", Role::Param),
        RoleRule::new(EMBED, Role::LinearInt8),
        RoleRule::new("*.ffn.experts.*", Role::Linear),
        RoleRule::new("*.attn.wo_a.weight", Role::Linear),
        RoleRule::new("*.markov_head.embed.weight", Role::LinearInt8),
        RoleRule::new("*.confidence_head.proj.weight", Role::Param),
];

struct Hooks {
    /// Only the DSpark stages ([`CONVERT_DSPARK`]).
    draft_only: bool,
}

fn cfg_of(c: &Value) -> Result<Deepseek41Config> {
    Deepseek41Config::from_json(c)
}

/// Layer index of `layers.{i}.*`.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("layers.")?.split('.').next()?.parse().ok()
}

/// `layers.{i}.ffn.experts.{e}.{w}.weight` -> (`layers.{i}`, e, w).
fn per_expert(name: &str) -> Option<(&str, usize, &str)> {
    let (prefix, rest) = name.split_once(".ffn.experts.")?;
    let (e, rest) = rest.split_once('.')?;
    let w = rest.strip_suffix(".weight")?;
    Some((prefix, e.parse().ok()?, w))
}

impl ConvertHooks for Hooks {
    /// The flattened text config (validated).
    fn hf_config(&self, c: &Value) -> Result<Value> {
        let flat = text_config(c);
        cfg_of(&flat).map(|_| flat)
    }

    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        if self.draft_only {
            return Ok(HfExtras::default());
        }
        let tok_cfg = files.optional("tokenizer_config.json")?.as_deref().map(read_json).transpose()?;
        cfg["special_tokens"] = special_tokens(tok_cfg.as_ref(), None);
        cfg["generation"] = match files.optional("generation_config.json")? {
            Some(p) => read_json(&p)?,
            None => json!({}),
        };
        let p = files.required("tokenizer.json")?;
        let tok = std::fs::read(&p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())))?;
        let c = cfg_of(cfg)?;
        let mut blobs = Vec::new();
        if let Some(ec) = &c.engram {
            let (meta, map) = engram::derive(ec, c.vocab_size, &tok)?;
            cfg["engram_hash"] = meta;
            blobs.push((engram::TOKEN_MAP_BLOB.into(), "application/octet-stream".into(), map));
        }
        cfg["tokenizer_blob"] = json!(TOKENIZER_BLOB);
        blobs.push((TOKENIZER_BLOB.into(), "application/json".into(), tok));
        Ok(HfExtras { blobs, ..Default::default() })
    }

    /// The Engram tables stay FP8 bytes (no scale pairing).
    fn hf_unpaired(&self, hf_name: &str) -> bool {
        hf_name.contains(".engram.embed.")
    }

    fn expected(&self, c: &Value) -> Result<Vec<String>> {
        let cfg = cfg_of(c)?;
        let mut v = dspark_names(&cfg);
        if self.draft_only {
            if v.is_empty() {
                return Err(Error::new("unsupported_model", "deepseek41: the checkpoint has no DSpark draft (num_nextn_predict_layers / dspark_*)"));
            }
            return Ok(v);
        }
        v.extend([EMBED, HEAD, "norm.weight"].map(String::from));
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let mut names: Vec<String> = ["attn_norm.weight", "ffn_norm.weight", "ffn.gate.weight", "ffn.gate.bias"].map(String::from).to_vec();
            names.extend(["attn", "ffn"].iter().flat_map(|k| ["fn", "base", "scale"].map(|s| format!("hc_{k}_{s}"))));
            names.extend(["attn_sink", "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "wo_a.weight", "wo_b.weight"].map(|n| format!("attn.{n}")));
            if cfg.kv_source_layers.contains(&i) {
                names.extend(["compressor.wkv.weight", "compressor.norm.weight", "indexer.wk.weight", "indexer.k_norm.weight"].map(|n| format!("attn.{n}")));
                if cfg.compress_ratios[i] > 1 {
                    names.push("attn.compressor.wgate.weight".into());
                }
            }
            if cfg.index_source_layers.contains(&i) {
                names.extend(["attn.indexer.wq_b.weight".into(), "attn.indexer.weights_proj.weight".into()]);
            }
            if cfg.engram_module(i).is_some() {
                names.extend(["wkv.weight", "q_weight", "k_weight", "embed.fp8", "embed.e8m0"].map(|n| format!("engram.{n}")));
            }
            names.extend(["w1", "w2", "w3"].iter().flat_map(|w| [format!("ffn.experts.{w}"), format!("ffn.shared_experts.{w}.weight")]));
            v.extend(names.iter().map(|n| format!("{p}.{n}")));
        }
        Ok(v)
    }

    /// `attn.wo_a` `[G * o_lora, in]` -> `[G, o_lora, in]`.
    fn reshape(&self, name: &str, shape: &[usize], c: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        if !(name.ends_with(".attn.wo_a.weight") && shape.len() == 2) {
            return None;
        }
        let cfg = cfg_of(c).ok()?;
        Some(vec![cfg.o_groups, cfg.o_lora_rank, shape[1]])
    }

    /// Drops layers past the backbone (and all but the draft for [`CONVERT_DSPARK`]);
    /// Engram tables -> blobs; routed experts -> stacks.
    fn transform(&self, name: &str, _src: &dyn TensorSource, c: &Value) -> Result<Emit> {
        let cfg = cfg_of(c)?;
        if layer_of(name).is_some_and(|l| l >= cfg.n_layers) || (self.draft_only && !name.starts_with("mtp.")) {
            return Ok(Emit::Drop);
        }
        if let Some(s) = name.strip_prefix("mtp.").and_then(|r| r.split('.').next()).and_then(|s| s.parse::<usize>().ok())
            && cfg.dspark.as_ref().is_none_or(|d| s >= d.stages)
        {
            return Ok(Emit::Drop);
        }
        for (suffix, kind) in [(".engram.embed.weight", "fp8"), (".engram.embed.scale", "e8m0")] {
            if let Some(p) = name.strip_suffix(suffix) {
                return Ok(Emit::Blob { name: format!("{p}.engram.embed.{kind}"), media_type: "application/octet-stream" });
            }
        }
        if let Some((prefix, e, w)) = per_expert(name) {
            if e != 0 {
                return Ok(Emit::Drop);
            }
            let n = if prefix.starts_with("mtp.") { cfg.dspark.as_ref().map_or(0, |d| d.n_experts) } else { cfg.n_experts };
            let parts = (0..n).map(|x| format!("{prefix}.ffn.experts.{x}.{w}.weight")).collect();
            return Ok(Emit::Stack { name: format!("{prefix}.ffn.experts.{w}"), parts });
        }
        Ok(Emit::Keep)
    }
}

/// `lapid convert --dspark`: only the DSpark draft of a DeepSeek V4.1 checkpoint (`src`: HF
/// directory or repo) to `out`.
pub fn convert_dspark(src: &str, dtype: &str, out: &std::path::Path) -> Result<()> {
    CONVERT_DSPARK.convert(src, lapid_convert::OutDType::parse(dtype)?, out)
}

/// Tensors of the DSpark stages (none without a draft config).
fn dspark_names(cfg: &super::Deepseek41Config) -> Vec<String> {
    let Some(d) = &cfg.dspark else { return Vec::new() };
    let mut v = vec!["mtp.0.main_proj.weight".to_string(), "mtp.0.main_norm.weight".to_string()];
    for s in 0..d.stages {
        let mut names: Vec<String> = ["attn_norm.weight", "ffn_norm.weight", "ffn.gate.weight", "ffn.gate.bias"].map(String::from).to_vec();
        names.extend(["attn", "ffn"].iter().flat_map(|k| ["fn", "base", "scale"].map(|x| format!("hc_{k}_{x}"))));
        names.extend(["attn_sink", "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "wo_a.weight", "wo_b.weight"].map(|n| format!("attn.{n}")));
        names.extend(["w1", "w2", "w3"].iter().flat_map(|w| [format!("ffn.experts.{w}"), format!("ffn.shared_experts.{w}.weight")]));
        if s + 1 == d.stages {
            names.extend(["norm.weight", "markov_head.embed.weight", "markov_head.head.weight", "confidence_head.proj.weight"].map(String::from));
        }
        v.extend(names.iter().map(|n| format!("mtp.{s}.{n}")));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::mapping::{Renamed, rename};

    #[test]
    fn names_and_roles() {
        let h = |n: &str| rename(CONVERT.hf_names, n);
        assert_eq!(h("mtp.0.ffn.gate.bias_vl"), Renamed::Drop);
        assert_eq!(per_expert("mtp.2.ffn.experts.5.w1.weight"), Some(("mtp.2", 5, "w1")));
        assert_eq!(CONVERT.role("mtp.2.confidence_head.proj.weight", &[1, 5376]), Role::Param);
        assert_eq!(h("layers.3.ffn.gate.bias_vl"), Renamed::Drop);
        assert_eq!(per_expert("layers.3.ffn.experts.17.w2.weight"), Some(("layers.3", 17, "w2")));
        assert_eq!(CONVERT.role(EMBED, &[129280, 5120]), Role::LinearInt8);
        assert_eq!(CONVERT.role("layers.2.attn.compressor.wkv.weight", &[512, 5120]), Role::Param);
        assert_eq!(CONVERT.role("layers.1.engram.q_weight", &[4, 5120]), Role::Param);
        assert_eq!(CONVERT.role("layers.1.engram.wkv.weight", &[25600, 6144]), Role::Linear);
        assert_eq!(CONVERT.role("layers.1.ffn.experts.w1", &[384, 2304, 5120]), Role::Linear);
    }
}
