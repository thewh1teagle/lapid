//! Developer entry points the `lapid` binary routes to: `convert`, `verify <model>`,
//! `bench <model>`. Registered models (`runtime::registry`) get the generic harness from
//! their descriptor; the others implement `verify` / `bench` in their own `<model>/cmd.rs`.
//! This module only routes by model name. Apps (transcription, ...) are examples.

use std::path::Path;

use lapid::{Result, bail};
use serde_json::Value;

pub use crate::convert::{convert, convert_with};

/// Model families `verify` / `bench` route to.
pub const MODELS: &[&str] = &["whisper", "embeddinggemma", "parakeet", "lfm3", "phonon", "qwen35", "deepseek4", "qwen4exp", "gemma4", "gemma4moe", "gemma4vl", "qwen3vl", "llama", "vad", "zipformer"];

/// Common model selection of `lapid verify` / `lapid bench`; `None` = the model's default.
#[derive(Clone, Debug, Default)]
pub struct ModelArgs {
    /// HF repo id, local HF directory or `.lapid` file.
    pub model: Option<String>,
    /// Weight dtype ("fp16", "int8", ...).
    pub dtype: Option<String>,
    /// Device name ("cpu", "cuda", "vulkan:1", ...); `None` = best / `$LAPID_DEVICE`.
    pub device: Option<String>,
}

fn unknown(model: &str) -> lapid::Error {
    lapid::Error::new("invalid_argument", format!("unknown model {model:?} (expected one of {})", MODELS.join(", ")))
}

/// Verify `model` against a reference directory (default: the model's own). Check `"ok"`.
/// Device memory (GiB) for MoE experts that do not fit in memory (CLI `--expert-cache`).
pub fn set_expert_cache_gib(gib: f64) {
    lapid_nn::llm::expert_cache::set_budget_gib(gib);
}

pub fn verify(model: &str, reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    if let Some(e) = crate::runtime::registry::find(model) {
        return (e.verify)(reference, args);
    }
    match model {
        _ => Err(unknown(model)),
    }
}

/// Benchmark `model` on `input` (default: the model's own sample) over `runs` runs.
pub fn bench(model: &str, input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    if runs == 0 {
        bail!("invalid_argument", "--runs must be at least 1");
    }
    if let Some(e) = crate::runtime::registry::find(model) {
        return (e.bench)(input, args, runs);
    }
    match model {
        _ => Err(unknown(model)),
    }
}
