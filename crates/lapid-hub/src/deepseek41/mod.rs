//! DeepSeek V4.1 (`deepseek_v41`; llama.cpp `deepseek41`): DeepSeek V4's hyper-connections,
//! shared-KV MQA (head dim 512, sinks), grouped low-rank output and sqrt-softplus MoE, with
//! compressed KV of ratio 1 / 2 shared across layers (kv-source layers compress, index-source
//! layers pick a top-k, a candidate-source layer pre-selects blocks), sparse top-k attention
//! on every compressed layer, no per-head q norm, the lagged ("single-pass") hyper-connection
//! pre-mix and Engram: n-gram hashed FP8 lookup tables gated into the residual streams
//! (hashed and gathered on the host from the mmap'd file). Text only (no vision tower, no
//! DSpark drafter). Prompts follow the upstream `encoding/encoding.py` ([`encoding`]).
//!
//! ```no_run
//! use lapid_hub::deepseek41::{ChatOptions, Deepseek41, Message};
//! let device = lapid::Device::best();
//! let llm = Deepseek41::load("deepseek41.lapid".as_ref(), &device)?;
//! let mut session = llm.new_session();
//! for token in llm.chat_stream(&mut session, &[Message::user("Hi!")], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod config;
pub mod convert;
pub mod encoding;
pub mod engram;
pub mod model;
mod numpy_rng;

use lapid_convert::engine::ConvertSpec;
use lapid_verify::Descriptor;

pub use config::{Deepseek41Config, EngramConfig};
pub use model::Deepseek41Model;
pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
use crate::runtime::{LmDef, Llm, ModelEntry};

/// A loaded DeepSeek V4.1 model (the generic causal-LM pipeline over [`Deepseek41Model`]).
pub type Deepseek41 = Llm<Deepseek41Model>;
pub type Session = crate::runtime::Session<model::State>;

impl LmDef for Deepseek41Model {
    /// The reference is a random-weight model with the V4.1-Flash structure in the official
    /// on-disk format (`deepseek41_ref.py --tiny`).
    const DESCRIPTOR: Descriptor = Descriptor {
        name: "deepseek41",
        default_model: "data/reference/deepseek41/tiny/model",
        default_dtype: "fp32",
        reference_dir: "data/reference/deepseek41/tiny",
        reference_script: "tools/reference/deepseek41_ref.py",
        reference_label: "PyTorch fp32 (official inference/model.py)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Deepseek41Model>();

impl Deepseek41 {
    pub fn config(&self) -> &Deepseek41Config {
        &self.model().cfg
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// The tiny reference (`deepseek41_ref.py --tiny`), when present: convert (fp32) and verify on the CPU.
    #[test]
    fn tiny_reference_cpu() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/deepseek41/tiny");
        if !root.join("result.json").is_file() || !root.join("model/config.json").is_file() {
            eprintln!("skipped: {} not found", root.display());
            return;
        }
        let out = std::env::temp_dir().join(format!("lapid-deepseek41-tiny-{}.lapid", std::process::id()));
        let model = root.join("model").display().to_string();
        convert::CONVERT.convert(&model, lapid_convert::OutDType::F32, &out).expect("convert");
        let args = crate::cmd::ModelArgs { model: Some(out.display().to_string()), dtype: Some("fp32".into()), device: Some("cpu".into()) };
        let report = (ENTRY.verify)(Some(&root), &args).expect("verify");
        let _ = std::fs::remove_file(&out);
        assert_eq!(report["ok"], serde_json::json!(true), "{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    }
}
