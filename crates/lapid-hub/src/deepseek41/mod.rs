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
        let chunked = chunked_prefill_matches(&root, &out);
        let cached = cached_experts_match(&root, &out);
        let _ = std::fs::remove_file(&out);
        chunked.expect("chunked prefill");
        cached.expect("cached experts");
        assert_eq!(report["ok"], serde_json::json!(true), "{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    }

    /// Prefill + greedy decode with the routed experts in a 2-slot device cache (misses,
    /// evictions, multi-pass prefill) give the logits of the experts as graph leaves, with
    /// the stored (mxfp4) experts, on the CPU and on CUDA when present.
    fn cached_experts_match(root: &Path, file: &Path) -> lapid::Result<()> {
        use lapid_nn::llm::expert_cache::set_forced_slots;
        let result: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("result.json"))?)?;
        let ids: Vec<u32> = serde_json::from_value(result["cases"]["long"]["prompt_tokens"].clone())?;
        let run = |dev: &lapid::Device, slots: Option<usize>| -> lapid::Result<(Vec<Vec<f32>>, Option<lapid_nn::llm::expert_cache::CacheStats>)> {
            set_forced_slots(slots);
            let llm = Deepseek41::load_with(file, dev, lapid_nn::Precision::Stored);
            set_forced_slots(None);
            let llm = llm?;
            assert_eq!(llm.model().expert_cache().is_some(), slots.is_some());
            let mut s = llm.new_session();
            let mut out = vec![llm.step(&mut s, &ids[..ids.len().min(150)])?];
            for _ in 0..4 {
                let next = lapid_nn::argmax(out.last().map(Vec::as_slice).unwrap_or_default()) as u32;
                out.push(llm.step(&mut s, &[next])?);
            }
            Ok((out, llm.model().expert_cache().map(|c| c.stats())))
        };
        let mut devices = vec![lapid::Device::cpu()];
        devices.extend(lapid::Device::cuda(0).ok());
        for dev in &devices {
            let (want, _) = run(dev, None)?;
            let (got, stats) = run(dev, Some(2))?;
            let stats = stats.expect("stats");
            assert!(stats.extra_passes > 0 && stats.misses > 2, "{stats:?}");
            for (step, (a, b)) in got.iter().zip(&want).enumerate() {
                let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                assert!(diff < 1e-3 * scale, "{}: step {step}: cached experts differ by {diff}", dev.name());
            }
        }
        Ok(())
    }

    /// The long prompt fed in uneven chunks (open compressor groups, selections and Engram
    /// look-back carried across them) gives the logits of one whole prefill.
    fn chunked_prefill_matches(root: &Path, file: &Path) -> lapid::Result<()> {
        let result: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("result.json"))?)?;
        let ids: Vec<u32> = serde_json::from_value(result["cases"]["long"]["prompt_tokens"].clone())?;
        let llm = Deepseek41::load_with(file, &lapid::Device::cpu(), lapid_nn::Precision::FakeQuantWeights)?;
        let whole = llm.step(&mut llm.new_session(), &ids)?;
        let mut s = llm.new_session();
        let mut last = Vec::new();
        for c in [&ids[..37], &ids[37..38], &ids[38..151], &ids[151..]] {
            last = llm.step(&mut s, c)?;
        }
        let diff = whole.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(diff < 1e-3, "chunked prefill differs by {diff}");
        Ok(())
    }
}
