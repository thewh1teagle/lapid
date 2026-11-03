//! The model registry: one [`ModelEntry`] per migrated model routes `lapid convert`
//! (HF `model_type`, GGUF `general.architecture`), `lapid verify` / `lapid bench` (by name)
//! and serving (`.lapid` metadata `architecture`). Adding a model = one line in [`MODELS`].
//! Models not yet migrated are still routed by the legacy tables in `crate::convert`,
//! `crate::cmd` and `serve::load_path`.

use std::path::Path;

use lapid::{Device, Result};
use lapid_convert::calibrate::{CalibrationSource, Calibrator};
use lapid_convert::engine::ConvertSpec;
use lapid_convert::gguf::GgufFile;
use lapid_verify::harness::Descriptor;
use serde_json::Value;

use super::causal_lm::CausalLm;
use super::lm_calibrate::LmCalibrator;
use super::serve::{AnyModel, LoadOptions, ModelMeta, serve_lm};
use crate::cmd::ModelArgs;

type VerifyFn = fn(Option<&Path>, &ModelArgs) -> Result<Value>;
type BenchFn = fn(Option<&Path>, &ModelArgs, usize) -> Result<Value>;
type ServeFn = fn(&Path, &Device, ModelMeta, &LoadOptions) -> Result<Box<dyn AnyModel>>;
type CalibrateFn = fn(&Path, &Device, &CalibrationSource) -> Result<Box<dyn Calibrator>>;

/// Everything the shared tooling needs to know about one model.
pub struct ModelEntry {
    /// `lapid verify <name>` / `lapid bench <name>`.
    pub name: &'static str,
    pub convert: &'static ConvertSpec,
    pub verify: VerifyFn,
    pub bench: BenchFn,
    pub serve: ServeFn,
    /// Calibrated int4 support.
    pub calibrate: Option<CalibrateFn>,
}

/// The descriptors of a causal LM (implemented next to its [`CausalLm`] impl).
pub trait LmDef: CausalLm {
    const DESCRIPTOR: Descriptor;
    const CONVERT: &'static ConvertSpec;
}

fn verify_lm<M: LmDef>(r: Option<&Path>, a: &ModelArgs) -> Result<Value> {
    super::lm_verify::verify::<M>(&M::DESCRIPTOR, r, a)
}

fn bench_lm<M: LmDef>(i: Option<&Path>, a: &ModelArgs, runs: usize) -> Result<Value> {
    super::lm_bench::bench::<M>(&M::DESCRIPTOR, i, a, runs)
}

fn calibrate_lm<M: LmDef>(p: &Path, d: &Device, s: &CalibrationSource) -> Result<Box<dyn Calibrator>> {
    Ok(Box::new(LmCalibrator::<M>::open(p, d, s, M::CONVERT)?))
}

impl ModelEntry {
    /// The full entry of a causal LM: generic verify, bench, serving and calibration.
    pub const fn causal_lm<M: LmDef>() -> ModelEntry {
        ModelEntry {
            name: M::DESCRIPTOR.name,
            convert: M::CONVERT,
            verify: verify_lm::<M>,
            bench: bench_lm::<M>,
            serve: serve_lm::<M>,
            calibrate: Some(calibrate_lm::<M>),
        }
    }
}

/// Migrated models.
pub static MODELS: &[&ModelEntry] = &[&crate::lfm3::ENTRY, &crate::deepseek4::ENTRY, &crate::deepseek41::ENTRY, &crate::embeddinggemma::ENTRY, &crate::phonon::ENTRY, &crate::llama::ENTRY, &crate::qwen35::ENTRY, &crate::qwen4exp::ENTRY, &crate::whisper::ENTRY, &crate::parakeet::ENTRY, &crate::zipformer::ENTRY, &crate::vad::ENTRY, &crate::gemma4::ENTRY, &crate::gemma4::MOE_ENTRY, &crate::gemma4::VL_ENTRY, &crate::qwen3vl::ENTRY];

/// Entry by verify/bench name or `.lapid` architecture (a spec's GGUF architectures name
/// the variants it writes too, e.g. `qwen3` / `qwen3moe` files of `llama`).
pub fn find(name: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().copied().find(|e| e.name == name || e.convert.architecture == name || e.convert.gguf_architectures.contains(&name))
}

/// Entry converting HF `config.json` `model_type`.
pub fn for_model_type(model_type: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().copied().find(|e| e.convert.accepts_model_type(model_type))
}

/// Entry converting a GGUF file.
pub fn for_gguf(f: &GgufFile) -> Option<&'static ModelEntry> {
    MODELS.iter().copied().find(|e| e.convert.accepts_gguf(f))
}

/// Names of the registered models.
pub fn names() -> Vec<&'static str> {
    MODELS.iter().map(|e| e.name).collect()
}
