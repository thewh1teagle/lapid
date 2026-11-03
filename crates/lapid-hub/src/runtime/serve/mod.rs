//! Serving any hub model through one object-safe trait: [`AnyModel`] exposes what a loaded
//! model can do (chat, completion, embedding, transcription, speech) through plain request
//! structs and callbacks; [`load`] / [`load_path`] turn a source (HF repo, directory,
//! `.gguf`, `.lapid`) into a `Box<dyn AnyModel>`, picking the model by the `.lapid`
//! metadata `architecture` (`runtime::registry`). Used by `lapid serve` (one model per worker
//! thread, requests one at a time, so adapters may keep mutable state such as sessions
//! without locking) and `lapid run`.

pub mod llm;
pub mod text;

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Error, LapidFile, Result, bail};
use serde::Serialize;
use serde_json::Value;

use super::resolve::resolve;

/// Options shared by every model of one `serve` / `run`.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Weight dtype for conversions ("int8", "fp16", "q4_k_m", ...); `None` = int8 for
    /// safetensors, the GGUF file's quant family for GGUF.
    pub dtype: Option<String>,
    /// Sessions kept per chat model for prefix reuse.
    pub sessions: usize,
    /// Server-side sampling defaults (used when a request leaves a field unset).
    pub defaults: GenParams,
    /// Activation steering files loaded with every chat model, on by default with their
    /// multipliers (requests choose with [`GenParams::steer`]).
    pub steer: Vec<super::steer::SteerSpec>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { dtype: None, sessions: 4, defaults: GenParams::default(), steer: Vec::new() }
    }
}

/// Base metadata of a `.lapid` file (capabilities and extras are filled by the adapter).
fn base_meta(file: &LapidFile, id: String, path: &Path, device: &Device) -> ModelMeta {
    let m = &file.metadata;
    let s = |v: &Value| v.as_str().unwrap_or("unknown").to_string();
    ModelMeta {
        id,
        name: s(&m["name"]),
        architecture: s(&m["architecture"]),
        task: String::new(),
        dtype: s(&m["dtype"]),
        license: s(&m["license"]),
        source: s(&m["source"]["repo"]),
        path: path.display().to_string(),
        file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        device: device.description(),
        capabilities: Vec::new(),
        load_ms: 0,
        extra: Default::default(),
    }
}

/// Load the converted model at `path` on `device` with the adapter its architecture needs.
pub fn load_path(path: &Path, id: String, device: &Device, opts: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    let t0 = Instant::now();
    let file = LapidFile::open(path)?;
    let meta = base_meta(&file, id, path, device);
    drop(file);
    let arch = meta.architecture.clone();
    let mut model: Box<dyn AnyModel> = match super::registry::find(&arch) {
        Some(entry) => (entry.serve)(path, device, meta, opts)?,
        None => bail!("unsupported_model", "{}: architecture {arch:?} cannot be served", path.display()),
    };
    let meta = model.meta_mut();
    meta.load_ms = t0.elapsed().as_millis() as u64;
    meta.task = match meta.capabilities.first() {
        Some(Capability::Chat | Capability::Completion) => "text-generation",
        Some(Capability::Embedding) => "feature-extraction",
        Some(Capability::Transcription) => "automatic-speech-recognition",
        Some(Capability::Speech) => "text-to-speech",
        None => "unknown",
    }
    .to_string();
    Ok(model)
}

/// Resolve `source` (converting into the cache on first use) and load it; `id` defaults to
/// the upstream repo / file name.
pub fn load(source: &str, id: Option<String>, device: &Device, opts: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    let r = resolve(source, opts.dtype.as_deref())?;
    load_path(&r.path, id.unwrap_or(r.id), device, opts)
}

/// Serve adapter of any causal LM (registry `serve` entry).
pub fn serve_lm<M: super::causal_lm::CausalLm>(path: &Path, device: &Device, meta: ModelMeta, opts: &LoadOptions) -> Result<Box<dyn AnyModel>> {
    Ok(Box::new(llm::LlmServed::new(super::causal_lm::Llm::<M>::load(path, device)?, meta, opts)?))
}

/// What a model serves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Chat,
    Completion,
    Embedding,
    Transcription,
    Speech,
}

/// Static description of a loaded model (`/v1/models`, `/api/info`).
#[derive(Clone, Debug, Serialize)]
pub struct ModelMeta {
    /// The id clients put in `"model"`.
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub task: String,
    pub dtype: String,
    pub license: String,
    pub source: String,
    pub path: String,
    pub file_size: u64,
    pub device: String,
    pub capabilities: Vec<Capability>,
    pub load_ms: u64,
    /// Model-specific extras (voices, sample_rate, embedding_dim, prompts, ...), flattened.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ModelMeta {
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Sampling and length controls shared by chat and text completion. `None` = the model's
/// recommended value.
#[derive(Clone, Debug, Default)]
pub struct GenParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Stop strings (the matched string is not returned).
    pub stop: Vec<String>,
    /// Steering files by name with multipliers, replacing the startup selection (empty:
    /// none); `None` = the startup selection.
    pub steer: Option<Vec<(String, f32)>>,
}

#[derive(Clone, Debug, Default)]
pub struct ChatParams {
    /// OpenAI messages normalized to `[{role, content: string, ...}]`.
    pub messages: Value,
    pub tools: Option<Value>,
    pub enable_thinking: Option<bool>,
    pub sampling: GenParams,
}

/// A piece of generated text.
#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Content(String),
    Reasoning(String),
}

/// How a generation ended, with token counts and timing.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GenSummary {
    /// "stop" | "length" | "cancelled".
    pub finish_reason: String,
    /// All prompt tokens (fed + reused from the session cache).
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: f64,
    pub decode_ms: f64,
}

impl GenSummary {
    pub fn prefill_tps(&self) -> f64 {
        rate(self.prompt_tokens - self.cached_tokens.min(self.prompt_tokens), self.ttft_ms)
    }

    pub fn decode_tps(&self) -> f64 {
        rate(self.completion_tokens.saturating_sub(1), self.decode_ms)
    }

    pub fn timings(&self) -> Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "generated_tokens": self.completion_tokens,
            "ttft_ms": round2(self.ttft_ms),
            "decode_ms": round2(self.decode_ms),
            "prefill_tps": round2(self.prefill_tps()),
            "decode_tps": round2(self.decode_tps()),
        })
    }
}

fn rate(n: usize, ms: f64) -> f64 {
    if ms > 0.0 { n as f64 / (ms / 1e3) } else { 0.0 }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[derive(Clone, Debug, Default)]
pub struct EmbedParams {
    pub dimensions: Option<usize>,
    /// Prompt name ("query", "document", ...) for models with a prompt table.
    pub prompt_name: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AsrParams {
    pub language: Option<String>,
    pub translate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AsrSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AsrWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptOut {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<AsrSegment>,
    pub words: Vec<AsrWord>,
}

#[derive(Clone, Debug, Default)]
pub struct TtsParams {
    pub voice: Option<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
}

fn unsupported(what: &str) -> Error {
    Error::new("unsupported", format!("this model does not support {what}"))
}

/// A loaded model as the server sees it. Default methods fail with "unsupported".
pub trait AnyModel {
    fn meta(&self) -> &ModelMeta;
    fn meta_mut(&mut self) -> &mut ModelMeta;

    /// Stream the assistant reply; `out` returns false when the client is gone (stop).
    fn chat(&mut self, _req: &ChatParams, _out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        Err(unsupported("chat completions"))
    }

    /// Continue raw `prompt` text.
    fn complete(&mut self, _prompt: &str, _gen: &GenParams, _out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        Err(unsupported("text completions"))
    }

    fn embed(&mut self, _texts: &[String], _opts: &EmbedParams) -> Result<Embeddings> {
        Err(unsupported("embeddings"))
    }

    /// Transcribe 16 kHz mono samples.
    fn transcribe(&mut self, _samples_16k: &[f32], _opts: &AsrParams) -> Result<TranscriptOut> {
        Err(unsupported("transcription"))
    }

    /// Stream PCM chunks (mono f32 at `meta().extra["sample_rate"]`); `out` returns false to stop.
    fn speak(&mut self, _text: &str, _opts: &TtsParams, _out: &mut dyn FnMut(&[f32]) -> bool) -> Result<()> {
        Err(unsupported("speech"))
    }

    /// Output sample rate of `speak` (0 when unsupported).
    fn sample_rate(&self) -> u32 {
        0
    }
}
