//! The causal LM pipeline. A model implements [`CausalLm`]: load its weights, make a state,
//! and build the lazy graph of one forward over new tokens (staging state, emitting named
//! taps for verification). [`Llm`] provides everything else: `.lapid` / hub loading, chunked
//! prefill + decode steps, sessions with prefix reuse, chat templates, sampling, stop tokens
//! and strings, token streaming, cancellation and timing, traced forwards for `lapid verify`,
//! lazy graphs for calibrated quantization.
//!
//! ```ignore
//! impl CausalLm for MyModel {
//!     const ARCH: &'static str = "mymodel";
//!     type State = LmCache;
//!     fn load(l: &Loader<'_>, meta: &Value) -> Result<Self> { ... }
//!     fn dims(&self) -> LmDims { ... }
//!     fn new_state(&self) -> LmCache { LmCache::new(KvCache::new(n_layers), 0) }
//!     fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> { ... }
//! }
//! pub type MyLlm = Llm<MyModel>;
//! ```

use std::path::Path;
use std::sync::Arc;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_nn::{Loader, Precision, SamplingParams, Tokenizer};
use serde_json::Value;

use super::chat::{ChatOptions, ChatTemplate, Message, TextAssets};
use super::generate::{GenerateOptions, TokenStream};
use super::session::{LmState, Session};
use super::steer::{Hook, SteerFile, Steering};
use crate::model::{Bytes, Model, ModelInfo};

/// Which logit rows a forward computes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The last position (decoding, prompt end).
    Last,
    /// Every position (verification).
    All,
    /// None (prefill chunks before the last).
    None,
}

/// Per-forward hooks: named intermediates collected during a traced forward (`embed`,
/// `layers.{i}`, `layers.{i}.op`, `norm`, ...; the pipeline adds `logits`; off during
/// generation), and the activation [`Steering`] applied at the model's hook points.
#[derive(Default)]
pub struct Taps {
    list: Option<Vec<(String, Tensor)>>,
    steering: Option<Arc<Steering>>,
}

impl Taps {
    pub fn off() -> Taps {
        Taps::default()
    }

    pub fn on() -> Taps {
        Taps { list: Some(Vec::new()), steering: None }
    }

    /// The same taps with `steering` applied by [`steer`](Self::steer).
    pub fn with_steering(mut self, steering: Option<Arc<Steering>>) -> Taps {
        self.steering = steering;
        self
    }

    pub fn steering(&self) -> Option<&Arc<Steering>> {
        self.steering.as_ref()
    }

    /// `x` steered at `hook` of `layer` (`x` itself without steering there).
    pub fn steer(&self, hook: Hook, layer: usize, x: &Tensor) -> Result<Tensor> {
        match &self.steering {
            Some(s) => s.apply(hook, layer, x),
            None => Ok(x.clone()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.list.is_some()
    }

    /// Record `t` under `name` (no-op when off; pass `format_args!(..)` to format lazily).
    pub fn tap(&mut self, name: impl std::fmt::Display, t: &Tensor) {
        if let Some(l) = &mut self.list {
            l.push((name.to_string(), t.clone()));
        }
    }

    pub fn into_vec(self) -> Vec<(String, Tensor)> {
        self.list.unwrap_or_default()
    }
}

/// Sizes the pipeline and the harness need.
#[derive(Copy, Clone, Debug)]
pub struct LmDims {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
}

/// A decoder-only language model: only its own math.
pub trait CausalLm: Sized + Send + Sync + 'static {
    /// `.lapid` metadata `architecture` and config object key.
    const ARCH: &'static str;
    type State: LmState;

    /// Build from weights; `meta` is the metadata object `ARCH`.
    fn load(l: &Loader<'_>, meta: &Value) -> Result<Self>;
    fn dims(&self) -> LmDims;
    fn new_state(&self) -> Self::State;
    /// Lazy graph for `ids` at positions `state.offset()..`, staging new state; returns the
    /// f32 logits of `rows` ([n_rows, vocab]), `None` for `Rows::None`.
    fn forward(&self, ids: &[u32], state: &mut Self::State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>>;

    /// Steering hook points the forward applies ([`Taps::steer`]); none by default.
    fn steer_hooks(&self) -> &'static [Hook] {
        &[]
    }
    /// Prompt tokens per evaluation during prefill (bounds attention scratch memory).
    fn prefill_chunk(&self) -> usize {
        512
    }
    /// Token strings that end a turn besides `eos_token_id` (e.g. `<|im_end|>`).
    fn end_of_turn(&self) -> &'static [&'static str] {
        &[]
    }
    /// Reasoning block markers (open, close) for splitting thinking from content.
    fn think_markers(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    /// Default chat-template globals beyond messages / tools / thinking / special tokens
    /// (per-call `ChatOptions::template_vars` override them).
    fn template_globals(&self) -> Vec<(&'static str, Value)> {
        Vec::new()
    }
    /// Whether a built-in speculative draft head is loaded ([`draft`](Self::draft)).
    fn can_draft(&self) -> bool {
        false
    }
    /// Up to `n_max` draft tokens after `anchor`, the sampled token at position
    /// `state.offset()` (not yet fed), from what `state` holds of the fed tokens; the draft
    /// stops at the first position whose confidence is below `p_min`. The state must
    /// [`truncate`](super::session::LmState::truncate) back over a verify forward of
    /// `n_max + 1` tokens.
    fn draft(&self, _state: &mut Self::State, _anchor: u32, _n_max: usize, _p_min: f32) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    /// Own prompt format instead of the stored Jinja template (messages / tools as JSON), e.g.
    /// DeepSeek V4's `encoding_dsv4`; `None` = render the template.
    fn render_prompt(&self, _messages_json: &str, _tools_json: Option<&str>, _add_generation_prompt: bool, _opts: &ChatOptions) -> Option<Result<String>> {
        None
    }
}

/// Evaluate `outputs` with the staged state and commit it (`n` tokens fed).
pub(crate) fn eval_commit<S: LmState>(device: &Device, outputs: &[&Tensor], state: &mut S, n: usize) -> Result<Vec<Tensor>> {
    eval_commit_owned(device, outputs.iter().map(|t| (*t).clone()).collect(), state, n)
}

/// [`eval_commit`] taking the outputs: with the staged state moved out too, nothing else
/// holds the step graph, so it is freed before it runs (`Device::eval_owned`).
pub(crate) fn eval_commit_owned<S: LmState>(device: &Device, mut outs: Vec<Tensor>, state: &mut S, n: usize) -> Result<Vec<Tensor>> {
    let n_out = outs.len();
    outs.extend(state.take_staged());
    let n_all = outs.len();
    let mut realized = match device.eval_owned(outs) {
        Ok(r) if r.len() == n_all => r,
        Ok(_) => {
            state.discard();
            return Err(Error::new("backend", "eval returned the wrong number of outputs"));
        }
        Err(e) => {
            state.discard();
            return Err(e);
        }
    };
    let st = realized.split_off(n_out);
    state.commit(st, n)?;
    Ok(realized)
}

/// Feed `ids` (a prompt, chunked, or one token) and return the f32 logits of the last position.
pub fn step<M: CausalLm>(model: &M, device: &Device, ids: &[u32], state: &mut M::State, steering: Option<&Arc<Steering>>) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(Error::new("invalid_argument", "step with no tokens"));
    }
    let chunks: Vec<&[u32]> = ids.chunks(model.prefill_chunk().max(1)).collect();
    let last = chunks.len() - 1;
    for (ci, chunk) in chunks.into_iter().enumerate() {
        let _step = tracing::trace_span!("lm.step").entered();
        let rows = if ci == last { Rows::Last } else { Rows::None };
        let build = tracing::trace_span!("lm.build").entered();
        let out = match model.forward(chunk, state, rows, &mut Taps::off().with_steering(steering.cloned())) {
            Ok(o) => o,
            Err(e) => {
                state.discard();
                return Err(e);
            }
        };
        drop(build);
        let realized = eval_commit_owned(device, out.into_iter().collect(), state, chunk.len())?;
        if ci == last {
            let logits = realized.first().ok_or_else(|| Error::new("backend", "no logits realized"))?;
            let v = tracing::trace_span!("lm.read").in_scope(|| device.to_vec_f32(logits));
            let _drop = tracing::trace_span!("lm.drop").entered();
            drop(realized);
            return v;
        }
    }
    Err(Error::new("internal", "prefill produced no logits"))
}

/// One uncached forward over `ids` from a fresh state with every tap realized (plus `logits`).
pub fn forward_traced<M: CausalLm>(model: &M, device: &Device, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
    let mut state = model.new_state();
    let mut taps = Taps::on();
    let logits = model.forward(ids, &mut state, rows, &mut taps)?;
    let mut list = taps.into_vec();
    if let Some(l) = logits {
        list.push(("logits".to_string(), l));
    }
    let refs: Vec<&Tensor> = list.iter().map(|(_, t)| t).collect();
    let realized = device.eval(&refs)?;
    list.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
}

/// Lazy last-position logits of one uncached forward (nothing evaluated; calibration).
pub fn forward_graph<M: CausalLm>(model: &M, ids: &[u32]) -> Result<Tensor> {
    let mut state = model.new_state();
    model.forward(ids, &mut state, Rows::Last, &mut Taps::off())?.ok_or_else(|| Error::new("internal", "forward graph has no logits"))
}

/// A loaded causal LM bound to a device, with its tokenizer and chat template.
pub struct Llm<M: CausalLm> {
    device: Device,
    info: ModelInfo,
    model: M,
    text: TextAssets,
}

impl<M: CausalLm> Llm<M> {
    /// Load `repo` from the lapid cache, converting it (hub converter routing) first if missing.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Llm<M>> {
        let convert = |r: &str, d: lapid_convert::OutDType, out: &Path| crate::convert::convert(r, d.name(), Some(out)).map(|_| ());
        Llm::load(&crate::model::hub_model_path(repo, dtype, convert)?, device)
    }

    /// Load a converted `.lapid` file.
    pub fn load(path: &Path, device: &Device) -> Result<Llm<M>> {
        Llm::load_with(path, device, Precision::Stored)
    }

    /// Load a `.lapid` image from memory (zero copy).
    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Llm<M>> {
        Llm::from_file(LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Llm<M>> {
        Llm::from_file(LapidFile::open(path)?, device, precision)
    }

    pub fn from_file(file: LapidFile, device: &Device, precision: Precision) -> Result<Llm<M>> {
        let meta = file
            .metadata
            .get(M::ARCH)
            .ok_or_else(|| Error::new("metadata", format!("{}: no {:?} metadata object", M::ARCH, M::ARCH)))?;
        let model = M::load(&Loader::new(&file, device, precision), meta)?;
        let mut text = TextAssets::from_file(&file, M::ARCH, meta, model.end_of_turn())?;
        for (k, v) in model.template_globals() {
            text.template.set_global(k, super::jinja::Value::from_json(&v));
        }
        Ok(Llm { device: device.clone(), info: ModelInfo::from_file(&file, "text-generation"), model, text })
    }

    /// The model's own math (config, ...).
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Mutable model access (e.g. image budgets of a vision-language model).
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.text.tokenizer
    }

    pub fn template(&self) -> &ChatTemplate {
        &self.text.template
    }

    /// Default stop tokens (eos ids + end-of-turn tokens).
    pub fn stop_tokens(&self) -> &[u32] {
        &self.text.stop
    }

    /// Sampling recommended by the model's generation defaults.
    pub fn recommended_sampling(&self) -> SamplingParams {
        self.text.recommended_sampling()
    }

    pub fn think_markers(&self) -> (&'static str, &'static str) {
        self.model.think_markers()
    }

    /// Render the chat template for typed messages.
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        self.render_chat_with(messages, add_generation_prompt, &ChatOptions { enable_thinking, ..Default::default() })
    }

    /// Render the chat template for HF-style messages / tools given as JSON.
    pub fn render_chat_json(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<String> {
        self.render_chat_json_with(messages_json, tools_json, add_generation_prompt, &ChatOptions { enable_thinking, ..Default::default() })
    }

    /// [`render_chat`](Self::render_chat) with per-call options (thinking, template vars).
    pub fn render_chat_with(&self, messages: &[Message], add_generation_prompt: bool, opts: &ChatOptions) -> Result<String> {
        if let Some(r) = self.model.render_prompt(&serde_json::to_string(messages)?, None, add_generation_prompt, opts) {
            return r;
        }
        self.text.template.render_with(messages, add_generation_prompt, opts)
    }

    /// [`render_chat_json`](Self::render_chat_json) with per-call options.
    pub fn render_chat_json_with(&self, messages_json: &str, tools_json: Option<&str>, add_generation_prompt: bool, opts: &ChatOptions) -> Result<String> {
        if let Some(r) = self.model.render_prompt(messages_json, tools_json, add_generation_prompt, opts) {
            return r;
        }
        self.text.template.render_json_with(messages_json, tools_json, add_generation_prompt, opts)
    }

    /// Token ids of rendered chat text (special tokens in the text are recognized; nothing added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.text.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.text.tokenizer.decode(ids, skip_special)
    }

    /// A fresh sequence.
    pub fn new_session(&self) -> Session<M::State> {
        Session::new(self.model.new_state())
    }

    /// Stream tokens after `prompt`; a cached prefix of `prompt` in `session` is not refed.
    pub fn generate<'a>(&'a self, session: &'a mut Session<M::State>, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let cached = session.reuse_prefix(prompt);
        let (model, device) = (&self.model, &self.device);
        if let Some(cfg) = super::spec::spec().filter(|_| model.can_draft()) {
            let spec = super::spec::LmSpec { model, device, session, p_min: cfg.p_min, last: 0 };
            return TokenStream::speculative(Box::new(spec), cfg.n_max, &self.text.tokenizer, prompt, cached, opts, &self.text.stop);
        }
        let steering = session.steering.clone();
        let step_fn = move |ids: &[u32]| -> Result<Vec<f32>> {
            let logits = step(model, device, ids, &mut session.state, steering.as_ref())?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step_fn, &self.text.tokenizer, prompt, cached, opts, &self.text.stop)
    }

    /// Stream the assistant reply to `messages` (template with the generation prompt).
    pub fn chat_stream<'a>(&'a self, session: &'a mut Session<M::State>, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let ids = self.encode(&self.render_chat_with(messages, true, opts)?)?;
        self.generate(session, &ids, opts.generate.clone())
    }

    /// The full assistant reply (non-streaming convenience).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Logits of the last position after feeding `ids` to `session` (low-level step).
    pub fn step(&self, session: &mut Session<M::State>, ids: &[u32]) -> Result<Vec<f32>> {
        let logits = step(&self.model, &self.device, ids, &mut session.state, session.steering.as_ref())?;
        session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    /// Read a steering file and check that it fits this model.
    pub fn load_steering(&self, path: &Path) -> Result<SteerFile> {
        let f = SteerFile::open(path)?;
        let d = self.model.dims();
        f.check(M::ARCH, d.hidden, d.layers, self.model.steer_hooks())?;
        Ok(f)
    }

    /// Steering of checked files with multipliers on this model's device (`None`: nothing
    /// applies); set it on a session with [`Session::set_steering`].
    pub fn steering(&self, parts: &[(&SteerFile, f32)]) -> Result<Option<Arc<Steering>>> {
        let d = self.model.dims();
        for (f, _) in parts {
            f.check(M::ARCH, d.hidden, d.layers, self.model.steer_hooks())?;
        }
        Steering::new(&self.device, parts)
    }

    /// Uncached forward over `ids` with every tap realized (verification).
    pub fn forward_traced(&self, ids: &[u32], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
        forward_traced(&self.model, &self.device, ids, rows)
    }
}

impl<M: CausalLm> Model for Llm<M> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
