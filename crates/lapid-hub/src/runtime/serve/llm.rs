//! Chat / completion adapter for the hub's text-generation models: every
//! `runtime::Llm<M: CausalLm>`. They share one generation surface (`TokenStream`,
//! `GenerateOptions`); [`ChatLm`] names it once so a single [`LlmServed`] serves all of them
//! with a small pool of sessions: a request reuses the session whose consumed tokens are a
//! prefix of its prompt (the earlier turns of a conversation), so only the new tokens are
//! prefilled. Steering files given at load are applied per request (by name with
//! multipliers); a session is only reused under the steering its state was computed with.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use lapid::{Error, Result};
use serde_json::{Value, json};
use crate::runtime::causal_lm::{CausalLm, Llm};
use crate::runtime::generate::{FinishReason, GenerateOptions, TokenStream};
use crate::runtime::session::Session;
use crate::runtime::steer::{SteerFile, Steering};
use crate::ModelInfo;
use lapid_nn::SamplingParams;

use super::{Capability, ChatParams, Delta, GenParams, GenSummary, ModelMeta, AnyModel};
use super::text::{OutputFilter, ThinkSplitter};

/// The generation surface the adapter needs from a hub LLM.
pub trait ChatLm {
    type Session;
    fn new_session(&self) -> Self::Session;
    fn session_tokens(s: &Self::Session) -> &[u32];
    fn render(&self, messages: &str, tools: Option<&str>, enable_thinking: Option<bool>) -> Result<String>;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn generate<'a>(&'a self, s: &'a mut Self::Session, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>>;
    fn recommended(&self) -> SamplingParams;
    fn info(&self) -> ModelInfo;
    /// Reasoning block markers (open, close).
    fn think_markers(&self) -> (&'static str, &'static str) {
        ("<think>", "</think>")
    }
    /// Read a steering file and check that it fits the model.
    fn load_steering(&self, path: &Path) -> Result<SteerFile>;
    /// Steering of `(file, multiplier)`s (`None`: nothing applies).
    fn steering(&self, parts: &[(&SteerFile, f32)]) -> Result<Option<Arc<Steering>>>;
    /// Steer a session (resets it when the steering differs from its state's).
    fn set_steering(s: &mut Self::Session, steering: Option<Arc<Steering>>);
    /// Key of a session's steering.
    fn session_steering(s: &Self::Session) -> Option<&str>;
}

impl<M: CausalLm> ChatLm for Llm<M> {
    type Session = Session<M::State>;
    fn new_session(&self) -> Self::Session {
        Llm::new_session(self)
    }
    fn session_tokens(s: &Self::Session) -> &[u32] {
        s.tokens()
    }
    fn render(&self, messages: &str, tools: Option<&str>, enable_thinking: Option<bool>) -> Result<String> {
        self.render_chat_json(messages, tools, true, enable_thinking)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Llm::encode(self, text)
    }
    fn generate<'a>(&'a self, s: &'a mut Self::Session, prompt: &[u32], opts: GenerateOptions) -> Result<TokenStream<'a>> {
        Llm::generate(self, s, prompt, opts)
    }
    fn recommended(&self) -> SamplingParams {
        self.recommended_sampling()
    }
    fn info(&self) -> ModelInfo {
        crate::Model::info(self)
    }
    fn think_markers(&self) -> (&'static str, &'static str) {
        Llm::think_markers(self)
    }
    fn load_steering(&self, path: &Path) -> Result<SteerFile> {
        Llm::load_steering(self, path)
    }
    fn steering(&self, parts: &[(&SteerFile, f32)]) -> Result<Option<Arc<Steering>>> {
        Llm::steering(self, parts)
    }
    fn set_steering(s: &mut Self::Session, steering: Option<Arc<Steering>>) {
        s.set_steering(steering);
    }
    fn session_steering(s: &Self::Session) -> Option<&str> {
        s.steering().map(|st| st.key())
    }
}


struct Slot<S> {
    session: S,
    last_used: u64,
}

/// A hub LLM with a session pool.
pub struct LlmServed<M: ChatLm> {
    model: M,
    meta: ModelMeta,
    slots: Vec<Slot<M::Session>>,
    max_slots: usize,
    clock: u64,
    defaults: GenParams,
    /// Steering files (unique names) and the startup selection (name, multiplier).
    steer_files: Vec<SteerFile>,
    steer_default: Vec<(String, f32)>,
    /// Built steerings by selection.
    steer_built: HashMap<String, Option<Arc<Steering>>>,
}

impl<M: ChatLm> LlmServed<M> {
    pub fn new(model: M, meta: ModelMeta, opts: &super::LoadOptions) -> Result<LlmServed<M>> {
        let mut meta = meta;
        meta.capabilities = vec![Capability::Chat, Capability::Completion];
        let max_slots = opts.sessions.max(1);
        let (mut steer_files, mut steer_default) = (Vec::<SteerFile>::new(), Vec::new());
        let mut listed = Vec::new();
        for spec in &opts.steer {
            let f = model.load_steering(&spec.path)?;
            if steer_files.iter().any(|g| g.name == f.name) {
                return Err(Error::new("invalid_argument", format!("two steering files are named {:?}", f.name)));
            }
            let mut d = f.describe();
            d["default_scale"] = json!(spec.scale);
            listed.push(d);
            steer_default.push((f.name.clone(), spec.scale));
            steer_files.push(f);
        }
        if !listed.is_empty() {
            meta.extra.insert("steering".into(), Value::Array(listed));
        }
        Ok(LlmServed { model, meta, slots: Vec::new(), max_slots, clock: 0, defaults: opts.defaults.clone(), steer_files, steer_default, steer_built: HashMap::new() })
    }

    /// The steering of a request's selection (`None`: the startup one).
    fn steering(&mut self, sel: Option<&[(String, f32)]>) -> Result<Option<Arc<Steering>>> {
        let sel = sel.unwrap_or(&self.steer_default);
        let key = sel.iter().map(|(n, m)| format!("{n}*{m}")).collect::<Vec<_>>().join("+");
        if let Some(s) = self.steer_built.get(&key) {
            return Ok(s.clone());
        }
        let mut parts = Vec::with_capacity(sel.len());
        for (name, mult) in sel {
            let f = self.steer_files.iter().find(|f| &f.name == name).ok_or_else(|| {
                let have: Vec<&str> = self.steer_files.iter().map(|f| f.name.as_str()).collect();
                Error::new("invalid_argument", format!("unknown steering {name:?} (loaded: {have:?})"))
            })?;
            if !mult.is_finite() {
                return Err(Error::new("invalid_argument", format!("steering {name:?}: scale must be finite")));
            }
            parts.push((f, *mult));
        }
        let built = self.model.steering(&parts)?;
        if self.steer_built.len() >= 16 {
            self.steer_built.clear();
        }
        self.steer_built.insert(key, built.clone());
        Ok(built)
    }

    /// The session to run `prompt` under `steering` on: the longest cached prefix computed
    /// with the same steering, else a new or the least recently used one (its state is reset
    /// by the generation).
    fn pick_slot(&mut self, prompt: &[u32], steering: Option<Arc<Steering>>) -> usize {
        self.clock += 1;
        let key = steering.as_ref().map(|s| s.key());
        let best = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                let t = M::session_tokens(&s.session);
                M::session_steering(&s.session) == key && !t.is_empty() && t.len() < prompt.len() && prompt[..t.len()] == *t
            })
            .max_by_key(|(_, s)| M::session_tokens(&s.session).len())
            .map(|(i, _)| i);
        let i = match best {
            Some(i) => i,
            None if self.slots.len() < self.max_slots => {
                self.slots.push(Slot { session: self.model.new_session(), last_used: 0 });
                self.slots.len() - 1
            }
            None => self.slots.iter().enumerate().min_by_key(|(_, s)| s.last_used).map_or(0, |(i, _)| i),
        };
        self.slots[i].last_used = self.clock;
        M::set_steering(&mut self.slots[i].session, steering);
        i
    }

    fn sampling(&self, g: &GenParams) -> SamplingParams {
        let d = &self.defaults;
        let g = &GenParams {
            max_tokens: g.max_tokens.or(d.max_tokens),
            temperature: g.temperature.or(d.temperature),
            top_p: g.top_p.or(d.top_p),
            top_k: g.top_k.or(d.top_k),
            repetition_penalty: g.repetition_penalty.or(d.repetition_penalty),
            seed: g.seed.or(d.seed),
            stop: Vec::new(),
            steer: None,
        };
        let mut s = self.model.recommended();
        if let Some(t) = g.temperature {
            s.temperature = t.max(0.0);
        }
        if let Some(p) = g.top_p {
            s.top_p = p.clamp(0.0, 1.0);
        }
        if let Some(k) = g.top_k {
            s.top_k = k;
        }
        if let Some(r) = g.repetition_penalty {
            s.repetition_penalty = r;
        }
        s.seed = g.seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
        });
        s
    }

    fn run(&mut self, prompt_text: &str, g: &GenParams, think: ThinkSplitter, out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        let ids = self.model.encode(prompt_text)?;
        if ids.is_empty() {
            return Err(Error::new("invalid_argument", "the prompt is empty"));
        }
        let opts = GenerateOptions {
            max_new_tokens: g.max_tokens.or(self.defaults.max_tokens).unwrap_or(2048).max(1),
            sampling: self.sampling(g),
            stop_tokens: None,
            stop_strings: Vec::new(),
            cancel: None,
        };
        let steering = self.steering(g.steer.as_deref())?;
        let slot = self.pick_slot(&ids, steering);
        let mut filter = OutputFilter::new(think, &g.stop);
        let model = &self.model;
        let session = &mut self.slots[slot].session;
        let mut stream = model.generate(session, &ids, opts)?;
        let mut cancelled = false;
        'decode: for tok in stream.by_ref() {
            let tok = tok?;
            for d in filter.push(&tok.text) {
                if !out(d) {
                    cancelled = true;
                    break 'decode;
                }
            }
            if filter.stopped() {
                break;
            }
        }
        if !cancelled {
            for d in filter.finish() {
                if !out(d) {
                    cancelled = true;
                    break;
                }
            }
        }
        let stats = stream.stats();
        let finish = if cancelled {
            "cancelled"
        } else if filter.stopped() {
            "stop"
        } else {
            match stream.finish_reason() {
                Some(FinishReason::Length) => "length",
                Some(FinishReason::Cancelled) => "cancelled",
                _ => "stop",
            }
        };
        let generated = stream.generated().len();
        let stop_tok = matches!(stream.finish_reason(), Some(FinishReason::Stop));
        Ok(GenSummary {
            finish_reason: finish.to_string(),
            prompt_tokens: stats.prompt_tokens + stats.cached_tokens,
            cached_tokens: stats.cached_tokens,
            completion_tokens: if stop_tok { generated.saturating_sub(1) } else { generated },
            ttft_ms: stats.time_to_first_token_ms,
            decode_ms: stats.decode_ms,
        })
    }
}

impl<M: ChatLm> AnyModel for LlmServed<M> {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn chat(&mut self, req: &ChatParams, out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        let messages = serde_json::to_string(&req.messages)?;
        let tools = req.tools.as_ref().map(serde_json::to_string).transpose()?;
        let text = self.model.render(&messages, tools.as_deref(), req.enable_thinking)?;
        let (open, close) = self.model.think_markers();
        let prompt_open = text.trim_end().ends_with(open.trim_end());
        let think = ThinkSplitter::new(open, close, prompt_open);
        self.run(&text, &req.sampling, think, out)
    }

    fn complete(&mut self, prompt: &str, sampling: &GenParams, out: &mut dyn FnMut(Delta) -> bool) -> Result<GenSummary> {
        self.run(prompt, sampling, ThinkSplitter::passthrough(), out)
    }
}
