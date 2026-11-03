//! The shared model framework: everything a hub model needs besides its own math.
//! See `docs/MODELS.md` for how to add a model.
//!
//! * Causal LMs: [`causal_lm::CausalLm`] (the model's trait) and [`causal_lm::Llm`] (the
//!   pipeline: loading, prefill / decode, sessions, chat, streaming), with [`chat`],
//!   [`session`], [`generate`] and the [`jinja`] chat-template interpreter.
//! * Verification and benchmarks: [`lm_verify`], [`lm_bench`] (driven by a
//!   `lapid_verify::Descriptor`), calibrated int4: [`lm_calibrate`].
//! * Serving: [`serve::AnyModel`] (one object-safe surface for `lapid serve` / `lapid run`),
//!   [`serve::load`].
//! * Other pipelines: [`transducer`] (RNN-T / TDT), [`seq2seq`] (Whisper-style),
//!   [`embedding`] (pooling, normalize, Matryoshka), [`tts`] (streaming speech),
//!   [`vlm`] (image + text LMs on top of [`causal_lm`], verified by [`vlm_verify`]).
//! * Routing: [`registry`] (convert / verify / bench / serve by model), [`resolve`] (sources
//!   to cached `.lapid` files).
//! * Conversion: `lapid_convert::engine` (declarative `ConvertSpec`).

pub mod audio;
pub mod audio_bench;
pub mod audio_verify;
pub mod causal_lm;
pub mod embedding;
pub mod embedding_harness;
pub mod endpoint;
pub mod chat;
pub mod generate;
pub mod jinja;
pub mod lm_bench;
pub mod lm_calibrate;
pub mod lm_verify;
pub mod registry;
pub mod resolve;
pub mod seq2seq;
pub mod serve;
pub mod session;
pub mod spec;
pub mod steer;
pub mod transducer;
pub mod tts;
pub mod tts_harness;
pub mod vlm;
pub mod vlm_verify;

pub use causal_lm::{CausalLm, LmDims, Llm, Rows, Taps};
pub use chat::{ChatOptions, ChatTemplate, Message, TextAssets};
pub use generate::{CancelToken, FinishReason, GenerateOptions, GenerationStats, Token, TokenStream};
pub use registry::{LmDef, ModelEntry};
pub use serve::{AnyModel, LoadOptions};
pub use session::{LmCache, LmState, Session};
pub use steer::{Hook, SteerFile, SteerSpec, Steering};
pub use vlm::{Media, MediaPrompt, VisionLm};
