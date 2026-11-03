//! Command-line arguments of `lapid serve` and `lapid run`, accepting llama.cpp's flags as
//! well (`-m`, `-hf repo[:QUANT]`, `-p`, `-n`, `-c`, `--temp`, `--top-k`, `--top-p`,
//! `--repeat-penalty`, `-s`, `-sys`, `--host`, `--port`, `-ngl`, `-t`, `-cnv`), so lapid
//! drops in where `llama-cli` / `llama-server` are used. Multi-letter single-dash flags are
//! rewritten to their long forms by [`normalize_args`] before clap parses them.

use std::path::PathBuf;

use clap::Args;
use lapid::{Error, Result};

use crate::load::LoadOptions;
use lapid_hub::runtime::serve::GenParams;
use lapid_hub::runtime::steer::SteerSpec;

/// llama.cpp's multi-letter single-dash flags and their long forms.
const LLAMA_SHORT: &[(&str, &str)] = &[
    ("-hf", "--hf-repo"),
    ("-hfr", "--hf-repo"),
    ("-ngl", "--n-gpu-layers"),
    ("-sys", "--system-prompt"),
    ("-cnv", "--conversation"),
    ("-no-cnv", "--no-conversation"),
    ("-fa", "--flash-attn"),
    ("-np", "--parallel"),
];

/// Rewrite llama.cpp-style multi-letter short flags (`-hf`, `-ngl`, ...) to long flags.
pub fn normalize_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
    args.into_iter()
        .map(|a| match LLAMA_SHORT.iter().find(|(s, _)| *s == a) {
            Some((_, long)) => long.to_string(),
            None => a,
        })
        .collect()
}

/// Sampling flags (llama.cpp names, OpenAI-style aliases).
#[derive(Args, Clone, Debug, Default)]
pub struct SamplingArgs {
    /// Maximum new tokens (default 2048).
    #[arg(short = 'n', long = "n-predict", visible_alias = "max-tokens")]
    pub n_predict: Option<i64>,
    /// Temperature (default: the model's recommendation).
    #[arg(long = "temp", visible_alias = "temperature")]
    pub temp: Option<f32>,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub repeat_penalty: Option<f32>,
    #[arg(short = 's', long)]
    pub seed: Option<i64>,
}

impl SamplingArgs {
    pub fn params(&self) -> GenParams {
        GenParams {
            // llama.cpp: -1 = unlimited.
            max_tokens: self.n_predict.and_then(|n| usize::try_from(n).ok()).map(|n| n.max(1)),
            temperature: self.temp,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repeat_penalty,
            seed: self.seed.and_then(|s| u64::try_from(s).ok()),
            stop: Vec::new(),
            steer: None,
        }
    }
}

/// Model loading flags shared by `serve` and `run`.
#[derive(Args, Clone, Debug, Default)]
pub struct LoadArgs {
    /// Weight dtype when converting: fp32, fp16, int8, int4, q4_k, q5_k, q6_k or q4_k_m
    /// (default: int8 for safetensors, the GGUF's own quant family for GGUF files).
    #[arg(long)]
    pub dtype: Option<String>,
    /// cpu, gpu, metal[:N], cuda[:N] or vulkan[:N] (default: best available, or $LAPID_DEVICE).
    #[arg(long)]
    pub device: Option<String>,
    /// Chat sessions kept per model for prompt-prefix reuse.
    #[arg(long, default_value_t = 4)]
    pub sessions: usize,
    /// Hugging Face repo `owner/repo[:QUANT]` (llama.cpp `-hf`): a GGUF repo's file for the
    /// quant tag (default Q4_K_M), or a safetensors repo.
    #[arg(long = "hf-repo")]
    pub hf_repo: Vec<String>,
    /// Context size (llama.cpp `-c`): accepted; lapid sizes the KV cache on demand.
    #[arg(short = 'c', long = "ctx-size")]
    pub ctx_size: Option<usize>,
    /// llama.cpp `-ngl`: accepted and ignored (lapid runs the whole model on the device).
    #[arg(long = "n-gpu-layers", visible_alias = "gpu-layers")]
    pub n_gpu_layers: Option<String>,
    /// CPU threads (sets LAPID_THREADS).
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// llama.cpp `-fa`: accepted and ignored (attention is always fused).
    #[arg(long = "flash-attn", num_args = 0..=1, default_missing_value = "on")]
    pub flash_attn: Option<String>,
    /// llama-server `--jinja`: accepted (chat templates are always rendered).
    #[arg(long)]
    pub jinja: bool,
    /// Read weights in place from the mmap'd model file (streamed from disk when larger
    /// than memory) instead of copying them to the device. Default: only when the model
    /// does not fit in available memory (or $LAPID_MMAP=0|1|auto).
    #[arg(long, overrides_with = "no_mmap")]
    pub mmap: bool,
    /// Always copy weights to the device (the default when the model fits).
    #[arg(long = "no-mmap", overrides_with = "mmap")]
    pub no_mmap: bool,
    /// Device memory (GiB) for MoE experts that do not fit in memory, streamed from the
    /// model file on demand (default: what is available minus a safety headroom;
    /// $LAPID_EXPERT_CACHE_GB).
    #[arg(long = "expert-cache", value_name = "GIB")]
    pub expert_cache: Option<f64>,
    /// Activation steering file (`lapid steer FILE` describes one), applied to chat models:
    /// FILE or FILE:SCALE (multiplies the file's scales; 0 = loaded but off). Repeatable.
    /// Server requests choose with `"steer": {"name": scale}`.
    #[arg(long = "steer", value_name = "FILE[:SCALE]")]
    pub steer: Vec<String>,
    /// Speculative decoding (llama.cpp `--spec-type`): `draft-dspark` uses the model's
    /// built-in DSpark draft head (DeepSeek V4.1). Off by default.
    #[arg(long = "spec-type", value_name = "TYPE")]
    pub spec_type: Option<String>,
    /// `--draft dspark`: same as `--spec-type draft-dspark`.
    #[arg(long = "draft", value_name = "KIND")]
    pub draft: Option<String>,
    /// Draft tokens per step at most (clamped to the draft's block size).
    #[arg(long = "spec-draft-n-max", visible_alias = "draft-max", value_name = "N")]
    pub spec_draft_n_max: Option<usize>,
    /// Cut the draft at the first position whose confidence is below P (0 = off).
    #[arg(long = "spec-draft-p-min", visible_alias = "draft-p-min", value_name = "P")]
    pub spec_draft_p_min: Option<f32>,
    /// Draft head weights (`lapid convert --dspark`) when the model file has none.
    #[arg(long = "spec-draft-model", visible_alias = "model-draft", value_name = "FILE")]
    pub spec_draft_model: Option<std::path::PathBuf>,
}

impl LoadArgs {
    pub fn options(&self, sampling: &SamplingArgs) -> Result<LoadOptions> {
        let steer = self.steer.iter().map(|s| SteerSpec::parse(s)).collect::<Result<_>>()?;
        Ok(LoadOptions { dtype: self.dtype.clone(), sessions: self.sessions, defaults: sampling.params(), steer })
    }

    /// Speculative decoding settings of the flags (`None`: off).
    pub fn spec(&self) -> Result<Option<lapid_hub::runtime::spec::SpecConfig>> {
        let kind = match (self.spec_type.as_deref(), self.draft.as_deref()) {
            (Some("draft-dspark") | None, Some("dspark")) | (Some("draft-dspark"), None) => true,
            (None | Some("none"), None) => false,
            (t, d) => return Err(Error::new("invalid_argument", format!("unsupported speculative decoding {:?} (only --spec-type draft-dspark / --draft dspark)", t.or(d).unwrap_or_default()))),
        };
        if !kind {
            return Ok(None);
        }
        let mut c = lapid_hub::runtime::spec::SpecConfig::default();
        if let Some(n) = self.spec_draft_n_max {
            c.n_max = n;
        }
        if let Some(p) = self.spec_draft_p_min {
            c.p_min = p;
        }
        c.draft_model = self.spec_draft_model.clone();
        Ok(Some(c))
    }

    /// Apply process-wide settings (threads, weight residency) and print notes on ignored flags.
    pub fn apply(&self) {
        if self.mmap {
            lapid::set_residency(lapid::Residency::Mmap);
        } else if self.no_mmap {
            lapid::set_residency(lapid::Residency::Upload);
        }
        if let Some(g) = self.expert_cache {
            lapid_nn::llm::expert_cache::set_budget_gib(g);
        }
        if let Some(t) = self.threads {
            // SAFETY: called at startup, before any other thread is spawned.
            unsafe { std::env::set_var("LAPID_THREADS", t.to_string()) };
        }
        if self.n_gpu_layers.is_some() {
            eprintln!("lapid: -ngl is ignored (the whole model runs on the selected device; use --device cpu for CPU)");
        }
        if self.ctx_size.is_some() {
            eprintln!("lapid: -c is accepted for compatibility; the KV cache grows on demand");
        }
    }

    /// Model sources from positional args, `-m` and `-hf` (downloads `-hf` GGUFs).
    pub fn sources(&self, positional: &[String], m: &[String]) -> Result<Vec<String>> {
        let mut out: Vec<String> = positional.iter().chain(m).cloned().collect();
        for spec in &self.hf_repo {
            out.push(crate::hf::resolve_hf(spec)?);
        }
        Ok(out)
    }
}

/// `lapid serve` (also llama-server's flags).
#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    /// Models: HF repo, local HF directory, .gguf or .lapid file; `id=source` names one.
    pub models: Vec<String>,
    /// Model file (llama.cpp `-m`; repeatable).
    #[arg(short = 'm', long = "model")]
    pub model: Vec<String>,
    /// Model id clients use for the first model (llama-server `--alias`).
    #[arg(short = 'a', long)]
    pub alias: Option<String>,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// llama-server `-np`: accepted; requests to one model are queued.
    #[arg(long)]
    pub parallel: Option<usize>,
    #[command(flatten)]
    pub load: LoadArgs,
    #[command(flatten)]
    pub sampling: SamplingArgs,
}

/// `lapid run` (also llama-cli's flags).
#[derive(Args, Clone, Debug)]
pub struct RunArgs {
    /// Model (HF repo, local HF directory, .gguf or .lapid file) followed by inputs:
    /// a prompt (one-shot chat), WAV files (transcription), or text (speech, embeddings).
    pub args: Vec<String>,
    /// Model file (llama.cpp `-m`).
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,
    /// Prompt: answer it and exit (with --conversation: the system prompt of an interactive chat).
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,
    /// System prompt.
    #[arg(long = "system-prompt", visible_alias = "system")]
    pub system_prompt: Option<String>,
    /// Interactive chat even when a prompt is given.
    #[arg(long)]
    pub conversation: bool,
    /// llama-cli `-no-cnv`: accepted (one-shot is the default with a prompt).
    #[arg(long)]
    pub no_conversation: bool,
    /// Turn the model's thinking on or off (default: the chat template's).
    #[arg(long)]
    pub think: Option<bool>,
    /// Transcription output: text, json, srt or vtt.
    #[arg(long)]
    pub format: Option<String>,
    /// Transcription language code (default: detect).
    #[arg(long)]
    pub language: Option<String>,
    /// Speech: output WAV file (default speech.wav).
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
    /// Speech: voice name.
    #[arg(long)]
    pub voice: Option<String>,
    #[command(flatten)]
    pub load: LoadArgs,
    #[command(flatten)]
    pub sampling: SamplingArgs,
}

pub fn serve(a: ServeArgs) -> Result<()> {
    a.load.apply();
    lapid_hub::runtime::spec::set_spec(a.load.spec()?);
    let mut models = a.load.sources(&a.models, &a.model)?;
    if let (Some(alias), Some(first)) = (&a.alias, models.first_mut()) {
        *first = format!("{alias}={first}");
    }
    crate::serve(crate::ServeOptions {
        models,
        host: a.host,
        port: a.port,
        device: a.load.device.clone(),
        load: a.load.options(&a.sampling)?,
    })
}

pub fn run(a: RunArgs) -> Result<()> {
    a.load.apply();
    lapid_hub::runtime::spec::set_spec(a.load.spec()?);
    let mut positional = a.args.clone();
    let explicit = a.model.iter().cloned().collect::<Vec<_>>();
    let model = if !explicit.is_empty() || !a.load.hf_repo.is_empty() {
        a.load.sources(&[], &explicit)?.into_iter().next()
    } else if positional.is_empty() {
        None
    } else {
        Some(positional.remove(0))
    };
    let model = model.ok_or_else(|| Error::new("invalid_argument", "give a model (positional, -m FILE or -hf owner/repo[:QUANT])"))?;
    // llama-cli: with -cnv the prompt is the system prompt of an interactive chat.
    let (system, interactive) = if a.conversation {
        (a.system_prompt.clone().or(a.prompt.clone()), true)
    } else {
        (a.system_prompt.clone(), false)
    };
    let mut inputs = positional;
    if let (Some(p), false) = (&a.prompt, a.conversation) {
        inputs.insert(0, p.clone());
    }
    crate::run(crate::RunOptions {
        model,
        inputs,
        interactive,
        device: a.load.device.clone(),
        load: a.load.options(&a.sampling)?,
        system,
        sampling: a.sampling.params(),
        enable_thinking: a.think,
        format: a.format,
        output: a.output,
        voice: a.voice,
        language: a.language,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_llama_flags() {
        let a = normalize_args(["lapid", "run", "-hf", "a/b:Q8_0", "-ngl", "99", "-p", "hi"].map(String::from));
        assert_eq!(a, ["lapid", "run", "--hf-repo", "a/b:Q8_0", "--n-gpu-layers", "99", "-p", "hi"]);
    }
}
