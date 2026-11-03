//! lapid-convert: generic building blocks for converting models to `.lapid`.
//!
//! Hugging Face Hub downloads (cached), a safetensors reader, a GGUF reader with ggml
//! dequantizers, the legacy ggml container (`ggml_legacy`), the output dtype policy and
//! quantized encoding, and a streaming, policy-applying writer, and the declarative
//! converter engine (`engine`, `mapping`, `transforms`, `assets`): a model describes its
//! tensor names and roles as tables plus small hooks, and the engine does the rest.
//! Model-specific tables and hooks live with their model in `lapid-hub`.

use std::path::PathBuf;

use lapid_tensor::{Result, bail};

pub mod assets;
pub mod calibrate;
mod emit;
pub mod engine;
pub mod ggml;
pub mod ggml_legacy;
mod gptq;
pub mod gguf;
mod hf;
pub mod mapping;
pub mod microscale;
pub mod parallel;
pub mod policy;
pub mod safetensors;
mod source;
pub mod transforms;
pub mod writer;

pub use hf::{download_hf, download_hf_optional};

/// Output weight precision. CLI names "fp32", "fp16", "int8", "int4", "q4_k", "q5_k",
/// "q6_k" and "q4_k_m" (llama.cpp's Q4_K_M mix; see `policy`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutDType {
    F32,
    F16,
    Int8,
    Int4,
    Q4K,
    Q5K,
    Q6K,
    Q4KM,
}

impl OutDType {
    pub fn parse(s: &str) -> Result<OutDType> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "fp32" | "f32" | "float32" => OutDType::F32,
            "fp16" | "f16" | "float16" => OutDType::F16,
            "int8" | "i8" | "q8" => OutDType::Int8,
            "int4" | "i4" | "q4" => OutDType::Int4,
            "q4_k" | "q4k" | "q4_k_s" => OutDType::Q4K,
            "q5_k" | "q5k" | "q5_k_s" => OutDType::Q5K,
            "q6_k" | "q6k" => OutDType::Q6K,
            "q4_k_m" | "q4km" => OutDType::Q4KM,
            _ => bail!(
                "unsupported_dtype",
                "unknown dtype {s:?} (expected fp32, fp16, int8, int4, q4_k, q5_k, q6_k or q4_k_m)"
            ),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            OutDType::F32 => "fp32",
            OutDType::F16 => "fp16",
            OutDType::Int8 => "int8",
            OutDType::Int4 => "int4",
            OutDType::Q4K => "q4_k",
            OutDType::Q5K => "q5_k",
            OutDType::Q6K => "q6_k",
            OutDType::Q4KM => "q4_k_m",
        }
    }

    pub const ALL: [OutDType; 8] = [
        OutDType::F32,
        OutDType::F16,
        OutDType::Int8,
        OutDType::Int4,
        OutDType::Q4K,
        OutDType::Q5K,
        OutDType::Q6K,
        OutDType::Q4KM,
    ];
}

/// `$LAPID_CACHE`, else `$XDG_CACHE_HOME/lapid`, else `~/.cache/lapid`.
pub fn cache_dir() -> PathBuf {
    let nonempty = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    if let Some(p) = nonempty("LAPID_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = nonempty("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("lapid");
    }
    if let Some(h) = nonempty("HOME").or_else(|| nonempty("USERPROFILE")) {
        return PathBuf::from(h).join(".cache").join("lapid");
    }
    std::env::temp_dir().join("lapid-cache")
}

/// `cache_dir()/models/{repo with / → --}-{dtype}.lapid`.
pub fn default_model_path(repo: &str, dtype: OutDType) -> PathBuf {
    let stem: String = repo.trim_matches('/').replace('/', "--");
    cache_dir().join("models").join(format!("{stem}-{}.lapid", dtype.name()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_names() {
        for d in OutDType::ALL {
            assert_eq!(OutDType::parse(d.name()).ok(), Some(d));
        }
        assert_eq!(OutDType::parse("INT8").ok(), Some(OutDType::Int8));
        assert_eq!(OutDType::parse("Q4_K_M").ok(), Some(OutDType::Q4KM));
        assert_eq!(OutDType::parse("q5").err().map(|e| e.code), Some("unsupported_dtype"));
    }

    #[test]
    fn model_path() {
        let p = default_model_path("openai/whisper-small", OutDType::Int8);
        assert!(p.ends_with("models/openai--whisper-small-int8.lapid"));
    }
}
