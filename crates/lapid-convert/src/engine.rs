//! The generic converter: Hugging Face safetensors (single or sharded) and GGUF checkpoints
//! become one `.lapid` file, driven by a model's declarative [`ConvertSpec`] (name rules,
//! role rules) plus small [`ConvertHooks`] for what a table cannot say (config parsing,
//! expected tensors, reshapes, row orders, value transforms).
//!
//! Output layout: our tensor names; metadata `name`, `architecture`, `dtype`, `license`,
//! `source`, the model's config object under its architecture key (plus text-asset keys
//! for language models) and a single-component declaration; text assets as two blobs.
//! Precision follows `crate::policy` through the role rules. GGUF tensors keep their bytes
//! whenever the policy allows it (Q8_0 / Q4_0 repacked to int8 / int4, K-quants copied).
//!
//! ```ignore
//! pub static CONVERT: ConvertSpec = ConvertSpec {
//!     architecture: "mymodel", license: "apache-2.0",
//!     hf_model_types: &["mymodel"], gguf_architectures: &["mymodel"],
//!     hf_names: &[Rule::drop("lm_head.weight")],
//!     gguf_names: &[Rule::new("blk.{i}.attn_q.{w}", "model.layers.{i}.self_attn.q_proj.{w}")],
//!     roles: &[RoleRule::new("model.embed_tokens.weight", Role::LinearInt8)],
//!     text_assets: true,
//!     hooks: &MyHooks,
//! };
//! CONVERT.convert("owner/repo", OutDType::Int8, Path::new("out.lapid"))?;
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lapid_tensor::{Result, bail};
use serde_json::{Value, json};

use crate::OutDType;
use crate::assets::{HfFiles, TextAssets};
use crate::gguf::{GgufFile, GgufValue, is_gguf};
use crate::mapping::{Renamed, RoleRule, Rule, rename, role_of};
use crate::policy::Role;
use crate::safetensors::SafeTensors;
use crate::source::{Sources, Src, hf_map};
use crate::writer::{ModelWriter, RoleArc, file_hash, metadata, read_json};

/// Where a tensor comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Hf,
    Gguf,
}

/// What to write for one tensor (see [`ConvertHooks::transform`]).
pub enum Emit {
    /// Write it as stored (raw passthrough when possible).
    Keep,
    /// Skip it.
    Drop,
    /// Write these f32 tensors (name, shape, values) instead.
    Replace(Vec<(String, Vec<usize>, Vec<f32>)>),
    /// Write parts made of whole stored rows (last axis; 1-D: elements) of this tensor:
    /// (name, shape, source row indices). Raw bytes, so ggml blocks are kept: splits fused
    /// tensors without requantizing. Rows index the stored layout (before `row_order`).
    Rows(Vec<(String, Vec<usize>, Vec<usize>)>),
    /// Write it as stored under another name (when the name depends on the config).
    Rename(String),
    /// Write the stored bytes of this source tensor unchanged as the blob `name` (streamed
    /// from the source mapping, never copied whole into memory).
    Blob { name: String, media_type: &'static str },
    /// Write tensor `name` shaped `[parts.len(), ..part shape]` from the listed source
    /// tensors (our names, all the same shape) back to back. Kept as stored when they share
    /// a quantized storage the output can hold (MXFP4 from official FP4 or GGUF type 39,
    /// ggml K / legacy quants the writer keeps), else via f32 and the dtype policy.
    Stack { name: String, parts: Vec<String> },
}

/// Read access to the source tensors by our names (for transforms that combine tensors,
/// e.g. BatchNorm folding).
pub trait TensorSource {
    fn kind(&self) -> SourceKind;
    fn names(&self) -> Vec<String>;
    /// Shape after [`ConvertHooks::reshape`].
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// Values (dequantized) in the [`shape`](Self::shape) layout.
    fn f32(&self, name: &str) -> Result<Vec<f32>>;
    /// Stored form: the safetensors dtype (`"BF16"`, ...), `"F8_E4M3+E8M0"` / `"F4_E2M1+E8M0"`
    /// for block-scaled weights paired with their scales, the ggml type name for GGUF.
    fn stored_dtype(&self, _name: &str) -> Option<String> {
        None
    }
}

/// Inputs of an HF checkpoint beyond `config.json` and the weights (see
/// [`ConvertHooks::hf_extras`]).
#[derive(Default)]
pub struct HfExtras {
    /// More safetensors files as (name prefix, path): tensor `n` is named `prefix + n`
    /// before the `hf_names` rules (e.g. sentence-transformers Dense modules, voice files).
    pub weights: Vec<(String, PathBuf)>,
    /// More blobs: (name, media type, bytes).
    pub blobs: Vec<(String, String, Vec<u8>)>,
    /// Top-level metadata keys set after [`ConvertHooks::metadata`] (license, components, ...).
    pub metadata: serde_json::Map<String, Value>,
}

/// Model-specific conversion logic beyond the name / role tables. Only `expected` is
/// required; `cfg` is the model config object as stored in the metadata.
pub trait ConvertHooks: Sync {
    /// Config object from HF `config.json` (default: stored as is). Validate here.
    fn hf_config(&self, config: &Value) -> Result<Value> {
        Ok(config.clone())
    }

    /// Other files of an HF checkpoint (module lists, extra weights, tokenizers without a chat
    /// template, model cards): may complete `cfg` (the [`hf_config`](Self::hf_config) result).
    fn hf_extras(&self, _files: &HfFiles, _cfg: &mut Value) -> Result<HfExtras> {
        Ok(HfExtras::default())
    }

    /// HF-style config object from GGUF metadata (required to convert GGUF).
    fn gguf_config(&self, f: &GgufFile) -> Result<Value> {
        bail!("unsupported_model", "no GGUF converter for architecture {:?}", f.architecture())
    }

    /// Blobs and metadata of a GGUF source (e.g. a tokenizer rebuilt from its vocabulary); may
    /// complete `cfg` (the [`gguf_config`](Self::gguf_config) result). `weights` is unused.
    fn gguf_extras(&self, _f: &GgufFile, _cfg: &mut Value) -> Result<HfExtras> {
        Ok(HfExtras::default())
    }

    /// A second GGUF converted into the same file, e.g. the vision `mmproj` of a llama.cpp
    /// model (see [`sibling_mmproj`]). Its tensors are renamed by
    /// [`companion_names`](Self::companion_names); unmatched ones are dropped.
    fn gguf_companion(&self, _model: &Path) -> Result<Option<PathBuf>> {
        Ok(None)
    }

    /// Name rules of the companion's tensors.
    fn companion_names(&self) -> &'static [Rule] {
        &[]
    }

    /// Complete `cfg` from the companion's metadata (vision config, processor, token ids).
    fn companion_config(&self, _companion: &GgufFile, _model: &GgufFile, _cfg: &mut Value) -> Result<()> {
        Ok(())
    }

    /// Every tensor (our names) the output must contain (blob names may be listed too).
    fn expected(&self, cfg: &Value) -> Result<Vec<String>>;

    /// Whether the HF weight `hf_name` (source name, before `hf_names`) that has an
    /// `F8_E8M0` `.scale` companion stays unpaired: both are then plain source tensors (e.g.
    /// to store them unchanged with [`Emit::Blob`]). Default: paired (see `microscale`).
    fn hf_unpaired(&self, _hf_name: &str) -> bool {
        false
    }

    /// New shape with the same element order (e.g. `[D, K]` -> `[D, 1, K]`).
    fn reshape(&self, _name: &str, _shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        None
    }

    /// Source row of every output row (applied to raw rows: quantized blocks are kept), e.g.
    /// `transforms::llama_qk_unpermute` for GGUF q/k projections.
    fn row_order(&self, _name: &str, _shape: &[usize], _cfg: &Value, _src: SourceKind) -> Option<Vec<usize>> {
        None
    }

    /// Whether a GGUF tensor stored as a float (f32 / f16 / bf16) stays a float (bf16 as
    /// f32) where the dtype policy would quantize it: precision-sensitive weights a quantized
    /// GGUF deliberately left unquantized.
    fn gguf_keep_float(&self, _name: &str) -> bool {
        false
    }

    /// Value transform of one tensor (dequantizes it): fold, split, combine.
    fn transform(&self, _name: &str, _src: &dyn TensorSource, _cfg: &Value) -> Result<Emit> {
        Ok(Emit::Keep)
    }

    /// Final edit of the whole metadata object.
    fn metadata(&self, _meta: &mut Value) {}
}

/// A model's declarative conversion description.
pub struct ConvertSpec {
    /// Metadata `architecture` and config object key.
    pub architecture: &'static str,
    /// SPDX-like license id recorded in the output.
    pub license: &'static str,
    /// HF `config.json` `model_type` values this spec converts.
    pub hf_model_types: &'static [&'static str],
    /// GGUF `general.architecture` values this spec converts.
    pub gguf_architectures: &'static [&'static str],
    /// HF name -> our name (unmatched names are kept as is).
    pub hf_names: &'static [Rule],
    /// GGUF name -> our name (unmatched names are an error).
    pub gguf_names: &'static [Rule],
    /// Our name -> role (unmatched: `mapping::default_role`).
    pub roles: &'static [RoleRule],
    /// Store tokenizer, chat template, special tokens and generation defaults.
    pub text_assets: bool,
    pub hooks: &'static dyn ConvertHooks,
}

impl ConvertSpec {
    /// Dtype-policy role of one of our tensors.
    pub fn role(&self, name: &str, shape: &[usize]) -> Role {
        role_of(self.roles, name, shape)
    }

    /// The role table as a shared classifier (writers, calibrated conversion).
    pub fn role_arc(&'static self) -> RoleArc {
        Arc::new(move |n: &str, s: &[usize]| role_of(self.roles, n, s))
    }

    /// Whether this spec converts an HF config with `model_type`.
    pub fn accepts_model_type(&self, model_type: &str) -> bool {
        self.hf_model_types.contains(&model_type)
    }

    pub fn accepts_gguf(&self, f: &GgufFile) -> bool {
        f.architecture().is_some_and(|a| self.gguf_architectures.contains(&a))
    }

    /// Convert `src` (GGUF file, HF repo id or HF directory) to `out` at `dtype`.
    pub fn convert(&'static self, src: &str, dtype: OutDType, out: &Path) -> Result<()> {
        let p = Path::new(src);
        if p.is_file() && is_gguf(p)? {
            return self.convert_gguf(p, dtype, out);
        }
        self.convert_hf(src, dtype, out)
    }

    pub fn convert_hf(&'static self, src: &str, dtype: OutDType, out: &Path) -> Result<()> {
        let files = HfFiles::locate(src)?;
        let raw = read_json(&files.config)?;
        let mt = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !self.accepts_model_type(mt) {
            bail!("unsupported_model", "{}: config.json model_type {mt:?} is not {:?}", files.repo, self.architecture);
        }
        let mut cfg = self.hooks.hf_config(&raw)?;
        let extras = self.hooks.hf_extras(&files, &mut cfg)?;
        let assets = if self.text_assets { Some(TextAssets::from_hf(&files)?) } else { None };
        let mut paths: Vec<(String, &Path)> = files.weights.iter().map(|p| (String::new(), p.as_path())).collect();
        paths.extend(extras.weights.iter().map(|(prefix, p)| (prefix.clone(), p.as_path())));
        let shards: Vec<SafeTensors> = paths.iter().map(|(_, p)| SafeTensors::open(p)).collect::<Result<_>>()?;
        let prefixes: Vec<String> = paths.iter().map(|(p, _)| p.clone()).collect();
        let map = hf_map(&shards, &prefixes, self.hf_names, self.hooks)?;
        let mut source = json!({ "repo": files.repo });
        if files.remote {
            source["revision"] = json!("main");
        }
        if files.weights.len() == 1 {
            source["file_hash"] = json!(file_hash(&files.weights[0])?);
        }
        let srcs = Sources { kind: SourceKind::Hf, shards: &shards, gguf: None, companion: None, map, cfg: &cfg, hooks: self.hooks };
        self.write(&srcs, &files.name(self.architecture), source, assets.as_ref(), &extras, dtype, out)
    }

    pub fn convert_gguf(&'static self, path: &Path, dtype: OutDType, out: &Path) -> Result<()> {
        let f = GgufFile::open(path)?;
        if !self.accepts_gguf(&f) {
            bail!("unsupported_model", "{}: GGUF architecture {:?} is not {:?}", path.display(), f.architecture(), self.architecture);
        }
        let mut cfg = self.hooks.gguf_config(&f)?;
        let extras = self.hooks.gguf_extras(&f, &mut cfg)?;
        let assets = if self.text_assets { Some(TextAssets::from_gguf(&f)?) } else { None };
        let mut map = BTreeMap::new();
        for (i, t) in f.tensors.iter().enumerate() {
            match rename(self.gguf_names, &t.name) {
                Renamed::Drop => {}
                Renamed::To(o) => {
                    map.insert(o, Src::Gguf(i));
                }
                Renamed::NoMatch => bail!("unsupported_model", "GGUF: unknown {} tensor {:?}", self.architecture, t.name),
            }
        }
        let companion_path = self.hooks.gguf_companion(path)?;
        let companion = companion_path.as_deref().map(GgufFile::open).transpose()?;
        if let Some(c) = &companion {
            self.hooks.companion_config(c, &f, &mut cfg)?;
            for (j, t) in c.tensors.iter().enumerate() {
                if let Renamed::To(o) = rename(self.hooks.companion_names(), &t.name) {
                    map.insert(o, Src::Gguf(f.tensors.len() + j));
                }
            }
        }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or(self.architecture).to_string();
        let mut source = json!({
            "repo": f.get("general.name").and_then(GgufValue::as_str).map_or_else(|| name.clone(), |n| format!("gguf:{n}")),
            "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            "format": "gguf",
            "file_hash": file_hash(path)?,
        });
        if let Some(p) = &companion_path {
            source["companion"] = json!(p.file_name().and_then(|s| s.to_str()).unwrap_or(""));
        }
        let srcs = Sources { kind: SourceKind::Gguf, shards: &[], gguf: Some(&f), companion: companion.as_ref(), map, cfg: &cfg, hooks: self.hooks };
        self.write(&srcs, &name, source, assets.as_ref(), &extras, dtype, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn write(&'static self, srcs: &Sources<'_>, name: &str, source: Value, assets: Option<&TextAssets>, extras: &HfExtras, dtype: OutDType, out: &Path) -> Result<()> {
        let arch = self.architecture;
        let expected = self.hooks.expected(srcs.cfg)?;
        let mut cfg = srcs.cfg.clone();
        if let Some(a) = assets {
            a.annotate(&mut cfg);
        }
        let mut meta = metadata(name, arch, dtype, self.license, source);
        meta[arch] = cfg;
        meta["components"] = json!({ arch: {"architecture": arch, "config": arch, "license": self.license, "tensor_prefix": ""} });
        self.hooks.metadata(&mut meta);
        for (k, v) in &extras.metadata {
            meta[k.as_str()] = v.clone();
        }

        let expected_set: BTreeSet<&String> = expected.iter().collect();
        let mut order: Vec<String> = expected.iter().filter(|n| srcs.map.contains_key(*n)).cloned().collect();
        order.extend(srcs.map.keys().filter(|k| !expected_set.contains(k)).cloned());
        let mut w = ModelWriter::create_shared(out, meta, dtype, self.role_arc(), order.len())?;
        if let Some(a) = assets {
            a.write_blobs(&mut w)?;
        }
        for (blob, media, bytes) in &extras.blobs {
            w.add_blob(blob, media, bytes)?;
        }
        for n in &order {
            match self.hooks.transform(n, srcs, srcs.cfg)? {
                Emit::Drop => {}
                Emit::Replace(ts) => {
                    for (tn, shape, data) in ts {
                        w.add_f32(&tn, &shape, &data)?;
                    }
                }
                Emit::Rows(parts) => self.write_rows(&mut w, srcs, n, &parts)?,
                Emit::Keep => self.write_one(&mut w, srcs, n, n)?,
                Emit::Rename(to) => self.write_one(&mut w, srcs, n, &to)?,
                Emit::Blob { name, media_type } => self.write_blob(&mut w, srcs, n, &name, media_type)?,
                Emit::Stack { name, parts } => self.write_stack(&mut w, srcs, &name, &parts)?,
            }
        }
        w.finish(&expected)
    }
}

/// The vision projector paired with GGUF `model`: `$LAPID_MMPROJ` (a path; `none` = no
/// projector), else a `mmproj*.gguf` in the same directory (a float one when several exist).
pub fn sibling_mmproj(model: &Path) -> Option<PathBuf> {
    match std::env::var("LAPID_MMPROJ") {
        Ok(v) if v == "none" => return None,
        Ok(v) if !v.is_empty() => return Some(PathBuf::from(v)),
        _ => {}
    }
    let dir = model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            n.starts_with("mmproj") && n.ends_with(".gguf") && p.as_path() != model
        })
        .collect();
    found.sort();
    let float = found.iter().find(|p| p.to_string_lossy().contains("F16")).cloned();
    float.or_else(|| found.into_iter().next())
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
