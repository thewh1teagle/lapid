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

use lapid_tensor::{Error, Result, bail};
use serde_json::{Value, json};

use crate::OutDType;
use crate::assets::{HfFiles, TextAssets};
use crate::ggml::GgmlType;
use crate::gguf::{GgufFile, GgufTensor, GgufValue, is_gguf, write_ggml_tensor};
use crate::mapping::{Renamed, RoleRule, Rule, rename, role_of};
use crate::policy::{Role, Target};
use crate::safetensors::SafeTensors;
use crate::transforms::gather_rows;
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

    /// Every tensor (our names) the output must contain.
    fn expected(&self, cfg: &Value) -> Result<Vec<String>>;

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

enum Src {
    Hf(usize, String),
    Gguf(usize),
}

struct Sources<'a> {
    kind: SourceKind,
    shards: &'a [SafeTensors],
    gguf: Option<&'a GgufFile>,
    /// Companion GGUF: `Src::Gguf(i)` with `i >= gguf.tensors.len()` indexes its tensors.
    companion: Option<&'a GgufFile>,
    map: BTreeMap<String, Src>,
    cfg: &'a Value,
    hooks: &'static dyn ConvertHooks,
}

impl Sources<'_> {
    fn raw_shape(&self, src: &Src) -> Result<Vec<usize>> {
        match src {
            Src::Hf(i, n) => self.shards[*i].shape(n),
            Src::Gguf(i) => Ok(self.ggml(*i)?.1.shape.clone()),
        }
    }

    /// File and tensor of GGUF source index `i` (model tensors, then companion tensors).
    fn ggml(&self, i: usize) -> Result<(&GgufFile, &GgufTensor)> {
        let g = self.gguf.ok_or_else(|| Error::new("internal", "gguf source without a file"))?;
        if let Some(t) = g.tensors.get(i) {
            return Ok((g, t));
        }
        let c = self.companion.ok_or_else(|| Error::new("internal", "gguf source index out of range"))?;
        c.tensors.get(i - g.tensors.len()).map(|t| (c, t)).ok_or_else(|| Error::new("internal", "gguf source index out of range"))
    }
}

impl TensorSource for Sources<'_> {
    fn kind(&self) -> SourceKind {
        self.kind
    }

    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let s = self.raw_shape(self.map.get(name)?).ok()?;
        Some(self.hooks.reshape(name, &s, self.cfg, self.kind).unwrap_or(s))
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let src = self.map.get(name).ok_or_else(|| Error::new("missing_tensor", format!("no tensor {name:?}")))?;
        let data = match src {
            Src::Hf(i, n) => self.shards[*i].to_f32(n)?,
            Src::Gguf(i) => {
                let (g, t) = self.ggml(*i)?;
                g.to_f32(t)?
            }
        };
        let shape = self.raw_shape(src)?;
        match self.hooks.row_order(name, &shape, self.cfg, self.kind) {
            Some(order) => gather_rows(&data, &order),
            None => Ok(data),
        }
    }
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
        let mut map = BTreeMap::new();
        for (i, st) in shards.iter().enumerate() {
            for n in st.names() {
                let full = format!("{}{n}", paths[i].0);
                match rename(self.hf_names, &full) {
                    Renamed::Drop => {}
                    Renamed::To(o) => {
                        map.insert(o, Src::Hf(i, n));
                    }
                    Renamed::NoMatch => {
                        map.insert(full, Src::Hf(i, n));
                    }
                }
            }
        }
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

    fn write_rows(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, parts: &[(String, Vec<usize>, Vec<usize>)]) -> Result<()> {
        let src = srcs.map.get(name).ok_or_else(|| Error::new("internal", format!("no source for {name:?}")))?;
        let raw = srcs.raw_shape(src)?;
        let n_rows = if raw.len() <= 1 { raw.first().copied().unwrap_or(1) } else { raw[..raw.len() - 1].iter().product() };
        fn pick<T: Copy>(data: &[T], n_rows: usize, rows: &[usize], name: &str) -> Result<Vec<T>> {
            let row = data.len() / n_rows.max(1);
            if let Some(&r) = rows.iter().find(|&&r| r >= n_rows) {
                bail!("shape_mismatch", "{name}: row {r} of {n_rows}");
            }
            Ok(rows.iter().flat_map(|&r| &data[r * row..(r + 1) * row]).copied().collect())
        }
        for (out, shape, rows) in parts {
            match src {
                Src::Gguf(i) => {
                    let (g, gt) = srcs.ggml(*i)?;
                    let mut t = gt.clone();
                    t.shape = shape.clone();
                    self.write_ggml(w, out, &t, &pick(g.tensor_data(gt), n_rows, rows, name)?, shape)?;
                }
                Src::Hf(i, n) => w.add_f32(out, shape, &pick(&srcs.shards[*i].to_f32(n)?, n_rows, rows, name)?)?,
            }
        }
        Ok(())
    }

    /// Stored bytes of source `name` as blob `out`.
    fn write_blob(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, out: &str, media_type: &str) -> Result<()> {
        let src = srcs.map.get(name).ok_or_else(|| Error::new("internal", format!("no source for {name:?}")))?;
        let bytes = match src {
            Src::Hf(i, n) => srcs.shards[*i].bytes(n)?,
            Src::Gguf(i) => {
                let (g, t) = srcs.ggml(*i)?;
                g.tensor_data(t)
            }
        };
        w.add_blob(out, media_type, bytes)
    }

    /// Source tensors `parts` stacked along a new first axis as `out`.
    fn write_stack(&self, w: &mut ModelWriter, srcs: &Sources<'_>, out: &str, parts: &[String]) -> Result<()> {
        let mut inner: Option<Vec<usize>> = None;
        for p in parts {
            let s = srcs.shape(p).ok_or_else(|| Error::new("missing_tensor", format!("{out}: stack part {p:?} missing")))?;
            if inner.as_ref().is_some_and(|i| *i != s) {
                bail!("shape_mismatch", "{out}: stack parts disagree in shape ({inner:?} vs {s:?})");
            }
            inner = Some(s);
        }
        let Some(inner) = inner else { bail!("invalid_argument", "{out}: empty stack") };
        let shape = [vec![parts.len()], inner].concat();
        // GGUF parts of one ggml type: concatenated blocks, written as one ggml tensor.
        let ggml: Option<Vec<(&GgufFile, &GgufTensor)>> =
            parts.iter().map(|p| match srcs.map.get(p) { Some(Src::Gguf(i)) => srcs.ggml(*i).ok(), _ => None }).collect();
        if let Some(gs) = ggml
            && gs.iter().all(|(_, t)| t.ggml_type == gs[0].1.ggml_type)
        {
            let mut data = Vec::new();
            for (p, (g, t)) in parts.iter().zip(&gs) {
                match self.hooks.row_order(p, &t.shape, srcs.cfg, srcs.kind) {
                    Some(o) => data.extend(gather_rows(g.tensor_data(t), &o)?),
                    None => data.extend_from_slice(g.tensor_data(t)),
                }
            }
            let mut t = gs[0].1.clone();
            t.shape = shape.clone();
            t.nbytes = data.len() as u64;
            return self.write_ggml(w, out, &t, &data, &shape);
        }
        let mut data = Vec::with_capacity(shape.iter().product());
        for p in parts {
            data.extend(srcs.f32(p)?);
        }
        w.add_f32(out, &shape, &data)
    }

    /// Raw ggml data of `t` as `name` (see [`ConvertHooks::gguf_keep_float`]).
    fn write_ggml(&self, w: &mut ModelWriter, name: &str, t: &GgufTensor, data: &[u8], shape: &[usize]) -> Result<()> {
        let float = matches!(t.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
        if !(float && self.hooks.gguf_keep_float(name) && matches!(w.target(name, shape), Target::Quant(_))) {
            return write_ggml_tensor(w, name, t, data, Some(shape));
        }
        if t.ggml_type == GgmlType::F16 {
            return w.add_raw(name, lapid_tensor::DType::F16, shape, data);
        }
        let v = crate::ggml::dequantize(t.ggml_type, data, shape.iter().product())?;
        w.add_raw(name, lapid_tensor::DType::F32, shape, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
    }

    /// Source tensor `name` written as `out`.
    fn write_one(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, out: &str) -> Result<()> {
        let src = srcs.map.get(name).ok_or_else(|| Error::new("internal", format!("no source for {name:?}")))?;
        let raw = srcs.raw_shape(src)?;
        let shape = self.hooks.reshape(name, &raw, srcs.cfg, srcs.kind).unwrap_or_else(|| raw.clone());
        let order = self.hooks.row_order(name, &raw, srcs.cfg, srcs.kind);
        match src {
            Src::Gguf(i) => {
                let (g, t) = srcs.ggml(*i)?;
                let data = g.tensor_data(t);
                match order {
                    Some(o) => write_ggml_tensor(w, out, t, &gather_rows(data, &o)?, Some(&shape)),
                    None => self.write_ggml(w, out, t, data, &shape),
                }
            }
            _ => w.add_f32(out, &shape, &srcs.f32(name)?),
        }
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
