//! Activation steering files: small safetensors files that change a causal LM's behavior at
//! runtime by projecting directions out of (`ablate`) or adding vectors to (`add`) the
//! activations at named hook points of chosen layers. Spec: `docs/spec/steering.md`, usage:
//! `docs/STEERING.md`.
//!
//! ```text
//! file:     __metadata__ {format: "lapid-steer", version: "1", architecture, hidden_size,
//!                         name, description, source, license, scale?, scale.<tensor>?}
//!           <hook>.<layer>.<op>   f32 [hidden_size]     hook: attn_out | ffn_out | resid
//! ablate:   y <- y - s * d * (d . y)     (d normalized at load)
//! add:      y <- y + s * v
//! s = file scale (`scale.<tensor>`, else `scale`, else 1) x the user's multiplier
//! ```
//! A model declares the hooks it supports ([`CausalLm::steer_hooks`](super::CausalLm::steer_hooks))
//! and applies them through [`Taps::steer`](super::Taps::steer); with no steering the graph
//! is unchanged. A [`Steering`] (files x multipliers, uploaded to a device) has a key: a
//! session's cached state is only reused under the same key.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lapid::{DType, Device, Error, Result, Tensor, bail};
use lapid_convert::safetensors::SafeTensors;
use serde_json::{Value, json};

/// `__metadata__["format"]` of a steering file.
pub const FORMAT: &str = "lapid-steer";
/// Format version written and read.
pub const VERSION: u32 = 1;

/// Where in a block a steering vector applies (all act on the last axis, `hidden_size`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Hook {
    /// The attention (token mixer) output added to the residual stream, after any post-norm;
    /// with hyper-connections, the block output before it is expanded to the streams.
    AttnOut,
    /// The feed-forward / MoE output, likewise.
    FfnOut,
    /// The residual stream after the block (every stream with hyper-connections).
    Resid,
}

impl Hook {
    pub const ALL: [Hook; 3] = [Hook::AttnOut, Hook::FfnOut, Hook::Resid];

    pub fn name(self) -> &'static str {
        match self {
            Hook::AttnOut => "attn_out",
            Hook::FfnOut => "ffn_out",
            Hook::Resid => "resid",
        }
    }

    pub fn parse(s: &str) -> Option<Hook> {
        Hook::ALL.into_iter().find(|h| h.name() == s)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SteerOp {
    /// Project the (normalized) direction out, scaled: `y - s d (d . y)` (s = 1: remove it).
    Ablate,
    /// Add the vector, scaled: `y + s v`.
    Add,
}

impl SteerOp {
    pub fn name(self) -> &'static str {
        match self {
            SteerOp::Ablate => "ablate",
            SteerOp::Add => "add",
        }
    }

    fn parse(s: &str) -> Option<SteerOp> {
        [SteerOp::Ablate, SteerOp::Add].into_iter().find(|o| o.name() == s)
    }
}

/// One vector of a steering file.
#[derive(Clone, Debug)]
pub struct SteerEntry {
    pub hook: Hook,
    pub layer: usize,
    pub op: SteerOp,
    /// The file's scale of this entry.
    pub scale: f32,
    /// `hidden_size` values (unit norm for `ablate` once loaded).
    pub vector: Vec<f32>,
}

impl SteerEntry {
    pub fn tensor_name(&self) -> String {
        format!("{}.{}.{}", self.hook.name(), self.layer, self.op.name())
    }
}

/// A parsed steering file.
#[derive(Clone, Debug, Default)]
pub struct SteerFile {
    pub name: String,
    pub description: String,
    pub architecture: String,
    pub hidden_size: usize,
    pub source: String,
    pub license: String,
    /// In application order (by hook, layer, then file order of the op).
    pub entries: Vec<SteerEntry>,
}

fn steer_err(path: &Path, msg: impl std::fmt::Display) -> Error {
    Error::new("steering", format!("{}: {msg}", path.display()))
}

impl SteerFile {
    /// Read and validate a steering file (format, version, shapes, finite non-zero vectors).
    pub fn open(path: &Path) -> Result<SteerFile> {
        let st = SafeTensors::open(path)?;
        let m = st.metadata();
        let get = |k: &str| m.get(k).cloned().unwrap_or_default();
        if get("format") != FORMAT {
            return Err(steer_err(path, format!("not a steering file (metadata format is {:?}, want {FORMAT:?})", get("format"))));
        }
        let version: u32 = get("version").parse().map_err(|_| steer_err(path, "bad metadata version"))?;
        if version > VERSION {
            return Err(steer_err(path, format!("format version {version} is newer than this lapid ({VERSION})")));
        }
        let hidden_size: usize = get("hidden_size").parse().map_err(|_| steer_err(path, "metadata hidden_size missing or not a number"))?;
        let parse_scale = |k: &str, v: &str| v.parse::<f32>().ok().filter(|s| s.is_finite()).ok_or_else(|| steer_err(path, format!("metadata {k}: bad scale {v:?}")));
        let scale = match m.get("scale") {
            Some(v) => parse_scale("scale", v)?,
            None => 1.0,
        };
        let mut entries = Vec::new();
        for name in st.names() {
            let parts: Vec<&str> = name.split('.').collect();
            let (hook, layer, op) = match parts.as_slice() {
                [h, l, o] => (Hook::parse(h), l.parse::<usize>().ok(), SteerOp::parse(o)),
                _ => (None, None, None),
            };
            let (Some(hook), Some(layer), Some(op)) = (hook, layer, op) else {
                return Err(steer_err(path, format!("tensor {name:?} is not <attn_out|ffn_out|resid>.<layer>.<ablate|add>")));
            };
            if st.shape(&name)? != [hidden_size] {
                return Err(steer_err(path, format!("{name}: shape {:?}, want [{hidden_size}]", st.shape(&name)?)));
            }
            let mut vector = st.to_f32(&name)?;
            if vector.iter().any(|v| !v.is_finite()) {
                return Err(steer_err(path, format!("{name}: non-finite values")));
            }
            if op == SteerOp::Ablate {
                let norm = vector.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                if norm < 1e-12 {
                    return Err(steer_err(path, format!("{name}: zero direction")));
                }
                vector.iter_mut().for_each(|v| *v = (f64::from(*v) / norm) as f32);
            }
            let scale = match m.get(&format!("scale.{name}")) {
                Some(v) => parse_scale(&format!("scale.{name}"), v)?,
                None => scale,
            };
            entries.push(SteerEntry { hook, layer, op, scale, vector });
        }
        if entries.is_empty() {
            return Err(steer_err(path, "no steering vectors"));
        }
        entries.sort_by_key(|e| (e.hook, e.layer, e.op == SteerOp::Add));
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(SteerFile {
            name: m.get("name").filter(|s| !s.is_empty()).cloned().unwrap_or(stem),
            description: get("description"),
            architecture: get("architecture"),
            hidden_size,
            source: get("source"),
            license: get("license"),
            entries,
        })
    }

    /// Write as a steering file (the file scale is stored per entry).
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut meta = BTreeMap::from([
            ("format".to_string(), FORMAT.to_string()),
            ("version".to_string(), VERSION.to_string()),
            ("architecture".to_string(), self.architecture.clone()),
            ("hidden_size".to_string(), self.hidden_size.to_string()),
            ("name".to_string(), self.name.clone()),
            ("description".to_string(), self.description.clone()),
            ("source".to_string(), self.source.clone()),
            ("license".to_string(), self.license.clone()),
        ]);
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for e in &self.entries {
            if e.vector.len() != self.hidden_size {
                bail!("steering", "{}: {} values, want {}", e.tensor_name(), e.vector.len(), self.hidden_size);
            }
            let begin = data.len();
            data.extend(e.vector.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(e.tensor_name(), json!({"dtype": "F32", "shape": [self.hidden_size], "data_offsets": [begin, data.len()]}));
            meta.insert(format!("scale.{}", e.tensor_name()), e.scale.to_string());
        }
        header.insert("__metadata__".into(), json!(meta));
        let mut h = serde_json::to_vec(&Value::Object(header))?;
        h.resize(h.len().next_multiple_of(8), b' ');
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        std::fs::write(path, out).map_err(|e| steer_err(path, e))
    }

    /// Check that the file fits a model (`arch`, hidden size, layer count, supported hooks).
    pub fn check(&self, arch: &str, hidden: usize, layers: usize, hooks: &[Hook]) -> Result<()> {
        let err = |msg: String| Error::new("steering", format!("steering {:?}: {msg}", self.name));
        if self.architecture != arch {
            return Err(err(format!("made for architecture {:?}, the model is {arch:?}", self.architecture)));
        }
        if self.hidden_size != hidden {
            return Err(err(format!("hidden size {}, the model's is {hidden}", self.hidden_size)));
        }
        for e in &self.entries {
            if e.layer >= layers {
                return Err(err(format!("{}: layer {} out of range (the model has {layers})", e.tensor_name(), e.layer)));
            }
            if !hooks.contains(&e.hook) {
                let have: Vec<&str> = hooks.iter().map(|h| h.name()).collect();
                return Err(Error::new("unsupported", format!("steering {:?}: hook {} is not supported by {arch} (supported: {have:?})", self.name, e.hook.name())));
            }
        }
        Ok(())
    }

    /// Summary as JSON (`lapid steer`, `/v1/models`).
    pub fn describe(&self) -> Value {
        let mut groups: BTreeMap<(&str, &str, String), Vec<usize>> = BTreeMap::new();
        for e in &self.entries {
            groups.entry((e.hook.name(), e.op.name(), e.scale.to_string())).or_default().push(e.layer);
        }
        let entries: Vec<Value> = groups.into_iter().map(|((hook, op, scale), layers)| json!({"hook": hook, "op": op, "scale": scale.parse::<f64>().unwrap_or(1.0), "layers": ranges(&layers)})).collect();
        json!({
            "name": self.name,
            "description": self.description,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "source": self.source,
            "license": self.license,
            "entries": entries,
        })
    }
}

/// `[2, 3, 4, 7]` -> `"2-4,7"`.
fn ranges(v: &[usize]) -> String {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j { v[i].to_string() } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(",")
}

/// `lapid steer FILE`: the file's metadata and entries.
pub fn info(path: &Path) -> Result<Value> {
    Ok(SteerFile::open(path)?.describe())
}

/// A steering file given on the command line: `path[:multiplier]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SteerSpec {
    pub path: PathBuf,
    /// Multiplies the file's scales (1 = as authored, 0 = off).
    pub scale: f32,
}

impl SteerSpec {
    pub fn parse(s: &str) -> Result<SteerSpec> {
        if let Some((p, sc)) = s.rsplit_once(':')
            && !Path::new(s).exists()
        {
            let scale: f32 = sc.parse().ok().filter(|x: &f32| x.is_finite()).ok_or_else(|| Error::new("invalid_argument", format!("--steer {s:?}: bad scale {sc:?} (want FILE[:SCALE])")))?;
            return Ok(SteerSpec { path: PathBuf::from(p), scale });
        }
        Ok(SteerSpec { path: PathBuf::from(s), scale: 1.0 })
    }
}

/// One vector ready to apply: the direction (`ablate`) or nothing, and the scaled vector.
struct Applied {
    dir: Option<Tensor>,
    scaled: Tensor,
}

/// Steering files with multipliers, uploaded to a device: what a forward applies.
pub struct Steering {
    key: String,
    by: HashMap<(Hook, usize), Vec<Applied>>,
}

impl std::fmt::Debug for Steering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Steering({})", self.key)
    }
}

impl Steering {
    /// Combine `(file, multiplier)`s (applied in that order); `None` when nothing is left
    /// (no files, or every multiplier 0). Files must be checked against the model first.
    pub fn new(device: &Device, parts: &[(&SteerFile, f32)]) -> Result<Option<Arc<Steering>>> {
        let mut by: HashMap<(Hook, usize), Vec<Applied>> = HashMap::new();
        let mut key = Vec::new();
        for (f, mult) in parts.iter().filter(|(_, m)| *m != 0.0) {
            key.push(format!("{}*{}", f.name, mult));
            for e in &f.entries {
                let s = e.scale * mult;
                let up = |v: &[f32]| device.upload(&Tensor::from_f32(v, &[v.len()])?);
                let scaled: Vec<f32> = e.vector.iter().map(|v| v * s).collect();
                let dir = match e.op {
                    SteerOp::Ablate => Some(up(&e.vector)?),
                    SteerOp::Add => None,
                };
                by.entry((e.hook, e.layer)).or_default().push(Applied { dir, scaled: up(&scaled)? });
            }
        }
        if by.is_empty() {
            return Ok(None);
        }
        Ok(Some(Arc::new(Steering { key: key.join("+"), by })))
    }

    /// Identity of this combination (files and multipliers): sessions key their state on it.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Whether anything applies at `hook` of `layer`.
    pub fn has(&self, hook: Hook, layer: usize) -> bool {
        self.by.contains_key(&(hook, layer))
    }

    /// `x` (any shape, last axis `hidden_size`) steered at `hook` of `layer` (computed in
    /// f32, returned in `x`'s dtype); `x` itself when nothing applies.
    pub fn apply(&self, hook: Hook, layer: usize, x: &Tensor) -> Result<Tensor> {
        let Some(list) = self.by.get(&(hook, layer)) else { return Ok(x.clone()) };
        let dt = x.dtype();
        let mut y = if dt == DType::F32 { x.clone() } else { x.cast(DType::F32)? };
        for a in list {
            y = match &a.dir {
                Some(d) => y.sub(&y.mul(d)?.sum(-1, true)?.mul(&a.scaled)?)?,
                None => y.add(&a.scaled)?,
            };
        }
        if dt == DType::F32 { Ok(y) } else { y.cast(dt) }
    }
}

#[cfg(test)]
#[path = "steer_tests.rs"]
mod tests;
