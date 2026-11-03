//! The engine's view of the source tensors by our names: HF safetensors shards (with
//! block-scaled FP8 / FP4 weights paired with their `F8_E8M0` scales, see `microscale`) or a
//! GGUF file plus its companion.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use lapid_tensor::{Error, Result, bail};
use serde_json::Value;

use crate::engine::{ConvertHooks, SourceKind, TensorSource};
use crate::gguf::{GgufFile, GgufTensor};
use crate::mapping::{Renamed, Rule, rename};
use crate::microscale::{self, Micro};
use crate::safetensors::SafeTensors;
use crate::transforms::gather_rows;

/// A block-scale companion: shard, stored name, format.
pub(crate) struct Scale {
    pub shard: usize,
    pub name: String,
    pub micro: Micro,
}

pub(crate) enum Src {
    /// Shard, stored name, block scales (`X.weight` + `X.scale` pairs).
    Hf(usize, String, Option<Scale>),
    Gguf(usize),
}

pub(crate) struct Sources<'a> {
    pub kind: SourceKind,
    pub shards: &'a [SafeTensors],
    pub gguf: Option<&'a GgufFile>,
    /// Companion GGUF: `Src::Gguf(i)` with `i >= gguf.tensors.len()` indexes its tensors.
    pub companion: Option<&'a GgufFile>,
    pub map: BTreeMap<String, Src>,
    pub cfg: &'a Value,
    pub hooks: &'static dyn ConvertHooks,
}

/// Our names of HF shards (`prefixes[i]` prepended to shard `i`'s names): `X.scale`
/// (`F8_E8M0`) next to an `X.weight` of a block-scaled dtype is folded into it (unless
/// [`ConvertHooks::hf_unpaired`]), then the `rules` rename.
pub(crate) fn hf_map(shards: &[SafeTensors], prefixes: &[String], rules: &[Rule], hooks: &dyn ConvertHooks) -> Result<BTreeMap<String, Src>> {
    let mut all: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (i, st) in shards.iter().enumerate() {
        for n in st.names() {
            all.insert(format!("{}{n}", prefixes[i]), (i, n));
        }
    }
    let mut scales: BTreeMap<String, Scale> = BTreeMap::new();
    for (full, (i, n)) in &all {
        let Some(base) = full.strip_suffix(".scale") else { continue };
        let weight = format!("{base}.weight");
        let Some((wi, wn)) = all.get(&weight) else { continue };
        if shards[*i].dtype(n)? != "F8_E8M0" || hooks.hf_unpaired(&weight) {
            continue;
        }
        let wdt = shards[*wi].dtype(wn)?;
        let Some(micro) = Micro::of(&wdt) else {
            bail!("unsupported_dtype", "{weight}: {wdt} weight with F8_E8M0 block scales");
        };
        micro
            .logical_shape(&shards[*wi].shape(wn)?, &shards[*i].shape(n)?)
            .map_err(|e| Error::new(e.code, format!("{weight}: {}", e.message)))?;
        scales.insert(weight, Scale { shard: *i, name: n.clone(), micro });
    }
    let consumed: BTreeSet<String> = scales.keys().map(|w| format!("{}.scale", &w[..w.len() - ".weight".len()])).collect();
    let mut map = BTreeMap::new();
    for (full, (i, n)) in all {
        if consumed.contains(&full) {
            continue;
        }
        let src = Src::Hf(i, n, scales.remove(&full));
        match rename(rules, &full) {
            Renamed::Drop => {}
            Renamed::To(o) => {
                map.insert(o, src);
            }
            Renamed::NoMatch => {
                map.insert(full, src);
            }
        }
    }
    Ok(map)
}

impl Sources<'_> {
    pub fn src(&self, name: &str) -> Result<&Src> {
        self.map.get(name).ok_or_else(|| Error::new("missing_tensor", format!("no source tensor {name:?}")))
    }

    /// Logical stored shape (before `reshape`; FP4 counts unpacked elements).
    pub fn raw_shape(&self, src: &Src) -> Result<Vec<usize>> {
        match src {
            Src::Hf(i, n, None) => self.shards[*i].shape(n),
            Src::Hf(i, n, Some(s)) => s.micro.logical_shape(&self.shards[*i].shape(n)?, &self.shards[s.shard].shape(&s.name)?),
            Src::Gguf(i) => Ok(self.ggml(*i)?.1.shape.clone()),
        }
    }

    /// File and tensor of GGUF source index `i` (model tensors, then companion tensors).
    pub fn ggml(&self, i: usize) -> Result<(&GgufFile, &GgufTensor)> {
        let g = self.gguf.ok_or_else(|| Error::new("internal", "gguf source without a file"))?;
        if let Some(t) = g.tensors.get(i) {
            return Ok((g, t));
        }
        let c = self.companion.ok_or_else(|| Error::new("internal", "gguf source index out of range"))?;
        c.tensors.get(i - g.tensors.len()).map(|t| (c, t)).ok_or_else(|| Error::new("internal", "gguf source index out of range"))
    }

    /// Values in the stored row order (before `row_order`).
    pub fn stored_f32(&self, src: &Src) -> Result<Vec<f32>> {
        match src {
            Src::Hf(i, n, None) => self.shards[*i].to_f32(n),
            Src::Hf(i, n, Some(s)) => {
                let (st, ss) = (&self.shards[*i], &self.shards[s.shard]);
                let (w, sc) = (st.bytes(n)?, ss.bytes(&s.name)?);
                let (wsh, ssh) = (st.shape(n)?, ss.shape(&s.name)?);
                match s.micro {
                    Micro::Fp8 => microscale::fp8_to_f32(w, sc, [wsh[0], wsh[1]], [ssh[0], ssh[1]]),
                    Micro::Fp4 => Ok(microscale::mxfp4_to_f32(&microscale::fp4_to_mxfp4(w, sc, wsh[0], wsh[1] * 2)?)),
                }
            }
            Src::Gguf(i) => {
                let (g, t) = self.ggml(*i)?;
                g.to_f32(t)
            }
        }
    }

    /// Unpadded MXFP4 blocks `[rows, cols]` of an official FP4 tensor in output row order
    /// (`None`: not FP4).
    pub fn mxfp4(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let src = self.src(name)?;
        let Src::Hf(i, n, Some(s)) = src else { return Ok(None) };
        if s.micro != Micro::Fp4 {
            return Ok(None);
        }
        let shape = self.raw_shape(src)?;
        let (st, ss) = (&self.shards[*i], &self.shards[s.shard]);
        let blocks = microscale::fp4_to_mxfp4(st.bytes(n)?, ss.bytes(&s.name)?, shape[0], shape[1])?;
        Ok(Some(match self.hooks.row_order(name, &shape, self.cfg, self.kind) {
            Some(order) => {
                let row = blocks.len() / shape[0].max(1);
                if order.len() != shape[0] {
                    bail!("shape_mismatch", "{name}: row order of {} rows for {shape:?}", order.len());
                }
                let rows: Vec<&[u8]> = blocks.chunks_exact(row).collect();
                gather_rows(&rows, &order)?.concat()
            }
            None => blocks,
        }))
    }

    /// The stored bytes of source `name` as a shared image + range (for blobs).
    pub fn shared(&self, name: &str) -> Result<(lapid_format::Bytes, Range<usize>)> {
        match self.src(name)? {
            Src::Hf(i, n, None) => self.shards[*i].shared(n),
            Src::Hf(_, n, Some(_)) => bail!(
                "invalid_argument",
                "{name}: Blob of a block-scaled weight ({n} + scale); keep it unpaired with ConvertHooks::hf_unpaired"
            ),
            Src::Gguf(i) => {
                let (g, t) = self.ggml(*i)?;
                Ok(g.shared(t))
            }
        }
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
        let src = self.src(name)?;
        let data = self.stored_f32(src)?;
        let shape = self.raw_shape(src)?;
        match self.hooks.row_order(name, &shape, self.cfg, self.kind) {
            Some(order) => gather_rows(&data, &order),
            None => Ok(data),
        }
    }

    fn stored_dtype(&self, name: &str) -> Option<String> {
        match self.map.get(name)? {
            Src::Hf(_, _, Some(s)) => Some(s.micro.name().to_string()),
            Src::Hf(i, n, None) => self.shards[*i].dtype(n).ok(),
            Src::Gguf(i) => self.ggml(*i).ok().map(|(_, t)| t.ggml_type.name().to_string()),
        }
    }
}
