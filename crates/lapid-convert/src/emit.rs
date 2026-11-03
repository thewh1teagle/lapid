//! Writing one [`Emit`](crate::engine::Emit) of the engine: stored tensors (raw passthrough
//! where possible), row splits, stacks and blobs. Memory: one source tensor (or one stack
//! part's values) at a time, blobs streamed from the source mapping.

use lapid_quant::Format;
use lapid_tensor::{DType, Error, Result, bail};

use crate::engine::{ConvertSpec, TensorSource};
use crate::ggml::GgmlType;
use crate::gguf::{GgufFile, GgufTensor, write_ggml_tensor};
use crate::policy::{Target, encode, quantizable_as};
use crate::source::{Sources, Src};
use crate::transforms::gather_rows;
use crate::writer::ModelWriter;

/// Unpadded MXFP4 blocks of `shape` (2-D, or 3-D: matrices back to back) in the stored
/// layout: each matrix padded to `layout(..).total`.
fn mxfp4_stored(blocks: &[u8], shape: &[usize]) -> Result<Vec<u8>> {
    if !quantizable_as(Format::Mxfp4, shape) {
        bail!("shape_mismatch", "MXFP4 tensor must be 2-D (or a 3-D stack) with last dim % 32 == 0, got {shape:?}");
    }
    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let per = rows * cols / 32 * 17;
    let n: usize = shape[..shape.len() - 2].iter().product();
    if blocks.len() != per * n {
        bail!("shape_mismatch", "MXFP4: {} bytes for {shape:?}", blocks.len());
    }
    let total = lapid_quant::layout(Format::Mxfp4, rows, cols).total;
    let mut out = vec![0u8; total * n];
    for (m, o) in blocks.chunks_exact(per).zip(out.chunks_exact_mut(total)) {
        o[..per].copy_from_slice(m);
    }
    Ok(out)
}

impl ConvertSpec {
    pub(crate) fn write_rows(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, parts: &[(String, Vec<usize>, Vec<usize>)]) -> Result<()> {
        let src = srcs.src(name)?;
        let raw = srcs.raw_shape(src)?;
        let n_rows = if raw.len() <= 1 { raw.first().copied().unwrap_or(1) } else { raw[..raw.len() - 1].iter().product() };
        fn pick<T: Copy>(data: &[T], n_rows: usize, rows: &[usize], name: &str) -> Result<Vec<T>> {
            let row = data.len() / n_rows.max(1);
            if let Some(&r) = rows.iter().find(|&&r| r >= n_rows) {
                bail!("shape_mismatch", "{name}: row {r} of {n_rows}");
            }
            Ok(rows.iter().flat_map(|&r| &data[r * row..(r + 1) * row]).copied().collect())
        }
        let values = match src {
            Src::Hf(..) => Some(srcs.stored_f32(src)?),
            Src::Gguf(_) => None,
        };
        for (out, shape, rows) in parts {
            match (src, &values) {
                (Src::Gguf(i), _) => {
                    let (g, gt) = srcs.ggml(*i)?;
                    let mut t = gt.clone();
                    t.shape = shape.clone();
                    self.write_ggml(w, out, &t, &pick(g.tensor_data(gt), n_rows, rows, name)?, shape)?;
                }
                (_, Some(v)) => w.add_f32(out, shape, &pick(v, n_rows, rows, name)?)?,
                _ => bail!("internal", "{name}: no values"),
            }
        }
        Ok(())
    }

    /// Raw ggml data of `t` as `name` (see [`ConvertHooks::gguf_keep_float`](crate::engine::ConvertHooks::gguf_keep_float)).
    pub(crate) fn write_ggml(&self, w: &mut ModelWriter, name: &str, t: &GgufTensor, data: &[u8], shape: &[usize]) -> Result<()> {
        let float = matches!(t.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
        if !(float && self.hooks.gguf_keep_float(name) && matches!(w.target(name, shape), Target::Quant(_))) {
            return write_ggml_tensor(w, name, t, data, Some(shape));
        }
        if t.ggml_type == GgmlType::F16 {
            return w.add_raw(name, DType::F16, shape, data);
        }
        let v = crate::ggml::dequantize(t.ggml_type, data, shape.iter().product())?;
        w.add_owned(name, DType::F32, shape, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
    }

    /// Source tensor `name` written as `out`.
    pub(crate) fn write_one(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, out: &str) -> Result<()> {
        let src = srcs.src(name)?;
        let raw = srcs.raw_shape(src)?;
        let shape = self.hooks.reshape(name, &raw, srcs.cfg, srcs.kind).unwrap_or_else(|| raw.clone());
        if let Some(blocks) = srcs.mxfp4(name)? {
            if shape.last() != raw.last() || shape.iter().product::<usize>() != raw.iter().product::<usize>() {
                bail!("shape_mismatch", "{name}: FP4 {raw:?} cannot be reshaped to {shape:?} (rows only)");
            }
            return w.add_owned(out, DType::Mxfp4, &shape, mxfp4_stored(&blocks, &shape)?);
        }
        match src {
            Src::Gguf(i) => {
                let (g, t) = srcs.ggml(*i)?;
                let data = g.tensor_data(t);
                match self.hooks.row_order(name, &raw, srcs.cfg, srcs.kind) {
                    Some(o) => write_ggml_tensor(w, out, t, &gather_rows(data, &o)?, Some(&shape)),
                    None => self.write_ggml(w, out, t, data, &shape),
                }
            }
            _ => w.add_f32(out, &shape, &srcs.f32(name)?),
        }
    }

    /// Stored bytes of source `name` as blob `out` (streamed from the source mapping).
    pub(crate) fn write_blob(&self, w: &mut ModelWriter, srcs: &Sources<'_>, name: &str, out: &str, media_type: &str) -> Result<()> {
        let (data, range) = srcs.shared(name)?;
        w.add_blob_shared(out, media_type, data, range)
    }

    /// Source tensors `parts` stacked along a new first axis as `out`.
    pub(crate) fn write_stack(&self, w: &mut ModelWriter, srcs: &Sources<'_>, out: &str, parts: &[String]) -> Result<()> {
        let mut inner: Option<Vec<usize>> = None;
        for p in parts {
            let s = srcs.shape(p).ok_or_else(|| Error::new("missing_tensor", format!("{out}: stack part {p:?} missing")))?;
            if inner.as_ref().is_some_and(|i| *i != s) {
                bail!("shape_mismatch", "{out}: stack parts disagree in shape ({inner:?} vs {s:?})");
            }
            inner = Some(s);
        }
        let Some(inner) = inner else { bail!("invalid_argument", "{out}: empty stack") };
        let shape = [vec![parts.len()], inner.clone()].concat();
        // Official FP4 parts: MXFP4 blocks, kept.
        if parts.iter().all(|p| srcs.stored_dtype(p).as_deref() == Some(crate::microscale::Micro::Fp4.name())) {
            let per = lapid_quant::layout(Format::Mxfp4, inner[0], *inner.last().unwrap_or(&0)).total;
            if inner.len() != 2 {
                bail!("shape_mismatch", "{out}: FP4 stack parts must be 2-D, got {inner:?}");
            }
            let mut bytes = vec![0u8; per * parts.len()];
            let errors = std::sync::Mutex::new(Vec::new());
            crate::parallel::fill_rows(&mut bytes, per, 1, |r, o| {
                for (p, o) in parts[r].iter().zip(o.chunks_exact_mut(per)) {
                    let res = srcs
                        .mxfp4(p)
                        .and_then(|b| b.ok_or_else(|| Error::new("internal", format!("{p}: not FP4"))))
                        .and_then(|b| mxfp4_stored(&b, &inner))
                        .map(|b| o.copy_from_slice(&b));
                    if let Err(e) = res {
                        errors.lock().unwrap_or_else(|e| e.into_inner()).push(e);
                    }
                }
            });
            if let Some(e) = errors.into_inner().unwrap_or_else(|e| e.into_inner()).into_iter().next() {
                return Err(e);
            }
            return w.add_owned(out, DType::Mxfp4, &shape, bytes);
        }
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
        // Values, one part at a time, encoded per matrix (the stack layout is matrices back
        // to back), so only one part's f32 is held.
        if w.is_prequantized(out) {
            return w.add_f32(out, &shape, &[]);
        }
        let target = w.target(out, &shape);
        let mut bytes = Vec::new();
        for p in parts {
            let v = srcs.f32(p)?;
            let (_, b) = encode(target, &v, &inner)?;
            bytes.extend(b);
        }
        w.add_owned(out, target.dtype(), &shape, bytes)
    }
}
