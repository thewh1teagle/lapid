//! `lapid verify` / `lapid bench` for any [`VisionLm`]. The reference (written by the model's
//! `tools/reference/<model>_ref.py`) uses the causal-LM format with image cases:
//! `cases.<case>` has `messages_json`, `prompt_text`, `prompt_tokens` (placeholders
//! expanded), `generated_tokens`, optional `image` (file next to `result.json`) and
//! `enable_thinking`; activations are `<case>.pixel_values` (+ whatever
//! [`VisionLm::reference_image`] reads), `<case>.vision.*` taps, the decoder taps
//! (`<case>.<tap>`, logits as `<case>.logits` (all rows) or `<case>.logits.last`) and
//! `<case>.gen.logits`. Checks, per case:
//!
//! * template rendering and the expanded prompt ids exact (plus `template_cases` /
//!   `tokenizer_cases`);
//! * our preprocessing of the picture vs the reference pixels;
//! * the vision tower on the reference pixels, tap by tap;
//! * the decoder with the image rows (the reference's when it has all
//!   [`VisionLm::media_taps`], else ours), tap by tap, plus per-row argmax;
//! * teacher-forced cached decoding (`<case>.gen.logits`) and greedy end to end (our
//!   preprocessing), token-exact for the descriptor's strict dtypes; a divergence at a
//!   near-tie of the reference logits (gap within float noise, teacher forced) is reported
//!   with its gap and accepted.
//!
//! Quantized files are checked against the fake-quant reference (same file dequantized on
//! the CPU), the PyTorch comparison is then informational. Text references of a vision model
//! may add an `inject` case ([`verify_text`]).

use std::path::{Path, PathBuf};
use std::time::Instant;

use lapid::{Device, Error, Result, Tensor};
use lapid_convert::OutDType;
use lapid_nn::Precision;
use lapid_verify::harness::{Descriptor, Reference, ids, kernel_compare, named, str_of};
use lapid_verify::timing::{median, ms, stats};
use lapid_verify::{Comparison, Stages, argmax_rows, tolerance};
use serde_json::{Value, json};

use super::causal_lm::{Llm, Rows, step};
use super::chat::ChatOptions;
use super::generate::GenerateOptions;
use super::lm_verify::{check_text, model_and_dtype};
use super::resolve::{cache_path_tagged, cached_matches, file_dtype, is_lapid};
use super::vlm::{Media, MediaPrompt, forward_media_traced, step_media};
use super::vlm::{VisionLm, encode_image};
use crate::cmd::ModelArgs;
use crate::vision::{Image, Message};

/// The `.lapid` of `model` with its vision tower: a `.lapid` as is, else converted (HF repo
/// or directory, GGUF with its `mmproj`) into the cache under a `-vision` name.
pub fn resolve_vision(model: &str, dtype: &str) -> Result<(PathBuf, Option<f64>)> {
    let p = Path::new(model);
    if is_lapid(p) {
        return Ok((p.to_path_buf(), None));
    }
    let path = cache_path_tagged(model, OutDType::parse(dtype)?, "vision");
    if path.is_file() && cached_matches(&path, model) {
        return Ok((path, None));
    }
    let t = Instant::now();
    crate::convert::convert(model, dtype, Some(&path))?;
    Ok((path, Some(ms(t))))
}

/// One reference case, built from the reference's pixels.
struct Case<I> {
    name: String,
    c: Value,
    /// The reference prompt ids with our image placement.
    prompt: MediaPrompt<()>,
    /// The reference's prepared image.
    image: Option<I>,
}

/// The image rows of `case` for the decoder: the reference's media taps when it has them
/// all (shaped like ours), else our encoder's outputs.
fn media_of<M: VisionLm>(llm: &Llm<M>, r: &Reference, case: &Case<M::Image>, ours: &[Tensor]) -> Result<Vec<Media>> {
    let Some(im) = case.prompt.images.first() else { return Ok(Vec::new()) };
    let names: Vec<String> = llm.model().media_taps().iter().map(|t| format!("{}.{t}", case.name)).collect();
    let embeds = if names.len() == ours.len() && names.iter().all(|n| r.acts.has(n)) {
        names.iter().zip(ours).map(|(n, t)| Tensor::from_f32(&r.acts.get(n)?, t.shape())).collect::<Result<Vec<_>>>()?
    } else {
        ours.to_vec()
    };
    Ok(vec![Media { start: im.start, len: im.len, grid: im.grid, embeds }])
}

fn prefixed(case: &str, v: Vec<(String, Vec<f32>)>) -> Vec<(String, Vec<f32>)> {
    v.into_iter().map(|(n, x)| (format!("{case}.{n}"), x)).collect()
}

/// Vision taps (reference pixels) and decoder taps of a case, named `<case>.<tap>`, plus the
/// decoder's image rows.
fn traced<M: VisionLm>(llm: &Llm<M>, r: &Reference, case: &Case<M::Image>) -> Result<(Vec<(String, Vec<f32>)>, Vec<Media>)> {
    let dev = llm.device();
    let mut out = Vec::new();
    let mut ours = Vec::new();
    if let Some(img) = &case.image {
        let (o, taps) = encode_image(llm.model(), dev, img, true)?;
        out.extend(prefixed(&case.name, taps));
        ours = o;
    }
    let media = media_of(llm, r, case, &ours)?;
    let all_rows = r.acts.has(&format!("{}.logits", case.name));
    let rows = if all_rows { Rows::All } else { Rows::Last };
    for (n, v) in forward_media_traced(llm.model(), dev, &case.prompt.ids, &media, rows)? {
        let n = if n == "logits" && !all_rows { "logits.last".to_string() } else { n };
        out.push((format!("{}.{n}", case.name), v));
    }
    Ok((out, media))
}

/// Teacher-forced cached decoding: logits after the prompt, then after each forced token.
fn decode_logits<M: VisionLm>(llm: &Llm<M>, ids: &[u32], media: &[Media], forced: &[u32], steps: usize) -> Result<Vec<f32>> {
    let (m, dev) = (llm.model(), llm.device());
    let mut st = m.new_state();
    let mut out = step_media(m, dev, ids, media, &mut st, None)?;
    for &t in forced.iter().take(steps.saturating_sub(1)) {
        out.extend(step(m, dev, &[t], &mut st, None)?);
    }
    Ok(out)
}

/// Greedy diverged at step `k` (we picked `ours`): teacher-forced on the reference tokens,
/// the logit gap between our pick and the reference token there, and whether it is a tie
/// within float noise (`<= 1e-4 * max(1, |logit|)`; random tiny models have flat logits).
/// Past a tie the continuations differ without a model error.
fn near_tie<M: VisionLm>(llm: &Llm<M>, ids: &[u32], media: &[Media], want: &[u32], k: usize, ours: u32) -> Result<Option<(f32, bool)>> {
    let vocab = llm.model().dims().vocab;
    let logits = decode_logits(llm, ids, media, want, k + 1)?;
    let row = &logits[k * vocab..(k + 1) * vocab];
    let (a, b) = (row[ours as usize], row[want[k] as usize]);
    Ok(Some((a - b, a - b <= 1e-4 * a.abs().max(1.0))))
}

/// Full verification report as JSON; check `"ok"`.
pub fn verify<M: VisionLm>(desc: &Descriptor, reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let reference_dir = reference.unwrap_or(Path::new(desc.reference_dir));
    let r = Reference::open(reference_dir, desc.reference_script)?;
    let (model, dtype_arg) = model_and_dtype(desc, &r, args);
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let mut stages = Stages::new();
    let (path, convert_ms) = resolve_vision(&model, &dtype_arg)?;
    if let Some(t) = convert_ms {
        stages.record("convert", t);
    }
    let dtype = file_dtype(&path)?;
    let quantized = !matches!(dtype.as_str(), "fp32" | "fp16");
    let tol = tolerance(&dtype).or_else(|_| tolerance("int4"))?;
    let load = |d: &Device, p: Precision| -> Result<Llm<M>> {
        let mut llm = Llm::<M>::load_with(&path, d, p)?;
        llm.model_mut().configure(&r.result)?;
        Ok(llm)
    };
    let llm = stages.time("load", || load(&dev, Precision::Stored))?;
    let mut text_only = r.result.clone();
    text_only["cases"] = Value::Null;
    let (mut text_ok, text) = check_text(&llm, &text_only)?;

    // ---- prompts and preprocessing
    let (max_abs, mean_abs) = llm.model().preprocess_tolerance();
    let mut cases = Vec::new();
    let mut prompts = serde_json::Map::new();
    let mut pre_ok = true;
    for (name, c) in r.result.get("cases").and_then(Value::as_object).into_iter().flatten() {
        let opts = ChatOptions { enable_thinking: c.get("enable_thinking").and_then(Value::as_bool), ..Default::default() };
        let rendered = llm.render_chat_json(str_of(c, "messages_json")?, None, true, opts.enable_thinking)?;
        let template_ok = rendered == str_of(c, "prompt_text")?;
        let picture = c.get("image").and_then(Value::as_str).map(|f| Image::open(&reference_dir.join(f))).transpose()?;
        let ids_raw = llm.encode(&rendered)?;
        let ours = llm.mm_prompt_ids(&ids_raw, &picture.iter().collect::<Vec<_>>())?;
        let want = ids(c, "prompt_tokens")?;
        let ids_ok = ours.ids == want;
        let mut pre = Value::Null;
        let mut image = None;
        if let Some(im) = ours.images.first() {
            let got = llm.model().image_pixels(&im.image);
            let refpix = r.acts.get(&format!("{name}.pixel_values"))?;
            let ok = got.len() == refpix.len() && {
                let d = lapid_verify::diff(got, &refpix)?;
                pre = json!({"max_abs": d.max_abs, "mean_abs": d.mean_abs, "values": got.len()});
                d.max_abs <= max_abs && d.mean_abs <= mean_abs
            };
            if pre.is_null() {
                pre = json!({"values": got.len(), "reference_values": refpix.len()});
            }
            pre["ok"] = json!(ok);
            pre_ok &= ok;
            image = Some(llm.model().reference_image(&r, name)?);
        }
        let extra = llm.model().check_prompt(&ours, c);
        text_ok &= template_ok && ids_ok && extra != Some(false);
        prompts.insert(name.clone(), json!({"template": template_ok, "tokens": ids_ok, "model_check": extra, "preprocess": pre}));
        let prompt = MediaPrompt {
            ids: want,
            images: ours.images.iter().map(|i| super::vlm::PromptImage { start: i.start, len: i.len, grid: i.grid, image: (), hash: i.hash }).collect(),
        };
        cases.push((Case { name: name.clone(), c: c.clone(), prompt, image }, ours));
    }

    // ---- layer by layer vs the reference
    let vocab = llm.model().dims().vocab;
    let mut quality = Comparison::new();
    let mut argmax = Vec::new();
    let mut all_ours = Vec::new();
    let mut names_all = Vec::new();
    let mut medias = Vec::new();
    for (case, _) in &cases {
        let (ours, media) = stages.time("forward", || traced(&llm, &r, case))?;
        let mut names = r.taps(&case.name);
        names.retain(|n| !n.ends_with(".pixel_values") && !n.ends_with(".positions"));
        names.sort_by_key(|n| ours.iter().position(|(k, _)| k == n).unwrap_or(usize::MAX));
        quality.compare_named(&names, &ours, |n| r.acts.get(n), tol)?;
        let lg = if r.acts.has(&format!("{}.logits", case.name)) { format!("{}.logits", case.name) } else { format!("{}.logits.last", case.name) };
        argmax.extend(argmax_rows(named(&ours, &lg), &r.acts.get(&lg)?, vocab));
        names_all.extend(names);
        all_ours.extend(ours);
        medias.push(media);
    }

    // ---- cached decode (teacher forced) + greedy end to end
    let mut decode = Comparison::new();
    let mut decode_argmax = Vec::new();
    let mut greedy_json = serde_json::Map::new();
    let mut tokens_match = true;
    for ((case, ours), media) in cases.iter().zip(&medias) {
        let want = ids(&case.c, "generated_tokens")?;
        let key = format!("{}.gen.logits", case.name);
        if r.acts.has(&key) {
            let steps = r.rows(&key).unwrap_or(0);
            let got = stages.time("decode", || decode_logits(&llm, &case.prompt.ids, media, &want, steps))?;
            let reference = r.acts.get(&key)?;
            decode.compare(&key, &got, &reference, tol);
            decode_argmax.extend(argmax_rows(&got, &reference, vocab));
        }
        let t = Instant::now();
        let mut s = llm.new_session();
        let mut stream = llm.mm_generate(&mut s, ours, GenerateOptions { max_new_tokens: want.len().max(1), ..Default::default() })?;
        for tok in stream.by_ref() {
            tok?;
        }
        let tokens = stream.generated().to_vec();
        drop(stream);
        stages.record("greedy", ms(t));
        let first = tokens.iter().zip(&want).position(|(a, b)| a != b);
        let tie = match first {
            Some(k) => near_tie(&llm, &case.prompt.ids, media, &want, k, tokens[k])?,
            None => None,
        };
        let m = tokens == want || tie.is_some_and(|(_, is_tie)| is_tie);
        tokens_match &= m;
        greedy_json.insert(
            case.name.clone(),
            json!({"match": m, "first_mismatch": first, "near_tie_gap": tie.map(|t| t.0), "tokens": tokens,
                   "reference_tokens": want, "text": llm.decode(&tokens, false), "reference_text": case.c.get("generated_text")}),
        );
    }

    // ---- kernel correctness vs fake-quant (quantized only)
    let mut kernel_json = Value::Null;
    let mut kernel: Option<Comparison> = None;
    if quantized {
        let cpu = Device::cpu();
        let run = |p: Precision| -> Result<Vec<(String, Vec<f32>)>> {
            let m = load(&cpu, p)?;
            let mut v = Vec::new();
            for (case, _) in &cases {
                v.extend(traced(&m, &r, case)?.0);
            }
            Ok(v)
        };
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> { Ok((run(Precision::FakeQuantWeights)?, run(Precision::FakeQuantActivations)?)) })?;
        let (merged, which) = kernel_compare(&names_all, &all_ours, &fw, &fa)?;
        kernel_json = json!({"reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8 activations",
                             "tolerance": lapid_verify::KERNEL_TOLERANCE, "matched_reference": which, "comparison": merged.to_json()});
        kernel = Some(merged);
    }

    let strict = desc.tokens_must_match(&dtype);
    let all = |v: &[bool]| v.iter().all(|&b| b);
    let correctness_ok = match &kernel {
        Some(k) => k.ok(),
        None => quality.ok() && decode.ok() && all(&argmax) && all(&decode_argmax),
    };
    let ok = text_ok && pre_ok && correctness_ok && (!strict || tokens_match);
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "text_ok": text_ok,
        "preprocess_ok": pre_ok,
        "tokens_match": tokens_match,
        "tokens_must_match": strict,
        "first_failing_layer": kernel.as_ref().unwrap_or(&quality).first_failing(),
        "layers": quality.layers,
        "quality": {"reference": desc.reference_label, "informational": quantized, "tolerance": tol, "within_tolerance": quality.ok(),
                    "max_rel": quality.max_rel(), "logits_argmax_match": argmax, "decode": decode.to_json(), "decode_argmax_match": decode_argmax},
        "kernel": kernel_json,
        "greedy": greedy_json,
        "text": text,
        "prompts": prompts,
        "reference_dir": reference_dir.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": quality.notes,
    }))
}

/// `lapid verify` of a text reference ([`super::lm_verify`]) plus, when the reference has an
/// `inject` case, image rows injected at its placeholder tokens: `inject.embeds` [n, D]
/// replaces `prompt_tokens[start..start + n]`; the last logits of an uncached forward and of
/// the chunked cached prefill and every row's final norm are compared.
pub fn verify_text<M: VisionLm>(desc: &Descriptor, reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let mut report = super::lm_verify::verify::<M>(desc, reference, args)?;
    let r = Reference::open(reference.unwrap_or(Path::new(desc.reference_dir)), desc.reference_script)?;
    let Some(c) = r.result.get("inject") else { return Ok(report) };
    let path = PathBuf::from(report["model_file"].as_str().unwrap_or_default());
    let llm = Llm::<M>::load(&path, &lapid_verify::open_device(args.device.as_deref())?)?;
    let tol = report["quality"]["tolerance"].as_f64().unwrap_or(1e-2) as f32;
    let prompt = ids(c, "prompt_tokens")?;
    let start = c.get("start").and_then(Value::as_u64).ok_or_else(|| Error::new("reference", "inject: no start"))? as usize;
    let shape = r.acts.shape("inject.embeds")?;
    let media = [Media { start, len: shape[0], grid: (1, shape[0]), embeds: vec![Tensor::from_f32(&r.acts.get("inject.embeds")?, &shape)?] }];
    let (m, dev) = (llm.model(), llm.device());
    let want_last = r.acts.get("inject.logits.last")?;
    let mut cmp = Comparison::new();
    cmp.compare("inject.logits.last", named(&forward_media_traced(m, dev, &prompt, &media, Rows::Last)?, "logits"), &want_last, tol);
    let last = step_media(m, dev, &prompt, &media, &mut m.new_state(), None)?;
    cmp.compare("inject.prefill.last", &last, &want_last, tol);
    cmp.compare("inject.norm", named(&forward_media_traced(m, dev, &prompt, &media, Rows::All)?, "norm"), &r.acts.get("inject.norm")?, tol);
    let argmax = argmax_rows(&last, &want_last, m.dims().vocab);
    let quantized = !matches!(report["dtype"].as_str(), Some("fp32" | "fp16"));
    if !quantized && !(cmp.ok() && argmax.iter().all(|&b| b)) {
        report["ok"] = json!(false);
    }
    report["inject"] = json!({"ok": cmp.ok(), "argmax_match": argmax, "informational": quantized, "comparison": cmp.to_json()});
    Ok(report)
}

/// `lapid bench` of an image prompt: preprocessing and encoding time, time to first token
/// (encode + prefill) and decode speed. `input`: the picture (default: the reference's).
pub fn bench<M: VisionLm>(desc: &Descriptor, input: Option<&Path>, args: &ModelArgs, runs: usize) -> Result<Value> {
    let model = args.model.as_deref().unwrap_or(desc.default_model);
    let dtype = args.dtype.as_deref().unwrap_or(desc.default_dtype);
    let gen_tokens = desc.bench_sizes()?.1.max(2);
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let (path, convert_ms) = resolve_vision(model, dtype)?;
    let t = Instant::now();
    let llm = Llm::<M>::load(&path, &dev)?;
    let load_ms = ms(t);
    let image_path = match input {
        Some(p) => p.to_path_buf(),
        None => {
            let r = Reference::open(Path::new(desc.reference_dir), desc.reference_script)?;
            let f = r.result.pointer("/cases/image/image").and_then(Value::as_str).unwrap_or("image.png").to_string();
            r.dir.join(f)
        }
    };
    let img = Image::open(&image_path)?;
    let messages = [Message::user_with_images(vec![img.clone()], "Describe this image in detail.")];
    let opts = ChatOptions { enable_thinking: Some(false), ..Default::default() };
    let (mut prep, mut enc, mut ttft, mut decode) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut n_image, mut prompt_tokens) = (0, 0);
    for run in 0..=runs {
        let t = Instant::now();
        let p = llm.model().prepare_image(&img)?;
        let p_ms = ms(t);
        let t = Instant::now();
        let e = llm.encode_image(&p)?;
        dev.to_vec_f32(&e[0])?;
        let e_ms = ms(t);
        n_image = llm.model().image_size(&p).0;
        let t = Instant::now();
        let prompt = llm.mm_prompt(&messages, &opts)?;
        prompt_tokens = prompt.ids.len();
        let mut s = llm.new_session();
        let g = GenerateOptions { max_new_tokens: gen_tokens, stop_tokens: Some(Vec::new()), ..Default::default() };
        let mut stream = llm.mm_generate(&mut s, &prompt, g)?;
        let mut first = None;
        let mut count = 0usize;
        for tok in stream.by_ref() {
            tok?;
            count += 1;
            first.get_or_insert_with(Instant::now);
        }
        let end = Instant::now();
        if run == 0 {
            continue;
        }
        prep.push(p_ms);
        enc.push(e_ms);
        if let Some(f) = first {
            ttft.push((f - t).as_secs_f64() * 1e3);
            if count > 1 {
                decode.push((count - 1) as f64 / (end - f).as_secs_f64());
            }
        }
    }
    Ok(json!({
        "engine": "lapid", "model": model, "model_file": path.display().to_string(), "dtype": file_dtype(&path)?,
        "device": dev.description(), "backend": dev.name(), "convert_ms": convert_ms, "load_ms": load_ms,
        "image": image_path.display().to_string(), "image_tokens": n_image, "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
        "preprocess_ms": median(prep.clone()), "image_encode_ms": median(enc.clone()), "ttft_ms": median(ttft.clone()),
        "decode_tokens_per_s": median(decode.clone()),
        "runs": {"preprocess_ms": stats(&prep), "image_encode_ms": stats(&enc), "ttft_ms": stats(&ttft), "decode_tokens_per_s": stats(&decode)},
    }))
}
