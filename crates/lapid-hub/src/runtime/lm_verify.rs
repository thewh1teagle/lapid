//! `lapid verify <causal LM>` for any [`CausalLm`]: compares lapid with the reference
//! written by the model's `tools/reference/<model>_ref.py` (the shared causal-LM reference
//! format, see `lapid_verify::harness`). Checks, in order:
//!
//! * chat template: every `template_cases` rendering byte-exact; tokenizer: every
//!   `tokenizer_cases` id list and the case prompts exact;
//! * `short` prompt, uncached forward: every `short.<tap>` of the reference (embedding,
//!   per-layer taps, final norm, all logits; per-row argmax);
//! * cached decoding, teacher-forced on the reference tokens: `<case>.gen.logits`
//!   (prefill + KV cache + recurrent state path);
//! * greedy continuation: token-exact for the descriptor's strict dtypes, reported otherwise;
//! * `long` prompt: `long.<tap>.tail` rows, `long.logits.last`, decode steps, greedy.
//!
//! Reference conventions: `result.json` `template_globals` (object) and `date_string` are
//! passed to every template render (fixed values the reference rendered with); a top-level
//! `chat_template` that differs from the file's (e.g. a GGUF carrying an older revision)
//! reports `template_cases` without counting them. Without `--model` / `--dtype`, the
//! reference's own `model` (a checkpoint directory next to `result.json`, also as the first
//! word of a label like `"x-tiny (random)"`, or a repo id) and
//! `default_dtype` are used, so tiny references verify with just `--reference`.
//!
//! Quantized models (int8 / int4 / k-quants) are checked for kernel correctness against the
//! fake-quant reference (same file, weights dequantized to f32 on the CPU; per tensor the
//! closer of f32 and int8 activations); the PyTorch comparison is then informational.

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Result};
use lapid_nn::Precision;
use lapid_verify::harness::{Descriptor, Reference, ids, kernel_compare, named, str_of, tail_rows};
use lapid_verify::timing::ms;
use lapid_verify::{Comparison, Stages, argmax_rows, tolerance};
use serde_json::{Value, json};

use super::causal_lm::{CausalLm, Llm, Rows};
use super::chat::ChatOptions;
use super::generate::GenerateOptions;
use super::resolve::{file_dtype, resolve};
use crate::cmd::ModelArgs;

/// Template options of a reference case: its `enable_thinking` plus the reference's fixed
/// template globals (`template_globals`, `date_string`).
fn template_opts(r: &Value, c: &Value) -> ChatOptions {
    let mut vars: Vec<(String, Value)> = r.get("template_globals").and_then(Value::as_object).into_iter().flatten().map(|(k, v)| (k.clone(), v.clone())).collect();
    if let Some(d) = r.get("date_string").filter(|d| d.is_string()) {
        vars.push(("date_string".to_string(), d.clone()));
    }
    ChatOptions { enable_thinking: c.get("enable_thinking").and_then(Value::as_bool), template_vars: vars, ..Default::default() }
}

/// Template and tokenizer checks. Returns (ok, details).
pub(crate) fn check_text<M: CausalLm>(llm: &Llm<M>, r: &Value) -> Result<(bool, Value)> {
    let mut ok = true;
    let same_template = r.get("chat_template").and_then(Value::as_str).is_none_or(|t| t == llm.template().source());
    let mut templates_ok = true;
    let mut templates = Vec::new();
    for (i, c) in r.get("template_cases").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let tools = c.get("tools_json").and_then(Value::as_str);
        let add_gen = c.get("add_generation_prompt").and_then(Value::as_bool).unwrap_or(false);
        let want = str_of(c, "text")?;
        let got = llm.render_chat_json_with(str_of(c, "messages_json")?, tools, add_gen, &template_opts(r, c));
        let m = got.as_deref().is_ok_and(|g| g == want);
        templates_ok &= m;
        let mut e = json!({"case": i, "match": m});
        if !m {
            e["got"] = match got {
                Ok(g) => json!(g),
                Err(err) => json!(format!("error: {err}")),
            };
            e["want"] = json!(want);
        }
        templates.push(e);
    }
    ok &= templates_ok || !same_template;
    let mut tokenizer = Vec::new();
    for c in r.get("tokenizer_cases").and_then(Value::as_array).into_iter().flatten() {
        let want = ids(c, "ids")?;
        let got = llm.encode(str_of(c, "text")?)?;
        let m = got == want;
        ok &= m;
        tokenizer.push(if m { json!({"match": true}) } else { json!({"match": false, "got": got, "want": want}) });
    }
    let mut prompts = serde_json::Map::new();
    for (name, c) in r.get("cases").and_then(Value::as_object).into_iter().flatten() {
        let text = llm.render_chat_json_with(str_of(c, "messages_json")?, None, true, &template_opts(r, c))?;
        let text_ok = text == str_of(c, "prompt_text")?;
        let ids_ok = llm.encode(&text)? == ids(c, "prompt_tokens")?;
        ok &= text_ok && ids_ok;
        prompts.insert(name.clone(), json!({"template": text_ok, "tokens": ids_ok}));
    }
    Ok((ok, json!({"template_cases": templates, "template_cases_counted": same_template, "tokenizer_cases": tokenizer, "prompts": prompts})))
}

/// Teacher-forced cached decoding: logits after the prompt, then after each reference token.
fn decode_logits<M: CausalLm>(llm: &Llm<M>, prompt: &[u32], forced: &[u32], steps: usize) -> Result<Vec<f32>> {
    let mut s = llm.new_session();
    let mut out = llm.step(&mut s, prompt)?;
    for &t in forced.iter().take(steps.saturating_sub(1)) {
        out.extend(llm.step(&mut s, &[t])?);
    }
    Ok(out)
}

fn greedy<M: CausalLm>(llm: &Llm<M>, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
    let mut s = llm.new_session();
    let mut stream = llm.generate(&mut s, prompt, GenerateOptions { max_new_tokens: n.max(1), ..Default::default() })?;
    for t in stream.by_ref() {
        t?;
    }
    Ok(stream.generated().to_vec())
}

/// Taps named like the reference: `short.<tap>` (all rows); `long.<tap>.tail` (last rows,
/// as many as the reference holds) and `long.logits.last`.
fn run_traced<M: CausalLm>(llm: &Llm<M>, r: &Reference, short: &[u32], long: &[u32]) -> Result<Vec<(String, Vec<f32>)>> {
    let mut out: Vec<(String, Vec<f32>)> = llm.forward_traced(short, Rows::All)?.into_iter().map(|(n, v)| (format!("short.{n}"), v)).collect();
    if !long.is_empty() {
        for (n, v) in llm.forward_traced(long, Rows::Last)? {
            if n == "logits" {
                out.push(("long.logits.last".to_string(), v));
                continue;
            }
            let key = format!("long.{n}.tail");
            if let (Some(tail), Ok(shape)) = (r.rows(&key), r.acts.shape(&key)) {
                out.push((key, tail_rows(&v, shape.last().copied().unwrap_or(1), tail)));
            }
        }
    }
    Ok(out)
}

/// Model and dtype to verify: the arguments, else the reference's own `model` (a checkpoint
/// directory next to `result.json`, or a repo id) and `default_dtype`, else the descriptor's.
pub(crate) fn model_and_dtype(desc: &Descriptor, r: &Reference, args: &ModelArgs) -> (String, String) {
    let model = args.model.clone().unwrap_or_else(|| {
        let m = r.result.get("model").and_then(Value::as_str).unwrap_or("");
        // "<dir> (random)" labels name the checkpoint directory first.
        let local = r.dir.join(m.split_whitespace().next().unwrap_or(""));
        if !m.is_empty() && local.exists() {
            local.display().to_string()
        } else if r.dir.join("model/config.json").exists() {
            r.dir.join("model").display().to_string()
        } else if m.contains('/') && !m.contains(char::is_whitespace) {
            m.to_string()
        } else {
            desc.default_model.to_string()
        }
    });
    // A checkpoint shipped with the reference is checked at the reference's own dtype.
    let own = (args.model.is_none() && model.starts_with(&*r.dir.display().to_string())).then(|| r.result.get("dtype").and_then(Value::as_str)).flatten();
    let dtype = args.dtype.clone().unwrap_or_else(|| r.result.get("default_dtype").and_then(Value::as_str).or(own).unwrap_or(desc.default_dtype).to_string());
    (model, dtype)
}

/// Full verification report as JSON; check `"ok"`.
pub fn verify<M: CausalLm>(desc: &Descriptor, reference: Option<&Path>, args: &ModelArgs) -> Result<Value> {
    let reference_dir = reference.unwrap_or(Path::new(desc.reference_dir));
    let r = Reference::open(reference_dir, desc.reference_script)?;
    let (model, dtype_arg) = model_and_dtype(desc, &r, args);
    let dev = lapid_verify::open_device(args.device.as_deref())?;
    let mut stages = Stages::new();
    let res = resolve(&model, Some(&dtype_arg))?;
    if let Some(t) = res.convert_ms {
        stages.record("convert", t);
    }
    let path = res.path;
    let dtype = file_dtype(&path)?;
    let quantized = !matches!(dtype.as_str(), "fp32" | "fp16");
    // Other ggml types (q5_0, q8_k mixes, ...) get the 4-bit budget (informational anyway).
    let quality_tol = tolerance(&dtype).or_else(|_| tolerance("int4"))?;
    // A float file can still hold losslessly repacked quantized weights (official FP4 experts
    // as mxfp4): the reference check then runs on them dequantized exactly, as the reference
    // does (their kernels quantize activations, which `lapid verify --dtype` of a quantized
    // file checks).
    let lossless_quant = !quantized && lapid::LapidFile::open(&path)?.tensors.values().any(|t| t.dtype().is_quant());
    let precision = if lossless_quant { Precision::FakeQuantWeights } else { Precision::Stored };
    let llm = stages.time("load", || Llm::<M>::load_with(&path, &dev, precision))?;
    let (text_ok, text) = stages.time("text", || check_text(&llm, &r.result))?;

    let (short, long) = (r.case("short"), r.case("long"));
    let short_prompt = ids(&short, "prompt_tokens")?;
    let long_prompt = if long.is_null() { Vec::new() } else { ids(&long, "prompt_tokens")? };

    // ---- layer by layer vs the reference
    let mut names = r.taps("short");
    if !long_prompt.is_empty() {
        names.extend(r.taps("long"));
    }
    let ours = stages.time("forward", || run_traced(&llm, &r, &short_prompt, &long_prompt))?;
    // Report in model order (so `first_failing_layer` is the first one the data reaches).
    names.sort_by_key(|n| ours.iter().position(|(k, _)| k == n).unwrap_or(usize::MAX));
    let mut quality = Comparison::new();
    quality.compare_named(&names, &ours, |n| r.acts.get(n), quality_tol)?;
    let vocab = llm.model().dims().vocab;
    let mut argmax = argmax_rows(named(&ours, "short.logits"), &r.acts.get("short.logits")?, vocab);
    if !long_prompt.is_empty() {
        argmax.extend(argmax_rows(named(&ours, "long.logits.last"), &r.acts.get("long.logits.last")?, vocab));
    }

    // ---- cached decode (teacher forced) + greedy continuation
    let mut decode = Comparison::new();
    let mut decode_argmax = Vec::new();
    let mut greedy_json = serde_json::Map::new();
    let mut tokens_match = true;
    for (name, c, prompt) in [("short", &short, &short_prompt), ("long", &long, &long_prompt)] {
        if prompt.is_empty() {
            continue;
        }
        let want = ids(c, "generated_tokens")?;
        let key = format!("{name}.gen.logits");
        let steps = r.rows(&key).unwrap_or(0);
        let got = stages.time("decode", || decode_logits(&llm, prompt, &want, steps))?;
        let reference = r.acts.get(&key)?;
        decode.compare(&key, &got, &reference, quality_tol);
        decode_argmax.extend(argmax_rows(&got, &reference, vocab));
        let t = Instant::now();
        let ours_tokens = greedy(&llm, prompt, want.len())?;
        stages.record("greedy", ms(t));
        let m = ours_tokens == want;
        tokens_match &= m;
        let first_diff = ours_tokens.iter().zip(&want).position(|(a, b)| a != b);
        greedy_json.insert(
            name.to_string(),
            json!({"match": m, "first_mismatch": first_diff, "tokens": ours_tokens, "reference_tokens": want,
                   "text": llm.decode(&ours_tokens, false), "reference_text": c.get("generated_text")}),
        );
    }

    // ---- kernel correctness vs fake-quant (quantized only)
    let mut kernel_json = Value::Null;
    let mut kernel: Option<Comparison> = None;
    if quantized {
        let cpu = Device::cpu();
        let (fw, fa) = stages.time("fake_quant", || -> Result<_> {
            let w = Llm::<M>::load_with(&path, &cpu, Precision::FakeQuantWeights)?;
            let ow = w.forward_traced(&short_prompt, Rows::All)?;
            drop(w);
            let a = Llm::<M>::load_with(&path, &cpu, Precision::FakeQuantActivations)?;
            let oa = a.forward_traced(&short_prompt, Rows::All)?;
            let prefix = |v: Vec<(String, Vec<f32>)>| v.into_iter().map(|(n, x)| (format!("short.{n}"), x)).collect::<Vec<_>>();
            Ok((prefix(ow), prefix(oa)))
        })?;
        let short_names: Vec<String> = names.iter().filter(|n| n.starts_with("short.")).cloned().collect();
        let (merged, which) = kernel_compare(&short_names, &ours, &fw, &fa)?;
        let ka = argmax_rows(named(&ours, "short.logits"), named(&fw, "short.logits"), vocab);
        kernel_json = json!({
            "reference": "fake-quant: same weights dequantized to f32 on cpu; per tensor the closer of f32 and int8-per-32-block activations",
            "tolerance": lapid_verify::KERNEL_TOLERANCE,
            "matched_reference": which,
            "logits_argmax_match_f32_activations": ka,
            "comparison": merged.to_json(),
        });
        kernel = Some(merged);
    }

    let strict = desc.tokens_must_match(&dtype);
    let all_true = |v: &[bool]| v.iter().all(|&b| b);
    let correctness_ok = match &kernel {
        Some(k) => k.ok(),
        None => quality.ok() && decode.ok() && all_true(&argmax) && all_true(&decode_argmax),
    };
    let ok = text_ok && correctness_ok && (!strict || tokens_match);
    Ok(json!({
        "ok": ok,
        "device": dev.description(),
        "model": model,
        "model_file": path.display().to_string(),
        "dtype": dtype,
        "correctness_reference": if quantized { "fake-quant" } else { "pytorch-fp32" },
        "text_ok": text_ok,
        "tokens_match": tokens_match,
        "tokens_must_match": strict,
        "first_failing_layer": kernel.as_ref().unwrap_or(&quality).first_failing(),
        "layers": quality.layers,
        "quality": {
            "reference": desc.reference_label,
            "informational": quantized,
            "tolerance": quality_tol,
            "within_tolerance": quality.ok(),
            "max_rel": quality.max_rel(),
            "logits_argmax_match": argmax,
            "decode": decode.to_json(),
            "decode_argmax_match": decode_argmax,
        },
        "kernel": kernel_json,
        "greedy": greedy_json,
        "text": text,
        "reference_dir": reference_dir.display().to_string(),
        "timings_ms": stages.to_json(),
        "notes": quality.notes,
        "weights": if lossless_quant { "quantized tensors dequantized to f32 (fake-quant weights)" } else { "as stored" },
    }))
}
