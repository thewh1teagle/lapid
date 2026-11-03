//! Runtime steering on the tiny reference: ablating `attn_out` (after `wo_b`) gives the
//! logits of the model with `wo_b` edited to `W - s d dᵀ W` (the abliteration weight edit),
//! through the graph forward and the expert-cache forward, on the CPU and on CUDA (at the
//! device's own fp32 noise).

use std::path::Path;

use lapid::{Device, Tensor};
use lapid_nn::Linear;
use lapid_nn::llm::expert_cache::set_forced_slots;

use crate::deepseek41::{Deepseek41, convert};
use crate::runtime::{Hook, SteerFile};
use crate::runtime::steer::{SteerEntry, SteerOp};

const SCALE: f32 = 2.5;

/// Random unit directions for layers 1.. (xorshift).
fn directions(hidden: usize, layers: usize) -> Vec<(usize, Vec<f32>)> {
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    (1..layers)
        .map(|l| {
            let v: Vec<f32> = (0..hidden).map(|_| next()).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            (l, v.iter().map(|x| x / n).collect())
        })
        .collect()
}

fn load(file: &Path, dev: &Device, slots: Option<usize>) -> lapid::Result<Deepseek41> {
    set_forced_slots(slots);
    let llm = Deepseek41::load_with(file, dev, lapid_nn::Precision::Stored);
    set_forced_slots(None);
    llm
}

/// Prefill + 3 greedy steps (the first run's tokens are fed to later runs: `fixed`).
fn logits(llm: &Deepseek41, s: &mut crate::deepseek41::Session, ids: &[u32], fixed: &mut Vec<u32>) -> lapid::Result<Vec<Vec<f32>>> {
    let mut out = vec![llm.step(s, ids)?];
    for i in 0..3 {
        if fixed.len() == i {
            fixed.push(lapid_nn::argmax(out.last().map(Vec::as_slice).unwrap_or_default()) as u32);
        }
        out.push(llm.step(s, &[fixed[i]])?);
    }
    Ok(out)
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32) {
    let scale = b.iter().flatten().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    let d = a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    (d, scale)
}

#[test]
fn ablation_matches_weight_edit() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/deepseek41/tiny");
    if !root.join("result.json").is_file() || !root.join("model/config.json").is_file() {
        eprintln!("skipped: {} not found", root.display());
        return;
    }
    let out = std::env::temp_dir().join(format!("lapid-deepseek41-steer-{}.lapid", std::process::id()));
    convert::CONVERT.convert(&root.join("model").display().to_string(), lapid_convert::OutDType::F32, &out).expect("convert");
    let r = check(&root, &out);
    let served = served(&out);
    let _ = std::fs::remove_file(&out);
    r.expect("steering");
    served.expect("served steering");
}

/// Served with a steering file: per-request selection, sessions reused only under the
/// steering they were computed with.
fn served(file: &Path) -> lapid::Result<()> {
    use crate::runtime::serve::{GenParams, LoadOptions, load_path};
    use crate::runtime::steer::SteerSpec;
    let dev = Device::cpu();
    let llm = load(file, &dev, None)?;
    let (hidden, layers) = (llm.config().hidden_size, llm.model().layers.len());
    drop(llm);
    let entries = directions(hidden, layers).into_iter().map(|(l, d)| SteerEntry { hook: Hook::AttnOut, layer: l, op: SteerOp::Ablate, scale: SCALE, vector: d }).collect();
    let sf = std::env::temp_dir().join(format!("lapid-deepseek41-steer-{}.safetensors", std::process::id()));
    SteerFile { name: "rand".into(), architecture: "deepseek41".into(), hidden_size: hidden, entries, ..Default::default() }.save(&sf)?;
    let opts = LoadOptions { steer: vec![SteerSpec { path: sf.clone(), scale: 1.0 }], ..Default::default() };
    let m = load_path(file, "tiny".into(), &dev, &opts);
    let _ = std::fs::remove_file(&sf);
    let mut m = m?;
    assert_eq!(m.meta().extra["steering"][0]["name"], "rand");
    // One token: nothing generated is fed, so a session holds exactly its prompt.
    let params = |steer: Option<Vec<(String, f32)>>| GenParams { max_tokens: Some(1), temperature: Some(0.0), steer, ..Default::default() };
    let on = || Some(vec![("rand".to_string(), 1.0)]);
    let mut run = |prompt: &str, g: GenParams| m.complete(prompt, &g, &mut |_| true).map(|s| s.cached_tokens);
    let p = ["the quick brown fox", " jumps over", " the lazy dog", " and runs", " away"];
    let prompt = |n: usize| p[..n].concat();
    assert_eq!(run(&prompt(1), params(None))?, 0);
    assert!(run(&prompt(2), params(None))? > 0, "same (default) steering: prefix reused");
    assert_eq!(run(&prompt(3), params(Some(Vec::new())))?, 0, "steering off: not reused");
    assert!(run(&prompt(3), params(on()))? > 0, "explicit = default steering: reused");
    assert_eq!(run(&prompt(4), params(Some(vec![("rand".to_string(), 0.5)])))?, 0, "other scale: not reused");
    assert!(run(&prompt(5), params(Some(Vec::new())))? > 0, "steering off again: its session reused");
    assert_eq!(run(&prompt(1), params(Some(vec![("nope".to_string(), 1.0)]))).unwrap_err().code, "invalid_argument");
    Ok(())
}

fn check(root: &Path, file: &Path) -> lapid::Result<()> {
    let result: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("result.json"))?)?;
    let ids: Vec<u32> = serde_json::from_value(result["cases"]["long"]["prompt_tokens"].clone())?;
    let ids = &ids[..60];
    let mut devices = vec![Device::cpu()];
    devices.extend(Device::cuda(0).ok());
    for dev in &devices {
        // The CPU agrees to ~3e-6 of the logit range; CUDA's fp32 differs by ~1e-4 between
        // any two evaluation orders of the same model (graph vs expert-cache forward too).
        let tol = if dev.name() == "cpu" { 1e-4 } else { 2.5e-4 };
        for slots in [None, Some(2)] {
            let mut fixed = Vec::new();
            let plain = load(file, dev, slots)?;
            let (hidden, layers) = (plain.config().hidden_size, plain.model().layers.len());
            let dirs = directions(hidden, layers);
            let entries = dirs.iter().map(|(l, d)| SteerEntry { hook: Hook::AttnOut, layer: *l, op: SteerOp::Ablate, scale: SCALE, vector: d.clone() }).collect();
            let f = SteerFile { name: "rand".into(), architecture: "deepseek41".into(), hidden_size: hidden, entries, ..Default::default() };
            // Steered at runtime.
            let mut s = plain.new_session();
            s.set_steering(plain.steering(&[(&f, 1.0)])?);
            let steered = logits(&plain, &mut s, ids, &mut fixed)?;
            // Multiplier 0: no steering, the unedited model.
            let mut s0 = plain.new_session();
            s0.set_steering(plain.steering(&[(&f, 0.0)])?);
            assert!(s0.steering().is_none());
            let base = logits(&plain, &mut s0, ids, &mut fixed)?;
            drop(plain);
            // wo_b edited.
            let mut edited = load(file, dev, slots)?;
            for (l, d) in &dirs {
                let lin = &mut edited.model_mut().layers[*l].attn.wo_b;
                let (out_f, in_f) = (lin.weight.shape()[0], lin.weight.shape()[1]);
                assert_eq!(out_f, hidden);
                let w = dev.to_vec_f32(&lin.weight)?;
                let dw: Vec<f64> = (0..in_f).map(|j| (0..out_f).map(|o| f64::from(d[o]) * f64::from(w[o * in_f + j])).sum()).collect();
                let w2: Vec<f32> = (0..out_f * in_f).map(|k| (f64::from(w[k]) - f64::from(SCALE) * f64::from(d[k / in_f]) * dw[k % in_f]) as f32).collect();
                *lin = Linear::new(dev.upload(&Tensor::from_f32(&w2, &[out_f, in_f])?)?, lin.bias.clone());
            }
            let want = logits(&edited, &mut edited.new_session(), ids, &mut fixed)?;
            let (diff, scale) = max_diff(&steered, &want);
            let (moved, _) = max_diff(&base, &want);
            eprintln!("{} slots {slots:?}: steered vs edited {diff:.2e}, unsteered vs edited {moved:.2e} (scale {scale:.1})", dev.name());
            assert!(diff <= tol * scale, "{}: steered logits differ from the edited model by {diff}", dev.name());
            assert!(moved > 100.0 * diff.max(1e-6), "{}: steering had no effect ({moved})", dev.name());
        }
    }
    Ok(())
}
