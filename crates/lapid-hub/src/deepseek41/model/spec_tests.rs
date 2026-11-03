//! DSpark draft and speculative decoding on the tiny reference (`deepseek41_ref.py --tiny`):
//! draft logits against the official `forward_spec`, rewinds against plain forwards, and
//! speculative greedy output against plain greedy (random drafts, and oracle drafts with
//! mistakes so blocks are accepted in part and in full), with the experts as graph leaves
//! and in a 2-slot expert cache, on the CPU and on CUDA when present.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lapid::{Device, Result};
use lapid_nn::Sampler;
use lapid_nn::llm::expert_cache::set_forced_slots;
use lapid_verify::reference::Activations;
use serde_json::Value;

use crate::deepseek41::{Deepseek41, convert};
use crate::runtime::CausalLm;
use crate::runtime::generate::GenerateOptions;
use crate::runtime::session::LmState;
use crate::runtime::spec::{LmSpec, SpecConfig, SpecGen, SpecModel, set_spec};

/// The process-wide speculative setting is read at load and at generation: one test at a time.
static SPEC_LOCK: Mutex<()> = Mutex::new(());

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference/deepseek41/tiny")
}

/// The tiny model converted (fp32) with its DSpark stages; `None` without reference data.
fn tiny_file(tag: &str) -> Option<PathBuf> {
    let root = root();
    if !root.join("dspark.safetensors").is_file() {
        eprintln!("skipped: {} has no DSpark reference", root.display());
        return None;
    }
    let out = std::env::temp_dir().join(format!("lapid-deepseek41-dspark-{tag}-{}.lapid", std::process::id()));
    convert::CONVERT.convert(&root.join("model").display().to_string(), lapid_convert::OutDType::F32, &out).expect("convert");
    Some(out)
}

fn load(file: &Path, dev: &Device, spec: bool, slots: Option<usize>) -> Result<Deepseek41> {
    set_spec(spec.then(SpecConfig::default));
    set_forced_slots(slots);
    // Dequantized weights: exact f32 math on every path (the stored-quant kernels quantize
    // activations differently for one row and for many, which a random tiny model amplifies).
    let llm = Deepseek41::load_with(file, dev, lapid_nn::Precision::FakeQuantWeights);
    set_forced_slots(None);
    set_spec(None);
    llm
}

fn result() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("result.json")).expect("result.json")).expect("json")
}

fn ids(v: &Value) -> Vec<u32> {
    serde_json::from_value(v.clone()).expect("ids")
}

fn devices() -> Vec<Device> {
    let mut v = vec![Device::cpu()];
    v.extend(Device::cuda(0).ok());
    v
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

/// Draft logits / tokens / confidences of every recorded step against the official
/// `forward_spec` (the window filled from the prefill, then one position per decode step),
/// and the state-updating draft gives the same tokens.
#[test]
fn dspark_matches_reference() {
    let _one = SPEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(file) = tiny_file("ref") else { return };
    let r = result();
    let refs = Activations::open(&root().join("dspark.safetensors")).expect("dspark.safetensors");
    for (dev, slots) in devices().into_iter().flat_map(|d| [(d.clone(), None), (d, Some(2))]) {
        let llm = load(&file, &dev, true, slots).expect("load");
        let m = llm.model();
        let d = m.dspark.as_ref().expect("dspark loaded");
        let (bs, v) = (d.block_size, m.cfg.vocab_size);
        for case in ["short", "long"] {
            let prompt = ids(&r["cases"][case]["prompt_tokens"]);
            let cont = ids(&r["cases"][case]["generated_tokens"]);
            let steps = r["dspark"]["cases"][case]["steps"].as_u64().expect("steps") as usize;
            let want_logits = refs.get(&format!("{case}.dspark.logits")).expect("logits");
            let want_conf = refs.get(&format!("{case}.dspark.confidence")).expect("confidence");
            let mut s = llm.new_session();
            llm.step(&mut s, &prompt).expect("prefill");
            for (j, &anchor) in cont.iter().enumerate().take(steps) {
                let (logits, toks, conf) = d.trace(m, s.state(), anchor).expect("trace");
                let want = &want_logits[j * bs * v..(j + 1) * bs * v];
                let scale = want.iter().fold(0f32, |a, x| a.max(x.abs())).max(1.0);
                let diff = max_diff(&logits, want);
                assert!(diff < 2e-4 * scale, "{} slots {slots:?} {case} step {j}: draft logits differ by {diff} (scale {scale})", dev.name());
                let cd = max_diff(&conf, &want_conf[j * bs..(j + 1) * bs]);
                assert!(cd < 1e-4, "{} {case} step {j}: confidence differs by {cd}", dev.name());
                for (i, &t) in toks.iter().enumerate() {
                    assert_eq!(t as usize, lapid_nn::argmax(&want[i * v..(i + 1) * v]), "{} {case} step {j} token {i}", dev.name());
                }
                let drafted = m.draft(s.state_mut(), anchor, bs, 0.0).expect("draft");
                assert_eq!(drafted, toks.iter().map(|&t| t as u32).collect::<Vec<_>>());
                llm.step(&mut s, &[anchor]).expect("decode");
            }
        }
    }
    let _ = std::fs::remove_file(&file);
}

/// Feeding a block and rewinding part of it leaves the state of feeding only the kept part:
/// the next logits match a plain run (raw windows, ratio-1 / ratio-2 compressor rows and
/// open groups at odd and even positions, Engram look-back, target features).
#[test]
fn rewind_matches_plain_forwards() {
    let _one = SPEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(file) = tiny_file("rewind") else { return };
    let r = result();
    let prompt = ids(&r["cases"]["long"]["prompt_tokens"]);
    let cont = ids(&r["cases"]["long"]["generated_tokens"]);
    for dev in devices() {
        for slots in [None, Some(2)] {
            let llm = load(&file, &dev, true, slots).expect("load");
            for (cut, block, keep) in [(prompt.len(), 6, 1), (prompt.len(), 6, 4), (prompt.len() - 1, 5, 2), (prompt.len() - 1, 6, 6), (prompt.len(), 1, 0)] {
                let base = &prompt[..cut];
                let tail: Vec<u32> = cont.iter().chain(prompt.iter()).copied().take(block).collect();
                let next = cont[7];
                let mut a = llm.new_session();
                llm.step(&mut a, base).expect("prefill");
                llm.step(&mut a, &tail).expect("block");
                assert!(a.state_mut().truncate(cut + keep), "rewind of {} tokens", block - keep);
                assert_eq!(a.position(), cut + keep);
                let got = llm.step(&mut a, &[next]).expect("after rewind");
                let mut b = llm.new_session();
                llm.step(&mut b, base).expect("prefill");
                if keep > 0 {
                    llm.step(&mut b, &tail[..keep]).expect("kept");
                }
                let want = llm.step(&mut b, &[next]).expect("plain");
                let diff = max_diff(&got, &want);
                assert!(diff < 1e-3, "{} slots {slots:?} cut {cut} block {block} keep {keep}: {diff}", dev.name());
                // the draft windows agree as well
                let m = llm.model();
                let da = m.draft(a.state_mut(), cont[8], 5, 0.0).expect("draft a");
                let db = m.draft(b.state_mut(), cont[8], 5, 0.0).expect("draft b");
                assert_eq!(da, db);
            }
            // Beyond the last forward: refused, state unchanged.
            let mut s = llm.new_session();
            llm.step(&mut s, &prompt).expect("prefill");
            assert!(!s.state_mut().truncate(prompt.len() - 1));
        }
    }
    let _ = std::fs::remove_file(&file);
}

/// Drafts from a known continuation, wrong at chosen draft indices (`wrong`).
struct Oracle<'a> {
    inner: LmSpec<'a, super::Deepseek41Model>,
    want: Vec<u32>,
    prompt_len: usize,
    wrong: Vec<usize>,
    made: usize,
}

impl SpecModel for Oracle<'_> {
    fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.prefill(ids)
    }
    fn draft(&mut self, anchor: u32, n_max: usize) -> Result<Vec<u32>> {
        // keep the real draft running (its window bookkeeping) but replace its tokens
        let _ = self.inner.draft(anchor, n_max)?;
        let k = self.inner.session.position() - self.prompt_len + 1; // tokens emitted so far
        let mut out: Vec<u32> = self.want.iter().skip(k).take(n_max).copied().collect();
        for t in &mut out {
            if self.wrong.contains(&self.made) {
                *t = (*t + 1) % 1000;
            }
            self.made += 1;
        }
        Ok(out)
    }
    fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.inner.verify(ids)
    }
    fn rewind(&mut self, keep: usize) -> Result<()> {
        self.inner.rewind(keep)
    }
}

/// Speculative greedy gives the plain greedy tokens: with the (random-weight) DSpark drafts
/// through `Llm::generate`, and with oracle drafts accepted in full and in part.
#[test]
fn speculative_greedy_matches_plain() {
    let _one = SPEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(file) = tiny_file("greedy") else { return };
    let r = result();
    let n = 40;
    for dev in devices() {
        for slots in [None, Some(2)] {
            let llm = load(&file, &dev, true, slots).expect("load");
            for case in ["short", "long"] {
                let prompt = ids(&r["cases"][case]["prompt_tokens"]);
                let opts = GenerateOptions { max_new_tokens: n, stop_tokens: Some(Vec::new()), ..Default::default() };
                let mut s = llm.new_session();
                let want: Vec<u32> = llm.generate(&mut s, &prompt, opts.clone()).expect("plain").map(|t| t.expect("token").id).collect();
                assert_eq!(&want[..20], &ids(&r["cases"][case]["generated_tokens"])[..20], "{} {case}: plain greedy vs reference", dev.name());
                // the model's own drafts (spec on for the generation only)
                set_spec(Some(SpecConfig { p_min: 0.0, ..SpecConfig::default() }));
                let mut s = llm.new_session();
                let stream = llm.generate(&mut s, &prompt, opts.clone());
                set_spec(None);
                let mut stream = stream.expect("spec stream");
                let got: Vec<u32> = stream.by_ref().map(|t| t.expect("token").id).collect();
                let stats = stream.stats().spec.expect("speculative");
                drop(stream);
                assert_eq!(got, want, "{} slots {slots:?} {case}: speculative greedy differs", dev.name());
                assert!(stats.steps > 0 && stats.drafted > 0, "{stats:?}");
                assert_eq!(s.position(), prompt.len() + n - 1);
                // oracle drafts: all right, then wrong now and then
                for wrong in [vec![], vec![0, 3, 7, 8, 12, 19, 20, 21, 33]] {
                    let mut s = llm.new_session();
                    let inner = LmSpec { model: llm.model(), device: llm.device(), session: &mut s, p_min: 0.0, last: 0 };
                    let oracle = Oracle { inner, want: want.clone(), prompt_len: prompt.len(), wrong: wrong.clone(), made: 0 };
                    let mut g = SpecGen::new(oracle, &prompt, 0, Sampler::greedy(), n, &[], 5);
                    let mut got = Vec::new();
                    while let Some(t) = g.next_token() {
                        got.push(t.expect("token"));
                    }
                    let stats = g.stats();
                    assert_eq!(got, want, "{} slots {slots:?} {case} oracle wrong {wrong:?}", dev.name());
                    assert!(stats.accepted > stats.drafted / 2, "{stats:?}");
                    if wrong.is_empty() {
                        assert_eq!(stats.accepted, stats.drafted, "{stats:?}");
                    }
                    drop(g);
                    assert_eq!(s.position(), prompt.len() + n - 1);
                }
            }
        }
    }
    let _ = std::fs::remove_file(&file);
}
