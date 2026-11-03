//! Speculative decoding with a model's built-in draft head (DeepSeek V4.1's DSpark;
//! llama.cpp `--spec-type draft-dspark`).
//!
//! One step: the draft head proposes up to `n_max` tokens after the last sampled token (the
//! *anchor*, not yet fed); the target model evaluates `[anchor, drafts..]` in one forward
//! (logits of every position); drafts are accepted left to right, the first rejected
//! position is resampled, and an all-accepted block adds the next token for free. The
//! target state then keeps only the fed tokens that were accepted ([`SpecModel::rewind`]).
//!
//! Drafts are deterministic (the head's greedy chain), so a draft token `d` with target
//! probability `p(d)` is accepted with probability `p(d)` and a rejection samples from `p`
//! without `d` (renormalized): standard speculative sampling with a one-hot draft
//! distribution, which leaves the output distribution exactly the target's. Greedy: the
//! drafts are accepted while they equal the target's argmax, so the output is the
//! non-speculative greedy output (up to batch-shape rounding of the verify forward).
//!
//! Process-wide setting ([`set_spec`]; CLI `--spec-type draft-dspark`), off by default.
//! `LAPID_SPEC_STATS=1` logs acceptance per generation.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::RwLock;

use lapid::{Device, Error, Result};
use lapid_nn::Sampler;
use serde::Serialize;

use super::causal_lm::{CausalLm, Rows, Taps, eval_commit_owned, step};
use super::session::{LmState, Session};

/// Speculative decoding settings (llama.cpp `--spec-draft-n-max` / `--spec-draft-p-min`).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecConfig {
    /// Draft tokens per step at most (clamped to the head's block size).
    pub n_max: usize,
    /// Stop the draft at the first position whose confidence is below this (0 = off).
    pub p_min: f32,
    /// Draft head weights in a separate `.lapid` (`lapid convert --dspark`), when the
    /// model file has none.
    pub draft_model: Option<PathBuf>,
}

impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig { n_max: 5, p_min: 0.0, draft_model: None }
    }
}

static SPEC: RwLock<Option<SpecConfig>> = RwLock::new(None);

/// Turn speculative decoding on (for models loaded and generations started afterwards) or off.
pub fn set_spec(cfg: Option<SpecConfig>) {
    *SPEC.write().unwrap_or_else(|e| e.into_inner()) = cfg;
}

/// The process-wide speculative decoding settings (`None`: off).
pub fn spec() -> Option<SpecConfig> {
    SPEC.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Acceptance counts of one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SpecStats {
    /// Target forwards after the prompt (verify steps).
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Tokens emitted by the verify steps.
    pub emitted: usize,
}

impl SpecStats {
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 { 0.0 } else { self.accepted as f64 / self.drafted as f64 }
    }

    pub fn tokens_per_step(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.emitted as f64 / self.steps as f64 }
    }
}

/// What the speculative loop needs from a model and its sequence state.
pub(crate) trait SpecModel {
    /// Feed prompt tokens; logits of the last position.
    fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>>;
    /// Up to `n_max` draft tokens after `anchor` (sampled, not yet fed).
    fn draft(&mut self, anchor: u32, n_max: usize) -> Result<Vec<u32>>;
    /// Feed `ids`; logits of every position.
    fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>>;
    /// Keep only the first `keep` tokens of the last [`verify`](Self::verify).
    fn rewind(&mut self, keep: usize) -> Result<()>;
}

impl<T: SpecModel + ?Sized> SpecModel for Box<T> {
    fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        (**self).prefill(ids)
    }
    fn draft(&mut self, anchor: u32, n_max: usize) -> Result<Vec<u32>> {
        (**self).draft(anchor, n_max)
    }
    fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        (**self).verify(ids)
    }
    fn rewind(&mut self, keep: usize) -> Result<()> {
        (**self).rewind(keep)
    }
}

/// A causal LM's session as a [`SpecModel`]: drafts from [`CausalLm::draft`], verify
/// forwards of every row, rewinds through [`LmState::truncate`].
pub(crate) struct LmSpec<'a, M: CausalLm> {
    pub model: &'a M,
    pub device: &'a Device,
    pub session: &'a mut Session<M::State>,
    pub p_min: f32,
    /// Tokens fed by the last verify.
    pub last: usize,
}

impl<M: CausalLm> SpecModel for LmSpec<'_, M> {
    fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let logits = step(self.model, self.device, ids, &mut self.session.state, self.session.steering.as_ref())?;
        self.session.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    fn draft(&mut self, anchor: u32, n_max: usize) -> Result<Vec<u32>> {
        self.model.draft(&mut self.session.state, anchor, n_max, self.p_min)
    }

    fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let st = &mut self.session.state;
        let mut taps = Taps::off().with_steering(self.session.steering.clone());
        let out = match self.model.forward(ids, st, Rows::All, &mut taps) {
            Ok(o) => o.ok_or_else(|| Error::new("internal", "verify forward without logits"))?,
            Err(e) => {
                st.discard();
                return Err(e);
            }
        };
        let realized = eval_commit_owned(self.device, vec![out], st, ids.len())?;
        let flat = self.device.to_vec_f32(realized.first().ok_or_else(|| Error::new("backend", "no logits realized"))?)?;
        self.session.tokens.extend_from_slice(ids);
        self.last = ids.len();
        let vocab = flat.len() / ids.len().max(1);
        Ok(flat.chunks(vocab.max(1)).map(<[f32]>::to_vec).collect())
    }

    fn rewind(&mut self, keep: usize) -> Result<()> {
        let drop = self.last.saturating_sub(keep);
        let len = self.session.state.offset() - drop;
        if !self.session.state.truncate(len) {
            return Err(Error::new("internal", format!("speculative decoding: the state cannot rewind {drop} tokens")));
        }
        self.session.tokens.truncate(self.session.tokens.len() - drop);
        self.last = keep;
        Ok(())
    }
}

/// Accept `drafts` against the target logits `rows` (`drafts.len() + 1` rows), extending
/// `history` (prompt + generated so far, for the repetition penalty). Returns the emitted
/// tokens: the accepted drafts, then a resampled or bonus token.
pub(crate) fn accept(sampler: &mut Sampler, rows: &[Vec<f32>], drafts: &[u32], history: &mut Vec<u32>) -> Vec<u32> {
    let mut out = Vec::with_capacity(drafts.len() + 1);
    for (i, row) in rows.iter().enumerate() {
        let mut dist = sampler.distribution(row, history);
        let Some(&d) = drafts.get(i) else {
            let t = sampler.pick(&dist);
            history.push(t);
            out.push(t);
            break;
        };
        let pd = dist.iter().find(|e| e.0 == d).map_or(0.0, |e| e.1);
        if pd >= 1.0 || (pd > 0.0 && sampler.uniform() < pd) {
            history.push(d);
            out.push(d);
            continue;
        }
        // Rejected: the rest of the distribution (`max(0, p - q)` for a one-hot q).
        if pd > 0.0 {
            dist.retain(|e| e.0 != d);
            let total: f64 = dist.iter().map(|e| e.1).sum();
            dist.iter_mut().for_each(|e| e.1 /= total);
        }
        let t = sampler.pick(&dist);
        history.push(t);
        out.push(t);
        break;
    }
    out
}

/// Generated tokens of a speculative generation, one at a time (the `TokenStream` source).
pub(crate) struct SpecGen<S> {
    model: S,
    sampler: Sampler,
    tokens: Vec<u32>,
    prompt_len: usize,
    cached: usize,
    stop: Vec<u32>,
    remaining: usize,
    queue: VecDeque<u32>,
    started: bool,
    done: bool,
    n_max: usize,
    stats: SpecStats,
    logged: bool,
}

impl<S: SpecModel> SpecGen<S> {
    /// After `prompt` (its first `cached` tokens already in the state), up to `max_new` tokens.
    pub fn new(model: S, prompt: &[u32], cached: usize, sampler: Sampler, max_new: usize, stop: &[u32], n_max: usize) -> SpecGen<S> {
        SpecGen {
            model,
            sampler,
            tokens: prompt.to_vec(),
            prompt_len: prompt.len(),
            cached,
            stop: stop.to_vec(),
            remaining: max_new,
            queue: VecDeque::new(),
            started: false,
            done: false,
            n_max,
            stats: SpecStats::default(),
            logged: false,
        }
    }

    pub fn generated(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    /// Queue `out` (cut after a stop token and at the budget); the number queued.
    fn emit(&mut self, mut out: Vec<u32>) -> usize {
        if let Some(i) = out.iter().position(|t| self.stop.contains(t)) {
            out.truncate(i + 1);
            self.done = true;
        }
        out.truncate(self.remaining);
        self.remaining -= out.len();
        let n = out.len();
        self.tokens.extend_from_slice(&out);
        self.queue.extend(out);
        n
    }

    /// One draft + verify step; the emitted tokens are queued.
    fn step(&mut self) -> Result<()> {
        let Some(&anchor) = self.tokens.last() else { return Ok(()) };
        // Every accepted draft and the token after them must fit the budget.
        let n = self.n_max.min(self.remaining.saturating_sub(1));
        let mut drafts = if n > 0 { self.model.draft(anchor, n)? } else { Vec::new() };
        drafts.truncate(n);
        let mut ids = Vec::with_capacity(drafts.len() + 1);
        ids.push(anchor);
        ids.extend_from_slice(&drafts);
        let rows = self.model.verify(&ids)?;
        let mut history = self.tokens.clone();
        let out = accept(&mut self.sampler, &rows, &drafts, &mut history);
        self.stats.steps += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += out.len() - 1;
        let n = self.emit(out);
        self.stats.emitted += n;
        // Fed: the anchor and every draft; kept: the anchor and the emitted tokens but the
        // last (the next anchor).
        if n < ids.len() {
            self.model.rewind(n)?;
        }
        Ok(())
    }

    fn log(&mut self) {
        if self.logged || self.stats.steps == 0 {
            return;
        }
        self.logged = true;
        if std::env::var("LAPID_SPEC_STATS").is_ok_and(|v| !v.is_empty() && v != "0") {
            let s = self.stats;
            lapid::log_line(&format!(
                "[lapid] spec: {} steps, {}/{} drafts accepted ({:.1}%), {:.2} tokens/step",
                s.steps,
                s.accepted,
                s.drafted,
                100.0 * s.acceptance(),
                s.tokens_per_step()
            ));
        }
    }

    pub fn next_token(&mut self) -> Option<Result<u32>> {
        loop {
            if let Some(t) = self.queue.pop_front() {
                return Some(Ok(t));
            }
            if self.done || self.remaining == 0 {
                self.log();
                return None;
            }
            let r = if self.started {
                self.step()
            } else {
                self.started = true;
                self.model.prefill(&self.tokens[self.cached.min(self.tokens.len())..]).map(|logits| {
                    let t = self.sampler.sample(&logits, &self.tokens);
                    self.emit(vec![t]);
                })
            };
            if let Err(e) = r {
                self.done = true;
                self.queue.clear();
                return Some(Err(e));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_nn::SamplingParams;

    /// Toy LM over 7 tokens: next = (sum of fed tokens) % 7 as soft logits (one forward per
    /// fed token, like a real state), with a rewindable fed list. The draft is right with
    /// the given pattern of mistakes.
    struct Toy {
        fed: Vec<u32>,
        last_verify: usize,
        wrong_every: usize,
        drafts_made: usize,
    }

    fn logits_after(fed: &[u32]) -> Vec<f32> {
        let mut l = vec![0.0f32; 7];
        let s = fed.iter().sum::<u32>() % 7;
        l[s as usize] = 3.0;
        l[((s + 3) % 7) as usize] = 2.0;
        l
    }

    impl SpecModel for Toy {
        fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
            self.fed.extend_from_slice(ids);
            Ok(logits_after(&self.fed))
        }
        fn draft(&mut self, anchor: u32, n_max: usize) -> Result<Vec<u32>> {
            let mut seq = self.fed.clone();
            seq.push(anchor);
            let mut out = Vec::new();
            for _ in 0..n_max {
                self.drafts_made += 1;
                let mut t = lapid_nn::argmax(&logits_after(&seq)) as u32;
                if self.wrong_every > 0 && self.drafts_made.is_multiple_of(self.wrong_every) {
                    t = (t + 1) % 7;
                }
                seq.push(t);
                out.push(t);
            }
            Ok(out)
        }
        fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
            let mut rows = Vec::new();
            for &t in ids {
                self.fed.push(t);
                rows.push(logits_after(&self.fed));
            }
            self.last_verify = ids.len();
            Ok(rows)
        }
        fn rewind(&mut self, keep: usize) -> Result<()> {
            assert!(keep <= self.last_verify);
            let n = self.fed.len() - (self.last_verify - keep);
            self.fed.truncate(n);
            Ok(())
        }
    }

    fn plain(prompt: &[u32], max_new: usize, stop: &[u32]) -> Vec<u32> {
        let toy = |new: &[u32], fed: &mut Vec<u32>| {
            fed.extend_from_slice(new);
            Ok(logits_after(fed))
        };
        let mut fed = Vec::new();
        lapid_nn::generate::generate(prompt, |n: &[u32]| toy(n, &mut fed), Sampler::greedy(), max_new, stop).map(|t| t.unwrap()).collect()
    }

    fn spec_run(prompt: &[u32], max_new: usize, stop: &[u32], wrong_every: usize, n_max: usize, sampler: Sampler) -> (Vec<u32>, SpecStats, Vec<u32>) {
        let toy = Toy { fed: Vec::new(), last_verify: 0, wrong_every, drafts_made: 0 };
        let mut g = SpecGen::new(toy, prompt, 0, sampler, max_new, stop, n_max);
        let mut out = Vec::new();
        while let Some(t) = g.next_token() {
            out.push(t.unwrap());
        }
        let fed = g.model.fed.clone();
        (out, g.stats(), fed)
    }

    #[test]
    fn greedy_matches_plain_and_rewinds() {
        for wrong_every in [0, 1, 2, 3, 7] {
            for n_max in [1, 3, 5] {
                for max_new in [1, 2, 9, 30] {
                    let want = plain(&[1, 2], max_new, &[]);
                    let (got, stats, fed) = spec_run(&[1, 2], max_new, &[], wrong_every, n_max, Sampler::greedy());
                    assert_eq!(got, want, "wrong_every {wrong_every} n_max {n_max}");
                    // the state holds the prompt and every emitted token but the last
                    let mut expect_fed = vec![1, 2];
                    expect_fed.extend_from_slice(&want[..want.len() - 1]);
                    assert_eq!(fed, expect_fed);
                    if wrong_every == 0 && max_new == 30 {
                        assert_eq!(stats.accepted, stats.drafted);
                    }
                    if wrong_every == 1 {
                        assert_eq!(stats.accepted, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn stops_inside_an_accepted_block() {
        let want = plain(&[1, 2], 40, &[5]);
        assert_eq!(want.last(), Some(&5));
        let (got, _, fed) = spec_run(&[1, 2], 40, &[5], 0, 5, Sampler::greedy());
        assert_eq!(got, want);
        assert_eq!(fed.len(), 2 + want.len() - 1);
    }

    /// Sampling keeps the target distribution: the first sampled token after an accepted /
    /// rejected draft has the target's frequencies.
    #[test]
    fn sampling_preserves_distribution() {
        // target p = softmax([3, 2] at s, s+3) ~ 0.731 / 0.269 (others e^0 = 1 each: 5 more)
        let row = logits_after(&[0]);
        let p = lapid_nn::sampling::softmax(&row);
        for draft in [0u32, 3, 1] {
            let mut counts = [0usize; 7];
            let mut s = Sampler::new(SamplingParams { temperature: 1.0, seed: 5, ..Default::default() });
            let n = 40_000;
            for _ in 0..n {
                let out = accept(&mut s, std::slice::from_ref(&row), &[], &mut Vec::new());
                let _ = out;
                let out = accept(&mut s, &[row.clone(), row.clone()], &[draft], &mut Vec::new());
                counts[out[0] as usize] += 1;
            }
            for (t, &c) in counts.iter().enumerate() {
                let f = c as f64 / n as f64;
                assert!((f - p[t] as f64).abs() < 0.01, "draft {draft}: token {t} {f} vs {}", p[t]);
            }
        }
    }
}
