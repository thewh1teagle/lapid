//! Per-sequence state of a causal LM and the session around it.
//!
//! A step is *build → eval → commit*: while the model builds its graph it stages new state
//! (K/V rows, recurrent / conv windows); the pipeline evaluates the outputs together with
//! [`LmState::staged`] and commits the realized tensors, or discards them on error.
//! [`LmCache`] is the ready-made state: a `KvCache` plus per-layer recurrent slots.
//! [`Session`] adds the consumed tokens, so a new prompt that extends them (the earlier
//! turns of a chat) only prefills the rest; with a rewindable state the longest common
//! prefix is reused.

use std::sync::Arc;

use lapid::{Error, Result, Tensor};
use lapid_nn::KvCache;

use super::steer::Steering;

/// Mutable per-sequence model state.
pub trait LmState: Send {
    /// Tokens consumed (absolute position of the next token).
    fn offset(&self) -> usize;
    /// Tensors staged by the last graph build, evaluated with the step's outputs.
    fn staged(&self) -> Vec<&Tensor>;
    /// Move the staged tensors out (same order as `staged`), leaving whatever `commit` /
    /// `discard` still need, so the step graph can be freed before it runs. The default
    /// clones them (the graph then lives until commit).
    fn take_staged(&mut self) -> Vec<Tensor> {
        self.staged().into_iter().cloned().collect()
    }
    /// Store the realized staged tensors (same order) after `n_tokens` were fed.
    fn commit(&mut self, realized: Vec<Tensor>, n_tokens: usize) -> Result<()>;
    /// Drop staged tensors (committed state unchanged).
    fn discard(&mut self);
    /// Forget everything.
    fn clear(&mut self);
    /// Rewind to the first `len` tokens; false when the state cannot (it is then unchanged).
    fn truncate(&mut self, _len: usize) -> bool {
        false
    }
}

/// KV cache + per-layer recurrent slots (conv windows, SSM / linear-attention states).
/// Rewindable while no slot holds state.
pub struct LmCache {
    pub kv: KvCache,
    slots: Vec<Option<Tensor>>,
    staged_slots: Vec<(usize, Tensor)>,
    pos: usize,
}

impl LmCache {
    pub fn new(kv: KvCache, n_slots: usize) -> LmCache {
        LmCache { kv, slots: vec![None; n_slots], staged_slots: Vec::new(), pos: 0 }
    }

    /// Committed state of slot `i`.
    pub fn slot(&self, i: usize) -> Option<Tensor> {
        self.slots.get(i).cloned().flatten()
    }

    /// Stage the new state of slot `i` (committed with the step).
    pub fn stage_slot(&mut self, i: usize, t: Tensor) {
        self.staged_slots.push((i, t));
    }

    /// Run `f` on a copy of slot `i`'s state and stage what it leaves there (the
    /// `Conv1d::forward_stream` pattern).
    pub fn with_slot<R>(&mut self, i: usize, f: impl FnOnce(&mut Option<Tensor>) -> Result<R>) -> Result<R> {
        let mut st = self.slot(i);
        let r = f(&mut st)?;
        if let Some(t) = st {
            self.stage_slot(i, t);
        }
        Ok(r)
    }
}

impl LmState for LmCache {
    fn offset(&self) -> usize {
        self.pos
    }

    fn staged(&self) -> Vec<&Tensor> {
        let mut v = self.kv.staged();
        v.extend(self.staged_slots.iter().map(|(_, t)| t));
        v
    }

    fn take_staged(&mut self) -> Vec<Tensor> {
        let mut v = self.kv.take_staged();
        for (_, t) in &mut self.staged_slots {
            let ph = Tensor::zeros(t.shape(), t.dtype());
            v.push(std::mem::replace(t, ph));
        }
        v
    }

    fn commit(&mut self, mut realized: Vec<Tensor>, n_tokens: usize) -> Result<()> {
        let n_kv = self.kv.staged().len();
        if realized.len() != n_kv + self.staged_slots.len() {
            self.discard();
            return Err(Error::new("backend", "step: wrong number of realized state tensors"));
        }
        let slots = realized.split_off(n_kv);
        for ((i, _), t) in self.staged_slots.drain(..).zip(slots) {
            if let Some(s) = self.slots.get_mut(i) {
                *s = Some(t);
            }
        }
        self.kv.commit(realized)?;
        self.pos += n_tokens;
        Ok(())
    }

    fn discard(&mut self) {
        self.kv.discard();
        self.staged_slots.clear();
    }

    fn clear(&mut self) {
        self.kv.clear();
        self.slots.iter_mut().for_each(|s| *s = None);
        self.staged_slots.clear();
        self.pos = 0;
    }

    fn truncate(&mut self, len: usize) -> bool {
        if len > self.pos || self.slots.iter().any(Option::is_some) {
            return false;
        }
        if self.kv.offset() == self.pos && self.kv.truncate(len).is_err() {
            return false;
        }
        self.pos = len;
        true
    }
}

/// A sequence being generated: the model state plus the tokens it has consumed.
pub struct Session<S> {
    pub(crate) state: S,
    pub(crate) tokens: Vec<u32>,
    /// (first token, fingerprint) of each image whose embeddings are in the state
    /// (`runtime::vlm`); prefix reuse stops before a different image.
    pub(crate) media: Vec<(usize, u64)>,
    /// Activation steering of every forward; the state was computed under it.
    pub(crate) steering: Option<Arc<Steering>>,
}

impl<S: LmState> Session<S> {
    pub fn new(state: S) -> Session<S> {
        Session { state, tokens: Vec::new(), media: Vec::new(), steering: None }
    }

    /// Tokens fed to the model so far (prompt + generated, without a final unfed token).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Absolute position of the next token (= tokens consumed).
    pub fn position(&self) -> usize {
        self.state.offset()
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// The steering applied by this session's forwards.
    pub fn steering(&self) -> Option<&Arc<Steering>> {
        self.steering.as_ref()
    }

    /// Steer the following forwards; a different steering than the cached state was
    /// computed under starts the session over.
    pub fn set_steering(&mut self, steering: Option<Arc<Steering>>) {
        if self.steering.as_ref().map(|s| s.key()) != steering.as_ref().map(|s| s.key()) {
            self.reset();
        }
        self.steering = steering;
    }

    pub fn reset(&mut self) {
        self.state.clear();
        self.tokens.clear();
        self.media.clear();
    }

    /// Prepare for `prompt`: keep the state for the longest common prefix of the consumed
    /// tokens and `prompt` (at least one prompt token is left to feed), rewinding when the
    /// state supports it; otherwise start over. Returns how many prompt tokens are cached.
    pub fn reuse_prefix(&mut self, prompt: &[u32]) -> usize {
        let common = self.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1));
        if common > 0 && common == self.tokens.len() {
            return common;
        }
        if common > 0 && self.state.truncate(common) {
            self.tokens.truncate(common);
            self.media.retain(|&(start, _)| start < common);
            return common;
        }
        self.reset();
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_reuse() {
        let mut s = Session::new(LmCache::new(KvCache::new(0), 0));
        assert_eq!(s.reuse_prefix(&[1, 2, 3]), 0);
        s.tokens = vec![1, 2, 3];
        s.state.pos = 3;
        assert_eq!(s.reuse_prefix(&[1, 2, 3, 4]), 3);
        // Diverging prompt: rewound to the common prefix (no slots in use).
        assert_eq!(s.reuse_prefix(&[1, 2, 9, 9]), 2);
        assert_eq!((s.tokens(), s.position()), (&[1u32, 2][..], 2));
        // Same prompt again: one token is always left to feed.
        assert_eq!(s.reuse_prefix(&[1, 2]), 1);
        // Another steering: the state is not reused.
        let st = super::super::steer::Steering::new(&lapid::Device::cpu(), &[(&steer_file(), 1.0)]).unwrap();
        s.set_steering(st.clone());
        assert_eq!(s.tokens().len(), 0);
        s.tokens = vec![1, 2];
        s.state.pos = 2;
        s.set_steering(st);
        assert_eq!(s.reuse_prefix(&[1, 2, 3]), 2);
        s.set_steering(None);
        assert_eq!(s.reuse_prefix(&[1, 2, 3]), 0);
    }

    fn steer_file() -> super::super::steer::SteerFile {
        use super::super::steer::*;
        let e = SteerEntry { hook: Hook::Resid, layer: 0, op: SteerOp::Add, scale: 1.0, vector: vec![1.0; 4] };
        SteerFile { name: "t".into(), hidden_size: 4, entries: vec![e], ..Default::default() }
    }
}
