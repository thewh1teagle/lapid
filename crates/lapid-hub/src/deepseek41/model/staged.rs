//! The forward with the routed experts in an [`ExpertCache`]: the router's choice must be
//! on the host before the expert matmuls of a layer, so the network is evaluated in one
//! graph per layer, cut after each router:
//!
//! ```text
//! eval i:  experts(i - 1) + shared(i - 1) + mHC post     (slots made resident before)
//!          engram(i), attention(i), mHC, ffn norm, router(i)  ->  ids, weights (host ids)
//! cache.ensure(layer i, ids)                              (loads the misses)
//! last:    experts(L - 1) ..., norm, head                 ->  logits
//! ```
//! Everything a later graph reads (streams, mixes, layer states, shared compressed rows,
//! taps) is realized at each cut, so no graph is ever evaluated twice and no graph reads
//! slots after they are rewritten. One forward at a time (the slots are shared).

use std::time::{Duration, Instant};

use lapid::{DType, Device, Error, Op, Result, Tensor};
use lapid_nn::llm::expert_cache::ExpertCache;
use lapid_nn::llm::expert_stats::ForwardTimes;
use lapid_nn::llm::hyper::{collapse, expand};

use super::attention::Shared;
use super::{Deepseek41Model, State, stage};
use crate::runtime::{Rows, Taps};

/// Realize every lazy tensor of `ts` in one evaluation (leaves stay as they are).
fn realize(device: &Device, ts: Vec<&mut Tensor>) -> Result<()> {
    let lazy: Vec<&mut Tensor> = ts.into_iter().filter(|t| !matches!(t.op(), Op::Device(_) | Op::Host(_))).collect();
    if lazy.is_empty() {
        return Ok(());
    }
    let refs: Vec<&Tensor> = lazy.iter().map(|t| &**t).collect();
    let out = device.eval(&refs)?;
    if out.len() != lazy.len() {
        return Err(Error::new("backend", "deepseek41: eval returned the wrong number of outputs"));
    }
    for (t, r) in lazy.into_iter().zip(out) {
        *t = r;
    }
    Ok(())
}

/// The distinct non-negative ids of `ids`, in first-seen order.
fn distinct(ids: &[i32]) -> Vec<u32> {
    let mut seen = std::collections::HashSet::new();
    ids.iter().filter_map(|&e| u32::try_from(e).ok()).filter(|&e| seen.insert(e)).collect()
}

/// The MoE of a layer whose router ran: its input, host expert ids, weights and the mHC
/// mixes that take its output.
struct Pending {
    layer: usize,
    x: Tensor,
    ids: Tensor,
    w: Tensor,
    post: Tensor,
    comb: Tensor,
}

impl Deepseek41Model {
    pub(super) fn forward_cached(&self, cache: &ExpertCache, ids: &[u32], engram_rows: &[Option<Tensor>], st: &mut State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let (t0, before) = (Instant::now(), cache.stats());
        let mut eval = Duration::ZERO;
        let dev = &self.device;
        let (n, d, hc) = (ids.len(), self.cfg.hidden_size, self.cfg.hc_mult);
        let pos = st.pos;
        // Taps waiting for the next evaluation.
        let mut pend_taps: Vec<(String, Tensor)> = Vec::new();
        let on = taps.enabled();
        let tap = |list: &mut Vec<(String, Tensor)>, name: String, t: &Tensor| {
            if on {
                list.push((name, t.clone()));
            }
        };
        let e = self.embed.lookup(ids)?.cast(DType::F32)?.reshape(&[n, d])?;
        tap(&mut pend_taps, "embed".into(), &e);
        let mut x = e.unsqueeze(1)?.expand(&[n, hc, d])?.contiguous()?;
        let mut pre: Option<Tensor> = None;
        let mut shared = Shared::default();
        let mut next = Vec::with_capacity(self.layers.len());
        let mut pending: Option<(Pending, Vec<i32>)> = None;
        let finish = |p: Pending, ids: &[i32], x: &Tensor| -> Result<Tensor> {
            let m = self.layers[p.layer].moe.forward_cached(cache, &p.x, ids, &p.w)?;
            expand(&m, x, &p.post, &p.comb)
        };
        let speculate = cache.speculative();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((p, host_ids)) = pending.take() {
                let li = p.layer;
                x = finish(p, &host_ids, &x)?;
                tap(&mut pend_taps, format!("layers.{li}"), &x.reshape(&[n, hc * d])?);
            }
            if let Some(eg) = &layer.engram {
                x = Self::engram(eg, &x, engram_rows[i].as_ref())?;
            }
            let am = layer.attn_hc.mix(&x)?;
            let xa = match &pre {
                Some(p) => collapse(p, &x)?,
                None => x.slice(1, 0, 1)?.reshape(&[n, d])?,
            };
            let (a, s) = layer.attn.forward(&layer.attn_norm.forward(&xa)?, pos, &st.layers[i], &mut shared)?;
            next.push(s);
            tap(&mut pend_taps, format!("layers.{i}.op"), &a);
            x = expand(&a, &x, &am.post, &am.comb)?;
            let fm = layer.ffn_hc.mix(&x)?;
            let c = collapse(&am.pre, &x)?;
            let xn = layer.ffn_norm.forward(&c)?.cast(DType::F32)?;
            let (route_ids, w) = layer.moe.route(&xn)?;
            // Guess of the next layer's experts: its router on this layer's FFN input (the
            // streams change little per layer), read in the background while this layer's
            // experts load and the device runs.
            let mut guess = match self.layers.get(i + 1) {
                Some(nl) if speculate => Some(nl.moe.route(&nl.ffn_norm.forward(&c)?.cast(DType::F32)?)?.0),
                _ => None,
            };
            let mut p = Pending { layer: i, x: xn, ids: route_ids, w, post: fm.post, comb: fm.comb };
            let mut fm_pre = fm.pre;
            {
                let mut cut: Vec<&mut Tensor> = vec![&mut p.ids, &mut p.w, &mut p.x, &mut p.post, &mut p.comb, &mut x, &mut fm_pre];
                if let Some(s) = next.last_mut() {
                    cut.extend(s.slots_mut().into_iter().flatten());
                }
                cut.extend(shared.tensors_mut());
                cut.extend(pend_taps.iter_mut().map(|(_, t)| t));
                cut.extend(guess.as_mut());
                let t = Instant::now();
                realize(dev, cut)?;
                eval += t.elapsed();
            }
            for (name, t) in pend_taps.drain(..) {
                taps.tap(name, &t);
            }
            pre = Some(fm_pre);
            let t = Instant::now();
            let host_ids = dev.to_vec_i32(&p.ids)?;
            let guess = guess.map(|g| dev.to_vec_i32(&g)).transpose()?;
            eval += t.elapsed();
            if let Some(g) = guess {
                cache.prefetch(i + 1, &distinct(&g));
            }
            pending = Some((p, host_ids));
        }
        let pre = pre.ok_or_else(|| Error::new("internal", "deepseek41: no layers"))?;
        let mut logits = None;
        if rows != Rows::None || taps.enabled() {
            if let Some((p, host_ids)) = pending.take() {
                let li = p.layer;
                x = finish(p, &host_ids, &x)?;
                tap(&mut pend_taps, format!("layers.{li}"), &x.reshape(&[n, hc * d])?);
            }
            let (xs, ps) = match rows {
                Rows::Last if n > 1 => (x.slice(0, n - 1, n)?, pre.slice(0, n - 1, n)?),
                _ => (x, pre),
            };
            let h = self.norm.forward(&collapse(&ps, &xs)?)?;
            if rows == Rows::All {
                tap(&mut pend_taps, "norm".into(), &h);
            }
            let mut out = if rows == Rows::None { None } else { Some(self.head.forward(&h)?) };
            let mut cut: Vec<&mut Tensor> = out.iter_mut().collect();
            cut.extend(pend_taps.iter_mut().map(|(_, t)| t));
            let t = Instant::now();
            realize(dev, cut)?;
            eval += t.elapsed();
            for (name, t) in pend_taps.drain(..) {
                taps.tap(name, &t);
            }
            logits = out;
        }
        stage(st, next, ids, self.hash.as_ref());
        self.stats.record(cache, n, before, ForwardTimes { wall: t0.elapsed(), eval });
        Ok(logits)
    }
}
