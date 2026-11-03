//! The DSpark draft head of DeepSeek V4.1 (upstream `DSparkBlock` / `forward_spec`, stored
//! as `mtp.{0..S}`; llama.cpp `dflash.cpp` `graph_dsv41` + `build_dspark_markov_head`).
//!
//! ```text
//! f_p    = concat_t mean_hc(streams entering target layer t at position p)    (target features)
//! main_x = rmsnorm(main_proj f_p)                                             (mtp.0)
//! window = per stage, rope(kv_norm(wkv main_x)) of the last `sliding_window` positions
//! block  = embed([anchor, noise, ..]) at positions pos..pos + b, repeated to hc streams
//! per stage: the backbone block (lagged mHC, MoE of 128 experts) with attention over
//!            window ++ block (not causal), no compression, plain RoPE
//! h      = collapse(pre, x);  logits_i = head(norm(h_i)) + markov_head(markov_embed(t_{i-1}))
//! t_i    = argmax(logits_i)   (t_{-1} = anchor: a greedy chain, as llama.cpp)
//! conf_i = sigmoid(proj . [h_i ; markov_embed(t_{i-1})])
//! ```
//! The draft keeps `t_0..` up to the first confidence below `p_min`. One evaluation per
//! draft: the window update, the stages and the chain.

use lapid::{DType, Device, Error, LapidFile, Residency, Result, Tensor};
use lapid_nn::llm::hyper::{SinkhornHc, collapse, expand};
use lapid_nn::{Linear, Loader, RmsNorm};

use super::attention::Attention;
use super::moe::Moe;
use super::{Deepseek41Model, REWIND_MAX, State, load};
use crate::deepseek41::config::{Deepseek41Config, DsparkConfig};
use crate::runtime::spec::spec;

const FEATURE_ROWS_MAX: usize = 1024;

/// Per-sequence draft state: target features of the latest positions (ending at the state's
/// position) and each stage's window rows (ending at `ring_end`).
#[derive(Clone, Default)]
pub(crate) struct DraftState {
    pub(super) feats: Vec<Tensor>,
    ring: Vec<Tensor>,
    ring_end: usize,
}

fn rows(ts: &[Tensor]) -> usize {
    ts.iter().map(|t| t.shape()[0]).sum()
}

impl DraftState {
    /// Features of the positions a forward fed (its last rows).
    pub fn push_features(&mut self, f: Tensor) {
        self.feats.push(f);
        while self.feats.len() > 1 && rows(&self.feats[1..]) >= FEATURE_ROWS_MAX {
            self.feats.remove(0);
        }
    }

    /// Forget the last `drop` positions (now `len`).
    pub fn truncate(&mut self, mut drop: usize, len: usize) -> Result<()> {
        while drop > 0 {
            let Some(last) = self.feats.pop() else { break };
            let n = last.shape()[0];
            if n > drop {
                self.feats.push(last.slice(0, 0, n - drop)?);
                break;
            }
            drop -= n;
        }
        if self.ring_end > len {
            let cut = self.ring_end - len;
            for r in &mut self.ring {
                let m = r.shape()[0];
                *r = r.slice(0, 0, m.saturating_sub(cut))?;
            }
            self.ring_end = len;
        }
        Ok(())
    }
}

struct Stage {
    attn_hc: SinkhornHc,
    ffn_hc: SinkhornHc,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn: Attention,
    moe: Moe,
}

pub(crate) struct Dspark {
    pub targets: Vec<usize>,
    stages: Vec<Stage>,
    main_proj: Linear,
    main_norm: RmsNorm,
    norm: RmsNorm,
    /// Markov head: previous token -> `[R]` embedding -> logit bias `[V]`.
    markov_embed: Tensor,
    markov_head: Linear,
    /// Confidence projection `[1, D + R]` f32.
    conf: Tensor,
    pub block_size: usize,
    noise: u32,
    window: usize,
    /// A separate draft file the weights may map.
    _file: Option<LapidFile>,
}

/// The draft head when speculative decoding is on and the config has one: from the model
/// file, else from the configured draft file (`lapid convert --dspark`).
pub(super) fn load(l: &Loader<'_>, cfg: &Deepseek41Config) -> Result<Option<Dspark>> {
    let (Some(dc), Some(sc)) = (&cfg.dspark, spec()) else { return Ok(None) };
    if l.contains("mtp.0.main_proj.weight") {
        return load_from(l, cfg, dc).map(Some);
    }
    let Some(path) = &sc.draft_model else {
        lapid::log_line("[lapid] deepseek41: this file has no DSpark draft (mtp.*): speculative decoding off (convert again, or `lapid convert --dspark` and --spec-draft-model)");
        return Ok(None);
    };
    let file = LapidFile::open(path)?;
    if l.device.name() != "cpu" {
        file.set_residency(Residency::Upload);
    }
    let mut d = load_from(&Loader::new(&file, l.device, l.precision), cfg, dc)?;
    d._file = Some(file);
    Ok(Some(d))
}

fn load_from(l: &Loader<'_>, cfg: &Deepseek41Config, dc: &DsparkConfig) -> Result<Dspark> {
    let (d, eps, hc) = (cfg.hidden_size, cfg.norm_eps, cfg.hc_mult);
    let mut stages = Vec::with_capacity(dc.stages);
    for s in 0..dc.stages {
        let p = format!("mtp.{s}");
        let conn = |kind: &str| SinkhornHc::load(l, &format!("{p}.hc_{kind}"), [hc, d, cfg.hc_sinkhorn_iters], cfg.hc_eps, eps);
        let (mut moe, stacks) = load::moe(l, cfg, &format!("{p}.ffn"), (dc.n_experts, dc.top_k), usize::MAX)?;
        load::upload_experts(l, &mut moe, stacks)?;
        stages.push(Stage {
            attn_hc: conn("attn")?,
            ffn_hc: conn("ffn")?,
            attn_norm: l.rms_norm_checked(&format!("{p}.attn_norm"), d, eps)?,
            ffn_norm: l.rms_norm_checked(&format!("{p}.ffn_norm"), d, eps)?,
            attn: load::attention(l, cfg, &format!("{p}.attn"), None)?,
            moe,
        });
    }
    let last = format!("mtp.{}", dc.stages - 1);
    let (v, r) = (cfg.vocab_size, dc.markov_rank);
    Ok(Dspark {
        targets: dc.targets.clone(),
        stages,
        main_proj: l.linear_checked("mtp.0.main_proj", d, dc.targets.len() * d, false)?,
        main_norm: l.rms_norm_checked("mtp.0.main_norm", d, eps)?,
        norm: l.rms_norm_checked(&format!("{last}.norm"), d, eps)?,
        markov_embed: l.tensor_checked(&format!("{last}.markov_head.embed.weight"), &[v, r])?,
        markov_head: l.linear_checked(&format!("{last}.markov_head.head"), v, r, false)?,
        conf: l.tensor_checked(&format!("{last}.confidence_head.proj.weight"), &[1, d + r])?.cast(DType::F32)?,
        block_size: dc.block_size,
        noise: dc.noise_token,
        window: cfg.sliding_window,
        _file: None,
    })
}

/// Realize `ts` in one evaluation.
fn eval(device: &Device, ts: &[&Tensor]) -> Result<Vec<Tensor>> {
    let out = device.eval(ts)?;
    if out.len() != ts.len() {
        return Err(Error::new("backend", "dspark: eval returned the wrong number of outputs"));
    }
    Ok(out)
}

impl Dspark {
    /// Target feature rows a forward keeps (the window; a rewindable forward's own rows).
    pub fn feature_rows(&self) -> usize {
        self.window.max(REWIND_MAX)
    }

    /// Each stage's window after adding the features not in it yet (lazy), and its end.
    fn windows(&self, st: &State) -> Result<Vec<Option<Tensor>>> {
        let ds = &st.draft;
        let mut ring: Vec<Option<Tensor>> = (0..self.stages.len()).map(|s| ds.ring.get(s).cloned()).collect();
        let (pos, r) = (st.pos, rows(&ds.feats));
        let first = pos.checked_sub(r).ok_or_else(|| Error::new("internal", "dspark: more features than positions"))?;
        let from = ds.ring_end.max(first);
        if from >= pos {
            return Ok(ring);
        }
        let f = Tensor::cat(&ds.feats.iter().collect::<Vec<_>>(), 0)?.slice(0, from - first, r)?.cast(DType::F32)?;
        // Target features carry massive activations (~3e5 on some tokens), beyond f16 in
        // batched matmuls. main_norm(main_proj(f)) does not depend on each row's scale, so
        // project rows scaled to a max of 1.
        let peak = f.unary(lapid::UnaryOp::Abs)?.reduce(lapid::ReduceOp::Max, 1, true)?.add_scalar(1e-30)?;
        let main_x = self.main_norm.forward(&self.main_proj.forward(&f.div(&peak)?)?)?.cast(DType::F32)?;
        let positions: Vec<usize> = (from..pos).collect();
        for (s, stage) in self.stages.iter().enumerate() {
            let k = stage.attn.main_kv(&main_x, &positions)?;
            let all = match &ring[s] {
                Some(old) => Tensor::cat(&[old, &k], 0)?,
                None => k,
            };
            let m = all.shape()[0];
            let w = self.window;
            ring[s] = Some(if m > w { all.slice(0, m - w, m)? } else { all });
        }
        Ok(ring)
    }

    /// The lazy draft of a `b`-token block after `anchor` (at position `st.pos`).
    fn graph(&self, m: &Deepseek41Model, st: &State, anchor: u32, b: usize) -> Result<DraftGraph> {
        let (d, hc, pos) = (m.cfg.hidden_size, m.cfg.hc_mult, st.pos);
        let ring = self.windows(st)?;
        let mut ids = vec![self.noise; b];
        ids[0] = anchor;
        let e = m.embed.lookup(&ids)?.cast(DType::F32)?.reshape(&[b, d])?;
        let mut x = e.unsqueeze(1)?.expand(&[b, hc, d])?.contiguous()?;
        let mut pre: Option<Tensor> = None;
        for (s, stage) in self.stages.iter().enumerate() {
            let am = stage.attn_hc.mix(&x)?;
            let xa = match &pre {
                Some(p) => collapse(p, &x)?,
                None => x.slice(1, 0, 1)?.reshape(&[b, d])?,
            };
            let ain = stage.attn_norm.forward(&xa)?;
            let a = stage.attn.forward_block(&ain, pos, ring[s].as_ref())?;
            x = expand(&a, &x, &am.post, &am.comb)?;
            let fm = stage.ffn_hc.mix(&x)?;
            let xn = stage.ffn_norm.forward(&collapse(&am.pre, &x)?)?.cast(DType::F32)?;
            let y = stage.moe.forward(&xn)?;
            x = expand(&y, &x, &fm.post, &fm.comb)?;
            pre = Some(fm.pre);
        }
        let pre = pre.ok_or_else(|| Error::new("internal", "dspark: no stages"))?;
        let h = collapse(&pre, &x)?.cast(DType::F32)?; // [b, D]
        let base = m.head.forward(&self.norm.forward(&h)?)?.cast(DType::F32)?; // [b, V]
        // The Markov chain: each position's bias comes from the token before it.
        let mut prev = Tensor::from_i32(&[anchor as i32], &[1])?;
        let (mut toks, mut confs, mut logits) = (Vec::with_capacity(b), Vec::with_capacity(b), Vec::with_capacity(b));
        for i in 0..b {
            let me = self.markov_embed.gather(&prev)?.cast(DType::F32)?.reshape(&[1, self.markov_embed.shape()[1]])?;
            let l = base.slice(0, i, i + 1)?.add(&self.markov_head.forward(&me)?.cast(DType::F32)?)?;
            let t = l.argmax(1)?; // [1] i32
            confs.push(Tensor::cat(&[&h.slice(0, i, i + 1)?, &me], 1)?.matmul_t(&self.conf)?.reshape(&[1])?);
            toks.push(t.clone());
            logits.push(l);
            prev = t;
        }
        let cat = |v: &[Tensor]| Tensor::cat(&v.iter().collect::<Vec<_>>(), 0);
        Ok(DraftGraph { toks: cat(&toks)?, conf: cat(&confs)?, logits: cat(&logits)?, ring })
    }

    /// Up to `n_max` draft tokens after `anchor` (at position `st.pos`), cut at the first
    /// confidence below `p_min`; updates the draft window in `st`.
    pub fn draft(&self, m: &Deepseek41Model, st: &mut State, anchor: u32, n_max: usize, p_min: f32) -> Result<Vec<u32>> {
        let b = n_max.min(self.block_size);
        if b == 0 {
            return Ok(Vec::new());
        }
        let g = self.graph(m, st, anchor, b)?;
        let mut outs: Vec<&Tensor> = vec![&g.toks, &g.conf];
        outs.extend(g.ring.iter().flatten());
        let mut realized = eval(&m.device, &outs)?.into_iter();
        let (Some(toks), Some(conf)) = (realized.next(), realized.next()) else { return Err(Error::new("backend", "dspark: no outputs")) };
        st.draft.ring = realized.collect();
        st.draft.ring_end = st.pos;
        st.draft.feats.clear();
        let toks = m.device.to_vec_i32(&toks)?;
        let conf = m.device.to_vec_f32(&conf)?;
        let keep = conf.iter().position(|&c| 1.0 / (1.0 + (-c).exp()) < p_min).unwrap_or(b);
        Ok(toks[..keep].iter().map(|&t| t as u32).collect())
    }

    /// One full-block draft without changing `st`: logits `[b * V]` (Markov bias included),
    /// tokens and confidence logits (before the sigmoid), as the reference records them.
    #[cfg(test)]
    pub fn trace(&self, m: &Deepseek41Model, st: &State, anchor: u32) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>)> {
        let g = self.graph(m, st, anchor, self.block_size)?;
        let r = eval(&m.device, &[&g.logits, &g.toks, &g.conf])?;
        Ok((m.device.to_vec_f32(&r[0])?, m.device.to_vec_i32(&r[1])?, m.device.to_vec_f32(&r[2])?))
    }
}

/// The lazy outputs of one draft: tokens `[b]` i32, confidence logits `[b]`, logits
/// `[b, V]` and each stage's window after the update.
struct DraftGraph {
    toks: Tensor,
    conf: Tensor,
    #[cfg_attr(not(test), allow(dead_code))]
    logits: Tensor,
    ring: Vec<Option<Tensor>>,
}
