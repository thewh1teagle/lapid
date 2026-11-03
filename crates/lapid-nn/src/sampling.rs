//! Host-side logits processing and sampling: greedy, temperature, top-k, top-p, repetition
//! penalty, with a seeded pure-Rust RNG (xoshiro256**), so runs are reproducible.

/// Index of the maximum value (first one on ties; NaN never wins; all NaN gives the last).
pub fn argmax(v: &[f32]) -> usize {
    // Two vectorizable passes: the max over non-NaN values (`f32::max` ignores NaN), then
    // the first index holding it.
    const L: usize = 32;
    let mut acc = [f32::NEG_INFINITY; L];
    let chunks = v.chunks_exact(L);
    let rest = chunks.remainder();
    for c in chunks {
        let c: &[f32; L] = c.try_into().unwrap_or(&[f32::NEG_INFINITY; L]);
        for i in 0..L {
            // Compare-select (NaN never wins): vectorizes, unlike a `max` reduction chain.
            acc[i] = if c[i] > acc[i] { c[i] } else { acc[i] };
        }
    }
    let m = rest.iter().chain(&acc).fold(f32::NEG_INFINITY, |a, &x| if x > a { x } else { a });
    // First index of `m`: a branch-free test per block, a scan only in the block that has it.
    for (b, c) in v.chunks(64).enumerate() {
        if c.iter().fold(0u32, |f, &x| f | (x == m) as u32) != 0 {
            return b * 64 + c.iter().position(|&x| x == m).unwrap_or(0);
        }
    }
    v.len().saturating_sub(1)
}

/// log(sum(exp(v))), computed stably (all -inf gives -inf).
pub fn logsumexp(v: &[f32]) -> f32 {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = v.iter().map(|&x| ((x - max) as f64).exp()).sum();
    max + sum.ln() as f32
}

/// log_softmax over the whole vector (all -inf stays -inf).
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let lse = logsumexp(logits);
    if lse == f32::NEG_INFINITY {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    logits.iter().map(|&x| x - lse).collect()
}

/// softmax over the whole vector.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    log_softmax(logits).into_iter().map(f32::exp).collect()
}

/// HF / CTRL repetition penalty: every token seen in `history` has its logit divided by
/// `penalty` if positive, multiplied if negative.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &t in history {
        let i = t as usize;
        if i < logits.len() && !seen[i] {
            seen[i] = true;
            let l = &mut logits[i];
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

/// xoshiro256** seeded through splitmix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1) with 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `n` samples of N(0, std^2) (Box-Muller: each pair of uniforms gives a cos and a sin
    /// sample; for odd `n` the last sin sample is dropped).
    pub fn normal(&mut self, n: usize, std: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = self.next_f64().max(f64::MIN_POSITIVE);
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            out.push((r * a.cos()) as f32 * std);
            if out.len() < n {
                out.push((r * a.sin()) as f32 * std);
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0 = greedy.
    pub temperature: f32,
    /// Keep the k most likely tokens (0 = off).
    pub top_k: usize,
    /// Keep the smallest set with cumulative probability >= top_p (1.0 = off).
    pub top_p: f32,
    /// 1.0 = off.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    /// Greedy.
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    pub params: SamplingParams,
    rng: Rng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Sampler {
        let rng = Rng::new(params.seed);
        Sampler { params, rng }
    }

    pub fn greedy() -> Sampler {
        Sampler::new(SamplingParams::default())
    }

    /// Pick the next token from `logits` given the tokens so far (for the repetition
    /// penalty). Order as in HF: penalty, temperature, top-k, top-p.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let dist = self.distribution(logits, history);
        self.pick(&dist)
    }

    /// The distribution [`sample`](Self::sample) draws from: `(token, probability)` with
    /// positive probabilities summing to 1, most likely first when truncated (top-k / top-p).
    /// Greedy (and degenerate cases) is the argmax with probability 1.
    pub fn distribution(&self, logits: &[f32], history: &[u32]) -> Vec<(u32, f64)> {
        let p = &self.params;
        // Copy only when the penalty edits the logits.
        let l: std::borrow::Cow<'_, [f32]> = if p.repetition_penalty == 1.0 {
            logits.into()
        } else {
            let mut l = logits.to_vec();
            apply_repetition_penalty(&mut l, history, p.repetition_penalty);
            l.into()
        };
        if p.temperature <= 0.0 || l.len() <= 1 {
            return vec![(argmax(&l) as u32, 1.0)];
        }
        let inv_t = 1.0 / p.temperature;
        let mut cand: Vec<(u32, f32)> =
            l.iter().enumerate().filter(|(_, x)| !x.is_nan()).map(|(i, &x)| (i as u32, x * inv_t)).collect();
        if p.top_k > 0 || p.top_p < 1.0 {
            // Total order (ties by lower id), so selecting the top k first and sorting only
            // them gives exactly the prefix of the full sort.
            let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
            if p.top_k > 0 && p.top_k < cand.len() {
                cand.select_nth_unstable_by(p.top_k - 1, order);
                cand.truncate(p.top_k);
            }
            cand.sort_unstable_by(order);
        }
        let max = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return vec![(argmax(&l) as u32, 1.0)];
        }
        let mut probs: Vec<f64> = cand.iter().map(|c| ((c.1 - max) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= total);
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = probs.len();
            for (i, &x) in probs.iter().enumerate() {
                acc += x;
                if acc >= p.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let total: f64 = probs.iter().sum();
            probs.iter_mut().for_each(|x| *x /= total);
        }
        cand.iter().zip(probs).map(|(c, x)| (c.0, x)).collect()
    }

    /// Draw from a [`distribution`](Self::distribution) (one uniform when sampling; greedy
    /// consumes no randomness).
    pub fn pick(&mut self, dist: &[(u32, f64)]) -> u32 {
        if let ([(t, _)], true) = (dist, self.params.temperature <= 0.0) {
            return *t;
        }
        let u = self.rng.next_f64();
        let mut acc = 0.0;
        for &(t, x) in dist {
            acc += x;
            if u < acc {
                return t;
            }
        }
        dist.last().map_or(0, |c| c.0)
    }

    /// One uniform draw in `[0, 1)` from the sampler's stream (speculative acceptance tests).
    pub fn uniform(&mut self) -> f64 {
        self.rng.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_moments_and_pairs() {
        let v = Rng::new(1).normal(20_000, 2.0);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.05 && (var - 4.0).abs() < 0.15, "mean {mean} var {var}");
        // an odd count drops the last pair's sin sample: the next draw starts a new pair
        let mut a = Rng::new(3);
        let (x, y) = (a.normal(3, 1.0), a.normal(1, 1.0));
        let z = Rng::new(3).normal(6, 1.0);
        assert_eq!((&x[..], y[0]), (&z[..3], z[4]));
    }

    /// The original scalar argmax (reference semantics).
    fn argmax_ref(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] || v[best].is_nan() {
                best = i;
            }
        }
        best
    }

    #[test]
    fn argmax_first_max_and_nan() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, -1.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 0.5, 2.0]), 2);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
        let mut r = Rng::new(5);
        let specials = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, -0.0, 1.0];
        for n in [0usize, 1, 2, 15, 16, 17, 33, 100, 1000] {
            for _ in 0..200 {
                let v: Vec<f32> = (0..n)
                    .map(|_| {
                        let u = r.next_u64();
                        if u % 5 == 0 { specials[(u >> 8) as usize % specials.len()] } else { ((u >> 16) % 7) as f32 }
                    })
                    .collect();
                assert_eq!(argmax(&v), argmax_ref(&v), "{v:?}");
            }
        }
        assert_eq!(argmax(&[f32::NAN; 3]), argmax_ref(&[f32::NAN; 3]));
    }

    #[test]
    fn top_k_matches_full_sort() {
        let mut r = Rng::new(11);
        let logits: Vec<f32> = (0..5000).map(|_| (r.next_u64() % 50) as f32 * 0.1).collect();
        for k in [1usize, 2, 20, 4999, 5000, 9000] {
            for top_p in [1.0f32, 0.9] {
                let params = SamplingParams { temperature: 0.8, top_k: k, top_p, seed: 7, ..Default::default() };
                let (mut a, mut b) = (Sampler::new(params.clone()), Sampler::new(params.clone()));
                for _ in 0..50 {
                    assert_eq!(a.sample(&logits, &[]), sample_full_sort(&mut b, &logits));
                }
            }
        }
    }

    /// The original sampler (full sort), for equivalence checks.
    fn sample_full_sort(s: &mut Sampler, l: &[f32]) -> u32 {
        let p = s.params.clone();
        let inv_t = 1.0 / p.temperature;
        let mut cand: Vec<(u32, f32)> =
            l.iter().enumerate().filter(|(_, x)| !x.is_nan()).map(|(i, &x)| (i as u32, x * inv_t)).collect();
        cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if p.top_k > 0 {
            cand.truncate(p.top_k.max(1));
        }
        let max = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f64> = cand.iter().map(|c| ((c.1 - max) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= total);
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = probs.len();
            for (i, &x) in probs.iter().enumerate() {
                acc += x;
                if acc >= p.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let total: f64 = probs.iter().sum();
            probs.iter_mut().for_each(|x| *x /= total);
        }
        let u = s.rng.next_f64();
        let mut acc = 0.0;
        for (c, &x) in cand.iter().zip(&probs) {
            acc += x;
            if u < acc {
                return c.0;
            }
        }
        cand.last().map_or(0, |c| c.0)
    }

    #[test]
    fn log_softmax_normalises() {
        let lp = log_softmax(&[1.0, 2.0, f32::NEG_INFINITY]);
        let s: f32 = lp.iter().map(|v| v.exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert_eq!(lp[2], f32::NEG_INFINITY);
        assert!(log_softmax(&[f32::NEG_INFINITY; 2]).iter().all(|v| *v == f32::NEG_INFINITY));
        let p = softmax(&[0.0, 2f32.ln()]);
        assert!((p[0] - 1.0 / 3.0).abs() < 1e-6 && (p[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((logsumexp(&[0.0, 0.0]) - 2f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty() {
        let mut l = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut l, &[0, 1, 0], 2.0);
        assert_eq!(l, vec![1.0, -4.0, 1.0]);
        let mut s = Sampler::new(SamplingParams { repetition_penalty: 4.0, ..Default::default() });
        assert_eq!(s.sample(&[2.0, 1.0], &[0]), 1);
    }

    #[test]
    fn greedy_and_truncation_degenerate_to_argmax() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        assert_eq!(Sampler::greedy().sample(&logits, &[]), 1);
        let mut k1 = Sampler::new(SamplingParams { temperature: 1.0, top_k: 1, seed: 3, ..Default::default() });
        let mut p0 = Sampler::new(SamplingParams { temperature: 1.0, top_p: 1e-6, seed: 4, ..Default::default() });
        for _ in 0..50 {
            assert_eq!(k1.sample(&logits, &[]), 1);
            assert_eq!(p0.sample(&logits, &[]), 1);
        }
    }

    #[test]
    fn seeded_and_distribution() {
        let logits = [0.0f32, 3f32.ln(), f32::NEG_INFINITY]; // p = 0.25, 0.75, 0
        let params = SamplingParams { temperature: 1.0, seed: 42, ..Default::default() };
        let mut a = Sampler::new(params.clone());
        let mut b = Sampler::new(params);
        let draws: Vec<u32> = (0..4000).map(|_| a.sample(&logits, &[])).collect();
        assert!(draws.iter().zip((0..4000).map(|_| b.sample(&logits, &[]))).all(|(x, y)| *x == y));
        assert!(draws.iter().all(|&t| t < 2));
        let ones = draws.iter().filter(|&&t| t == 1).count() as f64 / 4000.0;
        assert!((ones - 0.75).abs() < 0.03, "{ones}");
        // top-k 2 of 3 equal logits never picks the dropped one (ties keep lower ids)
        let mut k = Sampler::new(SamplingParams { temperature: 0.7, top_k: 2, seed: 1, ..Default::default() });
        assert!((0..200).all(|_| k.sample(&[1.0, 1.0, 1.0], &[]) < 2));
        // top-p 0.5 on 0.25 / 0.75 keeps only the 0.75 token
        let mut tp = Sampler::new(SamplingParams { temperature: 1.0, top_p: 0.5, seed: 9, ..Default::default() });
        assert!((0..200).all(|_| tp.sample(&logits, &[]) == 1));
    }

    #[test]
    fn rng_uniform() {
        let mut r = Rng::new(7);
        let xs: Vec<f64> = (0..10_000).map(|_| r.next_f64()).collect();
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!((mean - 0.5).abs() < 0.02);
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }
}
