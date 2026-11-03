use super::*;
use crate::testutil::{assert_close, eval, rand_vec};

#[test]
fn policy_keeps_frequent_experts() {
    // 1 layer x 8 experts, 3 slots. Expert 0 and 1 are used every step, the third is a
    // different one-off each time: the one-offs replace each other, never 0 or 1.
    let mut st = State::new(8, 3, 1e9);
    for step in 0..20u32 {
        let keys = [0, 1, 2 + step % 6];
        st.touch(&keys);
        let (slots, loads) = st.assign(&keys);
        assert!(slots.iter().all(|&s| s < 3));
        let distinct: std::collections::BTreeSet<u32> = slots.iter().copied().collect();
        assert_eq!(distinct.len(), 3, "step {step}: {slots:?}");
        if step > 0 {
            assert!(loads.iter().all(|&(k, _)| k >= 2), "step {step}: evicted a frequent expert: {loads:?}");
        }
        for (&k, &s) in keys.iter().zip(&slots) {
            assert_eq!(st.owner[s as usize], k);
            assert_eq!(st.slot_of[k as usize], s);
        }
    }
    assert!(st.stats.hits >= 2 * 19);
}

#[test]
fn policy_decays_old_popularity() {
    // Experts 0..3 were hot long ago; 4..7 are now: with a short half-life the new set wins.
    let mut st = State::new(8, 4, 2.0);
    for _ in 0..50 {
        let k = [0, 1, 2, 3];
        st.touch(&k);
        st.assign(&k);
    }
    for _ in 0..30 {
        let k = [4, 5, 6, 7];
        st.touch(&k);
        st.assign(&k);
    }
    let k = [4, 5, 6, 7];
    st.touch(&k);
    let (_, loads) = st.assign(&k);
    assert!(loads.is_empty());
}

fn stacks(layers: usize, e: usize, i: usize, d: usize) -> Vec<Vec<Vec<f32>>> {
    (0..layers).map(|l| (0..3).map(|k| rand_vec(e * i * d, (10 * l + k + 1) as u64, 0.5)).collect()).collect()
}

/// The cache (tiny pools: evictions, misses, multi-pass) gives the same result as
/// `Experts::forward` over the full stacks, for several layers and steps.
#[test]
fn cached_forward_matches_direct() {
    let (layers, e, i, d, k) = (3usize, 8usize, 6usize, 8usize, 2usize);
    let dev = Device::cpu();
    let data = stacks(layers, e, i, d);
    let t = |v: &[f32], s: &[usize]| Tensor::from_f32(v, s).expect("tensor");
    let host: Vec<Option<Vec<Tensor>>> = data
        .iter()
        .map(|ks| Some(vec![t(&ks[0], &[e, i, d]), t(&ks[1], &[e, i, d]), t(&ks[2], &[e, d, i])]))
        .collect();
    for slots in [1usize, 2, 3, 16] {
        let cache = ExpertCache::new(&dev, host.clone(), slots).expect("cache");
        for step in 0..6usize {
            let n = 1 + step % 4;
            let x = rand_vec(n * d, 100 + step as u64, 1.0);
            let ids: Vec<i32> = (0..n * k).map(|j| ((j * 5 + step * 3 + j / k) % e) as i32).collect();
            let ids: Vec<i32> = ids.chunks(k).flat_map(|c| if c[0] == c[1] { vec![c[0], (c[1] + 1) % e as i32] } else { c.to_vec() }).collect();
            let w = rand_vec(n * k, 200 + step as u64, 1.0);
            for (l, ks) in host.iter().enumerate() {
                let ks = ks.as_ref().expect("layer");
                let direct = Experts { gate_up: GateUp::Split { gate: ks[0].clone(), up: ks[1].clone() }, down: ks[2].clone(), act: Activation::Silu, limit: Some(1.5), act_quant: false };
                let want = eval(&direct.forward(&t(&x, &[n, d]), &Tensor::from_i32(&ids, &[n, k]).expect("ids"), &t(&w, &[n, k])).expect("direct"));
                let got = cache.forward(l, Glu { act: Activation::Silu, limit: Some(1.5), act_quant: false }, &t(&x, &[n, d]), &ids, &t(&w, &[n, k])).expect("cached");
                assert_close(&eval(&got), &want, 1e-5);
            }
        }
        let s = cache.stats();
        assert!(s.misses > 0 && s.loaded_bytes > 0);
        if slots < 4 {
            assert!(s.extra_passes > 0, "slots {slots}: {s:?}");
        } else {
            assert_eq!(s.extra_passes, 0);
            assert!(s.hits > 0);
        }
    }
}

#[test]
fn rejects_bad_requests() {
    let dev = Device::cpu();
    let t = |s: &[usize]| Tensor::from_f32(&vec![0.0; s.iter().product()], s).expect("tensor");
    let host = vec![Some(vec![t(&[4, 2, 2]), t(&[4, 2, 2]), t(&[4, 2, 2])]), None];
    let cache = ExpertCache::new(&dev, host, 2).expect("cache");
    assert!(cache.ensure(0, &[0, 1, 2]).is_err());
    assert!(cache.ensure(1, &[0]).is_err());
    assert!(cache.ensure(0, &[4]).is_err());
    assert_eq!(cache.ensure(0, &[3, 1]).expect("ok").len(), 2);
    assert!(ExpertCache::new(&dev, vec![Some(vec![t(&[4, 2, 2])]), Some(vec![t(&[4, 2, 3])])], 2).is_err());
}

#[test]
fn budget_override_is_capped() {
    set_budget_gib(1e9);
    let (b, why) = budget_bytes(0);
    if let Some(a) = lapid::residency::available_memory() {
        // (available memory moves a little between the two reads)
        assert!(b <= a.saturating_sub(MIN_FREE) + (1 << 30), "{why}");
    }
    set_budget_gib(0.0);
}

/// Hit rates of the replacement policy for DeepSeek V4.1 Flash decode (40 layers x 384
/// experts, top 6) on synthetic routing traces, per pool size and count half-life (in
/// tokens; ~0 behaves like LRU). Ignored (prints a table):
/// `cargo test --release -p lapid-nn policy_hit_rates -- --ignored --nocapture`.
#[test]
#[ignore]
fn policy_hit_rates() {
    let (layers, e, k, tokens) = (40usize, 384usize, 6usize, 1600usize);
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    // trace: (name, zipf alpha, reuse probability of the previous token's experts, drift period)
    let traces: [(&str, f64, f64, usize); 5] = [
        ("uniform", 0.0, 0.0, 0),
        ("zipf 0.6", 0.6, 0.0, 0),
        ("zipf 1.0", 1.0, 0.0, 0),
        ("zipf 0.8 + reuse 0.3", 0.8, 0.3, 0),
        ("zipf 0.8 + reuse 0.3, drift/500", 0.8, 0.3, 500),
    ];
    let variants: [(f64, u64); 5] = [(0.05, 0), (64.0, 0), (512.0, 0), (512.0, 1), (256.0, 1)];
    println!("{:<34} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "trace", "pool", "hl=0.05", "hl=64", "hl=512", "512+rec", "256+rec", "static*");
    for (name, alpha, reuse, drift) in traces {
        // per layer: expert weights over a random order
        let mut weights: Vec<Vec<f64>> = (0..layers)
            .map(|_| {
                let mut order: Vec<usize> = (0..e).collect();
                for i in (1..e).rev() {
                    order.swap(i, (next() * (i + 1) as f64) as usize);
                }
                let mut w = vec![0.0; e];
                for (rank, &x) in order.iter().enumerate() {
                    w[x] = 1.0 / (1.0 + rank as f64).powf(alpha);
                }
                w
            })
            .collect();
        let mut trace: Vec<Vec<u32>> = Vec::with_capacity(tokens * layers);
        let mut prev: Vec<Vec<u32>> = vec![Vec::new(); layers];
        for t in 0..tokens {
            if drift > 0 && t > 0 && t % drift == 0 {
                for w in &mut weights {
                    // reshuffle a third of the experts' weights
                    for _ in 0..e / 3 {
                        let (a, b) = ((next() * e as f64) as usize, (next() * e as f64) as usize);
                        w.swap(a, b);
                    }
                }
            }
            for l in 0..layers {
                let mut pick: Vec<u32> = Vec::with_capacity(k);
                for &p in &prev[l] {
                    if pick.len() < k && next() < reuse {
                        pick.push(p);
                    }
                }
                let total: f64 = weights[l].iter().sum();
                while pick.len() < k {
                    let mut r = next() * total;
                    let mut x = 0;
                    while x + 1 < e && r >= weights[l][x] {
                        r -= weights[l][x];
                        x += 1;
                    }
                    if !pick.contains(&(x as u32)) {
                        pick.push(x as u32);
                    }
                }
                prev[l] = pick.clone();
                trace.push(pick.iter().map(|&x| (l * e) as u32 + x).collect());
            }
        }
        for frac in [0.2, 0.28, 0.35] {
            let slots = (frac * (layers * e) as f64) as usize;
            let mut row = format!("{name:<34} {:>5.0}%", frac * 100.0);
            for (hl, protect) in variants {
                let mut st = State::new(layers * e, slots, hl * layers as f64);
                st.protect = protect * layers as u64;
                let warm = trace.len() / 4;
                let mut hits = 0u64;
                let mut total = 0u64;
                for (i, keys) in trace.iter().enumerate() {
                    st.touch(keys);
                    let before = st.stats.hits;
                    st.assign(keys);
                    if i >= warm {
                        hits += st.stats.hits - before;
                        total += keys.len() as u64;
                    }
                }
                row.push_str(&format!(" {:>8.3}", hits as f64 / total as f64));
            }
            // static: top `slots` experts by true weight (k-sampling approximated by mass)
            let mut all: Vec<f64> = weights.iter().flat_map(|w| { let s: f64 = w.iter().sum(); w.iter().map(move |x| x / s / layers as f64) }).collect();
            all.sort_by(|a, b| b.total_cmp(a));
            row.push_str(&format!(" {:>8.3}", all[..slots].iter().sum::<f64>()));
            println!("{row}");
        }
    }
}
