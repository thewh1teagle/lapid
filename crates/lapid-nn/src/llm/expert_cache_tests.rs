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
                let got = cache.forward(l, Activation::Silu, Some(1.5), false, &t(&x, &[n, d]), &ids, &t(&w, &[n, k])).expect("cached");
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
