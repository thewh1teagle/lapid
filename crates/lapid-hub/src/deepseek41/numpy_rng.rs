//! numpy's default generator, bit-exact for the one call the Engram constants need:
//! `np.random.default_rng(seed).integers(0, high, size, dtype=np.int64)` (`SeedSequence`
//! entropy mixing, PCG64 XSL-RR 128/64, Lemire's bounded integers). Used at conversion time.

const MULT_A: u32 = 0x931e_8875;
const MULT_B: u32 = 0x58f3_8ded;
const INIT_A: u32 = 0x43b0_d7e5;
const INIT_B: u32 = 0x8b51_f9dd;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL: usize = 4;
const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341;

/// `SeedSequence(entropy).generate_state(words, np.uint32)` for an integer entropy.
fn seed_sequence(entropy: u128, words: usize) -> Vec<u32> {
    // the entropy as little-endian u32 words ([0] for 0)
    let mut ent = Vec::new();
    let mut e = entropy;
    while e > 0 {
        ent.push(e as u32);
        e >>= 32;
    }
    if ent.is_empty() {
        ent.push(0);
    }
    let mut hash_const = INIT_A;
    let mut hashmix = |v: u32| -> u32 {
        let mut v = v ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; POOL];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(ent.get(i).copied().unwrap_or(0));
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    for &v in ent.iter().skip(POOL) {
        for p in pool.iter_mut() {
            let h = hashmix(v);
            *p = mix(*p, h);
        }
    }
    let mut hash_const = INIT_B;
    (0..words)
        .map(|i| {
            let mut v = pool[i % POOL] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> 16)
        })
        .collect()
}

/// numpy `PCG64` (XSL-RR output of the advanced 128-bit LCG state).
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    /// `np.random.default_rng(seed)`'s bit generator.
    pub fn from_seed(seed: u128) -> Pcg64 {
        let w = seed_sequence(seed, 8);
        let word = |i: usize| u64::from(w[2 * i]) | (u64::from(w[2 * i + 1]) << 32);
        let init_state = (u128::from(word(0)) << 64) | u128::from(word(1));
        let init_seq = (u128::from(word(2)) << 64) | u128::from(word(3));
        let mut g = Pcg64 { state: 0, inc: (init_seq << 1) | 1 };
        g.step();
        g.state = g.state.wrapping_add(init_state);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    /// Lemire's method for `[0, rng]` (inclusive), numpy `bounded_lemire_uint64`.
    fn bounded(&mut self, rng: u64) -> u64 {
        let excl = u128::from(rng) + 1;
        let mut m = u128::from(self.next_u64()) * excl;
        let mut leftover = m as u64;
        if u128::from(leftover) < excl {
            let threshold = ((u64::MAX - rng) as u128 % excl) as u64;
            while leftover < threshold {
                m = u128::from(self.next_u64()) * excl;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `Generator.integers(0, high, size, dtype=np.int64)`. Only the 64-bit path (`high - 1`
    /// above `u32::MAX`) is implemented; `None` otherwise.
    pub fn integers(&mut self, high: u64, size: usize) -> Option<Vec<u64>> {
        let rng = high.checked_sub(1)?;
        if rng <= u64::from(u32::MAX) {
            return None;
        }
        Some((0..size).map(|_| if rng == u64::MAX { self.next_u64() } else { self.bounded(rng) }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from numpy 1.26: `default_rng(10007 * layer).integers(0, b, size=4, dtype=np.int64) * 2 + 1`
    /// with `b = (2**63 - 1) // 99092 // 2` (the real Engram multipliers).
    #[test]
    fn matches_numpy() {
        let b = (i64::MAX as u64 / 99092) / 2;
        let want: [(u128, [u64; 4]); 4] = [
            (1, [76632096046245, 4839876093313, 35959672319349, 73987337458391]),
            (14, [67716810739261, 51510806800915, 30921347202721, 82619226485591]),
            (0, [59287678272599, 25111444230055, 3813769577005, 1538373747317]),
            (2, [76507469730479, 37097805264573, 22647385550623, 71424551940645]),
        ];
        for (layer, w) in want {
            let v = Pcg64::from_seed(10007 * layer).integers(b, 4).expect("64-bit range");
            assert_eq!(v.iter().map(|x| x * 2 + 1).collect::<Vec<_>>(), w, "layer {layer}");
        }
    }
}
