//! Engram hash constants (upstream `inference/engram.py`), derived at conversion time from the
//! config and the tokenizer and stored in the `.lapid`: the bucket primes, the per-layer hash
//! multipliers (numpy's PCG64 stream, [`numpy_rng`](super::numpy_rng)) and the compressed
//! token map (tokens that normalize alike share an id). The metadata object
//! `"engram_hash"` holds the small parts, the blob [`TOKEN_MAP_BLOB`] the map (i32 LE).

use lapid::{Error, Result, bail};
use lapid_nn::llm::engram::NgramHash;
use lapid_nn::tokenizer::tokenizers::normalizers::{NFD, NFKC, Lowercase, Replace, Sequence, Strip, StripAccents};
use lapid_nn::tokenizer::tokenizers::{NormalizedString, Normalizer, NormalizerWrapper, Tokenizer};
use serde_json::{Value, json};

use super::config::EngramConfig;
use super::numpy_rng::Pcg64;

/// Blob holding the token map (`[vocab]` i32 little-endian).
pub const TOKEN_MAP_BLOB: &str = "engram/token_map";

fn is_prime(n: u64) -> bool {
    if n < 2 || n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// `EngramLayout.from_args`: per module, `(N - 1) * H` primes above `vocab - 1`, drawn in
/// order and never reused (across orders and modules), order-major.
pub fn primes(modules: usize, max_ngram: usize, heads: usize, vocab: u64) -> Vec<Vec<u64>> {
    let mut seen = std::collections::HashSet::new();
    (0..modules)
        .map(|_| {
            let mut out = Vec::with_capacity((max_ngram - 1) * heads);
            for _ in 1..max_ngram {
                let mut cur = vocab - 1;
                for _ in 0..heads {
                    let mut c = cur + 1;
                    while !is_prime(c) || seen.contains(&c) {
                        c += 1;
                    }
                    seen.insert(c);
                    out.push(c);
                    cur = c;
                }
            }
            out
        })
        .collect()
}

/// `compute_hash_multipliers`: odd, from `default_rng(10007 * layer)`, bounded so that
/// `token * multiplier` cannot overflow i64.
pub fn multipliers(layer_ids: &[usize], max_ngram: usize, compressed_vocab: usize) -> Result<Vec<Vec<u64>>> {
    let bound = ((i64::MAX as u64 / compressed_vocab.max(1) as u64) / 2).max(1);
    layer_ids
        .iter()
        .map(|&l| {
            let v = Pcg64::from_seed(10007 * l as u128)
                .integers(bound, max_ngram)
                .ok_or_else(|| Error::new("unsupported_model", format!("engram: multiplier bound {bound} needs numpy's 32-bit path")))?;
            Ok(v.into_iter().map(|x| x * 2 + 1).collect())
        })
        .collect()
}

/// The upstream normalizer of `build_compressed_token_map`.
fn normalizer() -> Result<Sequence> {
    let sentinel = "\u{e000}";
    let re = |p: &str, c: &str| -> Result<NormalizerWrapper> {
        Replace::new(lapid_nn::tokenizer::tokenizers::normalizers::replace::ReplacePattern::Regex(p.into()), c)
            .map(Into::into)
            .map_err(|e| Error::new("tokenizer", format!("engram normalizer: {e}")))
    };
    let lit = |p: &str, c: &str| -> Result<NormalizerWrapper> {
        Replace::new(p, c).map(Into::into).map_err(|e| Error::new("tokenizer", format!("engram normalizer: {e}")))
    };
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        re(r"[ \t\r\n]+", " ")?,
        re(r"^ $", sentinel)?,
        Strip::new(true, true).into(),
        lit(sentinel, " ")?,
    ]))
}

/// `build_compressed_token_map`: token id -> compressed id, and the compressed vocab size.
/// Every token is decoded alone (raw decoder, special tokens kept); partial UTF-8 tokens are
/// keyed by their raw piece, the rest by their normalized text (or the text when that is empty).
pub fn token_map(tok: &Tokenizer) -> Result<(Vec<i32>, usize)> {
    let norm = normalizer()?;
    let mut keys: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let n = tok.get_vocab_size(true);
    let mut map = Vec::with_capacity(n);
    for id in 0..n as u32 {
        let text = tok.decode(&[id], false).map_err(|e| Error::new("tokenizer", format!("engram token map: decode {id}: {e}")))?;
        let key = if text.contains('\u{fffd}') {
            tok.id_to_token(id).unwrap_or_default()
        } else {
            let mut s = NormalizedString::from(text.as_str());
            norm.normalize(&mut s).map_err(|e| Error::new("tokenizer", format!("engram token map: {e}")))?;
            if s.get().is_empty() { text } else { s.get().to_string() }
        };
        let next = keys.len() as i32;
        map.push(*keys.entry(key).or_insert(next));
    }
    Ok((map, keys.len()))
}

/// The conversion-time constants: metadata object + token map blob bytes. Checks them against
/// the config (compressed vocab size, table rows) like upstream does.
pub fn derive(cfg: &EngramConfig, vocab: usize, tokenizer_json: &[u8]) -> Result<(Value, Vec<u8>)> {
    let text = std::str::from_utf8(tokenizer_json).map_err(|_| Error::new("tokenizer", "tokenizer.json is not UTF-8"))?;
    let tok: Tokenizer = text.parse().map_err(|e| Error::new("tokenizer", format!("tokenizer.json: {e}")))?;
    let (map, cvocab) = token_map(&tok)?;
    if cvocab != cfg.compressed_vocab_size || map.len() != vocab {
        bail!("metadata", "engram: token map {} -> {cvocab} ids, config says {vocab} -> {}", map.len(), cfg.compressed_vocab_size);
    }
    let primes = primes(cfg.layer_ids.len(), cfg.max_ngram, cfg.n_heads, cfg.vocab_size as u64);
    for ((p, &rows), l) in primes.iter().zip(&cfg.num_embeddings).zip(&cfg.layer_ids) {
        if p.iter().sum::<u64>() != rows {
            bail!("metadata", "engram layer {l}: the primes sum to {}, config says {rows} rows", p.iter().sum::<u64>());
        }
    }
    let pad = *map.get(cfg.pad_token_id).ok_or_else(|| Error::new("metadata", "engram: pad token outside the vocab"))?;
    let meta = json!({
        "multipliers": multipliers(&cfg.layer_ids, cfg.max_ngram, cvocab)?,
        "primes": primes,
        "pad": pad,
        "token_map_blob": TOKEN_MAP_BLOB,
    });
    Ok((meta, map.iter().flat_map(|v| v.to_le_bytes()).collect()))
}

/// The hash layout from the stored metadata object and token map blob.
pub fn load(cfg: &EngramConfig, meta: &Value, token_map: &[u8]) -> Result<NgramHash> {
    let rows = |k: &str| -> Result<Vec<Vec<u64>>> {
        let bad = || Error::new("metadata", format!("engram_hash.{k}: expected arrays of integers"));
        meta.get(k)
            .and_then(Value::as_array)
            .ok_or_else(bad)?
            .iter()
            .map(|r| r.as_array().ok_or_else(bad)?.iter().map(|v| v.as_u64().ok_or_else(bad)).collect())
            .collect()
    };
    let pad = meta.get("pad").and_then(Value::as_i64).ok_or_else(|| Error::new("metadata", "engram_hash.pad missing"))?;
    if !token_map.len().is_multiple_of(4) {
        bail!("format", "engram token map: {} bytes", token_map.len());
    }
    let map = token_map.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
    let h = NgramHash::new(cfg.max_ngram, cfg.n_heads, map, pad, rows("multipliers")?, rows("primes")?)?;
    if h.modules() != cfg.layer_ids.len() {
        bail!("metadata", "engram: {} hash modules for {} layers", h.modules(), cfg.layer_ids.len());
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_layout() {
        // the V4.1-Flash config: engram_vocab_size 16M, 2 layers, 4-grams, 8 heads
        let p = primes(2, 4, 8, 16_000_000);
        assert_eq!(p[0].iter().sum::<u64>(), 384_006_168);
        assert_eq!(p[1].iter().sum::<u64>(), 384_016_682);
        assert!(p[0][0] >= 16_000_000 && p[0].windows(2).all(|w| w[0] < w[1]));
        let m = multipliers(&[1, 14], 4, 99092).expect("multipliers");
        assert_eq!(m[0], vec![76632096046245, 4839876093313, 35959672319349, 73987337458391]);
    }

    /// The real tokenizer (`LAPID_DS41_TOKENIZER` = a V4.1 `tokenizer.json`): 99092 compressed ids.
    #[test]
    fn real_token_map() {
        let Ok(path) = std::env::var("LAPID_DS41_TOKENIZER") else {
            eprintln!("skipped: LAPID_DS41_TOKENIZER not set");
            return;
        };
        let text = std::fs::read_to_string(path).expect("read tokenizer");
        let tok: Tokenizer = text.parse().expect("tokenizer");
        let (map, n) = token_map(&tok).expect("token map");
        assert_eq!((map.len(), n), (129280, 99092));
    }
}
