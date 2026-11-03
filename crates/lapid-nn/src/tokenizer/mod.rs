//! Tokenizers: a thin wrapper over Hugging Face [`tokenizers`] (pure Rust, `fancy-regex`
//! backend), loaded from a `tokenizer.json` (file, string or bytes, e.g. a `.lapid` blob) or
//! built from a byte-level vocabulary (decode-only vocabularies such as Whisper's metadata).
//!
//! The library does normalization, pre-tokenization, the model (BPE / Unigram / WordPiece /
//! WordLevel, byte-level or SentencePiece-style with byte fallback), the post-processor
//! template (`add_special_tokens`) and decoding. On top of it this keeps per-id tables:
//! piece strings, raw bytes (for streaming detokenization) and special flags (added tokens
//! marked `special`, the unknown token, and ids marked with [`Tokenizer::mark_special`]).
//! Chat templates live with the models.

use std::path::Path;
use std::str::FromStr;

use lapid::{Error, Result};
use serde_json::{Value, json};
/// The Hugging Face library itself (normalizers, decoders, ... for model-specific conversions).
pub use tokenizers;

/// GPT-2 `bytes_to_unicode`: byte -> printable char used in byte-level BPE token strings.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable =
        |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut n = 0u32;
    for b in 0u32..256 {
        let c = if printable(b) {
            b
        } else {
            n += 1;
            255 + n
        };
        table[b as usize] = char::from_u32(c).unwrap_or('\u{FFFD}');
    }
    table
}

/// Inverse of [`bytes_to_unicode`].
pub fn unicode_to_bytes() -> std::collections::HashMap<char, u8> {
    bytes_to_unicode().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect()
}

fn tk_err(e: impl std::fmt::Display) -> Error {
    Error::new("tokenizer", e.to_string())
}

/// `<0xAB>` -> 0xAB.
fn parse_byte_piece(p: &str) -> Option<u8> {
    let hex = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Does a serialized component (decoder) contain a component of type `ty` (also inside a `Sequence`)?
fn has_type(v: &Value, ty: &str) -> bool {
    match v {
        Value::Object(m) => {
            m.get("type").and_then(Value::as_str) == Some(ty)
                || ["decoders", "pretokenizers"]
                    .iter()
                    .any(|k| m.get(*k).and_then(Value::as_array).is_some_and(|a| a.iter().any(|d| has_type(d, ty))))
        }
        _ => false,
    }
}

#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Piece string per id ("" for unused ids).
    pieces: Vec<String>,
    /// Raw decoded bytes per id (byte-level mapping undone, `▁` -> space, `<0xXX>` -> byte).
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("vocab_size", &self.pieces.len()).finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Load a Hugging Face `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Tokenizer> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::new("io", format!("cannot read {}: {e}", path.display())))?;
        Tokenizer::from_bytes(&bytes).map_err(|e| Error::new(e.code, format!("{}: {}", path.display(), e.message)))
    }

    /// Parse a Hugging Face `tokenizer.json` given as bytes (e.g. a `.lapid` blob).
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
        let text = std::str::from_utf8(bytes).map_err(|_| Error::new("tokenizer", "tokenizer.json is not UTF-8"))?;
        Tokenizer::from_tokenizer_json(text)
    }

    /// Parse a Hugging Face `tokenizer.json`.
    pub fn from_tokenizer_json(json: &str) -> Result<Tokenizer> {
        let inner = tokenizers::Tokenizer::from_str(json).map_err(|e| tk_err(format!("tokenizer.json: {e}")))?;
        // The unknown token is special (never text), as in the reference tokenizers' decode.
        let v: Value = serde_json::from_str(json)?;
        let model = v.get("model");
        let unk = model
            .and_then(|m| m.get("unk_token"))
            .and_then(Value::as_str)
            .and_then(|s| inner.token_to_id(s))
            .or_else(|| model.and_then(|m| m.get("unk_id")).and_then(Value::as_u64).map(|i| i as u32));
        Tokenizer::wrap(inner, None, unk)
    }

    /// Byte-level BPE from an id-ordered vocabulary of byte-level strings and merges
    /// (`("Ġ", "t")`, ...) with the GPT-2 split. With no merges the tokenizer can still decode
    /// (and encodes to single-symbol pieces).
    pub fn byte_level<S: AsRef<str>>(pieces: &[S], merges: &[(S, S)]) -> Result<Tokenizer> {
        let mut vocab = serde_json::Map::new();
        for (i, p) in pieces.iter().enumerate() {
            vocab.entry(p.as_ref().to_string()).or_insert(json!(i));
        }
        let merges: Vec<Value> = merges.iter().map(|(a, b)| json!([a.as_ref(), b.as_ref()])).collect();
        let bl = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true});
        let tj = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": bl, "post_processor": null, "decoder": bl,
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": Value::Object(vocab), "merges": merges},
        });
        let inner = tokenizers::Tokenizer::from_str(&tj.to_string()).map_err(tk_err)?;
        let pieces = pieces.iter().map(|p| p.as_ref().to_string()).collect();
        Tokenizer::wrap(inner, Some(pieces), None)
    }

    fn wrap(inner: tokenizers::Tokenizer, pieces: Option<Vec<String>>, unk: Option<u32>) -> Result<Tokenizer> {
        let pieces = match pieces {
            Some(p) => p,
            None => {
                let n = inner.get_vocab(true).values().map(|&i| i as usize + 1).max().unwrap_or(0);
                (0..n as u32).map(|i| inner.id_to_token(i).unwrap_or_default()).collect()
            }
        };
        let n = pieces.len();
        let mut special = vec![false; n];
        let mut added = vec![false; n];
        for (id, t) in inner.get_added_tokens_decoder() {
            if let Some(a) = added.get_mut(id as usize) {
                *a = true;
                special[id as usize] = t.special;
            }
        }
        if let Some(s) = unk.and_then(|u| special.get_mut(u as usize)) {
            *s = true;
        }
        let decoder = inner.get_decoder().map(serde_json::to_value).transpose()?.unwrap_or(Value::Null);
        let byte_level = has_type(&decoder, "ByteLevel");
        let inv = unicode_to_bytes();
        let bytes = pieces
            .iter()
            .enumerate()
            .map(|(i, piece)| {
                if added[i] {
                    piece.as_bytes().to_vec()
                } else if byte_level {
                    let mut out = Vec::with_capacity(piece.len());
                    for c in piece.chars() {
                        match inv.get(&c) {
                            Some(&b) => out.push(b),
                            None => out.extend_from_slice(c.encode_utf8(&mut [0u8; 4]).as_bytes()),
                        }
                    }
                    out
                } else if let Some(b) = parse_byte_piece(piece) {
                    vec![b]
                } else {
                    piece.replace('\u{2581}', " ").into_bytes()
                }
            })
            .collect();
        Ok(Tokenizer { inner, pieces, bytes, special })
    }

    /// Mark ids as special (skipped by `decode(.., true)`, excluded from [`Self::token_for_bytes`]).
    pub fn mark_special(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            if let Some(s) = self.special.get_mut(id as usize) {
                *s = true;
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// Id of a piece string or added token (`"<|endoftext|>"`, `"▁the"`, `"Ġthe"`).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Piece string of an id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    /// Decoded bytes of one id.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize).map(Vec::as_slice)
    }

    /// First non-special id whose decoded bytes are exactly `bytes`.
    pub fn token_for_bytes(&self, bytes: &[u8]) -> Option<u32> {
        (0..self.bytes.len()).find(|&i| !self.special[i] && self.bytes[i] == bytes).map(|i| i as u32)
    }

    /// Ids of every special token.
    pub fn special_ids(&self) -> Vec<u32> {
        (0..self.special.len()).filter(|&i| self.special[i]).map(|i| i as u32).collect()
    }

    /// Concatenated raw bytes of `ids` (per-token bytes, no decoder post-processing such as
    /// SentencePiece's leading-space strip; unknown ids ignored; special ids skipped if asked).
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i < self.bytes.len() && !(skip_special && self.special[i]) {
                out.extend_from_slice(&self.bytes[i]);
            }
        }
        out
    }

    /// Text of `ids` through the tokenizer's decoder; invalid UTF-8 (a character split
    /// across tokens) becomes U+FFFD. Unknown ids are ignored.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let kept: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| (id as usize) < self.pieces.len() && !(skip_special && self.special[id as usize]))
            .collect();
        match self.inner.decode(&kept, false) {
            Ok(s) => s,
            Err(_) => String::from_utf8_lossy(&self.decode_bytes(&kept, false)).into_owned(),
        }
    }

    /// Encode `text`; `add_special_tokens` applies the tokenizer's post-processor template
    /// (e.g. `<bos>` … `<eos>`).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, add_special_tokens).map_err(tk_err)?;
        Ok(enc.get_ids().to_vec())
    }

    /// The underlying Hugging Face tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests;
