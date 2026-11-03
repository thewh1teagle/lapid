//! BLOBS section (kind 4, format 1.1): named byte blobs for non-tensor assets.
//!
//! Layout, offsets relative to the section start:
//! ```text
//! 0      u64 index_length (L)
//! 8      index JSON (L bytes): [{"name", "media_type", "offset", "nbytes", "hash"}]
//! ...    zero pad; each blob at a 256-aligned offset >= 8 + L, gaps zero
//! ```

use std::collections::HashSet;

use lapid_tensor::{Error, Result, bail};

use crate::BlobInfo;
use crate::imp::{align_up, check_disjoint, is_hex_hash};

/// BLOBS section limit (large tables, e.g. embedding hash tables, are streamed blobs).
pub(crate) const MAX_BLOBS_SECTION: u64 = 1 << 42;
/// Sections up to this size have their section hash checked on open; larger ones are
/// checked blob by blob by `verify_tensors` (hashing hundreds of GB on every open is not
/// an option).
pub(crate) const VERIFY_ON_OPEN: u64 = 256 << 20;
pub(crate) const MAX_BLOB_INDEX: u64 = 16 << 20;
pub(crate) const MAX_BLOBS: usize = 65_536;
pub(crate) const BLOB_ALIGN: u64 = 256;
const MAX_NAME: usize = 1024;
const MAX_MEDIA_TYPE: usize = 255;

/// Blob names: 1..=1024 bytes. Error code `code` otherwise.
pub(crate) fn check_name(name: &str, code: &'static str) -> Result<()> {
    if name.is_empty() || name.len() > MAX_NAME {
        bail!(code, "invalid blob name {name:?}");
    }
    Ok(())
}

/// Media types: 1..=255 printable ASCII bytes without spaces (`type/subtype[;params]`).
pub(crate) fn check_media_type(mt: &str, code: &'static str) -> Result<()> {
    if mt.is_empty() || mt.len() > MAX_MEDIA_TYPE || !mt.contains('/') || !mt.bytes().all(|b| b.is_ascii_graphic()) {
        bail!(code, "invalid media type {mt:?}");
    }
    Ok(())
}

/// Parses and validates a BLOBS section (its section-table hash is checked by the caller).
pub(crate) fn parse(sec: &[u8]) -> Result<Vec<BlobInfo>> {
    let fmt = |m: String| Error::new("format", m);
    if sec.len() < 8 {
        bail!("format", "BLOBS section too small ({} B)", sec.len());
    }
    let mut l = [0u8; 8];
    l.copy_from_slice(&sec[..8]);
    let index_len = u64::from_le_bytes(l);
    if index_len > MAX_BLOB_INDEX {
        bail!("limit_exceeded", "blob index is {index_len} B (limit {MAX_BLOB_INDEX})");
    }
    let data_start = 8 + index_len;
    if data_start > sec.len() as u64 {
        bail!("format", "blob index ({index_len} B) exceeds the BLOBS section ({} B)", sec.len());
    }
    let blobs: Vec<BlobInfo> =
        serde_json::from_slice(&sec[8..data_start as usize]).map_err(|e| fmt(format!("blob index JSON: {e}")))?;
    if blobs.len() > MAX_BLOBS {
        bail!("limit_exceeded", "{} blobs (limit {MAX_BLOBS})", blobs.len());
    }
    let mut names = HashSet::with_capacity(blobs.len());
    let mut spans = Vec::with_capacity(blobs.len());
    for b in &blobs {
        check_name(&b.name, "format")?;
        if !names.insert(b.name.as_str()) {
            bail!("format", "duplicate blob {:?}", b.name);
        }
        check_media_type(&b.media_type, "format").map_err(|e| fmt(format!("blob {:?}: {}", b.name, e.message)))?;
        if b.offset % BLOB_ALIGN != 0 {
            bail!("format", "blob {:?}: offset {} not {BLOB_ALIGN}-aligned", b.name, b.offset);
        }
        let end = b.offset.checked_add(b.nbytes).ok_or_else(|| fmt(format!("blob {:?} range overflows", b.name)))?;
        if b.offset < data_start || end > sec.len() as u64 {
            bail!("format", "blob {:?} [{}, {end}) outside the BLOBS data [{data_start}, {})", b.name, b.offset, sec.len());
        }
        if !is_hex_hash(&b.hash) {
            bail!("format", "blob {:?}: hash is not 64 hex chars", b.name);
        }
        spans.push((b.offset, end, format!("blob {:?}", b.name)));
    }
    check_disjoint(&mut spans)?;
    Ok(blobs)
}

/// Blob placement: the index entries and their JSON for blobs given as
/// `(name, media_type, nbytes, hash)` (already validated); returns also the section length.
pub(crate) fn layout(entries: &[(&str, &str, u64, &str)]) -> Result<(Vec<BlobInfo>, Vec<u8>, u64)> {
    // Offsets depend on the index length, which depends on the offsets' digits: iterate
    // until the layout is stable (converges in a couple of rounds).
    let mut index_len = 0u64;
    let (infos, index) = loop {
        let mut pos = align_up(8 + index_len, BLOB_ALIGN);
        let mut infos = Vec::with_capacity(entries.len());
        for &(name, media_type, nbytes, hash) in entries {
            infos.push(BlobInfo { name: name.into(), media_type: media_type.into(), offset: pos, nbytes, hash: hash.into() });
            pos = align_up(pos + nbytes, BLOB_ALIGN);
        }
        let index = serde_json::to_vec(&infos)?;
        if align_up(8 + index.len() as u64, BLOB_ALIGN) == align_up(8 + index_len, BLOB_ALIGN) {
            break (infos, index);
        }
        index_len = index.len() as u64;
    };
    if index.len() as u64 > MAX_BLOB_INDEX {
        bail!("limit_exceeded", "blob index is {} B (limit {MAX_BLOB_INDEX})", index.len());
    }
    let total = infos.last().map_or(8 + index.len() as u64, |b| b.offset + b.nbytes);
    if total > MAX_BLOBS_SECTION {
        bail!("limit_exceeded", "BLOBS section is {total} B (limit {MAX_BLOBS_SECTION})");
    }
    Ok((infos, index, total))
}

/// Encodes a whole BLOBS section in memory (tests).
#[cfg(test)]
pub(crate) fn encode(entries: &[(String, String, Vec<u8>)]) -> Result<Vec<u8>> {
    let hashes: Vec<String> = entries.iter().map(|e| blake3::hash(&e.2).to_hex().to_string()).collect();
    let list: Vec<(&str, &str, u64, &str)> =
        entries.iter().zip(&hashes).map(|(e, h)| (e.0.as_str(), e.1.as_str(), e.2.len() as u64, h.as_str())).collect();
    let (infos, index, total) = layout(&list)?;
    let mut out = vec![0u8; total as usize];
    out[..8].copy_from_slice(&(index.len() as u64).to_le_bytes());
    out[8..8 + index.len()].copy_from_slice(&index);
    for (b, (_, _, bytes)) in infos.iter().zip(entries) {
        out[b.offset as usize..(b.offset + b.nbytes) as usize].copy_from_slice(bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<(String, String, Vec<u8>)> {
        vec![
            ("tokenizer/tokens.json".into(), "application/json".into(), b"[\"a\",\"b\"]".to_vec()),
            ("empty".into(), "application/octet-stream".into(), vec![]),
            ("voice/default.bin".into(), "application/octet-stream".into(), (0..1000u32).map(|i| i as u8).collect()),
        ]
    }

    #[test]
    fn round_trip() {
        let e = entries();
        let sec = encode(&e).expect("encode");
        let blobs = parse(&sec).expect("parse");
        assert_eq!(blobs.len(), 3);
        for (b, (name, mt, bytes)) in blobs.iter().zip(&e) {
            assert_eq!((&b.name, &b.media_type), (name, mt));
            assert_eq!(b.offset % BLOB_ALIGN, 0);
            assert_eq!(&sec[b.offset as usize..(b.offset + b.nbytes) as usize], &bytes[..]);
        }
        assert!(parse(&encode(&[]).expect("empty")).expect("parse").is_empty());
    }

    #[test]
    fn bad_sections_rejected() {
        let sec = encode(&entries()).expect("encode");
        assert_eq!(parse(&sec[..4]).err().map(|e| e.code), Some("format"));
        // Truncated: last blob runs past the end.
        assert_eq!(parse(&sec[..sec.len() - 1]).err().map(|e| e.code), Some("format"));
        // Index length beyond the limit.
        let mut big = sec.clone();
        big[..8].copy_from_slice(&(MAX_BLOB_INDEX + 1).to_le_bytes());
        assert_eq!(parse(&big).err().map(|e| e.code), Some("limit_exceeded"));
        // Overlapping blobs (rewrite the index by hand).
        let mut blobs = parse(&sec).expect("parse");
        blobs[2].offset = blobs[0].offset;
        let idx = serde_json::to_vec(&blobs).expect("json");
        let mut s = vec![0u8; sec.len()];
        s[..8].copy_from_slice(&(idx.len() as u64).to_le_bytes());
        s[8..8 + idx.len()].copy_from_slice(&idx);
        assert_eq!(parse(&s).err().map(|e| e.code), Some("format"));
        assert!(check_media_type("json", "x").is_err());
        assert!(check_media_type("application/json; charset=utf-8", "x").is_err());
        assert!(check_media_type("application/json;charset=utf-8", "x").is_ok());
    }
}
