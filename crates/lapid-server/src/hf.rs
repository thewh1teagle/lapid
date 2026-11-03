//! llama.cpp-style `-hf owner/repo[:QUANT]`: list the repo's files, pick the GGUF matching
//! the quant tag (default `Q4_K_M`, like llama.cpp; `mmproj` files are skipped) and download
//! it into the lapid HF cache. Repos without GGUF files are returned as plain HF repo ids
//! (safetensors checkpoints the hub converts). Offline, an already-cached match is used.

use std::path::{Path, PathBuf};

use lapid::{Result, bail};

/// Split `owner/repo[:QUANT]`.
pub fn split_tag(spec: &str) -> (&str, Option<&str>) {
    match spec.rsplit_once(':') {
        Some((repo, tag)) if !tag.is_empty() && !tag.contains('/') => (repo, Some(tag)),
        _ => (spec, None),
    }
}

/// Choose the GGUF for `tag` among `files` (repo-relative paths).
pub fn pick_gguf(files: &[String], tag: Option<&str>) -> Result<Option<String>> {
    let ggufs: Vec<&String> = files
        .iter()
        .filter(|f| f.to_ascii_lowercase().ends_with(".gguf") && !f.to_ascii_lowercase().contains("mmproj"))
        .collect();
    if ggufs.is_empty() {
        return Ok(None);
    }
    let want = tag.unwrap_or("Q4_K_M").to_ascii_lowercase();
    let stem = |f: &str| f.rsplit('/').next().unwrap_or(f).to_ascii_lowercase().trim_end_matches(".gguf").to_string();
    // Exact suffix ("…-Q4_K_M.gguf" / "….Q4_K_M.gguf") first, then a substring match.
    let exact = ggufs.iter().find(|f| {
        let s = stem(f);
        s.ends_with(&format!("-{want}")) || s.ends_with(&format!(".{want}")) || s.ends_with(&format!("_{want}")) || s == want
    });
    let chosen = exact.or_else(|| ggufs.iter().find(|f| stem(f).contains(&want)));
    match chosen {
        Some(f) if f.contains("-of-0") => bail!("unsupported", "{f}: split GGUF files are not supported yet"),
        Some(f) => Ok(Some((*f).clone())),
        None if tag.is_none() && ggufs.len() == 1 => Ok(Some(ggufs[0].clone())),
        None => {
            let names: Vec<String> = ggufs.iter().map(|f| stem(f)).collect();
            bail!("not_found", "no GGUF for quant {:?} (available: {})", tag.unwrap_or("Q4_K_M"), names.join(", "))
        }
    }
}

/// Files already in the lapid HF cache for `repo`.
fn cached_files(repo: &str) -> Vec<String> {
    let root = lapid_convert::cache_dir().join("hf").join(repo);
    let mut out = Vec::new();
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    walk(&root, &root, &mut out);
    out
}

/// Resolve `-hf owner/repo[:QUANT]` to a model source: a downloaded GGUF path, or the repo
/// id itself when it has no GGUF files.
pub fn resolve_hf(spec: &str) -> Result<String> {
    let (repo, tag) = split_tag(spec);
    if repo.split('/').count() != 2 {
        bail!("invalid_argument", "-hf {spec:?}: expected owner/repo[:QUANT]");
    }
    let files = match list_repo(repo) {
        Ok(f) => f,
        Err(e) => {
            let cached = cached_files(repo);
            if cached.is_empty() {
                return Err(e);
            }
            eprintln!("lapid: cannot list {repo} ({}); using the cache", e.message);
            cached
        }
    };
    match pick_gguf(&files, tag)? {
        Some(file) => {
            let path: PathBuf = lapid_convert::download_hf(repo, &file)?;
            Ok(path.to_string_lossy().into_owned())
        }
        None if tag.is_some() => bail!("not_found", "{repo} has no GGUF files (drop the :{} tag)", tag.unwrap_or("")),
        None => Ok(repo.to_string()),
    }
}

/// The repo's files (`api/models/{repo}`), through lapid-convert's HF client (endpoint,
/// token, and the fallback when HF rejects a stale token).
fn list_repo(repo: &str) -> Result<Vec<String>> {
    let body = lapid_convert::hf_api(&format!("api/models/{repo}"))?;
    let v: serde_json::Value = serde_json::from_str(&body)?;
    let files = v["siblings"]
        .as_array()
        .map(|s| s.iter().filter_map(|f| f["rfilename"].as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    Ok(files)
}

pub use lapid_hub::runtime::resolve::dtype_for_gguf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_quant() {
        let files: Vec<String> = ["README.md", "m-Q4_K_M.gguf", "m-Q8_0.gguf", "mmproj-m-f16.gguf", "m-Q4_K_S.gguf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pick_gguf(&files, None).ok().flatten().as_deref(), Some("m-Q4_K_M.gguf"));
        assert_eq!(pick_gguf(&files, Some("q8_0")).ok().flatten().as_deref(), Some("m-Q8_0.gguf"));
        assert_eq!(pick_gguf(&files, Some("Q5_K_M")).err().map(|e| e.code), Some("not_found"));
        assert_eq!(pick_gguf(&["config.json".to_string()], None).ok().flatten(), None);
        assert_eq!(split_tag("a/b:Q4_K_M"), ("a/b", Some("Q4_K_M")));
        assert_eq!(split_tag("a/b"), ("a/b", None));
        assert_eq!(dtype_for_gguf("x-Q8_0.gguf"), "int8");
        assert_eq!(dtype_for_gguf("x-Q4_K_M.gguf"), "q4_k_m");
    }
}
