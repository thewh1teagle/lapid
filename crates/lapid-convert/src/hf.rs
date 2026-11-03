//! Hugging Face Hub downloads: `https://huggingface.co/{repo}/resolve/main/{file}`,
//! cached at `cache_dir()/hf/{repo}/{file}`. Streams to `<dest>.partial-<pid>` and
//! renames atomically. The bearer token is `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), else
//! the file `huggingface-cli login` writes: `$HF_TOKEN_PATH`, else `$HF_HOME/token`, else
//! `$XDG_CACHE_HOME/huggingface/token`, else `~/.cache/huggingface/token`.
//! Without the `download` feature only cached files are served ("download_disabled").

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use lapid_tensor::{Result, bail};

use crate::cache_dir;

fn valid_component(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Validates `owner/name` repo ids and relative file paths (no `..`, no absolute paths).
pub(crate) fn check_repo_file(repo: &str, file: &str) -> Result<()> {
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || !parts.iter().all(|p| valid_component(p)) {
        bail!("invalid_argument", "invalid Hugging Face repo id {repo:?} (expected owner/name)");
    }
    if !file.split('/').all(valid_component) {
        bail!("invalid_argument", "invalid repo file path {file:?}");
    }
    Ok(())
}

/// Download `file` from HF repo `repo` (revision `main`) into the cache; returns the
/// cached path. Already-cached files are returned without network access.
pub fn download_hf(repo: &str, file: &str) -> Result<PathBuf> {
    check_repo_file(repo, file)?;
    let dest = cache_dir().join("hf").join(repo).join(file);
    if dest.is_file() {
        return Ok(dest);
    }
    net::fetch(repo, file, &dest)?;
    Ok(dest)
}

/// GET a Hugging Face Hub API path (e.g. `api/models/{repo}`) as text, with the same
/// endpoint (`HF_ENDPOINT`) and token handling as the downloads.
pub fn hf_api(path: &str) -> Result<String> {
    net::api(path)
}

/// Download, treating 404 as "absent" (`Ok(None)`).
pub fn download_hf_optional(repo: &str, file: &str) -> Result<Option<PathBuf>> {
    match download_hf(repo, file) {
        Ok(p) => Ok(Some(p)),
        Err(e) if e.code == "not_found" => Ok(None),
        Err(e) => Err(e),
    }
}

/// Single-line progress on stderr (`\r`-updated on a terminal, sparse lines otherwise).
#[cfg_attr(not(feature = "download"), allow(dead_code))]
pub(crate) struct Progress {
    label: String,
    total: Option<u64>,
    start: Instant,
    last: Instant,
    tty: bool,
}

#[cfg_attr(not(feature = "download"), allow(dead_code))]
impl Progress {
    pub(crate) fn new(label: String, total: Option<u64>) -> Progress {
        let now = Instant::now();
        Progress { label, total, start: now, last: now, tty: std::io::stderr().is_terminal() }
    }

    fn line(&self, done: u64) -> String {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        let secs = self.start.elapsed().as_secs_f64().max(1e-3);
        let rate = mib(done) / secs;
        match self.total {
            Some(t) if t > 0 => format!(
                "downloading {}: {:.1}/{:.1} MiB ({:.0}%) {:.1} MiB/s",
                self.label,
                mib(done),
                mib(t),
                done as f64 * 100.0 / t as f64,
                rate
            ),
            _ => format!("downloading {}: {:.1} MiB {:.1} MiB/s", self.label, mib(done), rate),
        }
    }

    pub(crate) fn update(&mut self, done: u64) {
        let every = if self.tty { Duration::from_millis(200) } else { Duration::from_secs(5) };
        if self.last.elapsed() < every {
            return;
        }
        self.last = Instant::now();
        let mut err = std::io::stderr().lock();
        let _ = if self.tty { write!(err, "\r\x1b[2K{}", self.line(done)) } else { writeln!(err, "{}", self.line(done)) };
        let _ = err.flush();
    }

    pub(crate) fn finish(&mut self, done: u64) {
        let mut err = std::io::stderr().lock();
        let _ = if self.tty { writeln!(err, "\r\x1b[2K{}", self.line(done)) } else { writeln!(err, "{}", self.line(done)) };
    }
}

#[cfg(feature = "download")]
mod net {
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};
    use std::path::Path;
    use std::sync::{Arc, OnceLock};

    use lapid_tensor::{Error, Result, bail};
    use ureq::tls::{TlsConfig, TlsProvider};

    use super::Progress;

    const ENDPOINT: &str = "https://huggingface.co";

    fn token() -> Option<String> {
        super::hf_token(&|k| std::env::var_os(k))
    }

    fn endpoint() -> String {
        std::env::var("HF_ENDPOINT").ok().filter(|e| !e.is_empty()).unwrap_or_else(|| ENDPOINT.to_string())
    }

    /// HTTP agent with rustls over the pure-Rust RustCrypto provider (no C in the build).
    fn agent() -> &'static ureq::Agent {
        static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
        AGENT.get_or_init(|| {
            let tls = TlsConfig::builder()
                .provider(TlsProvider::Rustls)
                .unversioned_rustls_crypto_provider(Arc::new(rustls_rustcrypto::provider()))
                .build();
            ureq::Agent::config_builder().tls_config(tls).build().new_agent()
        })
    }

    /// GET `url` with the HF token when there is one. A token Hugging Face rejects (401:
    /// expired or revoked; public repos would work without it) is dropped with a warning and
    /// the request is sent again without it.
    pub(super) fn get(url: &str) -> Result<ureq::http::Response<ureq::Body>> {
        static REJECTED: OnceLock<()> = OnceLock::new();
        let send = |auth: bool| {
            let mut req = agent().get(url).header("User-Agent", concat!("lapid/", env!("CARGO_PKG_VERSION")));
            if let Some(t) = token().filter(|_| auth) {
                req = req.header("Authorization", format!("Bearer {t}"));
            }
            req.call()
        };
        let with_token = token().is_some() && REJECTED.get().is_none();
        match send(with_token) {
            Err(ureq::Error::StatusCode(401)) if with_token => {
                if REJECTED.set(()).is_ok() {
                    eprintln!(
                        "lapid: Hugging Face rejected your access token (expired or revoked? from HF_TOKEN or ~/.cache/huggingface/token); continuing without it"
                    );
                }
                send(false).map_err(|e| http_error(url, e))
            }
            r => r.map_err(|e| http_error(url, e)),
        }
    }

    /// GET an HF API path (`api/models/{repo}`) as text.
    pub(super) fn api(path: &str) -> Result<String> {
        let url = format!("{}/{path}", endpoint());
        get(&url)?.into_body().read_to_string().map_err(|e| Error::new("http", format!("{url}: {e}")))
    }

    fn http_error(url: &str, e: ureq::Error) -> Error {
        match e {
            ureq::Error::StatusCode(404) => Error::new("not_found", format!("{url}: 404 not found")),
            ureq::Error::StatusCode(c @ (401 | 403)) => Error::new(
                "unauthorized",
                format!("{url}: HTTP {c} (gated or private repo? set HF_TOKEN to a token with access)"),
            ),
            ureq::Error::StatusCode(c) => Error::new("http", format!("{url}: HTTP {c}")),
            ureq::Error::Io(io) => Error::new("io", format!("{url}: {io}")),
            other => Error::new("http", format!("{url}: {other}")),
        }
    }

    /// Stream `repo/file` into `dest` (atomic rename).
    pub(super) fn fetch(repo: &str, file: &str, dest: &Path) -> Result<()> {
        let dir = dest.parent().ok_or_else(|| Error::new("io", "cache path has no parent"))?;
        std::fs::create_dir_all(dir).map_err(|e| Error::new("io", format!("create {}: {e}", dir.display())))?;
        let url = format!("{}/{repo}/resolve/main/{file}", endpoint());
        let resp = get(&url)?;
        let total = resp.body().content_length();
        let mut reader = resp.into_body().into_reader();

        let tmp = dir.join(format!(
            ".{}.partial-{}",
            dest.file_name().and_then(|n| n.to_str()).unwrap_or("download"),
            std::process::id()
        ));
        let result = (|| -> Result<u64> {
            let f = File::create(&tmp).map_err(|e| Error::new("io", format!("create {}: {e}", tmp.display())))?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            let mut buf = vec![0u8; 1 << 20];
            let mut done: u64 = 0;
            let mut progress = Progress::new(format!("{repo}/{file}"), total);
            loop {
                let n = match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::new("io", format!("{url}: {e}"))),
                };
                w.write_all(&buf[..n])?;
                done += n as u64;
                progress.update(done);
            }
            progress.finish(done);
            let f = w.into_inner().map_err(|e| Error::new("io", e.error().to_string()))?;
            f.sync_all()?;
            if let Some(t) = total
                && t != done
            {
                bail!("io", "{url}: truncated download ({done} of {t} bytes)");
            }
            Ok(done)
        })();
        match result {
            Ok(_) => {
                std::fs::rename(&tmp, dest)
                    .map_err(|e| Error::new("io", format!("rename {} -> {}: {e}", tmp.display(), dest.display())))?;
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }
}

#[cfg(not(feature = "download"))]
mod net {
    use std::path::Path;

    use lapid_tensor::{Error, Result};

    pub(super) fn api(path: &str) -> Result<String> {
        Err(Error::new("download_disabled", format!("cannot read {path}: lapid was built without the \"download\" feature")))
    }

    pub(super) fn fetch(repo: &str, file: &str, dest: &Path) -> Result<()> {
        Err(Error::new(
            "download_disabled",
            format!(
                "{repo}/{file} is not cached ({}) and lapid was built without the \"download\" feature",
                dest.display()
            ),
        ))
    }
}

/// HF access token: env `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, else the token file of
/// `huggingface_hub` (see the module docs). `env` reads a variable (injected for tests).
#[cfg_attr(not(feature = "download"), allow(dead_code))]
pub(crate) fn hf_token(env: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Option<String> {
    let nonempty = |k: &str| env(k).filter(|v| !v.is_empty());
    for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = nonempty(k).and_then(|v| v.into_string().ok()).map(|t| t.trim().to_string())
            && !t.is_empty()
        {
            return Some(t);
        }
    }
    let path = if let Some(p) = nonempty("HF_TOKEN_PATH") {
        PathBuf::from(p)
    } else if let Some(h) = nonempty("HF_HOME") {
        PathBuf::from(h).join("token")
    } else if let Some(x) = nonempty("XDG_CACHE_HOME") {
        PathBuf::from(x).join("huggingface").join("token")
    } else {
        PathBuf::from(nonempty("HOME").or_else(|| nonempty("USERPROFILE"))?).join(".cache").join("huggingface").join("token")
    };
    let t = std::fs::read_to_string(path).ok()?;
    let t = t.trim();
    if t.is_empty() { None } else { Some(t.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_lookup() {
        let dir = std::env::temp_dir().join(format!("lapid-hf-token-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".cache/huggingface")).expect("mkdir");
        std::fs::write(dir.join(".cache/huggingface/token"), "hf_home_default\n").expect("write");
        std::fs::write(dir.join("token"), "hf_from_hf_home").expect("write");
        let d = dir.clone();
        let env = move |vars: Vec<(&'static str, std::ffi::OsString)>| {
            move |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone())
        };
        let home = d.clone().into_os_string();
        assert_eq!(hf_token(&env(vec![("HOME", home.clone())])).as_deref(), Some("hf_home_default"));
        assert_eq!(hf_token(&env(vec![("HOME", home.clone()), ("HF_HOME", d.clone().into_os_string())])).as_deref(), Some("hf_from_hf_home"));
        assert_eq!(hf_token(&env(vec![("HOME", home.clone()), ("HF_TOKEN", " hf_env ".into())])).as_deref(), Some("hf_env"));
        assert_eq!(hf_token(&env(vec![("HOME", home), ("HF_TOKEN", "".into()), ("HF_TOKEN_PATH", d.join("nope").into_os_string())])), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repo_validation() {
        assert!(check_repo_file("openai/whisper-small", "config.json").is_ok());
        assert!(check_repo_file("openai/whisper-small", "sub/dir/model.safetensors").is_ok());
        assert!(check_repo_file("openai", "config.json").is_err());
        assert!(check_repo_file("openai/../etc", "config.json").is_err());
        assert!(check_repo_file("openai/whisper-small", "../x").is_err());
        assert!(check_repo_file("openai/whisper-small", "/abs").is_err());
    }
}
