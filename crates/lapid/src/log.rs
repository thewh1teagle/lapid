//! Diagnostic output (debug schedules, profiles, device fallback notes). Goes to stderr unless
//! an embedder installs a hook (e.g. the C API forwards it to its log callback).

use std::sync::{Arc, RwLock};

type Hook = Arc<dyn Fn(&str) + Send + Sync>;

/// A log callback: one call per diagnostic message.
pub type LogHook = Box<dyn Fn(&str) + Send + Sync>;

static HOOK: RwLock<Option<Hook>> = RwLock::new(None);

/// Route lapid's diagnostic lines to `hook` (one call per message, which may span lines);
/// `None` restores stderr.
pub fn set_log_hook(hook: Option<LogHook>) {
    let hook: Option<Hook> = hook.map(Arc::from);
    match HOOK.write() {
        Ok(mut g) => *g = hook,
        Err(p) => *p.into_inner() = hook,
    }
}

/// Write one diagnostic message (to the hook, else stderr).
pub fn emit(msg: &str) {
    let hook = match HOOK.read() {
        Ok(g) => g.clone(),
        Err(p) => p.into_inner().clone(),
    };
    match hook {
        Some(h) => h(msg),
        None => eprintln!("{msg}"),
    }
}
