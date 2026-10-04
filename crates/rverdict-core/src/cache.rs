use std::path::PathBuf;
use std::sync::OnceLock;

static OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Makes [`cache_root`] return `dir` for the rest of the process, for apps
/// that keep everything in their own data directory. Only the first call
/// takes effect; returns whether this one did.
pub fn set_cache_root(dir: PathBuf) -> bool {
    OVERRIDE.set(dir).is_ok()
}

/// Where rverdict keeps converted weights and compiled kernels: the
/// directory given to [`set_cache_root`], else `$RVERDICT_CACHE`, else
/// `$XDG_CACHE_HOME/rverdict`, `~/.cache/rverdict`, or
/// `%LOCALAPPDATA%\rverdict`.
pub fn cache_root() -> PathBuf {
    if let Some(dir) = OVERRIDE.get() {
        return dir.clone();
    }
    if let Some(dir) = std::env::var_os("RVERDICT_CACHE") {
        return PathBuf::from(dir);
    }
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir)
        .join("rverdict")
}
