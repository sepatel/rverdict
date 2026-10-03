use std::path::PathBuf;

/// Where rverdict keeps converted weights and benchmark data:
/// `$RVERDICT_CACHE`, else `$XDG_CACHE_HOME/rverdict`, `~/.cache/rverdict`,
/// or `%LOCALAPPDATA%\rverdict`.
pub fn cache_root() -> PathBuf {
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
