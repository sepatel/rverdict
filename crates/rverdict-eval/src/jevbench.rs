//! JevBench v1's public set (MIT, `fstandhartinger/jevbench`): 231 typed
//! decisions in three tiers, scored with the benchmark's own rules.
//!
//! Temporary and isolated by design (plan section 9): the files are
//! downloaded at evaluation time into the user cache, never committed, never
//! trained on, and nothing else in rverdict depends on this module.

use std::path::{Path, PathBuf};

use crate::EvalError;
use crate::task::Task;

/// Commit the published items are read from, so results stay comparable.
pub const COMMIT: &str = "bb05a335bc809e61b20c0f745d25499a82b326fc";
pub const TIERS: [&str; 3] = ["easy", "original", "hard"];

/// Where JevBench files are cached, per pinned commit.
pub fn cache_dir() -> PathBuf {
    rverdict_core::cache_root().join("jevbench").join(COMMIT)
}

/// Loads the given tiers, downloading any that are not cached yet.
pub fn load(tiers: &[&str], dir: &Path) -> Result<Vec<Task>, EvalError> {
    std::fs::create_dir_all(dir).map_err(|source| EvalError::Io {
        path: dir.to_owned(),
        source,
    })?;
    let mut tasks = Vec::new();
    for tier in tiers {
        let path = dir.join(format!("{tier}.jsonl"));
        if !path.exists() {
            let url = format!(
                "https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public/{tier}.jsonl"
            );
            let body = reqwest::blocking::get(&url)?.error_for_status()?.text()?;
            std::fs::write(&path, body).map_err(|source| EvalError::Io {
                path: path.clone(),
                source,
            })?;
        }
        let text = std::fs::read_to_string(&path).map_err(|source| EvalError::Io {
            path: path.clone(),
            source,
        })?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut task: Task = serde_json::from_str(line)?;
            (*tier).clone_into(&mut task.subset);
            tasks.push(task);
        }
    }
    Ok(tasks)
}
