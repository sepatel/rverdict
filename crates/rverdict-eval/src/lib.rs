//! Benchmarks and metrics for rverdict decision models.

pub mod calibrate;
pub mod datasets;
pub mod jevbench;
pub mod score;
pub mod task;

use std::path::PathBuf;

pub use score::{Outcome, Summary, run, summarize};
pub use task::Task;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("downloading benchmark data")]
    Http(#[from] reqwest::Error),
    #[error("reading {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing benchmark data")]
    Json(#[from] serde_json::Error),
}

/// FNV-1a: a stable, platform-independent hash for sampling and splits.
pub(crate) fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}
