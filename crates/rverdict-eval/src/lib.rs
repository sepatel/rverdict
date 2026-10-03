//! Benchmarks and metrics for rverdict decision models.

pub mod jevbench;
pub mod score;

use std::path::PathBuf;

pub use score::{Outcome, Summary, run, summarize};

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
