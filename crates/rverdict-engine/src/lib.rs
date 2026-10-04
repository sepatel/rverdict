//! The embeddable rverdict engine: load a checkpoint, pick a backend at
//! runtime, and answer System One requests.
//!
//! ```no_run
//! use rverdict_engine::{BackendChoice, Checkpoint, Engine, ModelRef, Precision, select};
//!
//! let checkpoint = Checkpoint::from_hub(&ModelRef::parse("von"))?;
//! let engine = Engine::load(&checkpoint, select(BackendChoice::Auto), Precision::F32)?;
//! let request = serde_json::from_str(r#"{"state": "Refund my duplicate charge",
//!     "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}}"#)?;
//! let response = engine.decide(&request)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod checkpoint;
mod device;
mod engine;
mod pack;

use std::path::PathBuf;

pub use checkpoint::{Checkpoint, ModelRef, Settings, Weights};
pub use device::{BackendChoice, SelectedBackend, select};
pub use engine::{Engine, Evaluated, RawQuestion};
pub use rverdict_model::Precision;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    InvalidRequest(#[from] rverdict_core::InvalidQuestion),
    #[error(transparent)]
    Model(#[from] rverdict_model::Error),
    #[error("downloading from Hugging Face")]
    Hub(#[from] hf_hub::HFError),
    #[error("{dir} has no {file}")]
    MissingFile { dir: PathBuf, file: String },
    #[error("{0:?} is not a model alias or owner/name")]
    BadModel(String),
    #[error("reading {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing checkpoint settings")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}
