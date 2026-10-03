//! ModernBERT encoder and option-marker decision head, generic over any Burn
//! backend so the same code serves inference and training.

mod config;
mod decision;
mod encoder;
mod head;
mod layout;
mod load;
mod rope;

use std::path::PathBuf;

pub use config::{AttentionKind, EncoderConfig};
pub use decision::DecisionModel;
pub use encoder::{EncoderInput, ModernBert};
pub use head::{OptionScorer, OptionScorerConfig};
pub use layout::{OptionAttention, PackedSequence, build_input};
pub use load::{
    Precision, load_decision_pytorch, load_decision_safetensors, load_encoder_safetensors,
    save_decision_safetensors,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing model config")]
    Json(#[from] serde_json::Error),
    #[error("unsupported model config: {0}")]
    Config(String),
    #[error("loading weights from {path}: {detail}")]
    Weights { path: PathBuf, detail: String },
}
