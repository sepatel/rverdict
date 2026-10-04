use crate::wire::{InvalidQuestion, Request, Response};

/// Anything that answers System One requests: the local engine, or a stub
/// in tests. Servers and hosts depend on this rather than on a backend.
pub trait Decider: Send + Sync {
    fn decide(&self, request: &Request) -> Result<Response, DecideError>;

    /// The model name reported in responses.
    fn model(&self) -> &str;
}

#[derive(Debug, thiserror::Error)]
pub enum DecideError {
    /// The request itself is malformed; the caller must fix it (HTTP 422).
    #[error(transparent)]
    Invalid(#[from] InvalidQuestion),
    /// The decider failed on a valid request (HTTP 500).
    #[error("{0}")]
    Failed(String),
}
