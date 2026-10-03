//! Backend-free pieces of rverdict: the Jev-compatible wire format, question
//! rendering, and the math from logits to calibrated answers. No ML
//! dependencies, so clients and servers can share it cheaply.

mod cache;
mod ordered;
pub mod render;
pub mod score;
pub mod wire;

pub use cache::cache_root;
pub use ordered::OrderedMap;
pub use render::{Rendered, RenderedKind, render, state_text};
pub use score::{Calibration, NoulDecision, NoulPrior, TemperatureMap};
pub use wire::{
    Answer, InvalidQuestion, NoulCriteria, Question, Request, Response, Truncation, Usage,
};
