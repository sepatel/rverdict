use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

use crate::encoder::norm;

/// Scores each option marker's hidden state to one logit:
/// `LayerNorm → Linear(h, h/2) → GELU → LayerNorm → Dropout → Linear(h/2, 1)`.
/// The layout matches Von's `OptionMarkerScorer`, so its weights load as-is.
#[derive(Module, Debug)]
pub struct OptionScorer<B: Backend> {
    input_norm: LayerNorm<B>,
    dense: Linear<B>,
    norm: LayerNorm<B>,
    dropout: Dropout,
    out_proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct OptionScorerConfig {
    pub hidden_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl OptionScorerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> OptionScorer<B> {
        let half = self.hidden_size / 2;
        OptionScorer {
            input_norm: LayerNormConfig::new(self.hidden_size).init(device),
            dense: LinearConfig::new(self.hidden_size, half).init(device),
            norm: LayerNormConfig::new(half).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            out_proj: LinearConfig::new(half, 1).init(device),
        }
    }
}

impl<B: Backend> OptionScorer<B> {
    /// `[n, hidden]` marker states to `[n]` logits.
    pub fn forward(&self, markers: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = norm(&self.input_norm, markers);
        let x = norm(&self.norm, gelu(self.dense.forward(x)));
        self.out_proj
            .forward(self.dropout.forward(x))
            .squeeze_dim(1)
    }
}
