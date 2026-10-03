use burn::prelude::*;
use burn::tensor::DType;

use crate::config::EncoderConfig;
use crate::encoder::{EncoderInput, ModernBert};
use crate::head::{OptionScorer, OptionScorerConfig};

/// Encoder plus option scorer: one forward pass gives one logit per option
/// marker. Field names match Von's `OptionMarkerModel` state dict.
#[derive(Module, Debug)]
pub struct DecisionModel<B: Backend> {
    pub encoder: ModernBert<B>,
    pub scorer: OptionScorer<B>,
}

impl EncoderConfig {
    pub fn init_decision_model<B: Backend>(&self, device: &B::Device) -> DecisionModel<B> {
        DecisionModel {
            encoder: self.init(device),
            scorer: OptionScorerConfig::new(self.hidden_size).init(device),
        }
    }
}

impl<B: Backend> DecisionModel<B> {
    /// Logits for every marker of every row, flattened in row order, as f32
    /// whatever the model's precision.
    pub fn option_logits(&self, input: EncoderInput<B>, markers: &[Vec<usize>]) -> Tensor<B, 1> {
        let hidden = self.encoder.forward(input);
        let device = hidden.device();
        let [rows, width, size] = hidden.dims();
        debug_assert_eq!(rows, markers.len());

        let flat: Vec<i64> = markers
            .iter()
            .enumerate()
            .flat_map(|(row, positions)| positions.iter().map(move |&p| row * width + p))
            .map(|i| i64::try_from(i).expect("token index fits in i64"))
            .collect();
        let count = flat.len();
        let index = Tensor::<B, 1, Int>::from_data(TensorData::new(flat, [count]), &device);

        let states = hidden.reshape([rows * width, size]).select(0, index);
        self.scorer.forward(states).cast(DType::F32)
    }
}
