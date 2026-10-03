use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use burn::prelude::*;
use candle_transformers::models::modernbert as candle_modernbert;
use rverdict_model::{
    AttentionKind, EncoderConfig, OptionAttention, PackedSequence, build_input,
    load_encoder_safetensors,
};

use crate::fetch::ModelFiles;
use crate::von::Packer;

pub struct Diff {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub mean_magnitude: f32,
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |Δ| {:.3e}, mean |Δ| {:.3e}, mean |x| {:.3e}",
            self.max_abs, self.mean_abs, self.mean_magnitude
        )
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "element counts are far below f32 precision limits"
)]
pub fn diff(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len(), "compared tensors differ in size");
    let n = a.len() as f32;
    let deltas = a.iter().zip(b).map(|(x, y)| (x - y).abs());
    Diff {
        max_abs: deltas.clone().fold(0.0, f32::max),
        mean_abs: deltas.sum::<f32>() / n,
        mean_magnitude: b.iter().map(|x| x.abs()).sum::<f32>() / n,
    }
}

/// Encoder last hidden state from Burn on backend `B`.
pub fn burn_hidden<B: Backend>(
    device: &B::Device,
    config: &EncoderConfig,
    weights: &Path,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let mut encoder = config.init::<B>(device);
    load_encoder_safetensors(&mut encoder, weights)?;
    let seq = PackedSequence {
        token_ids: ids.to_vec(),
        markers: vec![],
    };
    let input = build_input::<B>(
        &[seq],
        OptionAttention::Shared,
        config.pad_token_id,
        config.sliding_window,
        device,
    );
    encoder
        .forward(input)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// The same hidden state from candle-transformers' independent ModernBERT
/// implementation: the reference the Burn port is checked against.
pub fn candle_hidden(config: &EncoderConfig, weights: &Path, ids: &[u32]) -> Result<Vec<f32>> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(weights, &device)?
        .into_iter()
        .map(|(name, t)| {
            let name = if name.starts_with("model.") {
                name
            } else {
                format!("model.{name}")
            };
            (name, t)
        })
        .collect();
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &device);
    let every = config
        .layers
        .iter()
        .skip(1)
        .position(|k| *k == AttentionKind::Global)
        .map_or(config.layers.len(), |i| i + 1);
    let cfg = candle_modernbert::Config {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.layers.len(),
        num_attention_heads: config.num_attention_heads,
        intermediate_size: config.intermediate_size,
        max_position_embeddings: config.max_position_embeddings,
        layer_norm_eps: config.norm_eps,
        pad_token_id: config.pad_token_id,
        global_attn_every_n_layers: every,
        global_rope_theta: config.global_rope_theta,
        local_attention: config.sliding_window * 2,
        local_rope_theta: config.local_rope_theta,
        classifier_config: None,
    };
    let model = candle_modernbert::ModernBert::load(vb, &cfg)?;
    let input = Tensor::new(ids, &device)?.unsqueeze(0)?;
    let mask = Tensor::ones((1, ids.len()), DType::F32, &device)?;
    Ok(model
        .forward(&input, &mask)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

pub const SAMPLE: &str = "Hi team, we were billed twice for the March invoice #4411 and the duplicate \
charge of $1,249.00 is still pending on our corporate card. Please refund it before Friday or we \
will have to escalate this with our account manager. The original order was placed on 2026-03-02 \
for 40 seats of the Business plan. Thanks, Dana from Acme Logistics.";

pub fn sample_ids(files: &ModelFiles, repeat: usize) -> Result<Vec<u32>> {
    let packer = Packer::from_file(&files.path("tokenizer.json"))?;
    let text = std::iter::repeat_n(SAMPLE, repeat)
        .collect::<Vec<_>>()
        .join(" ");
    packer.encode_text(&text)
}

pub fn run<B: Backend>(device: &B::Device, files: &ModelFiles, repeat: usize) -> Result<()> {
    let config = EncoderConfig::from_file(&files.path("config.json"))?;
    let weights = files.path("model.safetensors");
    let ids = sample_ids(files, repeat)?;
    println!("sequence of {} tokens", ids.len());

    let reference = candle_hidden(&config, &weights, &ids)?;
    let ours = burn_hidden::<B>(device, &config, &weights, &ids)?;
    println!("burn vs candle-transformers: {}", diff(&ours, &reference));
    Ok(())
}
