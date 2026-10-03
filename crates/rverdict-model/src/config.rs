use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// Which attention pattern a layer uses. ModernBERT alternates a full
/// (global) layer with sliding-window (local) layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Global,
    Sliding,
}

/// A validated ModernBERT configuration, independent of which
/// `transformers` version wrote the `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub pad_token_id: u32,
    pub layers: Vec<AttentionKind>,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    /// Half of `local_attention`: a sliding layer attends to keys whose
    /// position differs from the query's by at most this much.
    pub sliding_window: usize,
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// A two-layer toy encoder with one global and one sliding layer, for
    /// backend self-tests and parity tests that must run in milliseconds.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            hidden_size: 32,
            num_attention_heads: 2,
            intermediate_size: 48,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            norm_bias: false,
            attention_bias: false,
            mlp_bias: false,
            pad_token_id: 0,
            layers: vec![AttentionKind::Global, AttentionKind::Sliding],
            global_rope_theta: 160_000.0,
            local_rope_theta: 10_000.0,
            sliding_window: 4,
        }
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, Error> {
        let raw: RawConfig = serde_json::from_str(text)?;
        raw.validate()
    }
}

/// The subset of a Hugging Face `ModernBertConfig` that affects inference.
/// Older files carry `global_rope_theta`/`local_rope_theta`; files written by
/// transformers 5 carry `rope_parameters` and `layer_types` instead.
#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<HashMap<String, RopeParameters>>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    #[serde(default)]
    rope_type: Option<String>,
}

impl RawConfig {
    fn validate(self) -> Result<EncoderConfig, Error> {
        let invalid = |reason: String| Err(Error::Config(reason));

        if self.model_type != "modernbert" {
            return invalid(format!(
                "model_type is {:?}, expected \"modernbert\"",
                self.model_type
            ));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return invalid(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if let Some(act) = self.hidden_activation.as_deref()
            && act != "gelu"
        {
            return invalid(format!(
                "hidden_activation {act:?} is not supported, only \"gelu\""
            ));
        }

        let layers = match &self.layer_types {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    "full_attention" => Ok(AttentionKind::Global),
                    "sliding_attention" => Ok(AttentionKind::Sliding),
                    other => Err(Error::Config(format!("unknown layer type {other:?}"))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => (0..self.num_hidden_layers)
                .map(|i| {
                    if i % self.global_attn_every_n_layers == 0 {
                        AttentionKind::Global
                    } else {
                        AttentionKind::Sliding
                    }
                })
                .collect(),
        };
        if layers.len() != self.num_hidden_layers {
            return invalid(format!(
                "layer_types has {} entries for {} layers",
                layers.len(),
                self.num_hidden_layers
            ));
        }

        let theta = |kind: &str, legacy: Option<f64>| -> Result<f64, Error> {
            if let Some(params) = self.rope_parameters.as_ref().and_then(|p| p.get(kind)) {
                if let Some(rope_type) = params.rope_type.as_deref()
                    && rope_type != "default"
                {
                    return Err(Error::Config(format!(
                        "rope_type {rope_type:?} for {kind} is not supported"
                    )));
                }
                return Ok(params.rope_theta);
            }
            legacy.ok_or_else(|| Error::Config(format!("no rope theta for {kind}")))
        };

        Ok(EncoderConfig {
            global_rope_theta: theta("full_attention", self.global_rope_theta)?,
            local_rope_theta: theta("sliding_attention", self.local_rope_theta)?,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            norm_eps: self
                .norm_eps
                .or(self.layer_norm_eps)
                .ok_or_else(|| Error::Config("no norm_eps".into()))?,
            norm_bias: self.norm_bias,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            pad_token_id: self.pad_token_id,
            layers,
            sliding_window: self.local_attention / 2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY: &str = r#"{
        "model_type": "modernbert", "vocab_size": 50368, "hidden_size": 1024,
        "num_hidden_layers": 28, "num_attention_heads": 16, "intermediate_size": 2624,
        "max_position_embeddings": 8192, "layer_norm_eps": 1e-5, "norm_eps": 1e-5,
        "pad_token_id": 50283, "global_attn_every_n_layers": 3, "local_attention": 128,
        "global_rope_theta": 160000.0, "local_rope_theta": 10000.0, "hidden_activation": "gelu"
    }"#;

    #[test]
    fn legacy_and_v5_configs_resolve_identically() {
        let v5 = r#"{
            "model_type": "modernbert", "vocab_size": 50368, "hidden_size": 1024,
            "num_hidden_layers": 4, "num_attention_heads": 16, "intermediate_size": 2624,
            "max_position_embeddings": 8192, "norm_eps": 1e-5,
            "pad_token_id": 50283, "global_attn_every_n_layers": 3, "local_attention": 128,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let legacy = EncoderConfig::from_json(
            &LEGACY.replace("\"num_hidden_layers\": 28", "\"num_hidden_layers\": 4"),
        )
        .unwrap();
        assert_eq!(EncoderConfig::from_json(v5).unwrap(), legacy);
        assert_eq!(legacy.sliding_window, 64);
    }
}
