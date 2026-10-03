//! Von's request packing and calibration, ported from `von/models/option_marker.py`
//! and `von/backends/option_marker_backend.py` (Apache-2.0). Spike-local until
//! Phase 1 moves it into the engine.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rverdict_model::PackedSequence;
use serde_json::Value;
use tokenizers::Tokenizer;

const MASK: &str = "[MASK]";
const SEP: &str = "[SEP]";
pub const NOUL_TRUE: &str = "Yes, condition holds true.";
pub const NOUL_FALSE: &str = "No, condition is false.";

pub struct Packer {
    tokenizer: Tokenizer,
    mask_id: u32,
}

/// Breaks a special-token literal inside user text with a zero-width joiner,
/// so an email cannot forge an option marker or a separator.
fn neutralise(text: &str) -> String {
    [MASK, SEP].iter().fold(text.to_owned(), |text, special| {
        let (head, tail) = special.split_at(1);
        text.replace(special, &format!("{head}\u{200d}{tail}"))
    })
}

impl Packer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        let mask_id = tokenizer
            .token_to_id(MASK)
            .context("tokenizer has no [MASK] token")?;
        Ok(Self { tokenizer, mask_id })
    }

    pub fn text(state: &str, question: &str, options: &[&str]) -> String {
        let (state, question) = (neutralise(state), neutralise(question));
        let prefix = if question.is_empty() {
            state.trim().to_owned()
        } else {
            format!("{question} {state}").trim().to_owned()
        };
        let options = options
            .iter()
            .map(|o| format!("{MASK} {}", neutralise(o).trim()))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{prefix} {SEP} {options}")
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn pack(&self, state: &str, question: &str, options: &[&str]) -> Result<PackedSequence> {
        let token_ids = self.encode_text(&Self::text(state, question, options))?;
        let markers: Vec<usize> = token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.mask_id).then_some(i))
            .collect();
        anyhow::ensure!(
            markers.len() == options.len(),
            "packed {} markers for {} options",
            markers.len(),
            options.len()
        );
        Ok(PackedSequence { token_ids, markers })
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("decoding: {e}"))
    }

    pub fn state_tokens(&self, state: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(state, false)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(encoding.get_ids().len().max(1))
    }
}

/// `T = bias + entropy·H_norm + log_tokens·log10(tokens)/4 + n_options·K/8`,
/// clamped to `[lo, hi]`.
pub struct Calibration {
    bias: f64,
    entropy: f64,
    log_tokens: f64,
    n_options: f64,
    lo: f64,
    hi: f64,
    pub noul_prior: (f64, f64),
}

impl Calibration {
    pub fn from_file(path: &Path) -> Result<Self> {
        let json: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let map = &json["calibration_map"];
        let get = |key: &str| {
            map[key]
                .as_f64()
                .with_context(|| format!("calibration_map.{key}"))
        };
        let prior = &json["noul_zero_shot_prior"];
        Ok(Self {
            bias: get("bias")?,
            entropy: get("entropy")?,
            log_tokens: get("log_tokens")?,
            n_options: get("n_options")?,
            lo: get("lo")?,
            hi: get("hi")?,
            noul_prior: (
                prior["a"].as_f64().context("noul prior a")?,
                prior["b"].as_f64().context("noul prior b")?,
            ),
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "token and option counts are small"
    )]
    pub fn temperature(&self, logits: &[f32], state_tokens: usize) -> f64 {
        let n = logits.len();
        let probs = softmax(logits, 1.0);
        let entropy = if n > 1 {
            -probs.iter().map(|&p| p * p.max(1e-12).ln()).sum::<f64>() / (n as f64).ln()
        } else {
            0.0
        };
        let t = self.bias
            + self.entropy * entropy
            + self.log_tokens * (state_tokens as f64).log10() / 4.0
            + self.n_options * n as f64 / 8.0;
        t.clamp(self.lo, self.hi)
    }
}

pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from(l) / temperature.max(1e-4))
        .collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

/// `(n·p_max − 1)/(n − 1)`: 0 at uniform, 1 at one-hot, for any n.
#[expect(clippy::cast_precision_loss, reason = "option counts are small")]
pub fn confidence(probs: &[f64]) -> f64 {
    let n = probs.len() as f64;
    if probs.len() <= 1 {
        return 1.0;
    }
    let p_max = probs.iter().copied().fold(0.0, f64::max);
    ((n * p_max - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// Von's `band` rule: keeps the argmax but moves every yes/no out of the
/// 0.2–0.8 abstention band.
pub fn noul_band(p: f64) -> f64 {
    const EDGE: f64 = 0.8;
    const SLOPE: f64 = 0.1;
    let p = p.clamp(0.0, 1.0);
    let out = if p >= 0.5 {
        EDGE + SLOPE * (p - 0.5)
    } else {
        (1.0 - EDGE) - SLOPE * (0.5 - p)
    };
    out.clamp(0.0, 1.0)
}
