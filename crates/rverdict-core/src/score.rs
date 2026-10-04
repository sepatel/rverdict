//! From raw option logits to calibrated, typed answers. Pure math shared by
//! every backend.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::render::{Rendered, RenderedKind};
use crate::wire::Answer;

/// One question's model output before calibration. Captured once, it lets
/// calibration be refit without running the model again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Logits {
    pub logits: Vec<f32>,
    /// The same question asked of an empty state, for zero-shot `noul`
    /// debiasing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_logits: Option<Vec<f32>>,
    pub state_tokens: usize,
}

/// How a checkpoint's logits become calibrated probabilities. Read from the
/// checkpoint's calibration file and refit by `rverdict calibrate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Calibration {
    /// The scaling for any question type without its own.
    #[serde(flatten)]
    pub scaling: Scaling,
    /// Per question type (`noul`, `choice`, `score`): a model can be
    /// overconfident on one type and well calibrated on another, so one
    /// shared scaling would trade them off against each other.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub per_type: BTreeMap<String, Scaling>,
    /// Zero-shot `noul` debiasing, `correction = a·bias + b`, where `bias` is
    /// the true-minus-false logit gap of the question asked with no state.
    #[serde(
        default,
        rename = "noul_zero_shot_prior",
        skip_serializing_if = "Option::is_none"
    )]
    pub noul_prior: Option<NoulPrior>,
    #[serde(default)]
    pub noul_decision: NoulDecision,
}

/// A temperature, fixed or conditioned on the input.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scaling {
    /// Used when no `map` is present.
    #[serde(default = "one")]
    pub temperature: f64,
    #[serde(
        default,
        rename = "calibration_map",
        skip_serializing_if = "Option::is_none"
    )]
    pub map: Option<TemperatureMap>,
}

impl Default for Scaling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            map: None,
        }
    }
}

/// The key a question type's [`Scaling`] is stored under.
pub fn type_name(kind: &RenderedKind) -> &'static str {
    match kind {
        RenderedKind::Noul { .. } => "noul",
        RenderedKind::Choice { .. } => "choice",
        RenderedKind::Score => "score",
    }
}

fn one() -> f64 {
    1.0
}

/// Input-conditioned temperature,
/// `T = bias + entropy·H_norm + log_tokens·log10(state_tokens)/4 + n_options·K/8`,
/// clamped to `[lo, hi]`. Monotonic in the logits, so it changes confidence
/// but never the answer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemperatureMap {
    pub bias: f64,
    pub entropy: f64,
    pub log_tokens: f64,
    pub n_options: f64,
    #[serde(default = "map_lo")]
    pub lo: f64,
    #[serde(default = "map_hi")]
    pub hi: f64,
}

fn map_lo() -> f64 {
    0.5
}

fn map_hi() -> f64 {
    12.0
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoulPrior {
    pub a: f64,
    pub b: f64,
}

/// How a calibrated P(true) becomes the reported `noul`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "lowercase")]
pub enum NoulDecision {
    /// Report the calibrated probability.
    Raw,
    /// Von's band rule: keep the side of 0.5 but move every answer out of the
    /// `[1 − edge, edge]` abstention band.
    Band { edge: f64, slope: f64 },
}

impl Default for NoulDecision {
    fn default() -> Self {
        Self::Band {
            edge: 0.8,
            slope: 0.1,
        }
    }
}

impl Scaling {
    #[expect(
        clippy::cast_precision_loss,
        reason = "token and option counts are small"
    )]
    pub fn temperature_for(&self, logits: &[f32], state_tokens: usize) -> f64 {
        let Some(map) = self.map else {
            return self.temperature;
        };
        let n = logits.len();
        let entropy = if n > 1 {
            let probs = softmax(logits, 1.0);
            -probs.iter().map(|&p| p * p.max(1e-12).ln()).sum::<f64>() / (n as f64).ln()
        } else {
            0.0
        };
        let t = map.bias
            + map.entropy * entropy
            + map.log_tokens * (state_tokens.max(1) as f64).log10() / 4.0
            + map.n_options * n as f64 / 8.0;
        t.clamp(map.lo, map.hi)
    }
}

impl Calibration {
    pub fn scaling_for(&self, kind: &RenderedKind) -> &Scaling {
        self.per_type.get(type_name(kind)).unwrap_or(&self.scaling)
    }

    /// Calibrated probabilities over a question's options, in option order,
    /// before any `noul` decision rule. For a `noul` with `null_logits` and a
    /// configured prior, the state-free bias is removed first.
    pub fn distribution(&self, kind: &RenderedKind, raw: &Logits) -> Vec<f64> {
        let mut logits = raw.logits.clone();
        if let (RenderedKind::Noul { .. }, Some(null), Some(prior)) =
            (kind, &raw.null_logits, self.noul_prior)
        {
            let bias = f64::from(null[0] - null[1]);
            #[expect(clippy::cast_possible_truncation, reason = "logits are f32")]
            let correction = (prior.a * bias + prior.b) as f32;
            logits[0] -= correction;
        }
        softmax(
            &logits,
            self.scaling_for(kind)
                .temperature_for(&logits, raw.state_tokens),
        )
    }

    /// Turns one question's logits into its answer.
    pub fn answer(&self, rendered: &Rendered, raw: &Logits) -> Answer {
        let probs = self.distribution(&rendered.kind, raw);
        match &rendered.kind {
            RenderedKind::Noul { .. } => {
                let p = probs[0].clamp(0.0, 1.0);
                Answer::Noul {
                    noul: round4(self.noul_decision.apply(p)),
                    noul_raw: Some(round4(p)),
                }
            }
            RenderedKind::Choice { keys } => Answer::Choice {
                choice: keys[argmax(&probs)].clone(),
                confidence: margin_confidence(&probs),
                probabilities: keys
                    .iter()
                    .cloned()
                    .zip(probs.iter().map(|&p| round4(p)))
                    .collect(),
            },
            RenderedKind::Score => {
                #[expect(clippy::cast_precision_loss, reason = "at most 10 levels")]
                let score = probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum::<f64>();
                Answer::Score {
                    score: (score * 100.0).round() / 100.0,
                    confidence: margin_confidence(&probs),
                    legend: rendered
                        .options
                        .iter()
                        .enumerate()
                        .map(|(i, d)| (i.to_string(), d.clone()))
                        .collect(),
                    probabilities: probs
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (i.to_string(), round4(p)))
                        .collect(),
                }
            }
        }
    }
}

impl NoulDecision {
    pub fn apply(self, p: f64) -> f64 {
        match self {
            Self::Raw => p,
            Self::Band { edge, slope } => {
                let out = if p >= 0.5 {
                    edge + slope * (p - 0.5)
                } else {
                    (1.0 - edge) - slope * (0.5 - p)
                };
                out.clamp(0.0, 1.0)
            }
        }
    }
}

pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let t = temperature.max(1e-4);
    let scaled: Vec<f64> = logits.iter().map(|&l| f64::from(l) / t).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

/// `(n·p_max − 1)/(n − 1)`: 0 at uniform, 1 at one-hot, comparable across
/// option counts.
#[expect(clippy::cast_precision_loss, reason = "option counts are small")]
pub fn margin_confidence(probs: &[f64]) -> f64 {
    if probs.len() <= 1 {
        return 1.0;
    }
    let n = probs.len() as f64;
    let p_max = probs.iter().copied().fold(0.0, f64::max);
    (((n * p_max - 1.0) / (n - 1.0)).clamp(0.0, 1.0) * 1000.0).round() / 1000.0
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn round4(p: f64) -> f64 {
    (p * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn von_calibration_file_parses_with_its_field_names() {
        let calibration: Calibration = serde_json::from_str(
            r#"{"model_type": "option_marker", "independent_options": true, "temperature": 2.2,
                "calibration_map": {"bias": 2.0151, "entropy": -3.2369, "log_tokens": 11.4916,
                    "n_options": -3.5574, "lo": 0.3, "hi": 12.0},
                "noul_zero_shot_prior": {"a": -0.5, "b": 0.3}}"#,
        )
        .unwrap();
        assert_eq!(calibration.scaling.map.map(|m| m.lo), Some(0.3));
        assert_eq!(calibration.noul_prior, Some(NoulPrior { a: -0.5, b: 0.3 }));
        assert_eq!(
            calibration.noul_decision,
            NoulDecision::Band {
                edge: 0.8,
                slope: 0.1
            }
        );
    }
}
