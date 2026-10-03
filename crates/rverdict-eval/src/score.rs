use std::time::{Duration, Instant};

use rverdict_core::{Answer, Request, Response};
use serde::Serialize;
use serde_json::Map;

use crate::jevbench::{Task, label};

/// A distribution whose sum is this far from 1 is rescaled; further off, it
/// is invalid and counts as wrong. JevBench's `RENORM_TOL`.
const RENORM_TOL: f64 = 2e-2;

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub id: String,
    pub tier: String,
    pub family: String,
    pub kind: String,
    /// `None` when the task has no expected answer.
    pub correct: Option<bool>,
    pub predicted: Option<String>,
    pub probabilities: Vec<(String, f64)>,
    pub latency_ms: f64,
    pub error: Option<String>,
}

/// An answer as JevBench reads it: probabilities over the task's labels.
fn distribution(answer: &Answer) -> Vec<(String, f64)> {
    match answer {
        Answer::Noul { noul, .. } => vec![("yes".into(), *noul), ("no".into(), 1.0 - noul)],
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => probabilities
            .iter()
            .map(|(k, &p)| (k.to_owned(), p))
            .collect(),
    }
}

/// JevBench's `validate_probs`: exact label set, values in [0, 1], sum near 1.
fn validate(probs: &[(String, f64)], labels: &[String]) -> Result<Vec<(String, f64)>, String> {
    let mut got: Vec<&str> = probs.iter().map(|(k, _)| k.as_str()).collect();
    let mut want: Vec<&str> = labels.iter().map(String::as_str).collect();
    got.sort_unstable();
    want.sort_unstable();
    if got != want {
        return Err(format!("labels {got:?} do not match {want:?}"));
    }
    if probs
        .iter()
        .any(|(_, p)| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("probability outside [0, 1]".into());
    }
    let total: f64 = probs.iter().map(|(_, p)| p).sum();
    if (total - 1.0).abs() > RENORM_TOL {
        return Err(format!("probabilities sum to {total}"));
    }
    Ok(probs.iter().map(|(k, p)| (k.clone(), p / total)).collect())
}

/// Highest probability, ties to the lexicographically smallest label.
fn argmax(probs: &[(String, f64)]) -> String {
    let mut sorted: Vec<&(String, f64)> = probs.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted
        .into_iter()
        .fold(None::<&(String, f64)>, |best, cur| match best {
            Some(b) if b.1 >= cur.1 => Some(b),
            _ => Some(cur),
        })
        .map(|(k, _)| k.clone())
        .unwrap_or_default()
}

pub fn request(task: &Task) -> Request {
    let mut questions = Map::new();
    questions.insert("q".into(), serde_json::Value::Object(task.question.clone()));
    Request {
        state: task.state.clone(),
        model: None,
        questions,
    }
}

/// Asks every task through `decide`, one request per task as JevBench does.
pub fn run<E: std::fmt::Display>(
    tasks: &[Task],
    mut decide: impl FnMut(&Request) -> Result<Response, E>,
) -> Vec<Outcome> {
    tasks
        .iter()
        .map(|task| {
            let start = Instant::now();
            let result = decide(&request(task));
            outcome(task, result.map_err(|e| e.to_string()), start.elapsed())
        })
        .collect()
}

fn outcome(task: &Task, result: Result<Response, String>, latency: Duration) -> Outcome {
    let kind = task
        .question
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("?")
        .to_owned();
    let mut out = Outcome {
        id: task.id.clone(),
        tier: task.tier.clone(),
        family: task.family.clone(),
        kind,
        correct: task.expected.as_ref().map(|_| false),
        predicted: None,
        probabilities: Vec::new(),
        latency_ms: latency.as_secs_f64() * 1e3,
        error: None,
    };
    let answer = result.and_then(|r| {
        r.answers
            .get("q")
            .cloned()
            .ok_or_else(|| "no answer for q".to_owned())
    });
    match answer.and_then(|a| validate(&distribution(&a), &task.labels)) {
        Ok(probs) => {
            let predicted = argmax(&probs);
            out.correct = task.expected.as_ref().map(|e| label(e) == predicted);
            out.predicted = Some(predicted);
            out.probabilities = probs;
        }
        Err(e) => out.error = Some(e),
    }
    out
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub scored: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Mean multi-class Brier over the exact label set.
    pub brier: f64,
    /// Top-label expected calibration error, 10 equal-width bins.
    pub ece: f64,
    pub invalid: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

#[expect(clippy::cast_precision_loss, reason = "benchmark sizes are small")]
pub fn summarize<'a>(outcomes: impl IntoIterator<Item = &'a Outcome>, tasks: &[Task]) -> Summary {
    let outcomes: Vec<&Outcome> = outcomes.into_iter().collect();
    let scored: Vec<&&Outcome> = outcomes.iter().filter(|o| o.correct.is_some()).collect();
    let correct = scored.iter().filter(|o| o.correct == Some(true)).count();

    let mut brier = Vec::new();
    let mut bins = [(0usize, 0.0f64, 0usize); 10];
    for o in &scored {
        if o.probabilities.is_empty() {
            continue;
        }
        let expected = tasks
            .iter()
            .find(|t| t.id == o.id)
            .and_then(|t| t.expected.as_ref())
            .map(label);
        let target = |k: &str| {
            if Some(k) == expected.as_deref() {
                1.0
            } else {
                0.0
            }
        };
        brier.push(
            o.probabilities
                .iter()
                .map(|(k, p)| (p - target(k)).powi(2))
                .sum::<f64>(),
        );
        let confidence = o.probabilities.iter().map(|(_, p)| *p).fold(0.0, f64::max);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "confidence is in [0, 1]"
        )]
        let bin = ((confidence * 10.0) as usize).min(9);
        bins[bin].0 += 1;
        bins[bin].1 += confidence;
        bins[bin].2 += usize::from(o.correct == Some(true));
    }
    let binned: usize = bins.iter().map(|b| b.0).sum();
    let ece = bins
        .iter()
        .filter(|b| b.0 > 0)
        .map(|&(n, conf, ok)| {
            (n as f64 / binned as f64) * (ok as f64 / n as f64 - conf / n as f64).abs()
        })
        .sum();

    let mut latencies: Vec<f64> = outcomes.iter().map(|o| o.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);
    let percentile = |q: f64| {
        if latencies.is_empty() {
            return 0.0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "index into a short list"
        )]
        let i = ((latencies.len() - 1) as f64 * q).round() as usize;
        latencies[i]
    };

    Summary {
        scored: scored.len(),
        correct,
        accuracy: if scored.is_empty() {
            0.0
        } else {
            correct as f64 / scored.len() as f64
        },
        brier: if brier.is_empty() {
            0.0
        } else {
            brier.iter().sum::<f64>() / brier.len() as f64
        },
        ece,
        invalid: outcomes.iter().filter(|o| o.error.is_some()).count(),
        p50_ms: percentile(0.5),
        p95_ms: percentile(0.95),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_to_the_smallest_label() {
        let probs = vec![("b".to_owned(), 0.5), ("a".to_owned(), 0.5)];
        assert_eq!(argmax(&probs), "a");
    }

    #[test]
    fn distributions_far_from_one_are_invalid_and_near_ones_are_rescaled() {
        let labels = vec!["no".to_owned(), "yes".to_owned()];
        assert!(validate(&[("yes".into(), 0.9), ("no".into(), 0.2)], &labels).is_err());
        let ok = validate(&[("yes".into(), 0.6), ("no".into(), 0.405)], &labels).unwrap();
        assert!((ok.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
