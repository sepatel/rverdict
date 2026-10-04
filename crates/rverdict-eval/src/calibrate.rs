//! Refits a checkpoint's calibration on labelled decisions, with the model
//! weights frozen. Temperature is monotonic, so a refit changes how sure the
//! answers are, not which answers win; a refit zero-shot `noul` prior can
//! also move borderline yes/no answers.

use rverdict_core::{
    Calibration, Logits, NoulPrior, Rendered, RenderedKind, Request, Scaling, TemperatureMap,
    type_name,
};
use serde::{Deserialize, Serialize};

use crate::score::{ece, request};
use crate::task::Task;

/// One labelled question's model output, captured once so every refit is
/// instant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capture {
    pub id: String,
    pub group: String,
    pub kind: RenderedKind,
    /// Index of the right answer among the question's options.
    pub expected: usize,
    pub logits: Logits,
}

/// Runs every task with an expected answer through `evaluate` (one request
/// per task, as [`crate::run`]), returning the captures and any failures.
pub fn capture<E: std::fmt::Display>(
    tasks: &[Task],
    mut evaluate: impl FnMut(&Request) -> Result<Vec<(Rendered, Logits)>, E>,
) -> (Vec<Capture>, Vec<(String, String)>) {
    let mut captures = Vec::new();
    let mut failures = Vec::new();
    for task in tasks {
        let Some(expected) = task.expected_index() else {
            continue;
        };
        match evaluate(&request(task)) {
            Ok(mut questions) if questions.len() == 1 => {
                let (rendered, logits) = questions.remove(0);
                captures.push(Capture {
                    id: task.id.clone(),
                    group: task.subset.clone(),
                    kind: rendered.kind,
                    expected,
                    logits,
                });
            }
            Ok(_) => failures.push((task.id.clone(), "expected one answer".into())),
            Err(e) => failures.push((task.id.clone(), e.to_string())),
        }
    }
    (captures, failures)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    /// Mean negative log-likelihood of the right answer: the fitting target.
    pub nll: f64,
    /// Mean multi-class Brier score.
    pub brier: f64,
    pub ece: f64,
}

/// Per-capture `(confidence, correct, nll, brier)` under `calibration`.
fn scored(captures: &[&Capture], calibration: &Calibration) -> Vec<(f64, bool, f64, f64)> {
    captures
        .iter()
        .map(|c| {
            let probs = calibration.distribution(&c.kind, &c.logits);
            let (best, confidence) =
                probs
                    .iter()
                    .copied()
                    .enumerate()
                    .fold(
                        (0, f64::NEG_INFINITY),
                        |b, (i, p)| if p > b.1 { (i, p) } else { b },
                    );
            let brier = probs
                .iter()
                .enumerate()
                .map(|(i, p)| (p - if i == c.expected { 1.0 } else { 0.0 }).powi(2))
                .sum();
            (
                confidence,
                best == c.expected,
                -probs[c.expected].max(1e-12).ln(),
                brier,
            )
        })
        .collect()
}

#[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
pub fn metrics(captures: &[&Capture], calibration: &Calibration) -> Metrics {
    let rows = scored(captures, calibration);
    let n = rows.len().max(1) as f64;
    let tops: Vec<(f64, bool)> = rows.iter().map(|r| (r.0, r.1)).collect();
    Metrics {
        n: rows.len(),
        accuracy: rows.iter().filter(|r| r.1).count() as f64 / n,
        nll: rows.iter().map(|r| r.2).sum::<f64>() / n,
        brier: rows.iter().map(|r| r.3).sum::<f64>() / n,
        ece: ece(&tops),
    }
}

/// What a refit may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// Von's input-conditioned temperature map: 4 parameters.
    Map,
    /// One temperature for everything: 1 parameter, for small datasets.
    Scalar,
}

/// A question type needs this many labelled decisions to get its own fit;
/// rarer types keep the starting scaling.
const MIN_PER_TYPE: usize = 30;

/// A refit must lower a type's mean NLL by this much (nats) on the data it
/// is fitted on to replace the starting scaling; smaller gains are noise
/// that would not hold on new data.
const MIN_NLL_GAIN: f64 = 0.005;

/// Fits, separately for each question type, the scaling that minimises the
/// mean negative log-likelihood of the right answers, starting from `start`
/// and from a neutral temperature. The zero-shot `noul` prior is fitted with
/// the `noul` scaling when nouls were asked without criteria.
pub fn fit(captures: &[&Capture], form: Form, start: &Calibration) -> Calibration {
    let mut fitted = start.clone();
    for kind in ["noul", "choice", "score"] {
        let subset: Vec<&Capture> = captures
            .iter()
            .copied()
            .filter(|c| type_name(&c.kind) == kind)
            .collect();
        if subset.len() < MIN_PER_TYPE {
            continue;
        }
        let with_prior = kind == "noul"
            && subset.iter().any(|c| {
                c.kind == RenderedKind::Noul { explicit: false } && c.logits.null_logits.is_some()
            });
        let prior_form = if with_prior {
            prior_form(&subset)
        } else {
            PriorForm::None
        };
        let (scaling, prior) = fit_type(&subset, form, start, kind, prior_form);
        let mut candidate = fitted.clone();
        candidate.per_type.insert(kind.to_owned(), scaling);
        if prior_form != PriorForm::None {
            candidate.noul_prior = Some(prior);
        }
        if metrics(&subset, &fitted).nll - metrics(&subset, &candidate).nll > MIN_NLL_GAIN {
            fitted = candidate;
        }
    }
    fitted
}

/// Which zero-shot `noul` prior parameters the data can identify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriorForm {
    None,
    /// Only the offset `b`: with one or two distinct questions the
    /// state-free bias is (nearly) constant, so its slope `a` is not
    /// identifiable from the offset.
    Offset,
    Full,
}

fn prior_form(subset: &[&Capture]) -> PriorForm {
    let mut biases: Vec<i64> = subset
        .iter()
        .filter_map(|c| c.logits.null_logits.as_ref())
        .map(|n| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "bucketing logit gaps to 1e-3"
            )]
            let bucket = (f64::from(n[0] - n[1]) * 1000.0).round() as i64;
            bucket
        })
        .collect();
    biases.sort_unstable();
    biases.dedup();
    if biases.len() >= 3 {
        PriorForm::Full
    } else {
        PriorForm::Offset
    }
}

fn fit_type(
    subset: &[&Capture],
    form: Form,
    start: &Calibration,
    kind: &str,
    prior_form: PriorForm,
) -> (Scaling, NoulPrior) {
    let with_prior = prior_form != PriorForm::None;
    let start_prior = start.noul_prior.unwrap_or(NoulPrior { a: 0.0, b: 0.0 });
    let base = *start.per_type.get(kind).unwrap_or(&start.scaling);
    let bounds = base.map.map_or((0.3, 12.0), |m| (m.lo, m.hi));
    let build = |p: &[f64]| -> (Scaling, NoulPrior) {
        let (scaling, rest) = match form {
            Form::Map => (
                Scaling {
                    temperature: base.temperature,
                    map: Some(TemperatureMap {
                        bias: p[0],
                        entropy: p[1],
                        log_tokens: p[2],
                        n_options: p[3],
                        lo: bounds.0,
                        hi: bounds.1,
                    }),
                },
                &p[4..],
            ),
            Form::Scalar => (
                Scaling {
                    temperature: p[0].abs().clamp(0.05, 50.0),
                    map: None,
                },
                &p[1..],
            ),
        };
        let prior = match prior_form {
            PriorForm::None => start_prior,
            PriorForm::Offset => NoulPrior {
                a: start_prior.a,
                b: rest[0],
            },
            PriorForm::Full => NoulPrior {
                a: rest[0],
                b: rest[1],
            },
        };
        (scaling, prior)
    };
    let loss = |p: &[f64]| {
        let (scaling, prior) = build(p);
        let mut calibration = start.clone();
        calibration.per_type.insert(kind.to_owned(), scaling);
        if with_prior {
            calibration.noul_prior = Some(prior);
        }
        metrics(subset, &calibration).nll
    };

    let mut starts: Vec<Vec<f64>> = match form {
        Form::Map => vec![
            base.map.map_or(vec![1.0, 0.0, 0.0, 0.0], |m| {
                vec![m.bias, m.entropy, m.log_tokens, m.n_options]
            }),
            vec![1.0, 0.0, 0.0, 0.0],
        ],
        Form::Scalar => vec![vec![base.temperature], vec![1.0]],
    };
    for s in &mut starts {
        match prior_form {
            PriorForm::None => {}
            PriorForm::Offset => s.push(start_prior.b),
            PriorForm::Full => s.extend([start_prior.a, start_prior.b]),
        }
    }

    let (mut best, mut best_loss) = (starts[0].clone(), f64::INFINITY);
    for s in &starts {
        let (p, l) = nelder_mead(&loss, s, 0.5, 4000);
        if l < best_loss {
            (best, best_loss) = (p, l);
        }
    }
    let (best, _) = nelder_mead(&loss, &best, 0.05, 4000);
    build(&best)
}

/// Minimises `f` with the Nelder–Mead simplex method: derivative-free, which
/// suits the clamped temperature map.
fn nelder_mead(
    f: &impl Fn(&[f64]) -> f64,
    start: &[f64],
    step: f64,
    iterations: usize,
) -> (Vec<f64>, f64) {
    let n = start.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = (0..=n)
        .map(|i| {
            let mut p = start.to_vec();
            if i > 0 {
                p[i - 1] += step;
            }
            let v = f(&p);
            (p, v)
        })
        .collect();

    #[expect(clippy::cast_precision_loss, reason = "a handful of parameters")]
    let count = n as f64;
    for _ in 0..iterations {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        if (simplex[n].1 - simplex[0].1).abs() < 1e-10 {
            break;
        }
        let centroid: Vec<f64> = (0..n)
            .map(|d| simplex[..n].iter().map(|(p, _)| p[d]).sum::<f64>() / count)
            .collect();
        let towards = |t: f64| -> Vec<f64> {
            centroid
                .iter()
                .zip(&simplex[n].0)
                .map(|(c, w)| c + t * (w - c))
                .collect()
        };

        let reflected = towards(-1.0);
        let reflected_value = f(&reflected);
        if reflected_value < simplex[0].1 {
            let expanded = towards(-2.0);
            let expanded_value = f(&expanded);
            simplex[n] = if expanded_value < reflected_value {
                (expanded, expanded_value)
            } else {
                (reflected, reflected_value)
            };
        } else if reflected_value < simplex[n - 1].1 {
            simplex[n] = (reflected, reflected_value);
        } else {
            let contracted = towards(0.5);
            let contracted_value = f(&contracted);
            if contracted_value < simplex[n].1 {
                simplex[n] = (contracted, contracted_value);
            } else {
                let best = simplex[0].0.clone();
                for vertex in &mut simplex[1..] {
                    vertex.0 = best
                        .iter()
                        .zip(&vertex.0)
                        .map(|(b, v)| b + 0.5 * (v - b))
                        .collect();
                    vertex.1 = f(&vertex.0);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex.swap_remove(0)
}

/// Deterministic split by a hash of each id: `holdout` of the captures go
/// to the second (test) set. Stable across runs and machines.
pub fn split(captures: &[Capture], holdout: f64) -> (Vec<&Capture>, Vec<&Capture>) {
    captures.iter().partition(|c| unit_hash(&c.id) >= holdout)
}

/// FNV-1a of `id`, mapped to `[0, 1)`.
#[expect(clippy::cast_precision_loss, reason = "only the leading bits matter")]
fn unit_hash(id: &str) -> f64 {
    (crate::fnv(id) >> 11) as f64 / (1u64 << 53) as f64
}

/// 95% bootstrap intervals of the `after − before` change in ECE and in
/// mean NLL on the same items, from 2000 resamples with a fixed seed.
#[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
pub fn change_intervals(
    captures: &[&Capture],
    before: &Calibration,
    after: &Calibration,
) -> [(f64, f64); 2] {
    let (before, after) = (scored(captures, before), scored(captures, after));
    let n = before.len();
    if n == 0 {
        return [(0.0, 0.0); 2];
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut ece_deltas, mut nll_deltas) = (Vec::with_capacity(2000), Vec::with_capacity(2000));
    for _ in 0..2000 {
        #[expect(clippy::cast_possible_truncation, reason = "index below n")]
        let picks: Vec<usize> = (0..n).map(|_| (next() % n as u64) as usize).collect();
        let tops = |rows: &[(f64, bool, f64, f64)]| -> Vec<(f64, bool)> {
            picks.iter().map(|&i| (rows[i].0, rows[i].1)).collect()
        };
        let nll = |rows: &[(f64, bool, f64, f64)]| {
            picks.iter().map(|&i| rows[i].2).sum::<f64>() / n as f64
        };
        ece_deltas.push(ece(&tops(&after)) - ece(&tops(&before)));
        nll_deltas.push(nll(&after) - nll(&before));
    }
    [ece_deltas, nll_deltas].map(|mut d| {
        d.sort_by(f64::total_cmp);
        (d[49], d[1949])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(id: &str, logits: [f32; 2], expected: usize) -> Capture {
        Capture {
            id: id.into(),
            group: String::new(),
            kind: RenderedKind::Choice {
                keys: vec!["a".into(), "b".into()],
            },
            expected,
            logits: Logits {
                logits: logits.to_vec(),
                null_logits: None,
                state_tokens: 10,
            },
        }
    }

    #[test]
    fn an_overconfident_model_gets_a_higher_temperature() {
        // Always 8 logits apart, but right only 3 times in 4: the fitted
        // temperature must soften answers toward 75%.
        let captures: Vec<Capture> = (0..40)
            .map(|i| capture(&i.to_string(), [8.0, 0.0], usize::from(i % 4 == 0)))
            .collect();
        let refs: Vec<&Capture> = captures.iter().collect();
        let fitted = fit(&refs, Form::Scalar, &Calibration::default());
        let p = fitted.distribution(&captures[1].kind, &captures[1].logits)[0];
        assert!((p - 0.75).abs() < 0.01, "p = {p}");
        assert!(metrics(&refs, &fitted).nll < metrics(&refs, &Calibration::default()).nll);
    }
}
