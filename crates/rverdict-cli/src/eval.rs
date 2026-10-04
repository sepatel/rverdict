use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand, ValueEnum};
use rverdict_eval::calibrate::{self, Capture, Form, Metrics};
use rverdict_eval::{Task, datasets, jevbench, run, summarize};

use crate::model::ModelArgs;

#[derive(Subcommand)]
pub enum Eval {
    /// JevBench v1's public set (231 items), downloaded on first use.
    Jevbench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, value_delimiter = ',', default_value = "easy,original,hard")]
        tiers: Vec<String>,
        /// Write every outcome as JSON, for paired comparisons between runs.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Reverse every choice's option order; answers must not change.
        #[arg(long)]
        reverse_options: bool,
    },
    /// Run labelled decisions on `--backend` and on `--against`, and compare
    /// the raw logits. Exits non-zero if any prediction differs or a logit
    /// differs by more than `--tolerance`.
    Compare {
        #[command(flatten)]
        model: ModelArgs,
        /// The backend trusted as correct.
        #[arg(long, default_value = "reference")]
        against: String,
        #[arg(long, required = true)]
        data: Vec<PathBuf>,
        /// Compare only the first this many decisions.
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        tolerance: f32,
    },
}

impl Eval {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Jevbench {
                model,
                tiers,
                out,
                reverse_options,
            } => jevbench_run(&model, &tiers, out.as_deref(), reverse_options),
            Self::Compare {
                model,
                against,
                data,
                limit,
                tolerance,
            } => compare(&model, &against, &data, limit, tolerance),
        }
    }
}

fn compare(
    model: &ModelArgs,
    against: &str,
    data: &[PathBuf],
    limit: Option<usize>,
    tolerance: f32,
) -> Result<()> {
    let mut tasks = Vec::new();
    for path in data {
        tasks.extend(Task::read_jsonl(path)?);
    }
    tasks.truncate(limit.unwrap_or(usize::MAX));
    let checkpoint = model.checkpoint()?;
    let tested = model.load_checkpoint(&checkpoint)?;
    let trusted = model.load_on(&checkpoint, against.parse().map_err(anyhow::Error::msg)?)?;

    let mut worst: Vec<(f32, String)> = Vec::new();
    let mut flips = Vec::new();
    for task in &tasks {
        let request = rverdict_eval::score::request(task);
        let (a, b) = (
            tested.evaluate(&request, true)?,
            trusted.evaluate(&request, true)?,
        );
        for (x, y) in a.questions.iter().zip(&b.questions) {
            let rows = [(&x.logits.logits, &y.logits.logits)].into_iter().chain(
                x.logits
                    .null_logits
                    .as_ref()
                    .zip(y.logits.null_logits.as_ref()),
            );
            let diff = rows
                .flat_map(|(p, q)| p.iter().zip(q.iter()).map(|(u, v)| (u - v).abs()))
                .fold(0.0, f32::max);
            worst.push((diff, task.id.clone()));
            if argmax(&x.logits.logits) != argmax(&y.logits.logits) {
                flips.push(task.id.clone());
            }
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "{} on {} vs {}: {} decisions",
        tested.backend(),
        checkpoint.name,
        trusted.backend(),
        worst.len()
    );
    println!(
        "largest logit difference {:.4}; predictions differ on {}",
        worst.first().map_or(0.0, |w| w.0),
        flips.len()
    );
    for (diff, id) in worst.iter().take(5) {
        println!("  {diff:.4}  {id}");
    }
    let failing = worst.iter().filter(|w| w.0 > tolerance).count();
    if failing > 0 || !flips.is_empty() {
        bail!(
            "{failing} decisions exceed the {tolerance} tolerance, {} predictions differ: {flips:?}",
            flips.len()
        );
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn jevbench_run(
    model: &ModelArgs,
    tiers: &[String],
    out: Option<&Path>,
    reverse_options: bool,
) -> Result<()> {
    {
        let decider = model.decider()?;
        let tiers: Vec<&str> = tiers.iter().map(String::as_str).collect();
        let mut tasks = jevbench::load(&tiers, &jevbench::cache_dir())?;
        if reverse_options {
            tasks.iter_mut().for_each(Task::reverse_choice_options);
        }
        let outcomes = run(&tasks, |request| decider.decide(request));
        println!(
            "{:<10} {:>7} {:>9} {:>7} {:>7} {:>8} {:>8}",
            "tier", "correct", "accuracy", "brier", "ece", "p50 ms", "invalid"
        );
        for tier in tiers.iter().copied().chain(["all"]) {
            let summary = summarize(
                outcomes
                    .iter()
                    .filter(|o| tier == "all" || o.subset == tier),
                &tasks,
            );
            println!(
                "{tier:<10} {:>3}/{:<3} {:>9.3} {:>7.3} {:>7.3} {:>8.1} {:>8}",
                summary.correct,
                summary.scored,
                summary.accuracy,
                summary.brier,
                summary.ece,
                summary.p50_ms,
                summary.invalid
            );
        }
        if let Some(path) = out {
            std::fs::write(path, serde_json::to_string_pretty(&outcomes)?)
                .with_context(|| format!("writing {}", path.display()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum FormArg {
    /// Von's input-conditioned temperature map (4 parameters).
    Map,
    /// One temperature (1 parameter), for small datasets.
    Scalar,
}

/// Refit calibration on labelled decisions, with the model weights frozen.
#[derive(Args)]
pub struct Calibrate {
    #[command(flatten)]
    model: ModelArgs,
    /// Labelled decisions as JSONL (`id`, `state`, `question`, `expected`).
    #[arg(long, required = true)]
    data: Vec<PathBuf>,
    /// Cache of the model's outputs: reused when it exists, written when not,
    /// so refits do not run the model again.
    #[arg(long)]
    captures: Option<PathBuf>,
    /// Share of decisions held out to validate the fit.
    #[arg(long, default_value_t = 0.5)]
    holdout: f64,
    #[arg(long, value_enum, default_value = "scalar")]
    form: FormArg,
    /// Where to write the calibration fitted on all the data.
    #[arg(long)]
    out: Option<PathBuf>,
}

impl Calibrate {
    pub fn run(&self) -> Result<()> {
        let mut tasks = Vec::new();
        for path in &self.data {
            tasks.extend(Task::read_jsonl(path)?);
        }
        let checkpoint = self.model.checkpoint()?;
        let start = self.model.calibration(&checkpoint)?;
        let captures = match &self.captures {
            Some(path) if path.exists() => read_captures(path)?,
            cached => {
                let engine = self.model.load_checkpoint(&checkpoint)?;
                let (captures, failures) = calibrate::capture(&tasks, |request| {
                    engine.evaluate(request, true).map(|e| {
                        e.questions
                            .into_iter()
                            .map(|q| (q.rendered, q.logits))
                            .collect()
                    })
                });
                for (id, error) in &failures {
                    eprintln!("rverdict: {id}: {error}");
                }
                if let Some(path) = cached {
                    write_captures(path, &captures)?;
                }
                captures
            }
        };
        if captures.is_empty() {
            bail!("no labelled decisions with an expected answer");
        }

        let form = match self.form {
            FormArg::Map => Form::Map,
            FormArg::Scalar => Form::Scalar,
        };
        let (train, test) = calibrate::split(&captures, self.holdout);
        let fitted = calibrate::fit(&train, form, &start);
        println!(
            "fit on {} decisions, validated on {} held out",
            train.len(),
            test.len()
        );
        println!(
            "{:<16} {:>6} {:>9} {:>7} {:>7} {:>7}",
            "held out", "n", "accuracy", "nll", "brier", "ece"
        );
        let mut groups: Vec<&str> = test.iter().map(|c| c.group.as_str()).collect();
        groups.sort_unstable();
        groups.dedup();
        let mut intervals = Vec::new();
        for group in groups.iter().copied().chain(["all"]) {
            let subset: Vec<&Capture> = test
                .iter()
                .copied()
                .filter(|c| group == "all" || c.group == group)
                .collect();
            row(
                &format!("{group} before"),
                &calibrate::metrics(&subset, &start),
            );
            row(
                &format!("{group} after"),
                &calibrate::metrics(&subset, &fitted),
            );
            intervals.push((group, calibrate::change_intervals(&subset, &start, &fitted)));
        }
        println!("held-out change, 95% intervals:");
        for (group, [(ece_lo, ece_hi), (nll_lo, nll_hi)]) in intervals {
            println!(
                "  {group:<10} ECE [{ece_lo:+.3}, {ece_hi:+.3}]  NLL [{nll_lo:+.3}, {nll_hi:+.3}]"
            );
        }

        if let Some(path) = &self.out {
            let all: Vec<&Capture> = captures.iter().collect();
            let calibration = calibrate::fit(&all, form, &start);
            std::fs::write(path, serde_json::to_string_pretty(&calibration)?)
                .with_context(|| format!("writing {}", path.display()))?;
            println!(
                "wrote the calibration fitted on all {} decisions to {}",
                all.len(),
                path.display()
            );
        }
        Ok(())
    }
}

fn row(label: &str, m: &Metrics) {
    println!(
        "{label:<16} {:>6} {:>9.3} {:>7.3} {:>7.3} {:>7.3}",
        m.n, m.accuracy, m.nll, m.brier, m.ece
    );
}

fn read_captures(path: &Path) -> Result<Vec<Capture>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

fn write_captures(path: &Path, captures: &[Capture]) -> Result<()> {
    let mut text = String::new();
    for capture in captures {
        text.push_str(&serde_json::to_string(capture)?);
        text.push('\n');
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[derive(Subcommand)]
pub enum Data {
    /// Enron-Spam emails as a zero-shot `noul` and a described `choice` each.
    EnronSpam {
        #[arg(long, default_value = "test")]
        split: String,
        /// Emails to sample (two decisions each).
        #[arg(long, default_value_t = 600)]
        limit: usize,
        #[arg(long)]
        out: PathBuf,
    },
}

impl Data {
    pub fn run(&self) -> Result<()> {
        let Self::EnronSpam { split, limit, out } = self;
        let tasks = datasets::enron_spam(split, *limit)?;
        Task::write_jsonl(&tasks, out)?;
        println!("wrote {} decisions to {}", tasks.len(), out.display());
        Ok(())
    }
}
