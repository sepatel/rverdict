use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use rverdict_core::Request;
use rverdict_engine::{BackendChoice, Checkpoint, Engine, ModelRef, Precision, select};
use rverdict_eval::{jevbench, run, summarize};
use serde_json::{Map, Value, json};

#[derive(Parser)]
#[command(
    name = "rverdict",
    version,
    about = "Fast, calibrated, local decisions: noul, choice and score questions"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Answer questions about a state and print the System One response.
    Ask(Ask),
    /// Show which backend would be used, and why others were skipped.
    Devices(ModelArgs),
    /// Download a model into the local Hugging Face cache.
    Fetch(ModelArgs),
    /// Run a benchmark.
    #[command(subcommand)]
    Eval(Eval),
}

#[derive(Args)]
struct ModelArgs {
    /// A model alias (`von`), `owner/name`, or `owner/name@revision`.
    #[arg(long, default_value = "von")]
    model: String,
    /// Load a checkpoint from a local directory instead of Hugging Face.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// auto, cpu, wgpu, cuda or rocm. Defaults to `RVERDICT_BACKEND`, then auto.
    #[arg(long)]
    backend: Option<String>,
    /// Use f16 weights and activations: half the memory, for GPUs that support it.
    #[arg(long)]
    f16: bool,
}

#[derive(Args)]
struct Ask {
    #[command(flatten)]
    model: ModelArgs,
    /// The text to decide about.
    #[arg(long, conflicts_with_all = ["state_file", "request"])]
    state: Option<String>,
    #[arg(long, conflicts_with = "request")]
    state_file: Option<PathBuf>,
    /// A full JSON request; `-` reads standard input.
    #[arg(long)]
    request: Option<String>,
    /// A yes/no question.
    #[arg(long)]
    noul: Vec<String>,
    /// `"question=option,option,…"`.
    #[arg(long)]
    choice: Vec<String>,
    /// `"question=level,level,…"`, lowest level first.
    #[arg(long)]
    score: Vec<String>,
}

#[derive(Subcommand)]
enum Eval {
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
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Ask(ask) => {
            let request = ask.request()?;
            let engine = load(&ask.model)?;
            let response = engine.decide(&request)?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Command::Devices(args) => {
            let selected = select(backend(&args)?);
            for (name, reason) in &selected.skipped {
                println!("skipped  {name}: {reason}");
            }
            println!("selected {}", selected.name);
        }
        Command::Fetch(args) => {
            let checkpoint = checkpoint(&args)?;
            println!("{} ({:?})", checkpoint.name, checkpoint.weights);
        }
        Command::Eval(Eval::Jevbench {
            model,
            tiers,
            out,
            reverse_options,
        }) => {
            let engine = load(&model)?;
            let tiers: Vec<&str> = tiers.iter().map(String::as_str).collect();
            let mut tasks = jevbench::load(&tiers, &jevbench::cache_dir())?;
            if reverse_options {
                tasks
                    .iter_mut()
                    .for_each(jevbench::Task::reverse_choice_options);
            }
            let outcomes = run(&tasks, |request| engine.decide(request));
            println!(
                "{:<10} {:>7} {:>9} {:>7} {:>7} {:>8} {:>8}",
                "tier", "correct", "accuracy", "brier", "ece", "p50 ms", "invalid"
            );
            for tier in tiers.iter().copied().chain(["all"]) {
                let summary = summarize(
                    outcomes.iter().filter(|o| tier == "all" || o.tier == tier),
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
                std::fs::write(&path, serde_json::to_string_pretty(&outcomes)?)
                    .with_context(|| format!("writing {}", path.display()))?;
            }
        }
    }
    Ok(())
}

fn backend(args: &ModelArgs) -> Result<BackendChoice> {
    match &args.backend {
        Some(name) => name.parse().map_err(anyhow::Error::msg),
        None => BackendChoice::from_env().map_err(anyhow::Error::msg),
    }
}

fn checkpoint(args: &ModelArgs) -> Result<Checkpoint> {
    Ok(match &args.model_dir {
        Some(dir) => Checkpoint::from_dir(dir)?,
        None => Checkpoint::from_hub(&ModelRef::parse(&args.model))?,
    })
}

fn load(args: &ModelArgs) -> Result<Engine> {
    let checkpoint = checkpoint(args)?;
    let selected = select(backend(args)?);
    for (name, reason) in &selected.skipped {
        eprintln!("rverdict: skipped {name}: {reason}");
    }
    let precision = if args.f16 {
        Precision::F16
    } else {
        Precision::F32
    };
    eprintln!(
        "rverdict: {} on {} ({precision:?})",
        checkpoint.name, selected.name
    );
    Ok(Engine::load(&checkpoint, selected, precision)?)
}

impl Ask {
    fn request(&self) -> Result<Request> {
        if let Some(source) = &self.request {
            let text = if source == "-" {
                let mut text = String::new();
                std::io::stdin().read_to_string(&mut text)?;
                text
            } else {
                std::fs::read_to_string(source).with_context(|| format!("reading {source}"))?
            };
            return Ok(serde_json::from_str(&text)?);
        }
        let state = match (&self.state, &self.state_file) {
            (Some(text), _) => text.clone(),
            (None, Some(path)) => std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
            (None, None) => bail!("give --state, --state-file or --request"),
        };

        let mut questions = Map::new();
        for (i, text) in self.noul.iter().enumerate() {
            questions.insert(
                format!("noul_{i}"),
                json!({"type": "noul", "instructions": text}),
            );
        }
        for (i, spec) in self.choice.iter().enumerate() {
            let (instructions, options) = split_spec(spec)?;
            let criteria: Map<String, Value> =
                options.into_iter().map(|o| (o, Value::Null)).collect();
            questions.insert(
                format!("choice_{i}"),
                json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            );
        }
        for (i, spec) in self.score.iter().enumerate() {
            let (instructions, levels) = split_spec(spec)?;
            questions.insert(
                format!("score_{i}"),
                json!({"type": "score", "instructions": instructions, "criteria": levels}),
            );
        }
        if questions.is_empty() {
            bail!("ask at least one --noul, --choice or --score question");
        }
        Ok(Request {
            state: Value::String(state),
            model: None,
            questions,
        })
    }
}

fn split_spec(spec: &str) -> Result<(String, Vec<String>)> {
    let (instructions, options) = spec
        .rsplit_once('=')
        .with_context(|| format!("{spec:?} is not \"question=option,option\""))?;
    let options = options
        .split(',')
        .map(|o| o.trim().to_owned())
        .filter(|o| !o.is_empty())
        .collect();
    Ok((instructions.trim().to_owned(), options))
}
