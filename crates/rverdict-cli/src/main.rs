use anyhow::Result;
use clap::{Parser, Subcommand};
use rverdict_engine::select;

mod ask;
mod eval;
mod model;
mod serve;

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
    Ask(ask::Ask),
    /// Show which backend would be used, and why others were skipped.
    Devices(model::ModelArgs),
    /// Download a model into the local Hugging Face cache.
    Fetch(model::ModelArgs),
    /// Install a pinned model into a directory, verified and converted to
    /// safetensors, as apps embedding rverdict do.
    Install {
        /// A model alias or `owner/name@revision`.
        #[arg(long, default_value = "von")]
        model: String,
        #[arg(long)]
        dir: std::path::PathBuf,
    },
    /// Run a benchmark.
    #[command(subcommand)]
    Eval(eval::Eval),
    /// Refit calibration on labelled decisions.
    Calibrate(eval::Calibrate),
    /// Serve the model over HTTP on the System One wire format.
    Serve(serve::Serve),
    /// Build labelled decisions from public datasets.
    #[command(subcommand)]
    Data(eval::Data),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Ask(ask) => ask.run()?,
        Command::Devices(args) => {
            let selected = select(args.backend()?);
            for (name, reason) in &selected.skipped {
                println!("skipped  {name}: {reason}");
            }
            println!("selected {}", selected.name);
        }
        Command::Fetch(args) => {
            let checkpoint = args.checkpoint()?;
            println!("{} ({:?})", checkpoint.name, checkpoint.weights);
        }
        Command::Install { model, dir } => {
            let model = rverdict_engine::ModelRef::parse(&model);
            let mut last = String::new();
            let checkpoint = rverdict_engine::install::install(&dir, &model, |step| {
                let line = match step {
                    rverdict_engine::install::InstallStep::Downloading { file, done, total } => {
                        format!("{file}: {} of {} MB", done >> 20, total >> 20)
                    }
                    other => format!("{other:?}"),
                };
                if line != last {
                    eprintln!("{line}");
                    last = line;
                }
                true
            })?;
            println!("{} installed ({:?})", checkpoint.name, checkpoint.weights);
        }
        Command::Eval(eval) => eval.run()?,
        Command::Calibrate(calibrate) => calibrate.run()?,
        Command::Data(data) => data.run()?,
        Command::Serve(serve) => serve.run()?,
    }
    Ok(())
}
