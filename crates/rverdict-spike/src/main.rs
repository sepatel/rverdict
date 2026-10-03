use anyhow::Result;
use clap::Parser;

mod backend;
mod bench;
mod fetch;
mod micro;
mod parity;
mod smoke;
mod von;

use backend::{BackendName, dispatch};

#[derive(Parser)]
#[command(about = "Phase 0 spike checks for rverdict-model")]
enum Command {
    /// Download a model's files from Hugging Face into the local HF cache.
    Fetch {
        #[arg(default_value = "wfzyx/von")]
        repo: String,
    },
    /// Compare the Burn encoder's hidden states with candle-transformers.
    Parity {
        #[arg(long, value_enum)]
        backend: BackendName,
        /// How many times to repeat the sample text; above ~2 exercises the sliding window.
        #[arg(long, default_value_t = 3)]
        repeat: usize,
    },
    /// Load Von's weights and answer a few decision questions.
    Smoke {
        #[arg(long, value_enum)]
        backend: BackendName,
    },
    /// Time inference at several sequence lengths.
    Infer {
        #[arg(long, value_enum)]
        backend: BackendName,
        #[arg(long, value_delimiter = ',', default_value = "128,512,1024,2048")]
        lengths: Vec<usize>,
        #[arg(long, default_value_t = 5)]
        runs: usize,
    },
    /// Time the encoder's individual kernels.
    Micro {
        #[arg(long, value_enum)]
        backend: BackendName,
        #[arg(long, default_value_t = 512)]
        seq: usize,
    },
    /// Time full fine-tuning steps (forward, backward, AdamW).
    Train {
        #[arg(long, value_enum)]
        backend: BackendName,
        #[arg(long, default_value_t = 512)]
        length: usize,
        #[arg(long, default_value_t = 4)]
        rows: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
    },
}

fn main() -> Result<()> {
    let von = || fetch::fetch("wfzyx/von");
    match Command::parse() {
        Command::Fetch { repo } => println!("{}", fetch::fetch(&repo)?.dir.display()),
        Command::Parity { backend, repeat } => dispatch!(backend, parity::run(&von()?, repeat))?,
        Command::Smoke { backend } => dispatch!(backend, smoke::run(&von()?))?,
        Command::Infer {
            backend,
            lengths,
            runs,
        } => dispatch!(backend, bench::inference(&von()?, &lengths, runs))?,
        Command::Micro { backend, seq } => dispatch!(backend, micro::run(seq)),
        Command::Train {
            backend,
            length,
            rows,
            steps,
        } => dispatch!(backend, bench::train(&von()?, length, rows, steps))?,
    }
    Ok(())
}
