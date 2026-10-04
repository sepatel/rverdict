use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use rverdict_core::{Calibration, DecideError, Decider, Request, Response};
use rverdict_engine::{BackendChoice, Checkpoint, Engine, ModelRef, Precision, select};
use rverdict_remote::RemoteClient;

/// Which model to run, and where.
#[derive(Args)]
pub struct ModelArgs {
    /// A model alias (`von`), `owner/name`, or `owner/name@revision`.
    #[arg(long, default_value = "von")]
    pub model: String,
    /// Load a checkpoint from a local directory instead of Hugging Face.
    #[arg(long)]
    pub model_dir: Option<PathBuf>,
    /// auto, cpu, wgpu, cuda, rocm or reference (slow CPU kernels for checking
    /// other backends). Defaults to `RVERDICT_BACKEND`, then auto.
    #[arg(long)]
    pub backend: Option<String>,
    /// Use f16 weights and activations: half the memory, for GPUs that support it.
    #[arg(long)]
    pub f16: bool,
    /// Replace the checkpoint's calibration with one from `rverdict calibrate`.
    #[arg(long)]
    pub calibration: Option<PathBuf>,
    /// Keep at most this many state tokens, cutting the middle of longer ones.
    #[arg(long, default_value_t = 8192)]
    pub max_state_tokens: usize,
    #[command(flatten)]
    pub remote: RemoteArgs,
}

/// Ask a hosted System One API instead of a local model. Each one sends the
/// state off this machine.
#[derive(Args)]
#[group(multiple = false)]
pub struct RemoteArgs {
    /// Any `/v1/systemone` server; its key, if any, from `RVERDICT_REMOTE_API_KEY`.
    #[arg(long)]
    remote: Option<String>,
    /// TypeSafe's Jev; key from `TYPESAFE_API_KEY`.
    #[arg(long)]
    typesafe: bool,
    /// A Workers AI model path such as `@cf/cloudflare/clef`; account from
    /// `CLOUDFLARE_ACCOUNT_ID`, token from `CLOUDFLARE_API_TOKEN`.
    #[arg(long)]
    cloudflare: Option<String>,
}

impl RemoteArgs {
    fn client(&self) -> Result<Option<RemoteClient>> {
        let env = |name: &str| std::env::var(name).with_context(|| format!("{name} is not set"));
        Ok(if let Some(url) = &self.remote {
            let client = RemoteClient::system_one(url);
            Some(match std::env::var("RVERDICT_REMOTE_API_KEY") {
                Ok(key) => client.with_api_key(key),
                Err(_) => client,
            })
        } else if self.typesafe {
            Some(RemoteClient::typesafe(env("TYPESAFE_API_KEY")?))
        } else if let Some(model) = &self.cloudflare {
            Some(RemoteClient::cloudflare(
                &env("CLOUDFLARE_ACCOUNT_ID")?,
                env("CLOUDFLARE_API_TOKEN")?,
                model,
            ))
        } else {
            None
        })
    }
}

/// A remote client behind the blocking [`Decider`] interface the CLI uses.
struct BlockingRemote {
    runtime: tokio::runtime::Runtime,
    client: RemoteClient,
}

impl Decider for BlockingRemote {
    fn decide(&self, request: &Request) -> Result<Response, DecideError> {
        self.runtime
            .block_on(self.client.decide(request))
            .map_err(|e| DecideError::Failed(e.to_string()))
    }

    fn model(&self) -> &str {
        self.client.endpoint()
    }
}

impl ModelArgs {
    pub fn backend(&self) -> Result<BackendChoice> {
        match &self.backend {
            Some(name) => name.parse().map_err(anyhow::Error::msg),
            None => BackendChoice::from_env().map_err(anyhow::Error::msg),
        }
    }

    pub fn checkpoint(&self) -> Result<Checkpoint> {
        Ok(match &self.model_dir {
            Some(dir) => Checkpoint::from_dir(dir)?,
            None => Checkpoint::from_hub(&ModelRef::parse(&self.model))?,
        })
    }

    /// The calibration in effect: `--calibration` if given, else the checkpoint's.
    pub fn calibration(&self, checkpoint: &Checkpoint) -> Result<Calibration> {
        match &self.calibration {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                Ok(serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?)
            }
            None => Ok(checkpoint.settings.calibration.clone()),
        }
    }

    /// The remote API when one was asked for, else the local engine.
    pub fn decider(&self) -> Result<Box<dyn Decider>> {
        if let Some(client) = self.remote.client()? {
            if !client.is_local() {
                eprintln!("rverdict: sending states to {}", client.endpoint());
            }
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            return Ok(Box::new(BlockingRemote { runtime, client }));
        }
        Ok(Box::new(self.load()?))
    }

    pub fn load(&self) -> Result<Engine> {
        let checkpoint = self.checkpoint()?;
        self.load_checkpoint(&checkpoint)
    }

    pub fn load_checkpoint(&self, checkpoint: &Checkpoint) -> Result<Engine> {
        self.load_on(checkpoint, self.backend()?)
    }

    pub fn load_on(&self, checkpoint: &Checkpoint, backend: BackendChoice) -> Result<Engine> {
        let selected = select(backend);
        for (name, reason) in &selected.skipped {
            eprintln!("rverdict: skipped {name}: {reason}");
        }
        let precision = if self.f16 {
            Precision::F16
        } else {
            Precision::F32
        };
        eprintln!(
            "rverdict: {} on {} ({precision:?})",
            checkpoint.name, selected.name
        );
        let mut engine = Engine::load(checkpoint, selected, precision)?;
        engine.set_calibration(self.calibration(checkpoint)?);
        engine.set_max_state_tokens(self.max_state_tokens);
        Ok(engine)
    }
}
