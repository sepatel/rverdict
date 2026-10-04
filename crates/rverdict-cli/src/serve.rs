use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Args;
use rverdict_server::{Options, router};

use crate::model::ModelArgs;

/// Serve the local model on the System One wire format.
#[derive(Args)]
pub struct Serve {
    #[command(flatten)]
    model: ModelArgs,
    /// Address to listen on. Loopback by default; anything else exposes the
    /// model to the network, so set `RVERDICT_API_KEY` too.
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: SocketAddr,
    /// Requests decided at once.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

impl Serve {
    pub fn run(&self) -> Result<()> {
        let engine = self.model.load()?;
        let api_key = std::env::var("RVERDICT_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        if !self.bind.ip().is_loopback() && api_key.is_none() {
            eprintln!(
                "rverdict: warning: listening on {} without RVERDICT_API_KEY; anyone who can reach it can use it",
                self.bind
            );
        }
        let app = router(
            Arc::new(engine),
            Options {
                api_key,
                concurrency: self.concurrency,
            },
        );
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let listener = tokio::net::TcpListener::bind(self.bind)
                .await
                .with_context(|| format!("binding {}", self.bind))?;
            eprintln!("rverdict: serving POST http://{}/v1/systemone", self.bind);
            rverdict_server::serve(listener, app)
                .await
                .context("serving")
        })
    }
}
