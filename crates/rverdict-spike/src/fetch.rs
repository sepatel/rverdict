use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::HFClientSync;

/// Files every option-marker checkpoint needs. `option_marker.pt` holds the
/// full trained model; `model.safetensors` is the encoder it started from.
const FILES: [&str; 5] = [
    "config.json",
    "tokenizer.json",
    "marker_calibration.json",
    "model.safetensors",
    "option_marker.pt",
];

pub struct ModelFiles {
    pub dir: PathBuf,
}

impl ModelFiles {
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

pub fn fetch(repo: &str) -> Result<ModelFiles> {
    let (owner, name) = repo
        .split_once('/')
        .with_context(|| format!("repo {repo:?} is not owner/name"))?;
    let model = HFClientSync::new()?.model(owner, name);
    let mut dir = None;
    for file in FILES {
        eprintln!("fetching {repo}:{file}");
        let path = model
            .download_file()
            .filename(file)
            .send()
            .with_context(|| format!("downloading {repo}:{file}"))?;
        dir = path.parent().map(PathBuf::from);
    }
    Ok(ModelFiles {
        dir: dir.context("no files downloaded")?,
    })
}
