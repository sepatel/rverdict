use std::path::{Path, PathBuf};

use hf_hub::{HFClientSync, HFError};
use rverdict_core::Calibration;
use rverdict_model::EncoderConfig;
use serde::Deserialize;

use crate::EngineError;

/// A Hugging Face model pinned to a revision, so a download today and a
/// download next year load identical weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub repo: String,
    pub revision: Option<String>,
}

/// Short names for known checkpoints.
const ALIASES: [(&str, &str, &str); 1] = [(
    "von",
    "wfzyx/von",
    "498ceba33390b32cfefaab6422ec380318ba9b99",
)];

impl ModelRef {
    /// Accepts an alias (`von`), `owner/name`, or `owner/name@revision`.
    pub fn parse(spec: &str) -> Self {
        if let Some((_, repo, revision)) = ALIASES.iter().find(|(alias, ..)| *alias == spec) {
            return Self {
                repo: (*repo).to_owned(),
                revision: Some((*revision).to_owned()),
            };
        }
        match spec.split_once('@') {
            Some((repo, revision)) => Self {
                repo: repo.to_owned(),
                revision: Some(revision.to_owned()),
            },
            None => Self {
                repo: spec.to_owned(),
                revision: None,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Weights {
    /// rverdict's own format: encoder and scorer in one safetensors file.
    Safetensors(PathBuf),
    /// A PyTorch state dict with `encoder.*` and `scorer.*`, such as Von's.
    PyTorch(PathBuf),
}

/// Model options read from the checkpoint's `marker_calibration.json`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Settings {
    /// Score options independently of each other (order invariant).
    #[serde(default)]
    pub independent_options: bool,
    /// Space out every digit before tokenizing.
    #[serde(default)]
    pub digit_split: bool,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(flatten)]
    pub calibration: Calibration,
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Shown as `model` in responses.
    pub name: String,
    pub config: EncoderConfig,
    pub tokenizer: PathBuf,
    pub weights: Weights,
    pub settings: Settings,
}

pub(crate) const DECISION_SAFETENSORS: &str = "decision.safetensors";
pub(crate) const OPTION_MARKER_PT: &str = "option_marker.pt";
pub(crate) const CALIBRATION: &str = "marker_calibration.json";

impl Checkpoint {
    /// Loads a checkpoint from a local directory.
    pub fn from_dir(dir: &Path) -> Result<Self, EngineError> {
        let weights = [
            (
                DECISION_SAFETENSORS,
                Weights::Safetensors as fn(PathBuf) -> Weights,
            ),
            (OPTION_MARKER_PT, Weights::PyTorch),
        ]
        .into_iter()
        .find_map(|(file, kind)| {
            let path = dir.join(file);
            path.exists().then(|| kind(path))
        })
        .ok_or_else(|| EngineError::MissingFile {
            dir: dir.to_owned(),
            file: format!("{DECISION_SAFETENSORS} or {OPTION_MARKER_PT}"),
        })?;
        let name = dir.file_name().map_or_else(
            || "rverdict".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        Self::assemble(dir, weights, name)
    }

    /// Downloads (or reuses from the Hugging Face cache) what a checkpoint
    /// needs, then loads it.
    ///
    /// When everything is already cached the network is not touched, so a
    /// pinned model works offline after its first download.
    pub fn from_hub(model: &ModelRef) -> Result<Self, EngineError> {
        let (owner, name) = model
            .repo
            .split_once('/')
            .ok_or_else(|| EngineError::BadModel(model.repo.clone()))?;
        let repo = HFClientSync::new()?.model(owner, name);
        let get = |file: &str, local: bool| -> Result<Option<PathBuf>, EngineError> {
            let request = repo
                .download_file()
                .filename(file)
                .maybe_revision(model.revision.clone())
                .local_files_only(local);
            match request.send() {
                Ok(path) => Ok(Some(path)),
                Err(HFError::EntryNotFound { .. } | HFError::LocalEntryNotFound { .. }) => Ok(None),
                Err(e) => Err(e.into()),
            }
        };
        let cached = |file: &str| get(file, true).ok().flatten();
        let offline = cached("config.json").is_some()
            && cached("tokenizer.json").is_some()
            && (cached(DECISION_SAFETENSORS).is_some() || cached(OPTION_MARKER_PT).is_some());
        let fetch = |file: &str| {
            if offline {
                Ok(cached(file))
            } else {
                get(file, false)
            }
        };

        let missing = |file: &str| EngineError::MissingFile {
            dir: PathBuf::from(&model.repo),
            file: file.to_owned(),
        };
        let config = fetch("config.json")?.ok_or_else(|| missing("config.json"))?;
        fetch("tokenizer.json")?.ok_or_else(|| missing("tokenizer.json"))?;
        fetch(CALIBRATION)?;
        let weights = match fetch(DECISION_SAFETENSORS)? {
            Some(path) => Weights::Safetensors(path),
            None => {
                Weights::PyTorch(fetch(OPTION_MARKER_PT)?.ok_or_else(|| missing(OPTION_MARKER_PT))?)
            }
        };
        let dir = config
            .parent()
            .expect("a downloaded file has a parent directory");
        let label = match &model.revision {
            Some(rev) => format!("{}@{}", model.repo, &rev[..rev.len().min(8)]),
            None => model.repo.clone(),
        };
        Self::assemble(dir, weights, label)
    }

    fn assemble(dir: &Path, weights: Weights, name: String) -> Result<Self, EngineError> {
        let calibration = dir.join(CALIBRATION);
        let settings = if calibration.exists() {
            let text = std::fs::read_to_string(&calibration).map_err(|source| EngineError::Io {
                path: calibration.clone(),
                source,
            })?;
            serde_json::from_str(&text)?
        } else {
            Settings::default()
        };
        Ok(Self {
            name: settings.model_id.clone().unwrap_or(name),
            config: EncoderConfig::from_file(&dir.join("config.json"))?,
            tokenizer: dir.join("tokenizer.json"),
            weights,
            settings,
        })
    }
}
