//! Installs a pinned checkpoint into a directory the caller owns, verified
//! file by file, for apps that keep models in their own data directory
//! rather than the shared Hugging Face cache.
//!
//! PyTorch weights are converted to `decision.safetensors` on install and
//! the pickle is removed, so an installed checkpoint is half the disk space
//! and loads directly with [`Checkpoint::from_dir`].

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use burn::{Dispatch, DispatchDevice};
use rverdict_model::{load_decision_pytorch, save_decision_safetensors};
use serde::{Deserialize, Serialize};
use sha1::Digest as _;

use crate::EngineError;
use crate::checkpoint::{
    CALIBRATION, Checkpoint, DECISION_SAFETENSORS, ModelRef, OPTION_MARKER_PT,
};

const MANIFEST: &str = "rverdict-manifest.json";
const HUB: &str = "https://huggingface.co";

/// What an install is doing, reported as it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallStep {
    Downloading { file: String, done: u64, total: u64 },
    Converting,
    Verifying,
}

/// Written last, so its presence means the install completed. Records what
/// was installed and the sha256 of every file kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub repo: String,
    pub revision: String,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestFile {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Deserialize)]
struct RevisionInfo {
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    sha256: String,
}

/// The directory `model` installs into under `root`.
pub fn install_dir(root: &Path, model: &ModelRef) -> Option<PathBuf> {
    let revision = model.revision.as_ref()?;
    Some(root.join(format!("{}@{revision}", model.repo.replace('/', "--"))))
}

/// The installed checkpoint, if a complete install exists. Does not touch
/// the network or re-hash files; see [`verify`].
pub fn installed(root: &Path, model: &ModelRef) -> Option<Checkpoint> {
    let dir = install_dir(root, model)?;
    dir.join(MANIFEST)
        .exists()
        .then(|| Checkpoint::from_dir(&dir).ok())
        .flatten()
}

/// Downloads `model` (which must be pinned to a revision) into `root`,
/// checking every file's size and hash against what Hugging Face reports,
/// then converts PyTorch weights to safetensors. Returns at once if it is
/// already installed. `progress` returning `false` cancels the install.
pub fn install(
    root: &Path,
    model: &ModelRef,
    mut progress: impl FnMut(&InstallStep) -> bool,
) -> Result<Checkpoint, EngineError> {
    if let Some(checkpoint) = installed(root, model) {
        return Ok(checkpoint);
    }
    let revision = model
        .revision
        .clone()
        .ok_or_else(|| EngineError::BadModel(format!("{} has no pinned revision", model.repo)))?;
    let dir = install_dir(root, model).expect("revision is pinned");
    create_dir(&dir)?;

    let http = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|e| download_error(&model.repo, &e))?;
    let info: RevisionInfo = {
        let url = format!(
            "{HUB}/api/models/{}/revision/{revision}?blobs=true",
            model.repo
        );
        let text = http
            .get(&url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::text)
            .map_err(|e| download_error(&url, &e))?;
        serde_json::from_str(&text)?
    };
    let sibling = |name: &str| info.siblings.iter().find(|s| s.rfilename == name);
    let weights = [DECISION_SAFETENSORS, OPTION_MARKER_PT]
        .into_iter()
        .find(|w| sibling(w).is_some())
        .ok_or_else(|| EngineError::MissingFile {
            dir: PathBuf::from(&model.repo),
            file: format!("{DECISION_SAFETENSORS} or {OPTION_MARKER_PT}"),
        })?;

    let mut kept = Vec::new();
    for name in ["config.json", "tokenizer.json", CALIBRATION, weights] {
        let Some(sibling) = sibling(name) else {
            if name == CALIBRATION {
                continue;
            }
            return Err(EngineError::MissingFile {
                dir: PathBuf::from(&model.repo),
                file: name.to_owned(),
            });
        };
        let url = format!("{HUB}/{}/resolve/{revision}/{name}", model.repo);
        let sha256 = download(&http, &url, sibling, &dir.join(name), &mut progress)?;
        if name != OPTION_MARKER_PT {
            kept.push(ManifestFile {
                name: name.to_owned(),
                size: sibling.size.unwrap_or_default(),
                sha256,
            });
        }
    }

    if weights == OPTION_MARKER_PT {
        if !progress(&InstallStep::Converting) {
            return Err(EngineError::Cancelled);
        }
        let pickle = dir.join(OPTION_MARKER_PT);
        let target = dir.join(DECISION_SAFETENSORS);
        convert(&dir, &pickle, &target)?;
        let size = std::fs::metadata(&target)
            .map_err(|source| io(&target, source))?
            .len();
        kept.push(ManifestFile {
            name: DECISION_SAFETENSORS.to_owned(),
            size,
            sha256: sha256_file(&target)?,
        });
        std::fs::remove_file(&pickle).map_err(|source| io(&pickle, source))?;
    }

    let manifest = Manifest {
        repo: model.repo.clone(),
        revision,
        files: kept,
    };
    write_atomic(
        &dir.join(MANIFEST),
        serde_json::to_string_pretty(&manifest)?.as_bytes(),
    )?;
    Checkpoint::from_dir(&dir)
}

/// Re-hashes an installed checkpoint against its manifest.
pub fn verify(root: &Path, model: &ModelRef) -> Result<bool, EngineError> {
    let Some(dir) = install_dir(root, model) else {
        return Ok(false);
    };
    let path = dir.join(MANIFEST);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(false);
    };
    let manifest: Manifest = serde_json::from_str(&text)?;
    for file in &manifest.files {
        let path = dir.join(&file.name);
        if !path.exists() || sha256_file(&path)? != file.sha256 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Streams `url` to `path` via a partial file, checking size and hash.
/// Returns the file's sha256.
fn download(
    http: &reqwest::blocking::Client,
    url: &str,
    sibling: &Sibling,
    path: &Path,
    progress: &mut impl FnMut(&InstallStep) -> bool,
) -> Result<String, EngineError> {
    let mut response = http
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| download_error(url, &e))?;
    let total = sibling
        .size
        .or_else(|| response.content_length())
        .unwrap_or_default();
    let partial = path.with_extension("partial");
    let mut out = std::fs::File::create(&partial).map_err(|source| io(&partial, source))?;
    let mut sha256 = sha2::Sha256::new();
    // Files outside LFS are identified by their git blob id: the sha1 of a
    // `blob <size>\0` header followed by the content.
    let mut git = sha1::Sha1::new();
    git.update(format!("blob {total}\0").as_bytes());

    let name = sibling.rfilename.clone();
    let mut buffer = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = response
            .read(&mut buffer)
            .map_err(|source| io(path, source))?;
        if n == 0 {
            break;
        }
        out.write_all(&buffer[..n])
            .map_err(|source| io(&partial, source))?;
        sha256.update(&buffer[..n]);
        git.update(&buffer[..n]);
        done += n as u64;
        if !progress(&InstallStep::Downloading {
            file: name.clone(),
            done,
            total,
        }) {
            drop(out);
            let _ = std::fs::remove_file(&partial);
            return Err(EngineError::Cancelled);
        }
    }
    out.sync_all().map_err(|source| io(&partial, source))?;
    drop(out);

    let sha256 = hex(&sha256.finalize());
    let git = hex(&git.finalize());
    let expected_ok = match (&sibling.lfs, &sibling.blob_id) {
        (Some(lfs), _) => lfs.sha256 == sha256,
        (None, Some(blob)) => *blob == git,
        (None, None) => true,
    };
    if done != total || !expected_ok {
        let _ = std::fs::remove_file(&partial);
        return Err(EngineError::Corrupt {
            file: name,
            detail: format!("got {done} of {total} bytes, sha256 {sha256}"),
        });
    }
    std::fs::rename(&partial, path).map_err(|source| io(path, source))?;
    Ok(sha256)
}

/// Loads PyTorch weights on the CPU and writes them as safetensors.
fn convert(dir: &Path, pickle: &Path, target: &Path) -> Result<(), EngineError> {
    let config = rverdict_model::EncoderConfig::from_file(&dir.join("config.json"))?;
    let device = DispatchDevice::Flex(burn::backend::flex::FlexDevice);
    let mut model = config.init_decision_model::<Dispatch>(&device);
    load_decision_pytorch(&mut model, pickle)?;
    let partial = target.with_extension("partial");
    save_decision_safetensors(&model, &partial)?;
    std::fs::rename(&partial, target).map_err(|source| io(target, source))
}

fn sha256_file(path: &Path) -> Result<String, EngineError> {
    let mut file = std::fs::File::open(path).map_err(|source| io(path, source))?;
    let mut hash = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hash).map_err(|source| io(path, source))?;
    Ok(hex(&hash.finalize()))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|source| io(&partial, source))?;
    std::fs::rename(&partial, path).map_err(|source| io(path, source))
}

fn create_dir(dir: &Path) -> Result<(), EngineError> {
    std::fs::create_dir_all(dir).map_err(|source| io(dir, source))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn io(path: &Path, source: std::io::Error) -> EngineError {
    EngineError::Io {
        path: path.to_owned(),
        source,
    }
}

fn download_error(url: &str, error: &reqwest::Error) -> EngineError {
    EngineError::Download {
        url: url.to_owned(),
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_blob_ids_match_git() {
        // `printf 'hello\n' | git hash-object --stdin`
        let mut git = sha1::Sha1::new();
        git.update(b"blob 6\0hello\n");
        assert_eq!(
            hex(&git.finalize()),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    #[test]
    fn installs_live_in_a_directory_per_revision() {
        let model = ModelRef::parse("von");
        let dir = install_dir(Path::new("/data"), &model).unwrap();
        assert_eq!(
            dir,
            Path::new("/data/wfzyx--von@498ceba33390b32cfefaab6422ec380318ba9b99")
        );
        assert!(install_dir(Path::new("/data"), &ModelRef::parse("a/b")).is_none());
    }
}
