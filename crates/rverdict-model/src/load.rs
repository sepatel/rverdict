use std::path::Path;

use burn::prelude::*;
use burn_store::{
    ApplyResult, BurnToPyTorchAdapter, ChainAdapter, HalfPrecisionAdapter, ModuleSnapshot,
    PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};

use crate::Error;
use crate::decision::DecisionModel;
use crate::encoder::ModernBert;

/// Hugging Face names the projections `Wqkv`/`Wi`/`Wo`; Rust fields are
/// lower case. `^model\.` strips the `ForMaskedLM` wrapper prefix.
const REMAPS: [(&str, &str); 4] = [
    (r"^model\.", ""),
    (r"\.Wqkv\.", ".wqkv."),
    (r"\.Wi\.", ".wi."),
    (r"\.Wo\.", ".wo."),
];

fn checked(result: ApplyResult, path: &Path) -> Result<ApplyResult, Error> {
    if result.is_success() && result.missing.is_empty() {
        Ok(result)
    } else {
        Err(Error::Weights {
            path: path.to_owned(),
            detail: format!("{result}"),
        })
    }
}

/// Loads encoder weights from a Hugging Face ModernBERT safetensors file,
/// either a bare `ModernBertModel` or a `ModernBertForMaskedLM`. Tensors the
/// encoder does not use, such as the MLM head, are ignored.
pub fn load_encoder_safetensors<B: Backend>(
    model: &mut ModernBert<B>,
    path: &Path,
) -> Result<ApplyResult, Error> {
    let mut store = REMAPS
        .iter()
        .fold(SafetensorsStore::from_file(path), |store, &(from, to)| {
            store.with_key_remapping(from, to)
        })
        .with_from_adapter(PyTorchToBurnAdapter);
    let result = model.load_from(&mut store).map_err(|e| Error::Weights {
        path: path.to_owned(),
        detail: e.to_string(),
    })?;
    checked(result, path)
}

/// Loads a full option-marker model (encoder and scorer) from a PyTorch state
/// dict, such as Von's `option_marker.pt`.
pub fn load_decision_pytorch<B: Backend>(
    model: &mut DecisionModel<B>,
    path: &Path,
) -> Result<ApplyResult, Error> {
    let mut store = REMAPS
        .iter()
        .skip(1)
        .fold(PytorchStore::from_file(path), |store, &(from, to)| {
            store.with_key_remapping(from, to)
        });
    let result = model.load_from(&mut store).map_err(|e| Error::Weights {
        path: path.to_owned(),
        detail: e.to_string(),
    })?;
    checked(result, path)
}

/// Floating-point precision of the loaded weights and activations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    F32,
    /// Half the memory and bandwidth; for GPUs that support f16.
    F16,
}

/// Loads a full option-marker model from a safetensors file that uses the
/// PyTorch names (`encoder.*`, `scorer.*`), rverdict's own checkpoint format.
pub fn load_decision_safetensors<B: Backend>(
    model: &mut DecisionModel<B>,
    path: &Path,
    precision: Precision,
) -> Result<ApplyResult, Error> {
    let store = REMAPS
        .iter()
        .skip(1)
        .fold(SafetensorsStore::from_file(path), |store, &(from, to)| {
            store.with_key_remapping(from, to)
        });
    let mut store = match precision {
        Precision::F32 => store.with_from_adapter(PyTorchToBurnAdapter),
        Precision::F16 => store.with_from_adapter(ChainAdapter::new(
            PyTorchToBurnAdapter,
            HalfPrecisionAdapter::new().without_module("LayerNorm"),
        )),
    };
    let result = model.load_from(&mut store).map_err(|e| Error::Weights {
        path: path.to_owned(),
        detail: e.to_string(),
    })?;
    checked(result, path)
}

/// Saves a decision model as safetensors with the PyTorch names and layouts,
/// so [`load_decision_safetensors`] and PyTorch tooling can both read it.
pub fn save_decision_safetensors<B: Backend>(
    model: &DecisionModel<B>,
    path: &Path,
) -> Result<(), Error> {
    let mut store = [
        (r"\.wqkv\.", ".Wqkv."),
        (r"\.wi\.", ".Wi."),
        (r"\.wo\.", ".Wo."),
    ]
    .iter()
    .fold(SafetensorsStore::from_file(path), |store, &(from, to)| {
        store.with_key_remapping(from, to)
    })
    .with_to_adapter(BurnToPyTorchAdapter);
    model.save_into(&mut store).map_err(|e| Error::Weights {
        path: path.to_owned(),
        detail: e.to_string(),
    })
}
