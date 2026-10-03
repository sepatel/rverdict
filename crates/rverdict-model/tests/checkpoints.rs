//! Real checkpoints, too large for CI. Point the environment variable at a
//! directory holding the model's `config.json` and `model.safetensors`, then
//! run with `--ignored`.

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use rverdict_model::{EncoderConfig, load_encoder_safetensors};

/// `answerdotai/ModernBERT-large`, the base our own checkpoints train from:
/// a `ForMaskedLM` file with a `model.` prefix and an MLM head to ignore.
#[test]
#[ignore = "needs RVERDICT_MODERNBERT_LARGE pointing at the 1.6 GB checkpoint"]
fn modernbert_large_loads_every_encoder_weight() {
    let dir = PathBuf::from(
        std::env::var("RVERDICT_MODERNBERT_LARGE").expect("RVERDICT_MODERNBERT_LARGE is set"),
    );
    let config = EncoderConfig::from_file(&dir.join("config.json")).unwrap();
    let mut encoder = config.init::<NdArray>(&NdArrayDevice::default());
    let result = load_encoder_safetensors(&mut encoder, &dir.join("model.safetensors")).unwrap();
    assert!(result.missing.is_empty(), "missing: {:?}", result.missing);
    // burn-store also lists a norm's PyTorch `weight` as unused after
    // applying it as Burn's `gamma`, so only names with no applied
    // counterpart are truly unused, and those must be the MLM head.
    let applied_as_gamma = |k: &str| {
        k.strip_suffix(".weight")
            .is_some_and(|p| result.applied.contains(&format!("{p}.gamma")))
    };
    let unused: Vec<_> = result
        .unused
        .iter()
        .filter(|k| !applied_as_gamma(k))
        .collect();
    assert!(
        unused
            .iter()
            .all(|k| k.starts_with("head.") || k.starts_with("decoder.")),
        "unused: {unused:?}"
    );
}
