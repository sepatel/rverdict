use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use rverdict_model::{DecisionModel, EncoderConfig, OptionAttention, PackedSequence, build_input};

type B = NdArray;

fn logits(model: &DecisionModel<B>, config: &EncoderConfig, batch: &[PackedSequence]) -> Vec<f32> {
    let device = NdArrayDevice::default();
    let input = build_input::<B>(
        batch,
        OptionAttention::Independent,
        config.pad_token_id,
        config.sliding_window,
        &device,
    );
    let markers: Vec<_> = batch.iter().map(|s| s.markers.clone()).collect();
    model
        .option_logits(input, &markers)
        .into_data()
        .to_vec()
        .unwrap()
}

/// `[CLS] prefix [SEP] ([MASK] option)* [SEP]` with marker token 4.
fn packed(prefix: &[u32], options: &[&[u32]]) -> PackedSequence {
    let mut token_ids = vec![1];
    token_ids.extend(prefix);
    token_ids.push(2);
    let mut markers = Vec::new();
    for option in options {
        markers.push(token_ids.len());
        token_ids.push(4);
        token_ids.extend(*option);
    }
    token_ids.push(2);
    PackedSequence { token_ids, markers }
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
    }
}

#[test]
fn padding_a_row_in_a_batch_does_not_change_its_logits() {
    let config = EncoderConfig::tiny();
    let model = config.init_decision_model::<B>(&NdArrayDevice::default());
    let short = packed(&[10, 11], &[&[20], &[21, 22]]);
    let long = packed(
        &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31],
        &[&[20, 23, 24], &[21], &[25]],
    );

    let alone = [
        logits(&model, &config, std::slice::from_ref(&short)),
        logits(&model, &config, std::slice::from_ref(&long)),
    ]
    .concat();
    assert_close(&logits(&model, &config, &[short, long]), &alone);
}

#[test]
fn permuting_options_only_permutes_their_logits() {
    let config = EncoderConfig::tiny();
    let model = config.init_decision_model::<B>(&NdArrayDevice::default());
    let prefix = [10, 11, 12, 13, 14, 15];
    let (a, b, c): (&[u32], &[u32], &[u32]) = (&[20, 21], &[22], &[23, 24, 25]);

    let forward = logits(&model, &config, &[packed(&prefix, &[a, b, c])]);
    let permuted = logits(&model, &config, &[packed(&prefix, &[c, a, b])]);
    assert_close(&[permuted[1], permuted[2], permuted[0]], &forward);
}

#[test]
fn saved_safetensors_load_back_to_the_same_model() {
    let config = EncoderConfig::tiny();
    let device = NdArrayDevice::default();
    let model = config.init_decision_model::<B>(&device);
    let path = std::env::temp_dir().join(format!(
        "rverdict-roundtrip-{}.safetensors",
        std::process::id()
    ));
    rverdict_model::save_decision_safetensors(&model, &path).unwrap();

    let mut loaded = config.init_decision_model::<B>(&device);
    rverdict_model::load_decision_safetensors(&mut loaded, &path, rverdict_model::Precision::F32)
        .unwrap();
    std::fs::remove_file(&path).unwrap();

    let seq = packed(&[10, 11, 12], &[&[20], &[21, 22]]);
    assert_close(
        &logits(&loaded, &config, std::slice::from_ref(&seq)),
        &logits(&model, &config, &[seq]),
    );
}
