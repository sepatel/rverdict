use std::time::{Duration, Instant};

use anyhow::Result;
use burn::backend::Autodiff;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use rverdict_model::{EncoderConfig, OptionAttention, PackedSequence, build_input};

use crate::fetch::ModelFiles;
use crate::parity::SAMPLE;
use crate::von::Packer;

const QUESTION: &str = "Which department should handle this request?";
const OPTIONS: [&str; 3] = [
    "billing: invoices, payments, refunds",
    "technical: bugs, outages, system errors",
    "sales: pricing, new contracts",
];

/// A realistic option-marker sequence of roughly `length` tokens.
fn sequence(packer: &Packer, length: usize) -> Result<PackedSequence> {
    let overhead = packer.pack("", QUESTION, &OPTIONS)?.token_ids.len();
    let filler = std::iter::repeat_n(SAMPLE, length / 40 + 2)
        .collect::<Vec<_>>()
        .join(" ");
    let mut ids = packer.encode_text(&filler)?;
    ids.truncate(length.saturating_sub(overhead + 2));
    let state = packer.decode(&ids[1..])?;
    packer.pack(&state, QUESTION, &OPTIONS)
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

pub fn inference<B: Backend>(
    device: &B::Device,
    files: &ModelFiles,
    lengths: &[usize],
    runs: usize,
) -> Result<()> {
    let config = EncoderConfig::from_file(&files.path("config.json"))?;
    let packer = Packer::from_file(&files.path("tokenizer.json"))?;
    let model = config.init_decision_model::<B>(device);

    for &length in lengths {
        let seq = sequence(&packer, length)?;
        let mut samples = Vec::with_capacity(runs);
        for run in 0..runs + 2 {
            let start = Instant::now();
            let input = build_input::<B>(
                std::slice::from_ref(&seq),
                OptionAttention::Independent,
                config.pad_token_id,
                config.sliding_window,
                device,
            );
            let logits = model.option_logits(input, std::slice::from_ref(&seq.markers));
            let _ = logits.into_data();
            if run >= 2 {
                samples.push(start.elapsed());
            }
        }
        println!(
            "inference  {:>5} tokens  median {:>8.1} ms  (min {:.1} ms, {runs} runs after 2 warm-up)",
            seq.token_ids.len(),
            median(samples.clone()).as_secs_f64() * 1e3,
            samples.iter().min().map_or(0.0, |d| d.as_secs_f64() * 1e3),
        );
    }
    Ok(())
}

pub fn train<B: Backend>(
    device: &B::Device,
    files: &ModelFiles,
    length: usize,
    rows: usize,
    steps: usize,
) -> Result<()> {
    type Ad<B> = Autodiff<B>;
    let config = EncoderConfig::from_file(&files.path("config.json"))?;
    let packer = Packer::from_file(&files.path("tokenizer.json"))?;
    let mut model = config.init_decision_model::<Ad<B>>(device);
    let mut optim = AdamWConfig::new().init::<Ad<B>, _>();

    let seq = sequence(&packer, length)?;
    let batch = vec![seq.clone(); rows];
    let markers = vec![seq.markers.clone(); rows];
    let options = seq.markers.len();

    let mut samples = Vec::with_capacity(steps);
    for step in 0..=steps {
        let start = Instant::now();
        let input = build_input::<Ad<B>>(
            &batch,
            OptionAttention::Independent,
            config.pad_token_id,
            config.sliding_window,
            device,
        );
        let logits = model
            .option_logits(input, &markers)
            .reshape([rows, options]);
        let loss = log_softmax(logits, 1).narrow(1, 0, 1).neg().mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-5, model, grads);
        let loss: f32 = loss.into_scalar().elem();
        let elapsed = start.elapsed();
        println!(
            "train step {step}: loss {loss:.4} in {:.2} s",
            elapsed.as_secs_f64()
        );
        if step >= 1 {
            samples.push(elapsed);
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let tokens = (rows * seq.token_ids.len()) as f64;
    let step = median(samples).as_secs_f64();
    println!(
        "train      {rows} × {} tokens  median step {step:.2} s  ≈ {:.0} tokens/s (AdamW, full fine-tune)",
        seq.token_ids.len(),
        tokens / step
    );
    Ok(())
}
