use std::time::Instant;

use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Distribution, s};

fn time<B: Backend, const D: usize>(name: &str, flops: f64, mut op: impl FnMut() -> Tensor<B, D>) {
    let _ = op().into_data();
    let start = Instant::now();
    let runs = 5;
    for _ in 0..runs {
        let _ = op().into_data();
    }
    let secs = start.elapsed().as_secs_f64() / f64::from(runs);
    println!(
        "{name:<34} {:>8.2} ms  {:>7.1} GFLOP/s",
        secs * 1e3,
        flops / secs / 1e9
    );
}

/// Times the encoder's building blocks at one sequence length to find which
/// kernel dominates.
#[expect(clippy::cast_precision_loss, reason = "shapes are small")]
pub fn run<B: Backend>(device: &B::Device, seq: usize) {
    let (hidden, heads, head_dim, inter) = (1024, 16, 64, 2624);
    let x = Tensor::<B, 3>::random([1, seq, hidden], Distribution::Default, device);
    let wi = Tensor::<B, 2>::random([hidden, 2 * inter], Distribution::Default, device);
    let wqkv = Tensor::<B, 2>::random([hidden, 3 * hidden], Distribution::Default, device);
    let q = Tensor::<B, 4>::random([1, heads, seq, head_dim], Distribution::Default, device);
    let mask = Tensor::<B, 4, Int>::zeros([1, 1, seq, seq], device)
        .equal_elem(1)
        .expand([1, heads, seq, seq]);
    let n = seq as f64;

    time(
        "linear Wi  [s,1024]x[1024,5248]",
        2.0 * n * 1024.0 * 5248.0,
        || x.clone().matmul(wi.clone().unsqueeze()),
    );
    time(
        "linear Wqkv [s,1024]x[1024,3072]",
        2.0 * n * 1024.0 * 3072.0,
        || x.clone().matmul(wqkv.clone().unsqueeze()),
    );
    time("attention (bool mask)", 4.0 * 16.0 * n * n * 64.0, || {
        attention(
            q.clone(),
            q.clone(),
            q.clone(),
            Some(mask.clone()),
            None,
            AttentionModuleOptions::default(),
        )
    });
    time("attention (no mask)", 4.0 * 16.0 * n * n * 64.0, || {
        attention(
            q.clone(),
            q.clone(),
            q.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        )
    });
    time(
        "attention (matmul+softmax, mask)",
        4.0 * 16.0 * n * n * 64.0,
        || {
            let scores = q.clone().matmul(q.clone().swap_dims(2, 3)) * 0.125;
            let probs = burn::tensor::activation::softmax(
                scores.mask_fill(mask.clone(), f32::NEG_INFINITY),
                3,
            );
            probs.matmul(q.clone())
        },
    );
    time("gelu [s,2624]", 0.0, || {
        gelu(x.clone().slice(s![.., .., 0..1024]))
    });
    let norm = burn::nn::LayerNormConfig::new(hidden)
        .with_bias(false)
        .init::<B>(device);
    time("layer norm [s,1024]", 0.0, || norm.forward(x.clone()));
    time("qkv split + swap_dims", 0.0, || {
        x.clone()
            .matmul(wqkv.clone().unsqueeze())
            .reshape([1, seq, 3, heads, head_dim])
            .narrow(2, 0, 1)
            .reshape([1, seq, heads, head_dim])
            .swap_dims(1, 2)
    });
    time("q via weight narrow + swap_dims", 0.0, || {
        x.clone()
            .matmul(wqkv.clone().narrow(1, 0, hidden).unsqueeze())
            .reshape([1, seq, heads, head_dim])
            .swap_dims(1, 2)
    });
    let cos = Tensor::<B, 4>::random([1, 1, seq, head_dim], Distribution::Default, device);
    let rotation = Tensor::<B, 2>::random([head_dim, head_dim], Distribution::Default, device);
    time("rope apply via [64,64] matmul", 0.0, || {
        q.clone() * cos.clone() + q.clone().matmul(rotation.clone().unsqueeze()) * cos.clone()
    });
    time("rope apply [16,s,64]", 0.0, || {
        let first = q.clone().narrow(3, 0, 32);
        let second = q.clone().narrow(3, 32, 32);
        q.clone() * cos.clone() + Tensor::cat(vec![second.neg(), first], 3) * cos.clone()
    });
}
