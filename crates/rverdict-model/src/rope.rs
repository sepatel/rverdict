use burn::prelude::*;
use burn::tensor::DType;

/// Rotary position embedding in the Hugging Face "rotate half" layout, driven
/// by explicit position ids so a caller can reset positions per segment.
///
/// The angle math is done in f32 in the same order as `transformers`, because
/// the angles reach `8191 * inv_freq` and any difference in rounding there
/// shows up in the hidden states; only the results take the model's `dtype`.
pub(crate) fn cos_sin<B: Backend>(
    position_ids: Tensor<B, 2, Int>,
    theta: f64,
    head_dim: usize,
    dtype: DType,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let device = position_ids.device();
    let [batch, seq] = position_ids.dims();
    let half = head_dim / 2;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "head_dim is tiny and transformers computes these exponents in f32"
    )]
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let inv_freq =
        Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), &device).reshape([1, 1, half]);

    let positions = position_ids.float().reshape([batch, seq, 1]);
    let freqs = positions * inv_freq;
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 2).unsqueeze_dim::<4>(1);
    (emb.clone().cos().cast(dtype), emb.sin().cast(dtype))
}

pub(crate) fn apply<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, _, dim] = x.dims();
    let half = dim / 2;
    let first = x.clone().narrow(3, 0, half);
    let second = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![second.neg(), first], 3);
    x * cos.clone() + rotated * sin.clone()
}
