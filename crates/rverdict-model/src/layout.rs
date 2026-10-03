use burn::prelude::*;

use crate::encoder::EncoderInput;

/// One tokenised sequence: `[CLS] prefix [SEP] [MASK] opt0 [MASK] opt1 … [SEP]`,
/// with the index of every option marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedSequence {
    pub token_ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// How option spans see each other inside the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionAttention {
    /// Plain bidirectional attention over the whole sequence.
    Shared,
    /// Each option attends only to the prefix and to itself, and every
    /// option's positions restart at the prefix length, so an option's logit
    /// depends on (prefix, that option) alone and never on option order.
    Independent,
}

/// The token span each position belongs to: the shared prefix or option `k`.
/// The trailing `[SEP]` and padding count as prefix.
fn segments(seq: &PackedSequence) -> Vec<Option<usize>> {
    let mut segment = vec![None; seq.token_ids.len()];
    let last_content = seq.token_ids.len().saturating_sub(1);
    for (k, &start) in seq.markers.iter().enumerate() {
        let end = seq.markers.get(k + 1).copied().unwrap_or(last_content);
        segment[start..end].fill(Some(k));
    }
    segment
}

fn position_ids(
    seq: &PackedSequence,
    segment: &[Option<usize>],
    mode: OptionAttention,
) -> Vec<usize> {
    let mut positions: Vec<usize> = (0..seq.token_ids.len()).collect();
    if mode == OptionAttention::Independent
        && let Some(&prefix_len) = seq.markers.first()
    {
        let mut offset = 0;
        for i in 0..positions.len() {
            match segment[i] {
                Some(k) if i == seq.markers[k] => offset = 0,
                Some(_) => offset += 1,
                None => continue,
            }
            positions[i] = prefix_len + offset;
        }
    }
    positions
}

/// Blocked-attention masks for one row padded to `width`: `(global, sliding)`.
fn row_masks(
    segment: &[Option<usize>],
    positions: &[usize],
    width: usize,
    mode: OptionAttention,
    sliding_window: usize,
) -> (Vec<bool>, Vec<bool>) {
    let len = segment.len();
    let mut global = vec![true; width * width];
    let mut sliding = vec![true; width * width];
    for i in 0..width {
        for j in 0..len {
            let allowed = match mode {
                OptionAttention::Shared => true,
                OptionAttention::Independent if i >= len => true,
                OptionAttention::Independent => match (segment[i], segment[j]) {
                    (None | Some(_), None) => true,
                    (Some(a), Some(b)) => a == b,
                    (None, Some(_)) => false,
                },
            };
            if allowed {
                global[i * width + j] = false;
                let pi = positions.get(i).copied().unwrap_or(i);
                if pi.abs_diff(positions[j]) <= sliding_window {
                    sliding[i * width + j] = false;
                }
            }
        }
        // A row with nothing to attend to would softmax to NaN.
        global[i * width + i] = false;
        sliding[i * width + i] = false;
    }
    (global, sliding)
}

/// Pads a batch of sequences into encoder tensors, returning the marker
/// indices per row alongside them.
pub fn build_input<B: Backend>(
    batch: &[PackedSequence],
    mode: OptionAttention,
    pad_token_id: u32,
    sliding_window: usize,
    device: &B::Device,
) -> EncoderInput<B> {
    let rows = batch.len();
    let width = batch.iter().map(|s| s.token_ids.len()).max().unwrap_or(0);

    let mut ids = Vec::with_capacity(rows * width);
    let mut positions = Vec::with_capacity(rows * width);
    let mut global = Vec::with_capacity(rows * width * width);
    let mut sliding = Vec::with_capacity(rows * width * width);

    for seq in batch {
        let segment = segments(seq);
        let row_positions = position_ids(seq, &segment, mode);
        let (g, s) = row_masks(&segment, &row_positions, width, mode, sliding_window);
        global.extend(g);
        sliding.extend(s);

        ids.extend(seq.token_ids.iter().map(|&t| i64::from(t)));
        ids.extend(std::iter::repeat_n(
            i64::from(pad_token_id),
            width - seq.token_ids.len(),
        ));
        positions.extend(
            row_positions
                .iter()
                .map(|&p| i64::try_from(p).expect("sequence length fits in i64")),
        );
        positions.extend(
            (seq.token_ids.len()..width)
                .map(|p| i64::try_from(p).expect("sequence length fits in i64")),
        );
    }

    EncoderInput {
        input_ids: Tensor::from_data(TensorData::new(ids, [rows, width]), device),
        position_ids: Tensor::from_data(TensorData::new(positions, [rows, width]), device),
        global_mask: Tensor::from_data(TensorData::new(global, [rows, 1, width, width]), device),
        sliding_mask: Tensor::from_data(TensorData::new(sliding, [rows, 1, width, width]), device),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [CLS] p p [SEP] [MASK] a a [MASK] b [SEP]
    fn sample() -> PackedSequence {
        PackedSequence {
            token_ids: vec![1, 10, 11, 2, 4, 20, 21, 4, 30, 2],
            markers: vec![4, 7],
        }
    }

    #[test]
    fn independent_positions_restart_each_option_and_keep_the_final_sep_raw() {
        let seq = sample();
        let segment = segments(&seq);
        assert_eq!(
            position_ids(&seq, &segment, OptionAttention::Independent),
            vec![0, 1, 2, 3, 4, 5, 6, 4, 5, 9]
        );
    }

    #[test]
    fn independent_options_cannot_see_each_other_but_see_the_prefix() {
        let seq = sample();
        let segment = segments(&seq);
        let positions = position_ids(&seq, &segment, OptionAttention::Independent);
        let (global, _) = row_masks(&segment, &positions, 10, OptionAttention::Independent, 64);
        let blocked = |i: usize, j: usize| global[i * 10 + j];
        assert!(!blocked(5, 1), "option token sees prefix");
        assert!(!blocked(5, 4), "option token sees its own marker");
        assert!(blocked(5, 8), "option a does not see option b");
        assert!(blocked(1, 5), "prefix does not see options");
        assert!(!blocked(8, 9), "options see the trailing [SEP]");
    }
}
