use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};

use crate::config::{AttentionKind, EncoderConfig};
use crate::rope;

/// One batch of encoder input. Masks are `true` where attention is blocked,
/// shaped `[batch, 1, seq, seq]`: sliding layers use `sliding_mask`, global
/// layers `global_mask`, so a caller controls both patterns, e.g. to keep
/// option spans from seeing each other.
#[derive(Debug, Clone)]
pub struct EncoderInput<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub position_ids: Tensor<B, 2, Int>,
    pub global_mask: Tensor<B, 4, Bool>,
    pub sliding_mask: Tensor<B, 4, Bool>,
}

#[derive(Module, Debug)]
pub struct Embeddings<B: Backend> {
    tok_embeddings: Embedding<B>,
    norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    wqkv: Linear<B>,
    wo: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    wi: Linear<B>,
    wo: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    /// Absent on the first layer, whose input is already normalised by the
    /// embedding norm.
    attn_norm: Option<LayerNorm<B>>,
    attn: Attention<B>,
    mlp_norm: LayerNorm<B>,
    mlp: Mlp<B>,
    sliding: bool,
}

/// The ModernBERT encoder. Field names mirror the Hugging Face checkpoint so
/// weights load with only a lower-casing remap.
#[derive(Module, Debug)]
pub struct ModernBert<B: Backend> {
    embeddings: Embeddings<B>,
    layers: Vec<Layer<B>>,
    final_norm: LayerNorm<B>,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModernBert<B> {
        let norm = |dim| {
            LayerNormConfig::new(dim)
                .with_epsilon(self.norm_eps)
                .with_bias(self.norm_bias)
                .init(device)
        };
        let linear =
            |d_in, d_out, bias| LinearConfig::new(d_in, d_out).with_bias(bias).init(device);
        let hidden = self.hidden_size;

        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let sliding = *kind == AttentionKind::Sliding;
                Layer {
                    attn_norm: (index > 0).then(|| norm(hidden)),
                    attn: Attention {
                        wqkv: linear(hidden, 3 * hidden, self.attention_bias),
                        wo: linear(hidden, hidden, self.attention_bias),
                        num_heads: self.num_attention_heads,
                        head_dim: self.head_dim(),
                        rope_theta: if sliding {
                            self.local_rope_theta
                        } else {
                            self.global_rope_theta
                        },
                    },
                    mlp_norm: norm(hidden),
                    mlp: Mlp {
                        wi: linear(hidden, 2 * self.intermediate_size, self.mlp_bias),
                        wo: linear(self.intermediate_size, hidden, self.mlp_bias),
                    },
                    sliding,
                }
            })
            .collect();

        ModernBert {
            embeddings: Embeddings {
                tok_embeddings: EmbeddingConfig::new(self.vocab_size, hidden).init(device),
                norm: norm(hidden),
            },
            layers,
            final_norm: norm(hidden),
        }
    }
}

impl<B: Backend> ModernBert<B> {
    /// Returns the last hidden state, `[batch, seq, hidden]`.
    pub fn forward(&self, input: EncoderInput<B>) -> Tensor<B, 3> {
        let head_dim = self.layers[0].attn.head_dim;
        let dtype = self.embeddings.tok_embeddings.weight.val().dtype();
        let rope_for = |theta| rope::cos_sin(input.position_ids.clone(), theta, head_dim, dtype);
        let global = self
            .layers
            .iter()
            .find(|l| !l.sliding)
            .map(|l| rope_for(l.attn.rope_theta));
        let local = self
            .layers
            .iter()
            .find(|l| l.sliding)
            .map(|l| rope_for(l.attn.rope_theta));

        let mut x = norm(
            &self.embeddings.norm,
            self.embeddings.tok_embeddings.forward(input.input_ids),
        );
        for layer in &self.layers {
            let (mask, rope) = if layer.sliding {
                (&input.sliding_mask, local.as_ref())
            } else {
                (&input.global_mask, global.as_ref())
            };
            let (cos, sin) = rope.expect("a layer of this kind exists, so its rope was computed");
            x = layer.forward(x, mask, cos, sin);
        }
        norm(&self.final_norm, x)
    }
}

impl<B: Backend> Layer<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: &Tensor<B, 4, Bool>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let normed = match &self.attn_norm {
            Some(attn_norm) => norm(attn_norm, x.clone()),
            None => x.clone(),
        };
        let x = x + self.attn.forward(normed, mask, cos, sin);
        let mlp = self.mlp.forward(norm(&self.mlp_norm, x.clone()));
        x + mlp
    }
}

impl<B: Backend> Attention<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: &Tensor<B, 4, Bool>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let (heads, head_dim) = (self.num_heads, self.head_dim);

        let qkv = self
            .wqkv
            .forward(x)
            .reshape([batch, seq, 3, heads, head_dim]);
        let part = |i| {
            qkv.clone()
                .narrow(2, i, 1)
                .reshape([batch, seq, heads, head_dim])
                .swap_dims(1, 2)
        };
        let q = rope::apply(part(0), cos, sin);
        let k = rope::apply(part(1), cos, sin);
        let v = part(2);

        let mask = mask.clone().expand([batch, heads, seq, seq]);
        let out = chunked_attention(&q, k, &v, &mask);
        self.wo
            .forward(out.swap_dims(1, 2).reshape([batch, seq, hidden]))
    }
}

/// Query rows per attention launch.
const QUERY_CHUNK: usize = 512;

/// Attention as explicit matmul → mask → softmax → matmul, over blocks of
/// queries.
///
/// Burn's fused attention autotunes between kernels, and on Vulkan one of
/// them returned wrong logits for some sequence lengths, differently from
/// run to run as tuning results changed; the explicit form uses only the
/// matmul and softmax primitives every backend already relies on. Blocking
/// is exact, since each query's softmax is independent, and keeps every GPU
/// launch short: one launch over thousands of queries can outlast a GPU
/// driver's job timeout, which resets the device and silently corrupts
/// results. It also bounds the score matrix's memory.
fn chunked_attention<B: Backend>(
    q: &Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: &Tensor<B, 4>,
    mask: &Tensor<B, 4, Bool>,
) -> Tensor<B, 4> {
    let [_, _, seq, head_dim] = q.dims();
    #[expect(clippy::cast_precision_loss, reason = "head_dim is small")]
    let scale = 1.0 / (head_dim as f64).sqrt();
    let keys = k.swap_dims(2, 3);
    let block = |start: usize, len: usize| {
        let scores = q.clone().narrow(2, start, len).matmul(keys.clone()) * scale;
        let scores = scores.mask_fill(mask.clone().narrow(2, start, len), f32::NEG_INFINITY);
        softmax(scores, 3).matmul(v.clone())
    };
    if seq <= QUERY_CHUNK {
        return block(0, seq);
    }
    let blocks = (0..seq)
        .step_by(QUERY_CHUNK)
        .map(|start| block(start, QUERY_CHUNK.min(seq - start)))
        .collect();
    Tensor::cat(blocks, 2)
}

impl<B: Backend> Mlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [input, gate] = self
            .wi
            .forward(x)
            .chunk(2, 2)
            .try_into()
            .expect("chunk(2) of an even dimension yields two halves");
        self.wo.forward(gelu(input) * gate)
    }
}

/// Layer norm in the norm's own precision, returning the input's. Norm
/// weights stay f32 in an f16 model: f16 mean and variance reductions lose
/// too much on some GPU backends.
pub(crate) fn norm<B: Backend, const D: usize>(
    layer: &LayerNorm<B>,
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    let dtype = x.dtype();
    layer.forward(x.cast(layer.gamma.val().dtype())).cast(dtype)
}
