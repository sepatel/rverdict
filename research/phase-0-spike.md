# Phase 0: technical spike

Date: 2026-10-03. Reference machine (one data point, not a target): AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24
threads, AVX-512), Radeon 890M iGPU (RADV, Vulkan 1.4), 64 GB RAM of which
the iGPU can map about 30 GB (GTT). Toolchain: Rust 1.96.1, Burn 0.21.

Reproduce with `cargo run -p rverdict-spike --release -- <command>`.

## Questions this phase had to answer

1. Can ModernBERT be ported to Burn so it matches an independent reference?
2. Can Von's weights (`option_marker.pt`) load directly, without Python?
3. Which Burn backend should serve inference and training on this machine?

## Result: Burn is confirmed

### 1. Encoder parity (`parity`)

Burn's encoder and candle-transformers' independent ModernBERT
implementation, on Von's `model.safetensors` and the same tokens:

| Backend | Tokens | max \|Δ\| | mean \|Δ\| | mean \|x\| |
|---|---|---|---|---|
| flex   | 242 | 3.9e-5 | 2.4e-6 | 0.78 |
| flex   | 962 | 5.2e-4 | 3.1e-6 | 0.80 |
| ndarray| 962 | 2.9e-4 | 3.6e-6 | 0.80 |
| vulkan | 962 | 3.9e-3 | 1.6e-5 | 0.80 |

At 962 tokens the sliding-window layers (window ±64) are fully exercised.
The CPU backends agree to f32 accumulation noise. Vulkan's larger max
difference is GPU reduction order; it does not change any decision (below).

### 2. Von end to end (`smoke`)

`option_marker.pt` loads directly through `burn-store`'s PyTorch reader with
three key renames (`Wqkv`/`Wi`/`Wo` → lower case); no conversion step.

| Check | Result |
|---|---|
| Billing ticket, department choice | billing 0.938, confidence 0.907 (Von's card shows 0.94 for a similar ticket) |
| Reversed option order | logit drift **0.0** on flex, < 1e-6 on vulkan |
| Newsletter vs receipt vs personal | newsletter 0.64 / receipt 0.85 on the right emails |
| Vulkan vs flex decisions | identical to 3 decimals |

The independent-options masks and per-option position ids are therefore
exact: an option's logit does not depend on where it sits.

**Finding: Von's `noul` answers are weak out of the box.** Raw logits point
the right way (e.g. "confirms a payment was charged": 4.53 vs 0.28), but the
JevBench-fitted calibration map sets T = 2.6–5.7 on short emails, which
flattens them to 0.51–0.72 before the band rule. Von's own card warns the map
does not carry across distributions. This is a calibration and training
problem, not a port bug, and is exactly what Phases 2–5 address. Final proof
of port correctness is reproducing Von's JevBench score in Phase 1.

### 3. Inference latency (`infer`), one question, median of 5

| Tokens | vulkan (890M) | flex (CPU) | ndarray | burn-cpu (MLIR) |
|---|---|---|---|---|
| 125  | **91 ms**  | 607 ms  | 1.2 s* | 3.5 s |
| 509  | **339 ms** | 2.47 s  | 4.5 s* | 13.9 s |
| 1021 | **827 ms** | 5.60 s  | 10.4 s* | – |
| 2045 | **2.30 s** | 14.9 s  | 26.4 s* | – |

\* ndarray measured before backend default features were re-enabled; shown
for scale only.

### 4. Training throughput (`train`), full fine-tune, AdamW, 4 × 509 tokens

| Backend | Step | Throughput |
|---|---|---|
| vulkan | 3.5 s (first step 24 s: kernel compile) | **≈ 570 tokens/s** |
| flex   | 36 s | ≈ 57 tokens/s |

Loss falls across steps, so gradients flow through the whole model,
including the custom masks and RoPE. At 570 tokens/s:

- personal data, e.g. 5,000 emails × 512 tokens: ≈ 1.2 h per epoch locally;
- a Von-sized public corpus (~370k rows × ~300 tokens): ≈ 54 h per epoch —
  feasible locally over a weekend, otherwise the same binary on a rented CUDA
  GPU, as the plan anticipated.

## Why the CPU numbers are slow (`micro`, 512 tokens)

| Kernel | flex | vulkan |
|---|---|---|
| GEMM `[512,1024]×[1024,5248]` | 8.5 ms, 650 GFLOP/s | 2.8 ms, 1.96 TFLOP/s |
| fused attention, bool mask | 29.5 ms, 36 GFLOP/s | 5.0 ms |
| RoPE apply `[16,512,64]` | 6.3–7.4 ms | 0.7 ms |
| QKV split + transpose (beyond GEMM) | ~7 ms | ~0.6 ms |

Flex's GEMM is good; its fused attention and broadcast/strided element-wise
kernels are an order of magnitude slower. Rewriting RoPE as a matmul and
splitting QKV on the weight side did not help. CPU is a working fallback, not
a fast path; improving it is a follow-up (upstream Flex work, or an attention
kernel of our own), not a blocker.

## Decisions

- **Framework: Burn 0.21.** One model definition serves inference and
  training; parity is proven; the AMD iGPU is usable through Vulkan.
- **Backends:** wgpu (Vulkan here; Metal and DX12 elsewhere) as the default
  GPU path, CUDA and ROCm optional, Flex for CPU-only machines, chosen at
  runtime (plan section 3.1a). burn-ndarray and burn-cpu are not used.
  burn-candle is deprecated upstream.
- **Weights:** load Hugging Face safetensors and PyTorch `.pt` directly
  through `burn-store`; no conversion tooling needed.
- **candle-transformers** stays only as the test reference for parity.
- Burn must be built with backend default features (rayon/SIMD for Flex,
  autotune/fusion for wgpu); without them Flex is 2× and Vulkan 5× slower.

## Not done in Phase 0

- Loading `answerdotai/ModernBERT-large` (the `ForMaskedLM` layout with the
  `model.` prefix). The remap is in place but untested on that file; needed
  before Phase 6.
- f16/bf16 inference on Vulkan (halves memory, likely faster).
- Batching several questions per forward pass.
