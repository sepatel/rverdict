# Phase 1: core, engine and Von parity

> **Correction (Phase 2, same day).** These runs used Burn's autotune,
> which caches one kernel choice per range of shapes. On longer email inputs
> the cached kernel returned wrong logits for some shapes in its range; no
> JevBench item was affected, so the JevBench results below stand. Autotune
> (and fusion, which turned out to be innocent) are now off. Vulkan medians
> are then 148 ms (f32) and 52 ms (f16) instead of 96 ms and 39 ms, with
> probabilities matching the CPU to 1e-4 in f32. See
> `phase-2-calibration.md`.

## What was built

| Crate | Contents |
|---|---|
| `rverdict-core` | Jev-compatible wire types (order-preserving), question rendering that reproduces Von's Python text exactly, calibration (Von's input-conditioned temperature map, zero-shot `noul` prior, band rule), cache paths |
| `rverdict-model` | Tiny-model config, safetensors save and load, f16 loading, PyTorch `.pt` loading |
| `rverdict-engine` | Checkpoints from Hugging Face (revision-pinned, offline once cached) or a directory; runtime backend selection on `burn::Dispatch` with a self-test; Von packing, digit splitting and middle truncation; batching of all questions in a request |
| `rverdict-eval` | JevBench public-set runner and scoring (JevBench's own validity, argmax and tie rules), Brier, ECE, latency |
| `rverdict-cli` | `rverdict ask`, `devices`, `fetch`, `eval jevbench` |

## Parity with Von

JevBench v1 public set, 231 items, one request per item:

| Run | easy | original | hard | all | p50 |
|---|---|---|---|---|---|
| Vulkan, f32 | 48/48 | 46/72 | 43/111 | **137/231** | 96 ms |
| CPU (Flex), f32 | 48/48 | 46/72 | 43/111 | 137/231 | 507 ms |
| Vulkan, f16 | 48/48 | 46/72 | 43/111 | 137/231 | **39 ms** |
| Vulkan, f32, options reversed | 48/48 | 46/72 | 43/111 | 137/231 | 96 ms |

- **Von's own number** on the public hard tier with its chains feature off
  is 0.378 (42/111); we get 43/111. One item is well inside the
  retrain noise Strands measured (σ ≈ 3.2 tasks). The board figures on
  Von's card (easy 0.931, standard 0.688) come from a different board
  revision and are not comparable item for item; no per-item Von results are
  published.
- **Backends agree:** CPU and Vulkan give identical predictions on all 231
  items (largest probability difference 0.005).
- **Order flips: 0.** Reversing every choice's options changes no prediction
  (largest probability difference 1.3e-4).
- **f16:** identical predictions to f32 (largest difference 0.014) at 2.5×
  the speed, once layer norms stay f32 (below).

## Findings

1. **f16 layer norms are unusable on Vulkan.** With every weight in f16,
   JevBench fell to 107/231 with 90 changed predictions, while f16 on CPU was
   exact. Keeping the norms' weights and arithmetic in f32 (cast in and out)
   restored exact parity. Attention in f16 was not the cause.
2. **Asking cubecl for an absent device crashes the process.** A wgpu adapter
   type that does not exist ends in a heap corruption abort, and a missing
   HIP runtime in a non-unwinding panic; neither can be caught. The engine
   therefore enumerates wgpu adapters first and checks CUDA
   (`device_count`, a catchable Rust panic) and HIP (`is_available`) before
   creating any device. CUDA and ROCm builds start and fall back cleanly on a
   machine with neither installed.
3. **Startup is dominated by one-time work.** First start 17.8 s (parse the
   1.6 GB PyTorch pickle, convert it to a content-addressed safetensors copy,
   compile GPU kernels). Every later start is **3.1 s**: 7 ms to resolve the
   pinned checkpoint offline, 63 ms to select and self-test the backend, and
   about 2.9 s to load the weights. That needed two caches: converted
   weights, and cubecl's autotune results and compiled SPIR-V, which default
   to the current directory or nowhere and are now under the rverdict cache.
   A long-lived host such as Post Office pays the 3 s once.
4. **Von's zero-shot `noul` remains weak.** "The customer asks for a
   refund." about "Please refund the duplicate charge" scores P(true) = 0.21.
   This is the model, not the port (the CPU, the GPU and Von's own
   calibration agree). Phases 2–5 (calibration on email, then training) are
   the fix; until then Post Office rules should prefer described `choice`
   options or `noul` with explicit criteria.
5. **ModernBERT-large loads completely** (the base for our own checkpoints).
   burn-store reports a LayerNorm's PyTorch `weight` as unused even after
   applying it as `gamma`; the test accounts for that.

## Not done

- f16 is opt-in (`--f16`). Making it the GPU default needs the self-test to
  cover f16 per device, because f16 support differs across GPUs and drivers.
- Lavapipe, Metal and Windows CI jobs are written but have not run yet; they
  run on the first push to GitHub.
- CUDA and ROCm have compiled, started without drivers and fallen back, but
  have not run on real hardware (`docs/backend-checklist.md`).
