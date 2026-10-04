# Backend checklist

CI runs the CPU backend everywhere and the Vulkan code path through Mesa's
lavapipe software driver. Real GPUs, CUDA and ROCm cannot run in CI, so check
them by hand on matching hardware (a rented GPU is fine) before a release, and
record the result in the table below.

## Steps

`scripts/backend-check.sh [wgpu|cuda|rocm]` runs steps 1–6 on Linux and
prints a summary; the steps below say what it checks.

1. Build for the target:

   | Target | Command |
   |---|---|
   | Vulkan (Linux, Windows) | `cargo build --release -p rverdict-cli` |
   | Metal (macOS) | `cargo build --release -p rverdict-cli --no-default-features --features metal` |
   | CUDA | `cargo build --release -p rverdict-cli --features cuda` |
   | ROCm | `cargo build --release -p rverdict-cli --features rocm` |

2. `rverdict devices` must select the GPU, not fall back. With a CUDA build,
   also check `rverdict devices` on a machine **without** an NVIDIA driver: it
   must start and select another backend.
3. `cargo test -p rverdict-engine` with `RVERDICT_EXPECT_BACKEND` set to the
   backend (`wgpu`, `cuda`, `rocm`) must pass: the tiny-model self-test
   matches the CPU.
4. `rverdict eval jevbench --out run.json` must give the same predictions as
   the reference CPU run (`--backend cpu`); compare the two `run.json` files.
5. JevBench's states are short, and GPU kernel bugs have shown up only on
   longer inputs. Also run `rverdict eval compare --against cpu --data
   long.jsonl` on labelled decisions with states of 1,000–8,000 tokens (for
   example the longest emails from `rverdict data enron-spam`); it fails on
   any changed prediction or a logit off by more than 0.05.
6. `rverdict eval jevbench --reverse-options` must give the same predictions
   again (order invariance).
7. Add a row below with the hardware, driver, accuracy and p50 latency.

## Results

| Date | Hardware | Backend | Driver | JevBench (all) | p50 | Notes |
|---|---|---|---|---|---|---|
| 2026-10-03 | Radeon 890M iGPU | Vulkan | Mesa 26.2.4 (RADV) | 137/231 | 148 ms (f32), 52 ms (f16) | Reference machine; 0 order flips; identical to CPU on JevBench and 16 long emails; autotune and fusion off |
| 2026-10-04 | Radeon 680M iGPU | Vulkan | Mesa 25.2.8 (RADV) | 137/231 | 227 ms (f32), 136 ms (f16) | 0 order flips; identical to CPU on JevBench and 24 long emails (f32) |
