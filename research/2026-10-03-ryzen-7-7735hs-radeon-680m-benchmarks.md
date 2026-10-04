# Ryzen 7 7735HS / Radeon 680M benchmark capture

Date: 2026-10-03.

This capture is for a no-discrete-GPU laptop profile: no NVIDIA card and no
ROCm access, with CPU-style Flex results plus the integrated Radeon 680M
through Vulkan/RADV. It should not be described as “no AMD GPU”; the GPU is
integrated.

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 7735HS with Radeon Graphics |
| Cores / threads | 8 / 16 |
| Cache | 256 KiB L1d per core, 4 MiB L2 per core, 16 MiB L3 |
| Kernel | Linux 7.0.0-31-generic |
| Memory | 57 GiB total |
| GPU | AMD Radeon 680M, Rembrandt |
| Vulkan | 1.4.318, RADV REMBRANDT, Mesa 25.2.8-0ubuntu0.24.04.2 |
| CPU governor | powersave during capture |
| Rust | 1.96.1 |
| Git commit | af542fd |
| Model | `wfzyx/von` snapshot `498ceba33390b32cfefaab6422ec380318ba9b99` |
| Spike features | default: `flex`, `ndarray`, `vulkan`, `x86-v4` |

ROCm was not benchmarked: `rocminfo` reports `sepatel is not member of
"render" group` and cannot open `/dev/kfd` read-write. The usable AMD path on
this machine is Vulkan through RADV.

Reproduce with:

```bash
cargo run -p rverdict-spike --release -- <command>
```

## Inference latency

One option-marker question, median of 5 runs after 2 warm-up runs.

| Tokens | Radeon 680M Vulkan | Flex CPU |
|---|---|---|
| 125 | 194.3 ms | 759.0 ms |
| 509 | 791.7 ms | 4.01 s |
| 1021 | 1.79 s | 8.80 s |
| 2045 | 4.79 s | 24.6 s |

Flex is useful as the CPU-only path here, but it is currently 4–5× slower than
the integrated GPU.

## Micro kernels, 512 tokens

| Kernel | Flex CPU | Radeon 680M Vulkan |
|---|---|---|
| linear Wi `[s,1024]x[1024,5248]` | 21.23 ms, 259.1 GFLOP/s | 9.55 ms, 576.0 GFLOP/s |
| linear Wqkv `[s,1024]x[1024,3072]` | 10.03 ms, 321.0 GFLOP/s | 10.91 ms, 295.4 GFLOP/s |
| attention, bool mask | 43.68 ms, 24.6 GFLOP/s | 6.24 ms, 172.1 GFLOP/s |
| attention, no mask | 31.63 ms, 34.0 GFLOP/s | 5.86 ms, 183.3 GFLOP/s |
| attention matmul+softmax, mask | 51.60 ms, 20.8 GFLOP/s | 5.14 ms, 208.9 GFLOP/s |
| gelu `[s,2624]` | 2.51 ms | 0.78 ms |
| layer norm `[s,1024]` | 0.16 ms | 3.09 ms |
| QKV split + swap_dims | 13.76 ms | 4.06 ms |
| q via weight narrow + swap_dims | 9.26 ms | 2.22 ms |
| RoPE apply via `[64,64]` matmul | 11.55 ms | 2.40 ms |
| RoPE apply `[16,s,64]` | 10.91 ms | 1.02 ms |

Attention and RoPE dominate the gap; Flex's small helper kernels are much
slower than Vulkan. The single linear-algebra case it handles better is one
Wqkv GEMM shape.

## Training throughput

Full fine-tune with AdamW, batch `4 × 509` tokens, 3 measured steps after
warm-up.

| Backend | Step times | Median step | Throughput |
|---|---|---|---|
| Flex CPU | 53.21 s, 51.01 s, 49.73 s | 51.01 s | ≈ 40 tokens/s |
| Radeon 680M Vulkan | 9.01 s, 8.71 s, 8.69 s | 8.71 s | ≈ 234 tokens/s |

Vulkan's first step includes kernel compilation (`42.89 s`); subsequent steps
are about `8.7 s`. Loss decreases across the short run on both backends, but
the exact loss trajectories are backend-specific because kernel order and
numerical details differ.

## Interpretation

This is a useful second data point after the Radeon 890M reference:

| Profile | 125 tokens | 509 tokens | 1021 tokens | 2045 tokens |
|---|---|---|---|---|
| Radeon 890M Vulkan | 91 ms | 339 ms | 827 ms | 2.30 s |
| Radeon 680M Vulkan | 194 ms | 792 ms | 1.79 s | 4.79 s |
| Ryzen 7 7735HS Flex CPU | 759 ms | 4.01 s | 8.80 s | 24.6 s |

For people without a discrete GPU, the integrated 680M is still a practical
Vulkan target. Flex establishes that CPU-only operation works, but at current
kernel performance it is a fallback, not a desktop-class path.

## Artifacts

The raw harness output for this capture is:

```text
flex-infer.log
vulkan-infer.log
flex-micro.log
vulkan-micro.log
flex-train.log
vulkan-train.log
```

These logs were generated with the commands shown above and reduced into the
tables here.
