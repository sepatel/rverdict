# Hardware benchmarks

Performance numbers for `rverdict-spike`, run with the release binary and the
default Cargo features (`flex`, `ndarray`, `vulkan`, `x86-v4`). One question,
median of 5 inference runs after 2 warm-up runs.

## 2026-10-03: Ryzen 7 7735HS + Radeon 680M

Linux 7.0.0-31-generic, Mesa 25.2.8-0ubuntu0.24.04.2, Vulkan/RADV, Rust
1.96.1, no NVIDIA GPU and no usable ROCm device.

| Machine / backend | 125 tokens | 509 tokens | 1021 tokens | 2045 tokens | Train, 4×509 |
|---|---|---|---|---|---|
| Radeon 680M Vulkan | 194.3 ms | 791.7 ms | 1.79 s | 4.79 s | ≈ 234 tokens/s |
| Flex CPU | 759.0 ms | 4.01 s | 8.80 s | 24.6 s | ≈ 40 tokens/s |

Reference point from Phase 0:

| Machine / backend | 125 tokens | 509 tokens | 1021 tokens | 2045 tokens | Train, 4×509 |
|---|---|---|---|---|---|
| Radeon 890M Vulkan | 91 ms | 339 ms | 827 ms | 2.30 s | ≈ 570 tokens/s |
| Flex CPU | 607 ms | 2.47 s | 5.60 s | 14.9 s | ≈ 57 tokens/s |

Details:
[`research/2026-10-03-ryzen-7-7735hs-radeon-680m-benchmarks.md`](../research/2026-10-03-ryzen-7-7735hs-radeon-680m-benchmarks.md)
