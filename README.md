# rverdict

A native Rust decision engine in the "System One" family (Jev, Strands
Decider, Von, Laya, Clef). Give it some text and typed questions (`noul`
yes/no, `choice` one-of-N, `score` on a scale), and it returns calibrated
probabilities from one forward pass. It never generates text.

It runs fully locally on [Burn](https://burn.dev): inference and training
share one model definition, and the backend is chosen at runtime: CUDA or
ROCm when compiled in and present, a GPU through Vulkan or Metal, otherwise
the CPU. No Python at any stage. It speaks the Jev-compatible
`/v1/systemone` wire format.

Status: **Phase 1 complete**: engine, CLI, runtime backend selection and
JevBench parity with Von (137/231, identical on CPU, Vulkan f32 and f16, 0
order flips). See [`docs/plan.md`](docs/plan.md) and
[`research/`](research/).

## Try it

```sh
cargo build --release -p rverdict-cli
./target/release/rverdict devices                     # which backend, and why
./target/release/rverdict ask \
  --state "We were billed twice for March. Refund it or we cancel." \
  --choice "Which team should handle this?=billing,technical,sales" \
  --noul "The customer threatens to cancel." \
  --score "How urgent is this?=not urgent,soon,blocking"
./target/release/rverdict eval jevbench               # downloads the public set on first use
```

The default model is Von 1.2 (`wfzyx/von`, Apache-2.0), pinned to a
revision and downloaded from Hugging Face on first use. After that it starts
offline. `--model-dir` loads a local checkpoint, `--backend` or
`RVERDICT_BACKEND` overrides the backend, `--f16` halves memory on GPUs.

## Builds

| Target | Features |
|---|---|
| Linux, Windows (Vulkan + CPU) | default |
| macOS (Metal + CPU) | `--no-default-features --features metal` |
| NVIDIA | add `--features cuda` (starts fine without a driver) |
| AMD ROCm | add `--features rocm` |

## Crates

| Crate | Purpose |
|---|---|
| `rverdict-core` | Wire format, question rendering, calibration math. No ML dependencies |
| `rverdict-model` | ModernBERT encoder and option-marker head on Burn, weight loading and saving |
| `rverdict-engine` | Embeddable engine: checkpoints, backend selection and self-test, packing, batching |
| `rverdict-eval` | Benchmarks and metrics (JevBench public set) |
| `rverdict-cli` | The `rverdict` command |
| `rverdict-spike` | Phase 0 measurement tools (not published) |

## Development

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

GPU backends that CI cannot run are checked by hand with
[`docs/backend-checklist.md`](docs/backend-checklist.md).

## License

MIT OR Apache-2.0. Von's weights and its ported packing and calibration
logic are Apache-2.0 ([wfzyx/von](https://github.com/wfzyx/von)). JevBench
data is MIT and is downloaded at evaluation time, never redistributed.
