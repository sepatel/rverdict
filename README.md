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

Status: **Phase 2 complete**: engine, CLI, runtime backend selection,
JevBench parity with Von (137/231, identical on CPU and GPU, 0 order flips),
calibration refitting, an HTTP server, and clients for hosted APIs. See [`docs/plan.md`](docs/plan.md) and
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
./target/release/rverdict serve                       # POST http://127.0.0.1:8090/v1/systemone
./target/release/rverdict install --dir ./models      # verified install into your own directory
```

Existing Jev, Clef or Von clients can point at `rverdict serve` by changing
their base URL. The other way round, `ask` and `eval` can ask a hosted API
instead of the local model: `--remote <url>` (any `/v1/systemone` server),
`--typesafe` (TypeSafe Jev, `TYPESAFE_API_KEY`) or
`--cloudflare @cf/cloudflare/clef` (Workers AI, `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN`).

Apps that embed rverdict and keep models in their own data directory use
`rverdict_engine::install` (download with progress and cancellation, size
and hash checks against Hugging Face, PyTorch weights converted to
safetensors) and `rverdict_core::set_cache_root`.

To refit calibration on your own labelled decisions, see
[`docs/calibration.md`](docs/calibration.md).

The default model is Von 1.2 (`wfzyx/von`, Apache-2.0), pinned to a
revision and downloaded from Hugging Face on first use. After that it starts
offline. `--model-dir` loads a local checkpoint, `--backend` or
`RVERDICT_BACKEND` overrides the backend, `--f16` halves memory on GPUs,
`--max-state-tokens` trades context for speed on long inputs.

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
| `rverdict-eval` | Benchmarks, metrics, labelled-data format, calibration fitting |
| `rverdict-server` | `/v1/systemone` HTTP server over any decider |
| `rverdict-remote` | Async client for TypeSafe Jev, Cloudflare Workers AI and any System One server |
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
