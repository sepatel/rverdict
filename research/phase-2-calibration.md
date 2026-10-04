# Phase 2: calibration, server and remote APIs

Date: 2026-10-03. Reference machine as before (Radeon 890M iGPU, Mesa 26.2.4
RADV); its numbers are one data point, not targets.

## What was built

| Piece | Contents |
|---|---|
| `rverdict calibrate` | Captures raw logits once, fits calibration on labelled decisions with the weights frozen, validates on a held-out half with bootstrap intervals, writes a calibration file |
| `rverdict data enron-spam` | Real emails (Enron-Spam) as labelled decisions: a zero-shot `noul` and a described `choice` each |
| Calibration format | Von-compatible top-level scaling, plus one scaling per question type and the zero-shot `noul` prior |
| `rverdict serve` / `rverdict-server` | `POST /v1/systemone`, `GET /v1/models`, `GET /health`; optional bearer key; a concurrency limit; 422 for malformed requests |
| `rverdict-remote` | Async client for TypeSafe Jev, Cloudflare Workers AI (Clef, Clef-flash) and any System One server, with retries on 429/502/503/529 and `Retry-After` |
| `ask`/`eval --remote`, `--typesafe`, `--cloudflare` | The same commands against hosted APIs |
| `rverdict eval compare` | Runs labelled decisions on two backends and fails on any changed prediction or a logit off by more than a tolerance |
| `--backend reference` | burn-ndarray: slow, simple CPU kernels for checking other backends |
| `--max-state-tokens` | Middle-truncates long states earlier; attention cost grows with the square of the length |

## Calibration result

Von 1.2 on 2,000 Enron emails (4,000 decisions, states capped at 2,048
tokens), fitted on 2,012 and validated on the 1,988 held out:

| Held out | n | Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| noul, Von's calibration | 997 | 0.677 | 0.636 | 0.440 | 0.059 |
| noul, refit | 997 | **0.737** | **0.565** | **0.379** | 0.046 |
| choice, Von's calibration | 991 | 0.635 | 0.625 | 0.437 | 0.053 |
| choice, refit | 991 | 0.635 | 0.625 | 0.437 | 0.053 |
| all, Von's calibration | 1988 | 0.656 | 0.630 | 0.438 | 0.011 |
| all, refit | 1988 | **0.686** | **0.595** | **0.408** | 0.011 |

95% bootstrap intervals for the change, held out:

| | ECE | NLL |
|---|---|---|
| noul | [−0.039, +0.022] | **[−0.090, −0.052]** |
| all | [−0.019, +0.010] | **[−0.045, −0.026]** |

- **What was wrong was a bias, not overconfidence.** Von's zero-shot `noul`
  leaned the wrong way on this question. The refit prior offset (b = −3.25)
  removes it: noul accuracy +6.0 points, and NLL and Brier improve with
  intervals well clear of zero.
- **Choice was already well calibrated**, so the refit left it alone (its gain
  was below the 0.005-nat threshold).
- **The phase's criterion, a measurable ECE drop, was not met, and could not
  be on this data:** overall ECE was 0.011 before the refit, and the noul
  interval spans zero. ECE is the wrong yardstick for a bias. NLL is a proper
  scoring rule that captures both, so it is the criterion we should use (plan
  updated).

### What the fitting work found

1. **One shared scaling trades question types off against each other.** The
   first fit improved nouls and made choices worse (held-out choice ECE
   0.028 → 0.113). Fitting each type separately fixed that.
2. **A prior fitted on one question is just an offset.** With a single noul
   template the state-free bias is constant, so the prior's slope and offset
   cannot be told apart (the fit returned a = 12.5, b = 21.1). With fewer than
   three distinct questions only the offset is fitted now. The general lesson
   for Post Office: calibrate per rule, from that rule's own history.
3. **ECE needs far more data than NLL.** At 600 decisions its interval spanned
   ±0.04. Reports now give intervals for both.

## GPU correctness: two bugs found and fixed

The calibration capture surfaced problems JevBench had not.

1. **Long inputs reset the GPU and returned garbage.** One attention launch
   over thousands of queries outlasted the amdgpu job timeout. The kernel log
   showed ring resets, and the answer came back as a uniform 0.50/0.50 without
   any error. *Fix:* attention runs in blocks of 512 queries (exact, since each
   query's softmax is independent), which also bounds its memory. No GPU
   timeouts in any run since, including 4,000 decisions over 35 minutes.
2. **Autotune replayed a wrong kernel.** Burn's autotune caches one kernel
   choice per range of shapes, tuned on the first input it sees. On Vulkan
   the cached choice returned wrong logits for some other shapes in the
   range, by up to 3.9, enough to flip predictions. It depended on input
   order, so single-email tests passed while batch runs failed, and it
   affected the null rows of 5 of 2,000 emails and several long ones.
   Disabling autotune removed every difference.
   - Fusion was suspected first and is innocent: with autotune off, fused and
     unfused builds both match the CPU. Every "fusion fixes it" result came
     from a fresh autotune cache.
   - Burn's fused attention kernel was replaced by explicit
     matmul → mask → softmax → matmul (uses only primitives every backend
     relies on, and at least as fast here).
   - The compiled-kernel cache is now keyed by the running binary, because
     cubecl keys it by its own version only.

### Verification and cost

| Build (Vulkan) | 16 hard emails vs CPU | JevBench vs CPU | p50 f32 | p50 f16 |
|---|---|---|---|---|
| autotune + fusion (Phase 1) | max Δ 3.91, 1 flip | identical | 96 ms | 39 ms |
| autotune, no fusion | max Δ 3.91, 1 flip | identical | 115 ms | 50 ms |
| **no autotune, no fusion (now)** | **max Δ 0.0003, 0 flips** | **identical, Δp ≤ 1e-4** | 148 ms | 52 ms |
| no autotune, fusion | max Δ 0.0003, 0 flips | identical | 149 ms | 49 ms |

Correctness costs f32 about 50% in latency; f16 is barely affected, so on GPUs
f16 is now about 3× faster than f32 with the same predictions.

## Server and remote client

- Answers through `rverdict serve` match local answers exactly. A Jev
  client's `"model": "jev-latest"` is accepted, and malformed or non-JSON
  bodies get a 422 with the reason.
- Five round-trip tests run a real server on a loopback port with a stub
  decider: success, 422, 401 without the key, a retried 429, and Cloudflare's
  response envelope.
- TypeSafe and Cloudflare have not been called for real (no API keys here).
  Jev's model id on Workers AI is not documented, so the client takes any
  Workers AI model path and defaults to Clef.

## Not done

- A real call to TypeSafe or Cloudflare.
- Autotune back on: needs a cubecl fix (an upstream report with this
  reproduction) or exact-shape tuning keys. Training throughput without
  autotune is measured in Phase 3.
- The startup self-test uses a tiny model and cannot catch shape-specific
  kernel bugs. `rverdict eval compare` on long inputs is now part of
  `docs/backend-checklist.md`.
- A GPU reset can still corrupt an answer silently if one happens for other
  reasons; detecting device loss is a follow-up.
