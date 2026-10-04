# Calibrating on your own decisions

A checkpoint's probabilities are only as trustworthy as the data its
calibration was fitted on. Von's map was fitted on JevBench; on other text it
can be over- or under-confident, and its zero-shot `noul` answers lean the
wrong way on some questions. `rverdict calibrate` refits calibration on
decisions you have labelled, with the model weights frozen.

## 1. Label some decisions

One JSON object per line:

```json
{"id": "m1", "state": "Subject: Invoice 4411 …", "question": {"type": "noul", "instructions": "Is this a bill?"}, "expected": "yes"}
{"id": "m2", "state": "…", "question": {"type": "choice", "instructions": "Which folder?", "criteria": {"bills": "Invoices and receipts", "work": "Work email"}}, "expected": "bills"}
```

`expected` is `"yes"`/`"no"` (or `true`/`false`) for a `noul`, an option key
for a `choice`, and a level index for a `score`. `subset` is optional and
splits the report. Public data works too:
`rverdict data enron-spam --limit 600 --out enron.jsonl`.

## 2. Fit and validate

```sh
rverdict calibrate --data labels.jsonl --captures labels.captures.jsonl --out calibration.json
```

- The model runs once; `--captures` keeps its outputs, so refits are instant.
- Half the decisions (by a stable hash of `id`, `--holdout`) are held out.
  The report compares the checkpoint's calibration with the refit on them,
  per group, with 95% bootstrap intervals for the change in ECE and NLL.
- The file written by `--out` is fitted on all the decisions.

## 3. Use it

```sh
rverdict ask --calibration calibration.json …
rverdict serve --calibration calibration.json
```

Embedders call `Engine::set_calibration`.

## What gets fitted

- **One scaling per question type.** A model can be well calibrated on
  choices and overconfident on yes/no questions; a shared temperature would
  trade one against the other. Each type with at least 30 decisions gets its
  own, and keeps the checkpoint's unless the refit lowers its NLL by at least
  0.005 nats on the fitting data.
- **`--form scalar`** (default) fits one temperature per type. **`--form map`**
  fits Von's input-conditioned map (4 parameters per type); it needs more
  data and overfits small sets.
- **The zero-shot `noul` prior** (for nouls asked without criteria) is fitted
  with the `noul` scaling. With fewer than three distinct questions, only its
  offset is identifiable, so only the offset is fitted.

Temperature never changes which option wins. The `noul` prior can: it is a
per-question bias correction, which is how calibration also raises accuracy.

## Caveats

- **Calibrate for the questions you will ask.** A fit on one question
  template learns that template's bias. Post Office should therefore
  calibrate per rule, from that rule's own labelled history.
- **ECE is noisy below about a thousand decisions** (its interval easily
  spans zero); NLL and Brier are proper scoring rules and settle faster.
  Judge a refit by NLL first.
- Capture at the `--max-state-tokens` you will run with: truncation changes
  the model's outputs.
