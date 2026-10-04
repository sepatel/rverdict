#!/usr/bin/env bash
# Runs docs/backend-checklist.md on this machine's GPU and prints a summary.
# Usage: scripts/backend-check.sh [wgpu|cuda|rocm]   (default wgpu)
# Needs network on first run (model, JevBench, Enron emails) and python3.
set -euo pipefail

backend="${1:-wgpu}"
cd "$(dirname "$0")/.."
out="${RVERDICT_CHECK_OUT:-target/backend-check}"
mkdir -p "$out"
features=()
case "$backend" in
  cuda) features=(--features cuda) ;;
  rocm) features=(--features rocm) ;;
esac

echo "== build"
cargo build --release -p rverdict-cli "${features[@]}"
bin=target/release/rverdict

echo "== devices"
$bin devices | tee "$out/devices.txt"
if ! grep -q "^selected" "$out/devices.txt" || grep -q "^selected cpu" "$out/devices.txt"; then
  echo "FAIL: no GPU selected" && exit 1
fi

echo "== self-test against the CPU"
RVERDICT_EXPECT_BACKEND="$backend" cargo test --release -p rverdict-engine "${features[@]}" 2>&1 | tail -3

echo "== JevBench: cpu, $backend f32, $backend f16"
$bin eval jevbench --backend cpu --out "$out/jb-cpu.json" | tail -1
$bin eval jevbench --backend "$backend" --out "$out/jb-f32.json" | tail -1
$bin eval jevbench --backend "$backend" --f16 --out "$out/jb-f16.json" | tail -1
$bin eval jevbench --backend "$backend" --reverse-options --out "$out/jb-reversed.json" | tail -1

echo "== long emails: $backend vs cpu"
$bin data enron-spam --limit 2000 --out "$out/enron.jsonl" >/dev/null
python3 - "$out" <<'EOF'
import json, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/enron.jsonl")]
longest = sorted((r for r in rows if r["id"].endswith("-noul")), key=lambda r: -len(r["state"]))[:24]
open(f"{out}/long.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in longest))
EOF
long_ok=pass
$bin eval compare --backend "$backend" --against cpu --data "$out/long.jsonl" --max-state-tokens 2048 \
  | tee "$out/compare-long.txt" || long_ok=FAIL

echo "== summary"
python3 - "$out" "$long_ok" <<'EOF'
import json, sys
out, long_ok = sys.argv[1], sys.argv[2]
load = lambda n: {o["id"]: o for o in json.load(open(f"{out}/{n}.json"))}
cpu = load("jb-cpu")
for name in ["jb-f32", "jb-f16", "jb-reversed"]:
    run = load(name)
    flips = [i for i in cpu if cpu[i]["predicted"] != run[i]["predicted"]]
    diff = max(abs(dict(cpu[i]["probabilities"]).get(k, 0) - p) for i in cpu for k, p in run[i]["probabilities"])
    lat = sorted(o["latency_ms"] for o in run.values())[len(run) // 2]
    print(f"{name:12} predictions differing from cpu: {len(flips):3}  max prob diff {diff:.4f}  p50 {lat:.0f} ms")
print(f"long emails vs cpu: {long_ok}")
EOF
