#!/bin/bash
# Usage: ./run/held-out-ranked.sh <session_id> <model> [--max N] [--from <backup_dir>] [extra held-out flags]
#   ./run/held-out-ranked.sh 20260909_054308 openrouter/openai/gpt-4o
#   ./run/held-out-ranked.sh 20260909_054308 openrouter/openai/gpt-4o --max 5
#   ./run/held-out-ranked.sh 20260909_054308 openrouter/openai/gpt-4o --dry-run
#
# Cross-model transfer of every prompt in an optimization session, in order
# from most to least promising: the seed (iteration 0) runs first as the
# baseline, then the Pareto-frontier iterations (summary.json
# pareto_frontier_iterations: validation pairwise vs loc F1), then the rest,
# each group sorted by validation score (history[].score, descending; ties
# broken by later iteration).
# Held-out results never influence the order, so the ranking stays a fair
# validation-based selection.
#
# If the session is not in meta-vul/outputs/ it is staged from a backup
# (--from, default ../results/<session_id>/meta-vul/outputs/<session_id>).
# Each iteration goes through ./run/held-out.sh, so every launch is recorded
# in run/held-out/ and finished samples are skipped on rerun (skip_existing).
# Score afterwards with ../evaluation/metrics.sh <session_id>.

set -u
RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$RUNS_DIR/.." && pwd)"

if [ $# -lt 2 ]; then
  echo "usage: $0 <session_id> <model> [--max N] [--from <backup_session_dir>] [extra held-out flags]" >&2
  exit 1
fi
SESSION_ID="$1"; MODEL="$2"; shift 2

MAX=""; FROM=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --max) MAX="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

SESSION_DIR="$REPO_DIR/meta-vul/outputs/$SESSION_ID"
if [ ! -f "$SESSION_DIR/summary.json" ]; then
  FROM="${FROM:-$REPO_DIR/../../results/$SESSION_ID/meta-vul/outputs/$SESSION_ID}"
  if [ ! -f "$FROM/summary.json" ]; then
    echo "error: session $SESSION_ID not in meta-vul/outputs/ and no backup at $FROM" >&2
    exit 1
  fi
  echo "[ranked] staging session prompts from $FROM"
  mkdir -p "$SESSION_DIR"
  rsync -a "$FROM/" "$SESSION_DIR/"
fi

ORDER="$(python3 - "$SESSION_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
pareto = set(s.get("pareto_frontier_iterations") or [])
hist = [h for h in s["history"] if h["iteration"] != 0]
# Seed first; then the Pareto-frontier iterations (validation pairwise vs joint1)
# by validation score; then the rest by validation score.
hist.sort(key=lambda h: (h["iteration"] not in pareto, -h["score"], -h["iteration"]))
print(" ".join(["0"] + [str(h["iteration"]) for h in hist]))
PY
)" || exit 1

# Output dir tag: model name without provider prefix, "/" and ":" -> "-" (as the runner names it).
MODEL_TAG="$(printf '%s' "${MODEL#*/}" | tr '/:' '--')"
TOTAL=870   # PrimeVul held-out samples
echo "[ranked] session $SESSION_ID -> $MODEL"
echo "[ranked] order (seed, then Pareto set, then the rest, each by validation score): $ORDER"

n=0
for it in $ORDER; do
  if [ -n "$MAX" ] && [ "$n" -ge "$MAX" ]; then break; fi
  n=$((n+1))
  # Skip iterations whose held-out run already has a verdict for every sample.
  PRED="$REPO_DIR/codex-self-improving-agent/outputs/held_out_primevul_${SESSION_ID}_iter${it}_${MODEL_TAG}/held-out/predictions.jsonl"
  DONE=0
  if [ -f "$PRED" ]; then
    DONE=$(python3 - "$PRED" <<'PY'
import json, sys
txt = open(sys.argv[1]).read(); dec = json.JSONDecoder(); i = 0; ids = set()
while i < len(txt):
    while i < len(txt) and txt[i].isspace(): i += 1
    if i >= len(txt): break
    r, i = dec.raw_decode(txt, i)
    if r.get("vulnerable") is not None: ids.add(r["id"])
print(len(ids))
PY
)
    if [ "$DONE" -ge "$TOTAL" ]; then echo "[ranked] skip iteration $it: $DONE/$TOTAL done"; continue; fi
  fi
  echo
  echo "============================================================"
  echo "[ranked] $n/${MAX:-all}: iteration $it ($DONE/$TOTAL done)"
  echo "============================================================"
  "$RUNS_DIR/held-out.sh" "$SESSION_ID" "$MODEL" --iteration "$it" ${EXTRA[@]+"${EXTRA[@]}"} \
    || { echo "[ranked] iteration $it failed; continuing" >&2; }
done
echo
echo "[ranked] done. Score with: ../evaluation/metrics.sh $SESSION_ID"
