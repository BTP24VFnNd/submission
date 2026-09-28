#!/bin/bash
# Usage: ./metrics-cross-transfer.sh <session_id> [model_tag]
#   ./metrics-cross-transfer.sh 20260909_054308                  every held-out model on that session
#   ./metrics-cross-transfer.sh 20260909_054308 openai-gpt-4o    one model only (tag as in the outputs dir name)
#
# Held-out (PrimeVul test) results of a session's harnesses, one row per iteration,
# grouped by evaluator model. Made for cross-model transfer runs, which are often
# partial while they run, so besides the official numbers (which divide by all 435
# pairs) it also reports the pairwise accuracy over the pairs that actually have a
# verdict for both halves.
#
# Columns:
#   iter      optimization iteration whose harness was evaluated
#   val       validation pairwise of that iteration (from the session's summary.json)
#   verdicts  samples with a verdict / 870
#   pairs     pairs where both halves have a verdict / 435
#   correct   correctly ordered pairs
#   pw_all    official pairwise accuracy: correct / 435   (what metrics.sh reports)
#   pw_done   pairwise over completed pairs: correct / pairs
#   parsefail rows without a parseable verdict (count as wrong in pw_all)
#   f1        binary F1 over samples with a verdict
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -z "$1" ] && { sed -n '2,4p' "$0" >&2; exit 1; }
exec python3 - "$ROOT" "$1" "${2:-}" <<'PY'
import json, re, sys
from collections import defaultdict
from pathlib import Path

root, session, only = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(root / "evaluation"))
from metrics import load_predictions, load_ground_truth, compute_all_metrics  # noqa: E402

gt = load_ground_truth(root / "data/primevul/test_with_vul_lines_uuid.jsonl")
summary = root / "meta-vul-code/meta-vul/outputs" / session / "summary.json"
val = {}
if summary.exists():
    s = json.loads(summary.read_text())
    val = {h["iteration"]: h["score"] for h in s.get("history", [])}
    best = s.get("best_iteration")
else:
    best = None

runs = defaultdict(dict)  # model_tag -> iter -> predictions path
pat = re.compile(rf"^held_out_primevul_{session}_iter(\d+)_(.+)$")
for agent_dir in sorted(root.glob("meta-vul-code/*-agent/outputs")):
    for d in agent_dir.iterdir():
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        it, tag = int(m.group(1)), m.group(2)
        if only and tag != only:
            continue
        preds = list(d.glob("*/predictions.jsonl"))
        if preds:
            runs[tag][it] = preds[0]

if not runs:
    print(f"no held-out runs for session {session}" + (f" and model {only}" if only else ""))
    sys.exit(0)

def complete_pairs(rows):
    by_index = {r.get("index"): r for r in rows}
    done = correct = 0
    for i in range(0, (max(by_index) if by_index else -1) + 1, 2):
        a, b = by_index.get(i), by_index.get(i + 1)
        if not a or not b or a.get("vulnerable") is None or b.get("vulnerable") is None:
            continue
        vuln = a if a.get("label") == 1 else b if b.get("label") == 1 else None
        patch = a if a.get("label") == 0 else b if b.get("label") == 0 else None
        if not vuln or not patch:
            continue
        done += 1
        if vuln.get("vulnerable") is True and patch.get("vulnerable") is False:
            correct += 1
    return done, correct

hdr = "%-5s %6s %9s %9s %8s %7s %8s %9s %6s"
for tag in sorted(runs):
    print(f"\n== Held-out (PrimeVul test), session {session}, harnesses evaluated by {tag}")
    print(hdr % ("iter", "val", "verdicts", "pairs", "correct", "pw_all", "pw_done", "parsefail", "f1"))
    for it in sorted(runs[tag]):
        rows = load_predictions(runs[tag][it])
        with_verdict = [r for r in rows if r.get("vulnerable") is not None]
        if not with_verdict:
            print(hdr % (it, f"{val.get(it, float('nan')):.2f}" if it in val else "-", "0/870", "0/435", 0, "-", "-", len(rows), "-"))
            continue
        m = compute_all_metrics(rows, gt)
        done, correct = complete_pairs(rows)
        mark = "*" if it == best else ""
        print(hdr % (
            f"{it}{mark}",
            f"{val[it]:.2f}" if it in val else "-",
            f"{len(with_verdict)}/870",
            f"{done}/435",
            correct,
            f"{m['pairwise_accuracy']:.2f}",
            f"{(correct / done * 100) if done else 0:.2f}",
            m.get("parse_failures", len(rows) - len(with_verdict)),
            f"{m['f1']:.3f}",
        ))
if best is not None:
    print(f"\n* = validation-selected iteration ({best}). pw_all divides by all 435 pairs; pw_done only by pairs with both verdicts.")
PY
