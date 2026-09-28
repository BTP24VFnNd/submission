#!/usr/bin/env python3
"""Paper metrics per optimization session: validation, PrimeVul held-out, and
cross-transfer to other models.

Output, one block per session:

    session_id: <id>

      dataset:         TitanVul/CWEval validation (404 samples)
      optimizer model: ...
      optimizer agent: ...
      results:
        <one row per optimizer iteration>

      dataset:         PrimeVul test (860 samples, 5 contaminated pairs removed)
      held-out model:  ...
      held-out agent:  ...
      results:
        <one row per iteration>
      (repeated for every held-out model, the optimizing model first)

    pareto from val (pairwise vs joint1): {iterations}
    ______________________________

Default columns: iter, n, pairwise, accuracy, precision, recall, F1, CWE
accuracy, localization F1 and hit rate. --full adds correct pairs, coverage,
localization precision/recall, joint CWE+localization rate (any listed CWE),
refined joint rate (first CWE only, "joint1") and parse failures. --csv
always writes every column.

Runs are collected from this repo's meta-vul-code, its .complete-backups and
.incomplete-backups, the sibling ../results folder, and any extra checkouts (METRICS_EXTRA_ROOTS or
--extra-root). When a run exists in several places, the copy with the most
valid predictions is used. Predictions are deduplicated per sample first.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_all_metrics, load_ground_truth, load_predictions  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VAL_GT = REPO / "data" / "titanvul_cweval_optimization_val_uuid.jsonl"
HELD_GT = REPO / "data" / "primevul" / "test_with_vul_lines_uuid.jsonl"
HELD_RE = re.compile(r"^held_out_primevul_(\d{8}_\d{6})_iter(\d+)_(.+)$")

# PrimeVul test pairs whose functions also appear in the TitanVul validation
# sample (same function, same fix), given as the row index of each pair's first
# row in HELD_GT. Held-out scores leave them out (430 pairs); pass
# --include-contaminated for the original 435. Pairs: TensorFlow CVE-2021-41203,
# Vim CVE-2022-2845, xserver CVE-2020-14361, shapelib CVE-2022-0699, newsbeuter
# CVE-2017-12904 (validation rows 2, 142, 154, 144, 198).
HELD_EXCLUDE_PAIRS = frozenset({124, 240, 290, 656, 820})
PROVIDER_PREFIXES = ("tinker", "openrouter", "openai")
AGENT_NAMES = {
    "claude-cli-agent": "claude-cli",
    "codex-self-improving-agent": "codex",
    "old-claude-cli-agent": "claude-cli",
}

# Held-out runs stored under a name the scan does not match. Each entry:
# (session, iteration, model tag, path relative to a meta-vul-code root).
EXTRA_HELD_RUNS = [
    # Opus 5 SCoT seed baseline, run standalone with the session's iteration-0
    # (SCoT seed) harness before held-out folders were named per session.
    ("20260716_181608", 0, "opus-5", "claude-cli-agent/outputs/20260726_103036/scot-seed/predictions.jsonl"),
    # March 2026 SCoT run with the CLI's "opus" alias (Opus 4.6 at the time),
    # from the pre-meta-vul claude-cli agent at the repo root (git-ignored).
    # Replaces the session's own partial iteration-0 run (692/870 valid).
    ("20260716_181608", 0, "opus-4-6", "../old-claude-cli-agent/outputs/20260319_225235/scot/predictions.jsonl"),
]

# Held-out runs that survive only as a scored summary JSON (a "metrics" dict from
# evaluation/metrics.py), with no raw predictions. Shown in the model's block;
# metrics the summary lacks print as "-". Same tuple shape as EXTRA_HELD_RUNS.
SCORED_ONLY_RUNS = [
    # (empty) Nemotron iteration 12 of 20260909_054308 used to be here; a 2026-09-24
    # rerun supplies its raw predictions.
]

# Optimizer model/agent for sessions without a run/discovery launch record.
# Sources: CHANGELOG and the backup READMEs for each session.
KNOWN_OPTIMIZERS = {
    "20260716_181608": ("claude-opus-4-6", "claude-cli"),
    "20260909_054308": ("thinkingmachines/Inkling-Small", "codex"),
    "20260919_171544": ("zai-org/GLM-5.3:peft:262144", "codex"),
    "20260919_175634": ("moonshotai/Kimi-K2.6:peft:131072", "codex"),
    "20260908_031934": ("not recorded", "claude-cli"),
    "20260922_170038": ("z-ai/glm-5.3 (OpenRouter)", "claude-cli"),
}

# (metric key, header, format). The first block is the default view.
SIMPLE = [
    ("n", "n", "{}"),
    ("pairwise_accuracy", "pairwise", "{:.1f}"),
    ("accuracy", "acc", "{:.3f}"),
    ("precision", "prec", "{:.3f}"),
    ("recall", "recall", "{:.3f}"),
    ("f1", "f1", "{:.3f}"),
    ("cwe_accuracy", "cwe", "{:.1f}"),
    ("loc_f1", "locF1", "{:.1f}"),
    ("loc_hit_rate", "locHit", "{:.1f}"),
]
EXTRA = [
    ("coverage", "cov%", "{:.0f}"),
    ("correct_pairs", "pairs", "{}"),
    ("loc_precision", "locP", "{:.1f}"),
    ("loc_recall", "locR", "{:.1f}"),
    ("joint_both_rate", "joint", "{:.1f}"),
    ("joint_top1_rate", "joint1", "{:.1f}"),
    ("parse_failures", "pfail", "{}"),
]
CSV_KEYS = [k for k, _, _ in SIMPLE + EXTRA] + [
    "total_pairs", "cwe_correct", "cwe_total", "tp", "fp", "fn", "tn"]


# ---------------------------------------------------------------- discovery

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def same_model(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    return bool(na and nb) and (na.endswith(nb) or nb.endswith(na))


def strip_provider(model: str) -> str:
    head, _, rest = model.partition("/")
    return rest if head in PROVIDER_PREFIXES and rest else model


def roots(extra: List[str]) -> List[Path]:
    """meta-vul-code roots: this repo, its backups, then extra checkouts and their backups."""
    out: List[Path] = []
    for base in [REPO] + [Path(e).expanduser().resolve().parent for e in extra]:
        if (base / "meta-vul-code").is_dir():
            out.append(base / "meta-vul-code")
        for kind in (".complete-backups", ".incomplete-backups", "../results"):
            out += [b for b in sorted((base / kind).glob("*")) if b.is_dir()]
    return list(dict.fromkeys(out))


def discover_sessions(rs: List[Path]) -> List[str]:
    ids = set()
    for r in rs:
        ids.update(d.name for d in (r / "meta-vul" / "outputs").glob("[0-9]" * 8 + "_" + "[0-9]" * 6))
        ids.update(m.group(1) for d in r.glob("*-agent/outputs/held_out_primevul_*")
                   if (m := HELD_RE.match(d.name)))
    return sorted(ids)


def evaluator(session: str, rs: List[Path]) -> Tuple[Optional[str], Optional[str]]:
    """(model, agent) that scored the session's validation runs."""
    for r in rs:
        for logs in sorted(r.glob(f"*-agent/outputs/opt_{session}_iter*/*/logs.jsonl")):
            with open(logs, errors="replace") as fh:
                m = re.search(r'"model":\s*"([^"]+)"', fh.read(200_000))
            if m:
                return m.group(1), AGENT_NAMES.get(logs.parts[-5], logs.parts[-5])
    return None, None


def optimizer(session: str, rs: List[Path], eval_model: Optional[str],
              eval_agent: Optional[str]) -> Tuple[str, str]:
    """(model, agent) of the optimizer: launch record, then known table, then evaluator."""
    for r in rs:
        rec = r / "run" / "discovery" / f"discover_{session}.sh"
        if rec.exists():
            text = rec.read_text()
            model = re.search(r"--optimizer-model\s+(\S+)", text) or re.search(r"--model\s+(\S+)", text)
            agent = re.search(r"--optimizer-agent\s+(\S+)", text) or re.search(r"--agent\s+(\S+)", text)
            if model:
                return strip_provider(model.group(1)), agent.group(1) if agent else (eval_agent or "unknown")
    if session in KNOWN_OPTIMIZERS:
        return KNOWN_OPTIMIZERS[session]
    return (eval_model or "unknown") + " (assumed, same as evaluator)", eval_agent or "unknown"


def session_summary(session: str, rs: List[Path]) -> Dict[str, Any]:
    for r in rs:
        f = r / "meta-vul" / "outputs" / session / "summary.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except ValueError:
                pass
    return {}


# Second objective of each validation Pareto frontier the report prints (the first
# is always pairwise_accuracy): (metric key, label). The paper uses joint1 only.
PARETO_SECOND = [("joint_top1_rate", "joint1")]


def pareto_from_val(session: str, rs: List[Path], summ: Dict[str, Any],
                    val_rows: List[Dict[str, Any]], second: str = "loc_f1") -> Tuple[List[int], bool]:
    """Pareto-frontier iterations on validation (pairwise_accuracy vs `second`).

    For loc_f1 this uses summary.json when the run recorded it; otherwise it is
    recomputed from history.json with the same rule as meta-vul/run_optimization.py
    (_pareto_frontier), or from the scored validation rows when there is no
    history. Any other second metric (joint_both_rate is not in history.json) always
    comes from the scored validation rows, so every metric is scored the same way. Returns (iterations, recomputed)."""
    hist: List[Dict[str, Any]] = []
    if second == "loc_f1":
        if summ.get("pareto_frontier_iterations") is not None:
            return sorted(summ["pareto_frontier_iterations"]), False
        for r in rs:
            f = r / "meta-vul" / "outputs" / session / "history.json"
            if f.exists():
                try:
                    hist = json.loads(f.read_text())
                    break
                except ValueError:
                    pass
    if not hist:
        hist = [{"iteration": r["_i"], "metrics": r} for r in val_rows]
    keys = ("pairwise_accuracy", second)
    vals = [tuple((h.get("metrics") or {}).get(k, 0.0) or 0.0 for k in keys) for h in hist]
    front, seen = [], set()
    for i, (h, v) in enumerate(zip(hist, vals)):
        dominated = any(all(o >= x for o, x in zip(ov, v)) and ov != v
                        for j, ov in enumerate(vals) if j != i)
        if not dominated and v not in seen:
            front.append(h["iteration"])
            seen.add(v)
    return sorted(front), True


# ------------------------------------------------------------------ scoring

def load_gt(path: Path) -> List[Dict[str, Any]]:
    """Ground truth as flat rows. Paired files (func_vulnerable/func_patched per
    row, as in the validation set) expand to a vulnerable then a patched row,
    matching meta-vul/runner/eval_prompt.py so prediction indices line up."""
    raw = load_ground_truth(path)
    if not (raw and "func_vulnerable" in raw[0] and "func_patched" in raw[0]):
        return raw
    rows = []
    for r in raw:
        cwe_id = r.get("cwe_id", "")
        shared = {
            "cve": r.get("cve_id", r.get("task_id", "")),
            "cwe": [c.strip() for c in cwe_id.split(",") if c.strip()] if cwe_id else [],
            "vuln_lines": r.get("vuln_lines", []),
            "vuln_line_contents": r.get("vuln_line_contents", []),
            "_source": r.get("_source", ""),
        }
        rows.append({**shared, "func": r["func_vulnerable"], "target": 1})
        rows.append({**shared, "func": r["func_patched"], "target": 0})
    return rows


def dedupe(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One record per sample. Retries append to predictions.jsonl, so keep the
    latest record that has a verdict, else the latest record."""
    by_key: Dict[Any, Dict[str, Any]] = {}
    for r in preds:
        key = r.get("index", r.get("id"))
        prev = by_key.get(key)
        if prev is None or r.get("vulnerable") in (True, False) or prev.get("vulnerable") not in (True, False):
            by_key[key] = r
    return list(by_key.values())


def score(pred_file: Path, gt: List[Dict[str, Any]],
          exclude: frozenset = frozenset()) -> Optional[Dict[str, Any]]:
    """Metrics for one predictions file. `exclude` holds pair start rows to drop
    (both rows of each pair); coverage is then out of the remaining samples."""
    try:
        preds = dedupe(load_predictions(pred_file))
        preds = [r for r in preds if (r.get("index", -2) // 2) * 2 not in exclude]
        m = compute_all_metrics(preds, ground_truth=gt) if preds else None
    except Exception as exc:
        print(f"warning: could not score {pred_file}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if not m or "n" not in m:
        return None
    m["coverage"] = 100.0 * m["n"] / max(len(gt) - 2 * len(exclude), 1)
    return m


def best_copy(cands: List[Path], gt,
              exclude: frozenset = frozenset()) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    best, best_m = None, None
    for p in cands:
        m = score(p, gt, exclude)
        if m and (best_m is None or m["n"] > best_m["n"]):
            best, best_m = p, m
    return best, best_m


def agent_of(path: Path) -> str:
    for part in path.parts:
        if part in AGENT_NAMES:
            return AGENT_NAMES[part]
    return "unknown"


# ------------------------------------------------------------------ output

def table(rows: List[Dict[str, Any]], cols) -> List[str]:
    widths = [max(len(h), 6) for _, h, _ in cols]
    lines = ["    " + "iter".ljust(6) + " ".join(h.rjust(w) for (_, h, _), w in zip(cols, widths))]
    for row in rows:
        cells = []
        for (k, _, f), w in zip(cols, widths):
            v = row.get(k)
            cells.append((f.format(v) if v is not None else "-").rjust(w))
        lines.append("    " + str(row["iter"]).ljust(6) + " ".join(cells))
    return lines


def block(fields: List[Tuple[str, str]], rows: List[Dict[str, Any]], cols) -> List[str]:
    out = [f"  {(k + ':').ljust(16)} {v}" for k, v in fields]
    out.append("  results:")
    out += table(rows, cols) if rows else ["    (none)"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="Session IDs (default: every session with results)")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="Another checkout's meta-vul-code dir to include (repeatable)")
    ap.add_argument("--csv", default="", help="Also write every row, all columns, to this CSV file")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="Hide runs (validation and held-out) below this coverage percent")
    ap.add_argument("--full", action="store_true", help="Show every metric column")
    ap.add_argument("--all", action="store_true",
                    help="Also show smoke-test sessions (no run above 50%% coverage)")
    ap.add_argument("--include-contaminated", action="store_true",
                    help="Score held-out runs on all 435 PrimeVul pairs, including the 5 that "
                         "overlap the validation set (default: leave them out, 430 pairs)")
    args = ap.parse_args()
    held_exclude = frozenset() if args.include_contaminated else HELD_EXCLUDE_PAIRS

    extra = list(args.extra_root)
    env = os.environ.get("METRICS_EXTRA_ROOTS")
    if env is not None:
        extra += [e for e in env.split(":") if e]
    rs = roots(extra)
    cols = SIMPLE + EXTRA if args.full else SIMPLE

    val_gt = load_gt(VAL_GT) if VAL_GT.exists() else []
    held_gt = load_gt(HELD_GT)
    csv_rows: List[Dict[str, Any]] = []
    printed = 0
    pareto_summary: List[Tuple[str, str, List[str]]] = []

    for s in args.sessions or discover_sessions(rs):
        summ = session_summary(s, rs)
        eval_model, eval_agent = evaluator(s, rs)
        opt_model, opt_agent = optimizer(s, rs, eval_model, eval_agent)

        def tag(i: int) -> str:
            return str(i)

        # Validation: one run per optimizer iteration.
        val: Dict[int, List[Path]] = {}
        for r in rs:
            for p in (r / "meta-vul" / "outputs" / s).glob("iter_[0-9][0-9][0-9]/predictions.jsonl"):
                val.setdefault(int(p.parent.name[5:]), []).append(p)
        all_val_rows = []  # unfiltered, so --min-coverage never changes the Pareto set
        for i in sorted(val):
            _, m = best_copy(val[i], val_gt)
            if m:
                all_val_rows.append({**m, "iter": tag(i), "_i": i})
        val_rows = [r for r in all_val_rows if r["coverage"] >= args.min_coverage]

        # Held-out: group runs by model; the agent comes from the folder they live in.
        held: Dict[Tuple[str, int], List[Path]] = {}
        for r in rs:
            for d in r.glob(f"*-agent/outputs/held_out_primevul_{s}_iter*"):
                m = HELD_RE.match(d.name)
                if m and m.group(1) == s:
                    held.setdefault((m.group(3), int(m.group(2))), []).extend(d.glob("*/predictions.jsonl"))
        for sess, it, model, rel in EXTRA_HELD_RUNS:
            if sess == s:
                held.setdefault((model, it), []).extend(p for r in rs if (p := r / rel).exists())
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        agents: Dict[str, set] = {}
        for (model, i), cands in sorted(held.items(), key=lambda kv: kv[0][1]):
            path, m = best_copy(cands, held_gt, held_exclude)
            if not m or m["coverage"] < args.min_coverage:
                continue
            by_model.setdefault(model, []).append({**m, "iter": tag(i), "_i": i})
            agents.setdefault(model, set()).add(agent_of(path))
        for sess, it, model, rel in SCORED_ONLY_RUNS:
            if sess != s or any(x["_i"] == it for x in by_model.get(model, [])):
                continue
            for r in rs:
                f = r / rel
                if not f.exists():
                    continue
                m = dict(json.loads(f.read_text()).get("metrics", {}))
                m["n"] = sum(m.get(k, 0) for k in ("tp", "fp", "fn", "tn"))
                # A summary has no per-sample rows, so the exclusion cannot apply here.
                m["coverage"] = 100.0 * m["n"] / max(len(held_gt), 1)
                if m["coverage"] >= args.min_coverage:
                    by_model.setdefault(model, []).append({**m, "iter": f"{tag(it)}s", "_i": it})
                    by_model[model].sort(key=lambda x: x["_i"])
                    agents.setdefault(model, set()).add(agent_of(f))
                break

        all_rows = val_rows + [x for v in by_model.values() for x in v]
        if not args.sessions:
            if not all_rows:
                continue
            if not args.all and not any(x["coverage"] >= 50 for x in all_rows):
                continue

        # The optimizing model's held-out block comes first, then other models by name.
        # Known-optimizer entries can carry a note, e.g. "z-ai/glm-5.3 (OpenRouter)"; match on the model id.
        ref = eval_model or opt_model.split(" ")[0]
        order = sorted(by_model, key=lambda mdl: (not same_model(ref, mdl), mdl))

        lines = [f"session_id: {s}", ""]
        val_fields = [("dataset", f"TitanVul/CWEval validation ({len(val_gt)} samples)"),
                      ("optimizer model", opt_model), ("optimizer agent", opt_agent)]
        if eval_model and not same_model(eval_model, opt_model.split(" ")[0]):
            val_fields.append(("evaluator model", eval_model))
        lines += block(val_fields, val_rows, cols)
        for r in val_rows:
            csv_rows.append({"session": s, "dataset": "validation", "model": eval_model or opt_model,
                             "agent": eval_agent or opt_agent, "iter": r["_i"], **r})
        for mdl in order:
            agent = ", ".join(sorted(agents[mdl]))
            label = mdl + ("" if same_model(ref, mdl) else "  (cross-transfer)")
            lines.append("")
            held_label = f"PrimeVul test ({len(held_gt) - 2 * len(held_exclude)} samples"
            held_label += f", {len(held_exclude)} contaminated pairs removed)" if held_exclude else ")"
            lines += block([("dataset", held_label),
                            ("held-out model", label), ("held-out agent", agent)],
                           by_model[mdl], cols)
            for r in by_model[mdl]:
                csv_rows.append({"session": s, "dataset": "primevul_test", "model": mdl,
                                 "agent": agent, "iter": r["_i"], **r})
        lines.append("")
        pareto_lines = []
        for key, label in PARETO_SECOND:
            pareto, recomputed = pareto_from_val(s, rs, summ, all_val_rows, key)
            line = "{" + ", ".join(str(i) for i in pareto) + "}" + ("  (recomputed)" if recomputed and pareto and key == "loc_f1" else "")
            lines.append(f"pareto from val (pairwise vs {label}): ".ljust(38) + line)
            pareto_lines.append(line)
        pareto_summary.append((s, opt_model, pareto_lines))

        if printed:
            print("\n" + "_" * 60 + "\n")
        print("\n".join(lines))
        printed += 1

    if pareto_summary:
        print("\n" + "=" * 60)
        print("pareto from val, every session above (pairwise accuracy vs joint1):")
        width = max(len(m) for _, m, _ in pareto_summary)
        cw = [max(len(label), *(len(ls[k]) for _, _, ls in pareto_summary)) for k, (_, label) in enumerate(PARETO_SECOND)]
        print("  " + "session".ljust(15) + "  " + "optimizer".ljust(width) + "  "
              + "  ".join(label.ljust(w) for (_, label), w in zip(PARETO_SECOND, cw)))
        for sid, model, ls in pareto_summary:
            print(f"  {sid}  {model.ljust(width)}  " + "  ".join(l.ljust(w) for l, w in zip(ls, cw)))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["session", "dataset", "model", "agent", "iter"] + CSV_KEYS,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
