#!/usr/bin/env python3
"""
Evaluate a prompt template by running it through claude-cli-agent's pipeline.

1. Writes the prompt to claude-cli-agent/prompts/self-improving.txt
2. Runs run_single_agent.py with --self-improving flag
3. Reads back predictions.jsonl from the session output
4. Computes metrics from those predictions
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# evaluation/ lives at the primevul_prompting_experiments repo root, one level
# above meta-vul-code/ (this package's parent after the meta-vul-code move).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "evaluation"))

from metrics import binary_metrics, pairwise_accuracy, cwe_metrics, localization_metrics, joint_top1_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AGENT_DIR = "claude-cli-agent"


def _agent_dir(config: Dict[str, Any]) -> Path:
    return REPO_ROOT / config.get("agent_dir", DEFAULT_AGENT_DIR)


def _expand_paired(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand paired format (func_vulnerable/func_patched per row) into
    two consecutive flat rows with 'func' and 'target' fields."""
    rows = []
    for r in raw_rows:
        cwe_id = r.get("cwe_id", "")
        cwe_list = [c.strip() for c in cwe_id.split(",") if c.strip()] if cwe_id else []
        shared = {
            "cve": r.get("cve_id", r.get("task_id", "")),
            "cwe": cwe_list,
            "vuln_lines": r.get("vuln_lines", []),
            "vuln_line_contents": r.get("vuln_line_contents", []),
            "_source": r.get("_source", ""),
        }
        vul = {**shared, "func": r["func_vulnerable"], "target": 1}
        pat = {**shared, "func": r["func_patched"], "target": 0}
        rows.append(vul)
        rows.append(pat)
    return rows


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    raw = _load_jsonl(path)
    if raw and "func_vulnerable" in raw[0] and "func_patched" in raw[0]:
        return _expand_paired(raw)
    return raw


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    raw = _load_jsonl(path)
    if raw and "func_vulnerable" in raw[0] and "func_patched" in raw[0]:
        return _expand_paired(raw)
    return raw


def _dedupe_predictions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate records for the same sample, preferring a real verdict.

    predictions.jsonl is opened in append mode, so re-running a failed sample
    leaves both the original blank record and the new one. Keep the record with
    a verdict; if several qualify, keep the last (most recent) write. Order is
    preserved by first appearance.
    """
    best: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for rec in records:
        key = rec.get("index")
        if key is None:
            key = ("id", rec.get("id"))
        if key not in best:
            best[key] = rec
            order.append(key)
            continue
        prev = best[key]
        # A verdict beats a blank; between two verdicts the later write wins.
        if prev.get("vulnerable") is None or rec.get("vulnerable") is not None:
            best[key] = rec
    return [best[k] for k in order]


def evaluate_prompt(
    template: str,
    opt_session_id: str,
    iteration: int,
    config: Dict[str, Any],
    config_path: str = None,
    ground_truth: List[Dict[str, Any]] = None,
    limit: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Run a prompt template through the configured agent (agent_dir in config,
    defaults to claude-cli-agent) and compute metrics.

    Returns (metrics_dict, predictions_list, logs_path).
    """
    agent_dir = _agent_dir(config)
    run_script = agent_dir / "runner" / "run_single_agent.py"

    # Write prompt to <agent_dir>/prompts/self-improving.txt
    prompt_path = agent_dir / "prompts" / "self-improving.txt"
    prompt_path.write_text(template)

    # Run run_single_agent.py
    cmd = [
        sys.executable, str(run_script),
        "--config", config_path,
        "--policy", "self-improving",
        "--run-discovery", opt_session_id, str(iteration),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"  Running {agent_dir.name}: session opt_{opt_session_id}_iter{iteration}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(agent_dir),
    )

    if result.returncode != 0:
        print(f"  run_single_agent.py failed: {result.stderr[:500]}")
        return {"error": result.stderr[:500]}, [], None

    print(result.stdout)

    # Read predictions
    session_name = f"opt_{opt_session_id}_iter{iteration}"
    pred_path = agent_dir / "outputs" / session_name / "self-improving" / "predictions.jsonl"

    if not pred_path.exists():
        return {"error": f"predictions not found: {pred_path}"}, []

    predictions = []
    with pred_path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()
        if raw:
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(raw):
                while pos < len(raw) and raw[pos] in ' \t\n\r':
                    pos += 1
                if pos >= len(raw):
                    break
                try:
                    rec, end = decoder.raw_decode(raw, pos)
                    predictions.append(rec)
                    pos = end
                except json.JSONDecodeError:
                    pos += 1

    predictions = _dedupe_predictions(predictions)

    # Locate logs.jsonl (per-sample prompts and raw responses)
    log_path = agent_dir / "outputs" / session_name / "self-improving" / "logs.jsonl"
    logs_path = str(log_path) if log_path.exists() else None

    if not predictions:
        return {"error": "no predictions parsed"}, [], logs_path

    # Compute metrics
    valid = [(p["label"], p["vulnerable"]) for p in predictions
             if p["label"] in (0, 1) and p["vulnerable"] is not None]

    if not valid:
        return {"error": "no valid predictions"}, predictions, logs_path

    labels, pred_flags = zip(*valid)
    metrics = binary_metrics(list(labels), list(pred_flags))
    metrics["n"] = len(valid)
    metrics["parse_failures"] = sum(1 for p in predictions if p["vulnerable"] is None)
    metrics["session"] = session_name

    pw = pairwise_accuracy(predictions)
    metrics.update(pw)

    if ground_truth:
        metrics.update(cwe_metrics(predictions, ground_truth))
        metrics.update(localization_metrics(predictions, ground_truth))
        # joint1: first predicted CWE correct AND a ground-truth line hit; the Pareto
        # frontier's second objective (run_optimization._pareto_frontier).
        metrics.update(joint_top1_metrics(predictions, ground_truth))

    return metrics, predictions, logs_path
