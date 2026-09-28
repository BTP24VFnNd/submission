#!/usr/bin/env python3
"""
Single-agent benchmark runner (agentic mode).

For each sample in the dataset, loads the prompt template for the chosen
policy, injects the code, calls Claude with full filesystem access to a
workspace containing the optimization trajectory, and parses the verdict.

The workspace gives Claude visibility into:
- Past iteration prompts and metrics
- Its own previous predictions and reasoning
- Dataset metadata

Ground truth labels are NEVER written to the workspace.

Supports concurrent execution with --concurrency (or config runner.concurrency).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from prompt_loader import load_template, render
from model_client import call_model, _claude_binary, cli_settings_args, model_env
from output_parser import parse_output


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset, expanding paired format if detected.

    Paired format (TitanVul/CWEval): each row has func_vulnerable and
    func_patched.  These are expanded into two consecutive flat rows
    with 'func' and 'target' fields so the rest of the pipeline (and
    pairwise_accuracy) works unchanged.
    """
    raw_rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

    if not raw_rows:
        return raw_rows

    if "func_vulnerable" in raw_rows[0] and "func_patched" in raw_rows[0]:
        rows = []
        for r in raw_rows:
            cwe_id = r.get("cwe_id", "")
            cwe_list = [c.strip() for c in cwe_id.split(",") if c.strip()] if cwe_id else []
            uuids = r.get("uuid", [None, None])
            shared = {
                "cve": r.get("cve_id", r.get("task_id", "")),
                "cwe": cwe_list,
                "vuln_lines": r.get("vuln_lines", []),
                "vuln_line_contents": r.get("vuln_line_contents", []),
                "_source": r.get("_source", ""),
                "val_idx": r.get("val_idx"),
                "cve_id": r.get("cve_id", ""),
                "task_id": r.get("task_id", ""),
            }
            vul = {**shared, "uuid": uuids[0], "func": r["func_vulnerable"], "target": 1}
            pat = {**shared, "uuid": uuids[1], "func": r["func_patched"], "target": 0}
            rows.append(vul)
            rows.append(pat)
        return rows

    return raw_rows


# ---------------------------------------------------------------------------
# Workspace builder
# ---------------------------------------------------------------------------



def build_workspace(
    base_dir: Path,
    session_id: str,
    iteration: Optional[int],
    config: Dict[str, Any],
    *,
    keep_labels: bool = False,
) -> Optional[Path]:
    """Build a workspace directory with trajectory context.

    Args:
        keep_labels: If True, predictions keep the ground truth label field
                     (for the optimizer). If False, labels are stripped
                     (for the evaluator).

    Returns the workspace path, or None if agentic mode is disabled.
    """
    agentic = config.get("agentic", {})
    if not agentic.get("enabled", False):
        return None

    suffix = "_optimizer" if keep_labels else ""
    workspace = base_dir / "workspaces" / f"{session_id}{suffix}"
    traj_dir = workspace / "trajectory"

    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "prompts").mkdir(exist_ok=True)
    (traj_dir / "metrics").mkdir(exist_ok=True)
    (traj_dir / "predictions").mkdir(exist_ok=True)

    # Copy trajectory from meta-vul outputs if available
    meta_vul_dir = base_dir.parent / "meta-vul" / "outputs"
    if meta_vul_dir.exists():
        _populate_trajectory(meta_vul_dir, traj_dir, session_id, iteration)

    # Copy past predictions
    outputs_dir = base_dir / config.get("outputs_dir", "outputs")
    _populate_past_predictions(
        outputs_dir, traj_dir / "predictions", session_id,
        strip_labels=not keep_labels,
    )

    return workspace


def _populate_trajectory(
    si_outputs: Path,
    traj_dir: Path,
    session_id: str,
    iteration: Optional[int],
) -> None:
    """Copy prompts, metrics, and history from self-improving outputs."""
    # Find the matching self-improving session
    # session_id for optimization runs looks like "opt_<ts>_iter<N>"
    # The self-improving session dir is just the <ts> part
    si_session = None
    if session_id.startswith("opt_"):
        parts = session_id.split("_")
        if len(parts) >= 3:
            si_session = f"{parts[1]}_{parts[2]}"

    if not si_session:
        # Look for the most recent self-improving session
        sessions = sorted(si_outputs.iterdir()) if si_outputs.exists() else []
        sessions = [s for s in sessions if s.is_dir() and not s.name.startswith(".")]
        if sessions:
            si_session = sessions[-1].name

    if not si_session:
        return

    si_dir = si_outputs / si_session
    if not si_dir.exists():
        return

    # Copy history.json
    history = si_dir / "history.json"
    if history.exists():
        shutil.copy2(history, traj_dir / "history.json")

    # Copy iteration prompts and metrics (only up to current iteration)
    max_iter = iteration if iteration is not None else 9999
    for f in sorted(si_dir.iterdir()):
        if f.suffix == ".txt" and f.name.startswith("iter_"):
            iter_num = _parse_iter_num(f.name)
            if iter_num is not None and iter_num < max_iter:
                shutil.copy2(f, traj_dir / "prompts" / f.name)

        if f.is_dir() and f.name.startswith("iter_"):
            iter_num = _parse_iter_num(f.name)
            if iter_num is not None and iter_num < max_iter:
                metrics = f / "metrics.json"
                if metrics.exists():
                    shutil.copy2(metrics, traj_dir / "metrics" / f"{f.name}_metrics.json")


def _populate_past_predictions(
    outputs_dir: Path,
    pred_dest: Path,
    current_session: str,
    *,
    strip_labels: bool = True,
) -> None:
    """Copy past predictions from the same optimization run into the workspace.

    Only includes sessions that share the same base session ID (e.g. opt_20260716_181608_iter0
    and opt_20260716_181608_iter1 share base "20260716_181608"). Ignores unrelated sessions.
    """
    if not outputs_dir.exists():
        return

    # Extract base session ID: "opt_20260716_181608_iter1" -> "20260716_181608"
    import re
    m = re.match(r'opt_(\d{8}_\d{6})_iter\d+', current_session)
    base_session = m.group(1) if m else None

    for session_dir in sorted(outputs_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        if session_dir.name == current_session:
            continue
        # Only include sessions from the same optimization run
        if base_session and not session_dir.name.startswith(f"opt_{base_session}_"):
            continue
        for policy_dir in session_dir.iterdir():
            if not policy_dir.is_dir():
                continue
            pred_file = policy_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            dest = pred_dest / f"{session_dir.name}_{policy_dir.name}.jsonl"
            if strip_labels:
                _copy_predictions_no_labels(pred_file, dest)
            else:
                _copy_predictions_with_labels(pred_file, dest)


def _copy_predictions_no_labels(src: Path, dest: Path) -> None:
    """Copy predictions JSONL, stripping the ground truth label field.

    Handles both single-line and pretty-printed (indent=2) JSON records.
    """
    raw = src.read_text(encoding="utf-8").strip()
    if not raw:
        return
    decoder = json.JSONDecoder()
    pos = 0
    with dest.open("w") as fout:
        while pos < len(raw):
            while pos < len(raw) and raw[pos] in ' \t\n\r':
                pos += 1
            if pos >= len(raw):
                break
            try:
                rec, end = decoder.raw_decode(raw, pos)
                pos = end
                if isinstance(rec, dict):
                    rec.pop("label", None)
                    fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            except json.JSONDecodeError:
                pos += 1


def _copy_predictions_with_labels(src: Path, dest: Path) -> None:
    """Copy predictions JSONL, keeping all fields including labels.

    Handles both single-line and pretty-printed (indent=2) JSON records.
    """
    raw = src.read_text(encoding="utf-8").strip()
    if not raw:
        return
    decoder = json.JSONDecoder()
    pos = 0
    with dest.open("w") as fout:
        while pos < len(raw):
            while pos < len(raw) and raw[pos] in ' \t\n\r':
                pos += 1
            if pos >= len(raw):
                break
            try:
                rec, end = decoder.raw_decode(raw, pos)
                pos = end
                if isinstance(rec, dict):
                    fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            except json.JSONDecodeError:
                pos += 1


def _parse_iter_num(name: str) -> Optional[int]:
    """Extract iteration number from filenames like 'iter_005.txt' or 'iter_005'."""
    import re
    m = re.search(r'iter_?(\d+)', name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sample_id(row: Dict[str, Any], idx: int, id_field: str) -> str:
    val = row.get(id_field)
    if val not in (None, ""):
        return f"{val}_{idx}"
    return f"idx-{idx}"


def get_label(row: Dict[str, Any], label_field: str) -> int:
    val = row.get(label_field)
    if val is None:
        return -1
    if isinstance(val, bool):
        return 1 if val else 0
    s = str(val).strip().lower()
    if s in {"1", "true", "vulnerable", "vul"}:
        return 1
    if s in {"0", "false", "not vulnerable", "non-vulnerable", "safe"}:
        return 0
    return -1


def get_or_create_session_id(base_dir: Path, force_new: bool = False) -> str:
    session_file = base_dir / ".session_id"
    if force_new or not session_file.exists():
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file.write_text(session_id)
        return session_id
    return session_file.read_text().strip()


def verify_model(config: Dict[str, Any]) -> str:
    """
    Ask the CLI which model it actually resolves to and compare against config.

    The config value is only a request. Aliases ("opus", "sonnet") resolve to
    the *latest* model of that tier, and a retired or unavailable id can fall
    back silently. logs.jsonl records the requested string, not the served one,
    so without this check a run can silently use the wrong model.

    Returns the resolved model id. Raises SystemExit on mismatch.
    """
    import subprocess as _sp

    requested = config["model"].get("name", "")
    if not requested:
        return ""

    try:
        env = model_env(config)
    except Exception as e:
        raise SystemExit(f"could not prepare model environment: {e}")
    cmd = [_claude_binary(config), "-p", *cli_settings_args(config), "--output-format", "json",
           "--system-prompt", "", "--tools", "", "--model", requested]
    try:
        res = _sp.run(cmd, input="hi", capture_output=True, text=True,
                      env=env, cwd="/tmp", timeout=120)
        served = list(json.loads(res.stdout).get("modelUsage", {}).keys())
    except Exception as e:
        print(f"  WARNING: could not verify model ({e}). Proceeding on trust.")
        return requested

    if not served:
        print("  WARNING: CLI returned no modelUsage. Proceeding on trust.")
        return requested

    resolved = served[0]
    if resolved != requested:
        raise SystemExit(
            f"MODEL MISMATCH: config requests '{requested}' but the CLI served "
            f"'{resolved}'.\nAliases resolve to the latest model of a tier; pin "
            f"the full id in the config. Refusing to run."
        )
    print(f"Model verified: {resolved}")
    return resolved


def resolve_held_out(base_dir: Path, opt_session_id: str, dataset_name: str,
                     iteration: Optional[int] = None,
                     model_name: str = "") -> Tuple[str, str, int]:
    """
    Look up a prompt from an optimization session for held-out evaluation.

    By default reads meta-vul/outputs/{opt_session_id}/summary.json to
    find which iteration produced the best validation score and uses its
    best_prompt.txt.  When `iteration` is given, that iteration's prompt is used
    instead, which is how the full trajectory (not just the winner) is
    evaluated on a test set.

    Returns (session_id, prompt_text, iteration) where session_id is
    held_out_{dataset}_{opt_session_id}_iter{iteration}.
    """
    opt_dir = base_dir.parent / "meta-vul" / "outputs" / opt_session_id
    summary_path = opt_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"No summary.json for optimization session '{opt_session_id}' "
            f"(looked in {opt_dir}). That session never completed, so it has "
            f"no best prompt to evaluate."
        )

    summary = json.loads(summary_path.read_text())
    best_iter = summary.get("best_iteration")
    if best_iter is None:
        raise SystemExit(f"{summary_path} has no 'best_iteration' field.")

    if iteration is None:
        # Prefer best_prompt.txt, but verify it matches the winning iteration's
        # prompt file so a stale copy can't silently be evaluated.
        target_iter = best_iter
        best_prompt_path = opt_dir / "best_prompt.txt"
        iter_prompt_path = opt_dir / f"iter_{best_iter:03d}.txt"
        if not best_prompt_path.exists():
            raise SystemExit(f"No best_prompt.txt in {opt_dir}.")
        prompt_text = best_prompt_path.read_text()
        if iter_prompt_path.exists() and iter_prompt_path.read_text() != prompt_text:
            raise SystemExit(
                f"best_prompt.txt does not match {iter_prompt_path.name} "
                f"(the recorded best iteration). Refusing to guess which is current."
            )
        source = best_prompt_path
    else:
        target_iter = iteration
        iter_prompt_path = opt_dir / f"iter_{target_iter:03d}.txt"
        if not iter_prompt_path.exists():
            available = sorted(
                _parse_iter_num(f.name) for f in opt_dir.iterdir()
                if f.suffix == ".txt" and f.name.startswith("iter_")
            )
            available = [a for a in available if a is not None]
            raise SystemExit(
                f"Session '{opt_session_id}' has no iteration {target_iter} "
                f"({iter_prompt_path.name} not found). Available: {available}"
            )
        prompt_text = iter_prompt_path.read_text()
        source = iter_prompt_path

    short_name = dataset_name.split("_")[0] if dataset_name else "dataset"
    # The model MUST be in the session id. Without it, evaluating the same
    # iteration on two models resolves to one directory, and skip_existing
    # silently returns the first model's predictions for the second run.
    model_tag = (model_name or "unknown").removeprefix("claude-").replace("/", "-")
    session_id = (f"held_out_{short_name}_{opt_session_id}"
                  f"_iter{target_iter}_{model_tag}")

    # Val score for the iteration actually being evaluated, from history.
    val_score = "?"
    for h in summary.get("history", []):
        if h.get("iteration") == target_iter:
            val_score = h.get("score", "?")
            break

    print(f"Held-out evaluation of optimization session {opt_session_id}")
    print(f"  iteration: {target_iter}"
          f"{' (best)' if target_iter == best_iter else f' (best was {best_iter})'}"
          f" | {summary.get('target_metric', 'score')} {val_score} on val")
    print(f"  prompt: {source}")
    return session_id, prompt_text, target_iter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent vulnerability detection benchmark (agentic mode)")
    parser.add_argument("--config", default="configs/config-test.yaml", help="Path to config YAML (defaults to PrimeVul test)")
    parser.add_argument("--policy", required=True,
                        help="Reasoning policy to use (vanilla, scot, mot, tot, or custom name)")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Number of concurrent calls (0 = use config value)")
    parser.add_argument("--dry-run", action="store_true", help="Render prompts but skip model calls")
    parser.add_argument("--timestamped", action="store_true", help="Use timestamped output directories")
    parser.add_argument("--new-session", action="store_true", help="Start a new session (ignore existing .session_id)")
    parser.add_argument("--run-discovery", nargs=2, metavar=("SESSION_ID", "ITERATION"),
                        help="Run as part of a harness discovery optimization loop. "
                             "Session is named opt_{SESSION_ID}_iter{ITERATION}.")
    parser.add_argument("--held-out", metavar="OPT_SESSION_ID",
                        help="Held-out evaluation of the best prompt from an "
                             "optimization session. Reads that session's summary.json "
                             "to find which iteration won, uses its best_prompt.txt, "
                             "and names the output session "
                             "held_out_{dataset}_{OPT_SESSION_ID}_iter{BEST_ITER}.")
    parser.add_argument("--iteration", type=int, default=None,
                        help="With --held-out, evaluate this iteration's prompt "
                             "instead of the session's best. Used to evaluate a "
                             "whole optimization trajectory on a test set.")
    parser.add_argument("--skip-model-check", action="store_true",
                        help="Skip the preflight check that the CLI serves the "
                             "model named in the config.")
    args = parser.parse_args()

    if args.run_discovery and args.held_out:
        parser.error("--run-discovery and --held-out are mutually exclusive")
    if args.iteration is not None and not args.held_out:
        parser.error("--iteration requires --held-out")

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load((base_dir / args.config).read_text())

    # Confirm the CLI actually serves the model the config asks for, before
    # spending a full run on the wrong one.
    if not args.skip_model_check:
        verify_model(config)

    # Get or create session ID
    iteration = None
    held_out_prompt = None
    if args.run_discovery:
        opt_session, iteration_str = args.run_discovery
        iteration = int(iteration_str)
        session_id = f"opt_{opt_session}_iter{iteration}"
    elif args.held_out:
        session_id, held_out_prompt, iteration = resolve_held_out(
            base_dir, args.held_out, config["dataset"].get("name", ""),
            args.iteration, config["model"].get("name", ""),
        )
    else:
        session_id = get_or_create_session_id(base_dir, args.new_session)
    print(f"Session ID: {session_id}")


    # Load prompt template. In held-out mode the prompt comes from the
    # optimization session's best_prompt.txt, not from prompts/{policy}.txt.
    prompts_dir = base_dir / config["prompts_dir"]
    template = held_out_prompt or load_template(prompts_dir, args.policy)

    # Load dataset
    dataset_path = base_dir / config["dataset"]["path"]
    rows = load_dataset(dataset_path)

    code_field = config["dataset"]["code_field"]
    label_field = config["dataset"]["label_field"]
    id_field = config["dataset"]["id_field"]

    # Prepare output dirs
    output_base = config["outputs_dir"]
    out_dir = base_dir / output_base / session_id / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    log_path = out_dir / "logs.jsonl"

    skip_existing = config["runner"].get("skip_existing", False)
    existing_ids = set()
    existing_indices = set()
    if skip_existing and pred_path.exists():
        raw = pred_path.read_text(encoding="utf-8").strip()
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
                    # Only count a sample as done if it actually produced a
                    # verdict. A record with vulnerable=None is a failure (e.g.
                    # the session limit returned empty output) and must be
                    # re-run, not skipped.
                    if rec.get("vulnerable") is not None:
                        existing_ids.add(rec.get("id"))
                        if rec.get("index") is not None:
                            existing_indices.add(rec["index"])
                    pos = end
                except json.JSONDecodeError:
                    pos += 1

    concurrency = args.concurrency or config["runner"].get("concurrency", 1)

    # Build task list
    tasks = []
    for idx, row in enumerate(rows):
        if args.limit and idx >= args.limit:
            break
        sid = get_sample_id(row, idx, id_field)
        if sid in existing_ids or idx in existing_indices:
            print(f"[{idx + 1}] SKIP {sid}")
            continue
        code = row.get(code_field, "")
        label = get_label(row, label_field)
        prompt = render(template, code)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        row_meta = {
            "uuid": row.get("uuid"),
            "val_idx": row.get("val_idx"),
            "_source": row.get("_source", ""),
            "cve_id": row.get("cve_id", ""),
            "task_id": row.get("task_id", ""),
        }
        tasks.append((idx, sid, label, prompt, prompt_hash, row_meta))

    total = len(tasks)
    if total == 0:
        print("Nothing to run.")
        return

    if args.dry_run:
        for idx, sid, label, prompt, prompt_hash, row_meta in tasks:
            print(f"[{idx + 1}] {sid} (dry-run, {len(prompt)} chars)")
        return

    write_lock = threading.Lock()
    pred_f = pred_path.open("a", encoding="utf-8")
    log_f = log_path.open("a", encoding="utf-8")
    completed = [0]

    def process_sample(idx, sid, label, prompt, prompt_hash, row_meta):
        result = call_model(prompt, config)
        parsed = parse_output(result["raw"])

        pred_rec = {
            "id": sid,
            "index": idx,
            "uuid": row_meta.get("uuid"),
            "val_idx": row_meta.get("val_idx"),
            "_source": row_meta.get("_source", ""),
            "cve_id": row_meta.get("cve_id", ""),
            "task_id": row_meta.get("task_id", ""),
            "policy": args.policy,
            "label": label,
            "vulnerable": parsed["vulnerable"],
            "confidence": parsed["confidence"],
            "verdict_json": parsed["parsed"],
            "parse_errors": parsed["errors"],
        }

        log_rec = {
            "id": sid,
            "index": idx,
            "uuid": row_meta.get("uuid"),
            "policy": args.policy,
            "prompt_hash": prompt_hash,
            "prompt_chars": len(prompt),
            "model": config["model"]["name"],
            "temperature": config["decoding"]["temperature"],
            "max_tokens": config["decoding"]["max_tokens"],
            "tokens_prompt": result["tokens_prompt"],
            "tokens_completion": result["tokens_completion"],
            "latency_s": result["latency_s"],
            "raw_chars": len(result["raw"]),
            "error": result["error"],
            "parse_errors": parsed["errors"],
            "prompt": prompt,
            "raw_output": result["raw"],
        }

        with write_lock:
            pred_f.write(json.dumps(pred_rec, indent=2, ensure_ascii=True) + "\n")
            pred_f.flush()
            log_f.write(json.dumps(log_rec, indent=2, ensure_ascii=True) + "\n")
            log_f.flush()
            completed[0] += 1
            status = "OK" if not parsed["errors"] and not result["error"] else "ERR"
            print(f"[{completed[0]}/{total}] {sid} [{status}] {result['latency_s']}s")

    try:
        if concurrency <= 1:
            for idx, sid, label, prompt, prompt_hash, row_meta in tasks:
                process_sample(idx, sid, label, prompt, prompt_hash, row_meta)
        else:
            print(f"Running {total} samples with concurrency={concurrency}")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [
                    ex.submit(process_sample, idx, sid, label, prompt, prompt_hash, row_meta)
                    for idx, sid, label, prompt, prompt_hash, row_meta in tasks
                ]
                for fut in as_completed(futures):
                    fut.result()
    finally:
        pred_f.close()
        log_f.close()

    print(f"Done. Predictions: {pred_path}")


if __name__ == "__main__":
    main()
