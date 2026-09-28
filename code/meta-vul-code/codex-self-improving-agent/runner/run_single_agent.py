#!/usr/bin/env python3
"""
Codex single-agent benchmark runner.

For each sample in the dataset, loads the prompt template for the chosen
policy, injects the code, calls Codex, and parses the verdict.

Per-sample evaluator calls are isolated from trajectory data and labels.

The optimizer workspace gives Codex visibility into:
- Past iteration prompts and metrics
- Its own previous predictions and reasoning
- Ground-truth labels used to identify prior failures

Supports concurrent execution with --concurrency (or config runner.concurrency).
"""
import argparse
import hashlib
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from prompt_loader import load_template, render
from model_client import call_model
from output_parser import parse_output


FATAL_MODEL_ERROR_MARKERS = (
    "workspace is out of credits",
    "usage limit has been reached",
    "you've hit your usage limit",
)

# Known model-provider prefixes for --model overrides (same syntax as
# run_harness_discovery.py's --model): "tinker/<model>" repoints Codex's
# model provider at Tinker's endpoint via the local litellm proxy.
KNOWN_PROVIDERS = {"tinker", "openrouter"}


def is_fatal_model_error(error: str) -> bool:
    normalized = (error or "").lower()
    return any(marker in normalized for marker in FATAL_MODEL_ERROR_MARKERS)


def is_rate_limit_error(error: str) -> bool:
    normalized = (error or "").lower()
    return "429" in normalized and "too many requests" in normalized


# Shared across worker threads: when any call is rate limited, all workers
# hold off until this timestamp so the provider's quota window can refill.
_rate_limit_lock = threading.Lock()
_rate_limit_until = 0.0


def _start_rate_limit_cooldown(cooldown_s: float) -> None:
    global _rate_limit_until
    with _rate_limit_lock:
        _rate_limit_until = max(_rate_limit_until, time.time() + cooldown_s)


def _wait_for_rate_limit_cooldown() -> None:
    while True:
        with _rate_limit_lock:
            remaining = _rate_limit_until - time.time()
        if remaining <= 0:
            return
        # Jitter so the workers don't all resume in the same instant.
        time.sleep(remaining + random.uniform(0.0, 5.0))


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
    """Copy predictions from earlier iterations in the same optimization run."""
    if not outputs_dir.exists():
        return

    lineage = _optimization_lineage(current_session)

    for session_dir in sorted(outputs_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        if session_dir.name == current_session:
            continue
        if lineage is not None:
            prefix, current_iteration = lineage
            candidate = _optimization_lineage(session_dir.name)
            if candidate is None:
                continue
            candidate_prefix, candidate_iteration = candidate
            if candidate_prefix != prefix or candidate_iteration >= current_iteration:
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


def _optimization_lineage(session_id: str):
    """Return (run prefix, iteration) for opt_<session>_iter<N> IDs."""
    import re
    match = re.fullmatch(r"(opt_.+)_iter(\d+)", session_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


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


def detect_opt_session_model(base_dir: Path, opt_session_id: str) -> Dict[str, str]:
    """Recover the model actually used to generate an optimization session's
    predictions, by reading it back out of a logged call rather than trusting
    whatever --config happens to be loaded for a later held-out invocation.

    Iterations can be served by different backends over a long optimization
    run (a local proxy, a hosted model, ...), so this is best-effort: it
    reports what iter_000 used. Returns {} if nothing could be recovered.
    """
    iter0_dir = base_dir.parent / "codex-self-improving-agent" / "outputs" / f"opt_{opt_session_id}_iter0"
    logs_path = iter0_dir / "self-improving" / "logs.jsonl"
    if not logs_path.exists():
        return {}
    # Despite the .jsonl extension, records are pretty-printed (multi-line),
    # concatenated back-to-back rather than one compact object per line.
    text = logs_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    model_name = None
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        try:
            record, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        idx = end
        if record.get("model"):
            model_name = record["model"]
            break
    if not model_name:
        return {}
    litellm_config = base_dir.parent / ".litellm" / "config.yaml"
    is_proxied = False
    if litellm_config.exists():
        served = yaml.safe_load(litellm_config.read_text()).get("model_list", [])
        is_proxied = any(m.get("model_name") == model_name for m in served)
    return {"name": model_name, "provider": "tinker" if is_proxied else ""}


def resolve_held_out(
    base_dir: Path,
    opt_session_id: str,
    dataset_name: str,
    iteration: Optional[int] = None,
    model_name: str = "",
) -> tuple[str, str, int]:
    """Load an optimization prompt and name a held-out evaluation session.

    With no explicit iteration, evaluate the recorded best prompt. With an
    iteration, evaluate that exact trajectory prompt. The model tag prevents
    ``skip_existing`` from mixing results from different served models.
    """
    opt_dir = base_dir.parent / "meta-vul" / "outputs" / opt_session_id
    summary_path = opt_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"No summary.json for optimization session '{opt_session_id}' "
            f"(looked in {opt_dir})."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_iter = summary.get("best_iteration")
    if best_iter is None:
        raise SystemExit(f"{summary_path} has no 'best_iteration' field.")

    if iteration is None:
        target_iter = int(best_iter)
        source = opt_dir / "best_prompt.txt"
        if not source.exists():
            raise SystemExit(f"No best_prompt.txt in {opt_dir}.")
        prompt_text = source.read_text(encoding="utf-8")
        iter_prompt = opt_dir / f"iter_{target_iter:03d}.txt"
        if iter_prompt.exists() and iter_prompt.read_text(encoding="utf-8") != prompt_text:
            raise SystemExit(
                f"best_prompt.txt does not match {iter_prompt.name}; refusing to guess."
            )
    else:
        target_iter = iteration
        source = opt_dir / f"iter_{target_iter:03d}.txt"
        if not source.exists():
            available = sorted(
                value for value in (_parse_iter_num(path.name) for path in opt_dir.glob("iter_*.txt"))
                if value is not None
            )
            raise SystemExit(
                f"Session '{opt_session_id}' has no iteration {target_iter}. "
                f"Available: {available}"
            )
        prompt_text = source.read_text(encoding="utf-8")

    dataset_tag = dataset_name.split("_")[0] if dataset_name else "dataset"
    # Tinker model names contain "/" (e.g. "thinkingmachines/Inkling-Small"),
    # which would otherwise be interpreted as a path separator and split this
    # tag across nested directories.
    model_tag = (model_name or "unknown").removeprefix("claude-").replace("/", "-")
    session_id = (
        f"held_out_{dataset_tag}_{opt_session_id}_iter{target_iter}_{model_tag}"
    )
    print(f"Held-out evaluation of optimization session {opt_session_id}")
    print(f"  iteration: {target_iter} | best validation iteration: {best_iter}")
    print(f"  prompt: {source}")
    return session_id, prompt_text, target_iter


def load_policy_template(
    base_dir: Path,
    config: Dict[str, Any],
    policy: str,
    prompt_file: Optional[Path] = None,
) -> str:
    """Load either a named policy or an explicit prompt template."""
    if prompt_file is not None:
        path = prompt_file.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")
    return load_template(base_dir / config["prompts_dir"], policy)


def call_and_parse_with_retries(
    prompt: str,
    config: Dict[str, Any],
):
    """Retry transport and schema failures, retaining the best parse."""
    retries = max(0, int(config["runner"].get("retries", 0)))
    retry_delay = max(0.0, float(config["runner"].get("retry_delay_s", 1.0)))
    rate_limit_cooldown = max(0.0, float(config["runner"].get("rate_limit_cooldown_s", 30.0)))
    max_rate_limit_waits = max(0, int(config["runner"].get("max_rate_limit_waits", 20)))
    best = None
    total_latency = 0.0

    rate_limit_waits = 0
    attempt = 0
    while attempt < retries + 1:
        attempt += 1
        _wait_for_rate_limit_cooldown()
        result = call_model(prompt, config)
        parsed = parse_output(result["raw"])
        total_latency += float(result.get("latency_s") or 0.0)

        # A 429 means the provider quota is exhausted, not that this sample
        # failed: pause every worker and retry without spending an attempt.
        if is_rate_limit_error(result.get("error", "")) and rate_limit_waits < max_rate_limit_waits:
            rate_limit_waits += 1
            _start_rate_limit_cooldown(rate_limit_cooldown)
            attempt -= 1
            continue

        candidate = (result, parsed, attempt)
        if best is None or (
            best[1]["vulnerable"] is None and parsed["vulnerable"] is not None
        ):
            best = candidate

        if not result["error"] and not parsed["errors"]:
            best = candidate
            break

        if is_fatal_model_error(result.get("error", "")):
            break

        if attempt <= retries:
            time.sleep(retry_delay * (2 ** (attempt - 1)))

    result, parsed, attempts = best
    result = {**result, "latency_s": round(total_latency, 3)}
    return result, parsed, attempts


def load_completed_prediction_keys(pred_path: Path):
    """Return IDs and indices for records that contain an actual verdict."""
    completed_ids = set()
    completed_indices = set()
    if not pred_path.exists():
        return completed_ids, completed_indices

    raw = pred_path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        while pos < len(raw) and raw[pos] in " \t\n\r":
            pos += 1
        if pos >= len(raw):
            break
        try:
            rec, end = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        pos = end
        if not isinstance(rec, dict) or rec.get("vulnerable") is None:
            continue
        completed_ids.add(rec.get("id"))
        if rec.get("index") is not None:
            completed_indices.add(rec["index"])
    return completed_ids, completed_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Codex vulnerability detection benchmark")
    parser.add_argument("--config", default="configs/config-test.yaml", help="Path to config YAML (defaults to PrimeVul test)")
    parser.add_argument("--policy", required=True,
                        help="Reasoning policy to use (vanilla, scot, mot, tot, or custom name)")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Number of concurrent calls (0 = use config value)")
    parser.add_argument("--timeout-s", type=float, default=0,
                        help="Per-call timeout in seconds (0 = use config value)")
    parser.add_argument("--retries", type=int, default=-1,
                        help="Retries after a failed call (-1 = use config value)")
    parser.add_argument("--dry-run", action="store_true", help="Render prompts but skip model calls")
    parser.add_argument("--timestamped", action="store_true", help="Use timestamped output directories")
    parser.add_argument("--new-session", action="store_true", help="Start a new session (ignore existing .session_id)")
    parser.add_argument("--prompt-file", type=Path,
                        help="Use an explicit prompt template instead of prompts/<policy>.txt")
    parser.add_argument("--run-discovery", nargs=2, metavar=("SESSION_ID", "ITERATION"),
                        help="Run as part of a harness discovery optimization loop. "
                             "Session is named opt_{SESSION_ID}_iter{ITERATION}.")
    parser.add_argument("--held-out", metavar="OPT_SESSION_ID",
                        help="Evaluate the best prompt from an optimization session on this held-out dataset.")
    parser.add_argument("--iteration", type=int, default=None,
                        help="With --held-out, evaluate this exact optimization iteration.")
    parser.add_argument("--model", default=None,
                        help="With --held-out, evaluate using this model instead of the one "
                             "auto-detected from the optimization session's logs. Prefix with "
                             "a known provider (e.g. 'tinker/thinkingmachines/Inkling-Small') "
                             "to route through a different model provider.")
    args = parser.parse_args()

    if args.run_discovery and args.held_out:
        parser.error("--run-discovery and --held-out are mutually exclusive")
    if args.iteration is not None and not args.held_out:
        parser.error("--iteration requires --held-out")
    if args.model is not None and not args.held_out:
        parser.error("--model requires --held-out")

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load((base_dir / args.config).read_text())
    if args.timeout_s:
        config["runner"]["timeout_s"] = args.timeout_s
    if args.retries >= 0:
        config["runner"]["retries"] = args.retries

    if args.held_out:
        if args.model:
            # Explicit override: e.g. to compare against a model that never
            # ran the optimization loop, or when logs can't be trusted.
            provider, name = (args.model.split("/", 1) if "/" in args.model
                               and args.model.split("/", 1)[0] in KNOWN_PROVIDERS
                               else (None, args.model))
            config["model"]["name"] = name
            if provider:
                config["model"]["provider"] = provider
            else:
                config["model"].pop("provider", None)
            print(f"Held-out model set explicitly via --model: {config['model']}")
        else:
            # Default: held-out must reproduce the validation run's model, not
            # whatever --config happens to be loaded here -- recover it from
            # the optimization session's own logs instead of trusting a guess.
            detected = detect_opt_session_model(base_dir, args.held_out)
            if not detected:
                raise SystemExit(
                    f"Could not recover the model used by optimization session "
                    f"'{args.held_out}' from its logs; refusing to guess by "
                    f"falling back to --config's model ({config['model'].get('name')}). "
                    f"Pass --model explicitly to override."
                )
            config["model"]["name"] = detected["name"]
            if detected["provider"]:
                config["model"]["provider"] = detected["provider"]
            else:
                config["model"].pop("provider", None)
            print(f"Held-out model auto-detected from session logs: {detected}")

    # Get or create session ID
    iteration = None
    held_out_prompt = None
    if args.run_discovery:
        opt_session, iteration_str = args.run_discovery
        iteration = int(iteration_str)
        session_id = f"opt_{opt_session}_iter{iteration}"
    elif args.held_out:
        session_id, held_out_prompt, iteration = resolve_held_out(
            base_dir,
            args.held_out,
            config["dataset"].get("name", ""),
            args.iteration,
            config["model"].get("name", ""),
        )
    else:
        session_id = get_or_create_session_id(base_dir, args.new_session)
    print(f"Session ID: {session_id}")


    # Load prompt template
    template = held_out_prompt or load_policy_template(
        base_dir, config, args.policy, args.prompt_file
    )

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
    if skip_existing:
        existing_ids, existing_indices = load_completed_prediction_keys(pred_path)

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
    stop_event = threading.Event()
    fatal_errors = []
    pred_f = pred_path.open("a", encoding="utf-8")
    log_f = log_path.open("a", encoding="utf-8")
    completed = [0]

    def process_sample(idx, sid, label, prompt, prompt_hash, row_meta):
        if stop_event.is_set():
            return
        result, parsed, attempts = call_and_parse_with_retries(prompt, config)
        fatal_error = is_fatal_model_error(result.get("error", ""))
        if fatal_error:
            stop_event.set()

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
            "attempts": attempts,
            "isolation": result.get("isolation"),
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
            if fatal_error:
                fatal_errors.append(result["error"])

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

    if fatal_errors:
        raise RuntimeError(
            "Fatal model-account error stopped the run; resume after access is restored: "
            f"{fatal_errors[0].splitlines()[-1]}"
        )

    print(f"Done. Predictions: {pred_path}")


if __name__ == "__main__":
    main()
