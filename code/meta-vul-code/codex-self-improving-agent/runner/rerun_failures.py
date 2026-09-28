#!/usr/bin/env python3
"""
Re-run only parse-failure samples with updated config (e.g. higher max_tokens/timeout).

Reads existing predictions.jsonl and logs.jsonl, finds samples where vulnerable is None,
re-calls the model using the original prompt (stored in logs), re-parses, and patches
the records in-place. Preserves file order.

Usage:
    python runner/rerun_failures.py --session 20260305_193906 --policy scot \
        --config config_kimi_k2.5_thinking.yaml \
        --max-tokens 16384 --timeout 900 --concurrency 2
"""
import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from model_client import call_model
from output_parser import parse_output


def load_pretty_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load pretty-printed JSONL using raw_decode."""
    records = []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return records
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        while pos < len(raw) and raw[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(raw):
            break
        try:
            rec, end = decoder.raw_decode(raw, pos)
            records.append(rec)
            pos = end
        except json.JSONDecodeError:
            pos += 1
    return records


def main():
    parser = argparse.ArgumentParser(description="Re-run parse failures with updated config")
    parser.add_argument("--session", required=True, help="Session ID (e.g. 20260305_193906)")
    parser.add_argument("--policy", required=True, help="Policy name (e.g. scot)")
    parser.add_argument("--config", required=True, help="Path to config YAML (relative to base_dir)")
    parser.add_argument("--max-tokens", type=int, default=0, help="Override max_tokens (0 = use config)")
    parser.add_argument("--timeout", type=int, default=0, help="Override timeout_s (0 = use config)")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent calls")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be re-run without calling model")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load((base_dir / args.config).read_text())

    if args.max_tokens:
        config["decoding"]["max_tokens"] = args.max_tokens
    if args.timeout:
        config["runner"]["timeout_s"] = args.timeout

    out_dir = base_dir / config["outputs_dir"] / args.session / args.policy
    pred_path = out_dir / "predictions.jsonl"
    log_path = out_dir / "logs.jsonl"

    if not pred_path.exists() or not log_path.exists():
        print(f"Error: missing {pred_path} or {log_path}", file=sys.stderr)
        sys.exit(1)

    preds = load_pretty_jsonl(pred_path)
    logs_list = load_pretty_jsonl(log_path)
    logs_by_index = {r["index"]: r for r in logs_list}

    # Find failures
    failure_indices = []
    for i, r in enumerate(preds):
        if r.get("vulnerable") is None:
            failure_indices.append(i)

    print(f"Found {len(failure_indices)} parse failures in {pred_path}")

    # Build tasks: (pred_list_index, sample_index, prompt)
    tasks = []
    skipped = 0
    for pi in failure_indices:
        sample_idx = preds[pi]["index"]
        log = logs_by_index.get(sample_idx)
        if not log or not log.get("prompt"):
            print(f"  SKIP idx={sample_idx}: no prompt in logs")
            skipped += 1
            continue
        tasks.append((pi, sample_idx, log["prompt"]))

    print(f"Will re-run {len(tasks)} samples (skipped {skipped} with no prompt)")
    print(f"Config: max_tokens={config['decoding']['max_tokens']}, timeout={config['runner']['timeout_s']}s, concurrency={args.concurrency}")

    if args.dry_run:
        for pi, sample_idx, prompt in tasks:
            print(f"  idx={sample_idx} label={preds[pi]['label']} prompt_len={len(prompt)}")
        return

    # Backup originals before re-running
    backup_dir = out_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(pred_path, backup_dir / f"predictions_pre_rerun_{ts}.jsonl")
    shutil.copy2(log_path, backup_dir / f"logs_pre_rerun_{ts}.jsonl")
    print(f"Backed up originals to {backup_dir}/")

    write_lock = threading.Lock()
    completed = [0]
    total = len(tasks)

    def _flush_files():
        """Write full predictions and logs to disk, plus post-rerun backups."""
        with pred_path.open("w", encoding="utf-8") as f:
            for rec in preds:
                f.write(json.dumps(rec, indent=2, ensure_ascii=True) + "\n")
        with log_path.open("w", encoding="utf-8") as f:
            for rec in logs_list:
                f.write(json.dumps(rec, indent=2, ensure_ascii=True) + "\n")
        shutil.copy2(pred_path, backup_dir / f"predictions_post_rerun_{ts}.jsonl")
        shutil.copy2(log_path, backup_dir / f"logs_post_rerun_{ts}.jsonl")

    def process(pi, sample_idx, prompt):
        result = call_model(prompt, config)
        parsed = parse_output(result["raw"])

        with write_lock:
            preds[pi]["vulnerable"] = parsed["vulnerable"]
            preds[pi]["confidence"] = parsed["confidence"]
            preds[pi]["verdict_json"] = parsed["parsed"]
            preds[pi]["parse_errors"] = parsed["errors"]

            # Update the log entry too
            log = logs_by_index.get(sample_idx)
            if log:
                log["raw_output"] = result["raw"]
                log["error"] = result["error"]
                log["latency_s"] = result["latency_s"]
                log["tokens_prompt"] = result["tokens_prompt"]
                log["tokens_completion"] = result["tokens_completion"]
                log["max_tokens"] = config["decoding"]["max_tokens"]
                log["timeout_s"] = config["runner"]["timeout_s"]

            # Flush to disk after each sample
            _flush_files()

            completed[0] += 1
            status = "OK" if parsed["vulnerable"] is not None else "FAIL"
            vuln = parsed["vulnerable"]
            print(f"[{completed[0]}/{total}] idx={sample_idx} [{status}] vuln={vuln} {result['latency_s']}s err={result['error'][:60] if result['error'] else '-'}")

    try:
        if args.concurrency <= 1:
            for pi, sample_idx, prompt in tasks:
                process(pi, sample_idx, prompt)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [
                    ex.submit(process, pi, sample_idx, prompt)
                    for pi, sample_idx, prompt in tasks
                ]
                for fut in as_completed(futures):
                    fut.result()
    except KeyboardInterrupt:
        print("\nInterrupted. Files already saved up to last completed sample.")

    print(f"Post-rerun backups saved to {backup_dir}/")

    remaining = sum(1 for r in preds if r.get("vulnerable") is None)
    print(f"Done. {len(tasks) - remaining} recovered, {remaining} still failing.")
    print(f"Updated: {pred_path}")


if __name__ == "__main__":
    main()
