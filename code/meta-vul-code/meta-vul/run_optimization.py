#!/usr/bin/env python3
"""
Main optimization loop for self-improving prompt optimization.

1. Load seed prompt and ground truth
2. Evaluate seed prompt on val set (through claude-cli-agent)
3. Feed metrics + failures to optimizer LLM
4. Optimizer proposes new prompt
5. Evaluate new prompt on val set
6. If improved, keep it; otherwise revert
7. Repeat until max_iterations or patience exhausted

Resume support:
  python run_optimization.py --resume 20260507_192030
  Reloads history, prompts, and metrics from the session dir and
  continues from the next iteration.
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "optimizer"))

from eval_prompt import evaluate_prompt, load_dataset, DEFAULT_AGENT_DIR
from metrics import joint_top1_metrics  # evaluation/metrics.py, put on sys.path by eval_prompt
from prompt_optimizer import propose_new_prompt

PARETO_KEYS = ("pairwise_accuracy", "joint_top1_rate")


def _import_build_workspace(agent_dir_name: str):
    """Import build_workspace from the configured agent's runner package."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / agent_dir_name / "runner"))
    from run_single_agent import build_workspace
    return build_workspace


def _load_resume_state(out_dir: Path):
    """Load state from a previous session for resuming."""
    history_path = out_dir / "history.json"
    if not history_path.exists():
        return None

    history = json.loads(history_path.read_text())
    if not history:
        return None

    last = history[-1]
    last_iteration = last["iteration"]

    # Load the prompt from the last iteration
    prompt_file = out_dir / last["prompt_file"]
    if not prompt_file.exists():
        return None
    current_prompt = prompt_file.read_text()

    # Load predictions from the last iteration
    iter_tag = f"iter_{last_iteration:03d}"
    pred_path = out_dir / iter_tag / "predictions.jsonl"
    predictions = []
    if pred_path.exists():
        with pred_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))

    # Reconstruct best state
    best_score = -1.0
    best_prompt = current_prompt
    best_iteration = 0
    no_improve_count = 0
    for h in history:
        s = h.get("score", 0.0)
        if s > best_score:
            best_score = s
            best_iteration = h["iteration"]
            best_prompt_file = out_dir / h["prompt_file"]
            if best_prompt_file.exists():
                best_prompt = best_prompt_file.read_text()

    # Count consecutive non-improvements at the tail
    for h in reversed(history):
        if h["iteration"] == 0:
            break
        if "no improvement" in h.get("change_summary", ""):
            no_improve_count += 1
        else:
            break

    return {
        "history": history,
        "last_iteration": last_iteration,
        "current_prompt": current_prompt,
        "metrics": last["metrics"],
        "predictions": predictions,
        "best_score": best_score,
        "best_prompt": best_prompt,
        "best_iteration": best_iteration,
        "no_improve_count": no_improve_count,
        "incomplete": last["metrics"].get("parse_failures", 0) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-improving prompt optimization")
    parser.add_argument("--config", default="configs/config-val.yaml", help="Path to config YAML")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Override max iterations (0 = use config)")
    parser.add_argument("--val-only", action="store_true",
                        help="Only run validation, skip final test evaluation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print setup info without running")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to evaluate per iteration (0 = all)")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="SESSION_ID",
                        help="Resume a previous session. If no session ID given, resumes the latest.")
    parser.add_argument("--session-id", default=None,
                        help="Session ID to use for a fresh run (default: current timestamp). "
                             "Ignored with --resume.")
    parser.add_argument("--finalize", metavar="SESSION_ID", default=None,
                        help="If the session is finished, rebuild its Pareto frontier and harnesses "
                             "from saved predictions (no model calls). With --dry-run, only report.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = str(base_dir / args.config)
    config = yaml.safe_load(Path(config_path).read_text())

    # Load val dataset (also serves as ground truth for CWE/localization metrics)
    repo_root = base_dir.parent
    dataset_rel = config["dataset"]["path"]
    val_dataset_path = (base_dir / dataset_rel).resolve()
    if not val_dataset_path.exists():
        val_dataset_path = (repo_root / dataset_rel).resolve()
    val_samples = load_dataset(val_dataset_path)
    gt = val_samples

    max_iterations = args.max_iterations or config["optimization"]["max_iterations"]
    patience = config["optimization"]["patience"]
    target_metric = config["optimization"]["target_metric"]

    if args.finalize:
        out_dir = base_dir / config["outputs_dir"] / args.finalize
        sys.exit(finalize_session(out_dir, args.finalize, gt, max_iterations, patience,
                                  target_metric, config, dry_run=args.dry_run))

    agent_dir_name = config.get("agent_dir", DEFAULT_AGENT_DIR)
    build_workspace = _import_build_workspace(agent_dir_name)

    # Resume or start fresh
    reuse_prompt_iteration = None
    if args.resume:
        if args.resume == "latest":
            outputs_root = base_dir / config["outputs_dir"]
            sessions = sorted(
                [d.name for d in outputs_root.iterdir()
                 if d.is_dir() and (d / "history.json").exists()],
                reverse=True,
            )
            if not sessions:
                print(f"No resumable sessions found in {outputs_root}")
                sys.exit(1)
            session_id = sessions[0]
        else:
            session_id = args.resume
        out_dir = base_dir / config["outputs_dir"] / session_id
        if not out_dir.exists():
            print(f"Session dir not found: {out_dir}")
            sys.exit(1)

        state = _load_resume_state(out_dir)
        if state is None:
            print(f"No valid history found in {out_dir}")
            sys.exit(1)

        history = state["history"]
        current_prompt = state["current_prompt"]
        metrics = state["metrics"]
        predictions = state["predictions"]
        best_prompt = state["best_prompt"]
        no_improve_count = state["no_improve_count"]
        score = state["history"][-1].get("score", 0.0)

        last_iter = state["last_iteration"]
        reuse_prompt_iteration = None

        if state["incomplete"] and last_iter > 0:
            # The last iteration recorded a score but some samples never
            # produced a verdict (e.g. a session limit cut the run short).
            # Redo that iteration instead of moving past it: drop its entry,
            # and reuse its saved prompt so the partial predictions still match.
            failed = state["metrics"].get("parse_failures", 0)
            n_done = state["metrics"].get("n", 0)
            print(f"Iteration {last_iter} is incomplete: {failed} sample(s) "
                  f"without a verdict ({n_done} scored).")
            print(f"Resuming inside iteration {last_iter} — re-running only the "
                  f"failed samples with its existing prompt.")
            history = [h for h in history if h["iteration"] != last_iter]
            start_iteration = last_iter
            reuse_prompt_iteration = last_iter
            # Recompute best/patience without the discarded entry.
            best_score, best_iteration = -1.0, 0
            for h in history:
                if h.get("score", 0.0) > best_score:
                    best_score = h["score"]
                    best_iteration = h["iteration"]
                    bp = out_dir / h["prompt_file"]
                    if bp.exists():
                        best_prompt = bp.read_text()
            no_improve_count = 0
            score = history[-1].get("score", 0.0) if history else 0.0
        elif (out_dir / f"iter_{last_iter + 1:03d}.txt").exists() and (
                base_dir.parent / "claude-cli-agent" / "outputs"
                / f"opt_{session_id}_iter{last_iter + 1}"
                / "self-improving" / "predictions.jsonl").exists():
            # The next iteration has a saved prompt and partial predictions on
            # disk but no history entry: the process was killed mid-evaluation
            # before it could record a score. Resume into that iteration and
            # reuse its prompt so the completed samples stay valid; skip_existing
            # + dedupe will finish the blank/missing ones.
            orphan = last_iter + 1
            print(f"Iteration {orphan} was interrupted mid-evaluation "
                  f"(prompt saved, no score recorded).")
            print(f"Resuming inside iteration {orphan} — completing its unfinished "
                  f"samples with the existing prompt.")
            start_iteration = orphan
            reuse_prompt_iteration = orphan
            best_score = state["best_score"]
            best_iteration = state["best_iteration"]
        else:
            start_iteration = last_iter + 1
            best_score = state["best_score"]
            best_iteration = state["best_iteration"]

        print(f"Resuming session: {session_id}")
        print(f"Last completed iteration: {last_iter}")
        print(f"Best {target_metric}: {best_score:.2f} @ iter {best_iteration}")
        print(f"Patience: {no_improve_count}/{patience}")
        print(f"Resuming from iteration {start_iteration}")
        print()
    else:
        session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / config["outputs_dir"] / session_id
        if args.session_id and out_dir.exists():
            raise SystemExit(f"Session {session_id} already exists; use --resume {session_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load seed prompt
        seed_path = base_dir / config["optimization"]["seed_prompt"]
        current_prompt = seed_path.read_text()

        print(f"Session: {session_id}")
        print(f"Val samples: {len(val_samples)} ({len(val_samples) // 2} pairs)")
        print(f"Max iterations: {max_iterations}")
        print(f"Target metric: {target_metric}")
        print(f"Patience: {patience}")
        print(f"Seed prompt: {seed_path.name}")
        print()

        if args.dry_run:
            print("Dry run, exiting.")
            return

        # Save seed prompt
        (out_dir / "iter_000.txt").write_text(current_prompt)

        history = []
        best_score = -1.0
        best_prompt = current_prompt
        best_iteration = 0
        no_improve_count = 0

        # Iteration 0: evaluate seed
        print("=" * 60)
        print("Iteration 0: evaluating seed prompt")
        print("=" * 60)
        metrics, predictions, logs_path = evaluate_prompt(current_prompt, session_id, 0, config, config_path=config_path, ground_truth=gt, limit=args.limit)
        score = metrics.get(target_metric, 0.0)

        iter_result = {
            "iteration": 0,
            "target_metric": target_metric,
            "score": score,
            "metrics": metrics,
            "prompt_file": "iter_000.txt",
            "change_summary": "seed prompt (SCoT baseline)",
            "agent_session": metrics.get("session", ""),
        }
        history.append(iter_result)
        _save_iteration(out_dir, 0, metrics, predictions, logs_path)

        best_score = score
        print(f"\nSeed {target_metric}: {score:.2f}")
        print(f"Full metrics: {json.dumps(metrics, indent=2)}")

        start_iteration = 1

    if args.dry_run:
        print("Dry run, exiting.")
        return

    # Optimization loop
    for iteration in range(start_iteration, max_iterations + 1):
        print()
        print("=" * 60)
        print(f"Iteration {iteration}/{max_iterations} (best={best_score:.2f} @ iter {best_iteration}, patience={no_improve_count}/{patience})")
        print("=" * 60)

        # Build workspace for agentic optimizer
        agent_base = base_dir.parent / agent_dir_name
        optimizer_workspace = None
        if config.get("agentic", {}).get("enabled", False):
            optimizer_workspace = build_workspace(
                agent_base, f"opt_{session_id}_iter{iteration}", iteration, config,
                keep_labels=True,
            )

        prompt_file = f"iter_{iteration:03d}.txt"

        if iteration == reuse_prompt_iteration and (out_dir / prompt_file).exists():
            # Finishing a partially-evaluated iteration: keep its prompt so the
            # already-completed samples stay valid. Don't call the optimizer.
            new_prompt = (out_dir / prompt_file).read_text()
            print(f"Reusing existing prompt {prompt_file} (completing a partial run).")
            reuse_prompt_iteration = None
        else:
            # Propose new prompt
            print("Calling optimizer...")
            try:
                new_prompt = propose_new_prompt(
                    current_prompt, metrics, predictions, val_samples, config, history,
                    workspace=optimizer_workspace,
                )
            except Exception as e:
                print(f"Optimizer failed: {e}")
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Patience exhausted after {iteration} iterations.")
                    break
                continue

            # Save proposed prompt
            (out_dir / prompt_file).write_text(new_prompt)

        # Evaluate new prompt through claude-cli-agent
        print(f"Evaluating iteration {iteration} prompt...")
        new_metrics, new_predictions, new_logs_path = evaluate_prompt(
            new_prompt, session_id, iteration, config, config_path=config_path, ground_truth=gt, limit=args.limit,
        )
        new_score = new_metrics.get(target_metric, 0.0)

        _save_iteration(out_dir, iteration, new_metrics, new_predictions, new_logs_path)

        # Compare
        improved = new_score > best_score
        delta = new_score - score
        print(f"\n{target_metric}: {score:.2f} -> {new_score:.2f} (delta={delta:+.2f})")

        if improved:
            print(f"  NEW BEST (was {best_score:.2f} @ iter {best_iteration})")
            best_score = new_score
            best_prompt = new_prompt
            best_iteration = iteration
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  No improvement (patience {no_improve_count}/{patience})")

        # Update state for next iteration
        current_prompt = new_prompt
        metrics = new_metrics
        predictions = new_predictions
        score = new_score

        change_summary = f"{'improved' if improved else 'no improvement'}, {target_metric} {new_score:.2f}"
        history.append({
            "iteration": iteration,
            "target_metric": target_metric,
            "score": new_score,
            "metrics": new_metrics,
            "prompt_file": prompt_file,
            "change_summary": change_summary,
            "agent_session": new_metrics.get("session", ""),
        })

        # Save history after each iteration
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if no_improve_count >= patience:
            print(f"\nPatience exhausted after {iteration} iterations.")
            break

    # Save best prompt
    (out_dir / "best_prompt.txt").write_text(best_prompt)
    # Resumed sessions may predate joint_top1_rate; score it from saved predictions.
    if _backfill_joint_metrics(history, out_dir, gt):
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    frontier = _pareto_frontier(history)
    harness_paths = _write_harnesses(out_dir, session_id, frontier, target_metric)
    print()
    print("=" * 60)
    print(f"Optimization complete. Best {target_metric}: {best_score:.2f} (iteration {best_iteration})")
    print(f"Best prompt saved to: {out_dir / 'best_prompt.txt'}")
    print(f"Pareto frontier ({len(frontier)} iteration(s), pairwise_accuracy vs joint_top1_rate): "
          f"{[h['iteration'] for h in frontier]}")
    for p in harness_paths:
        print(f"  harness: {p}")

    # Save final summary
    summary = {
        "session_id": session_id,
        "seed_prompt": str(config["optimization"]["seed_prompt"]),
        "target_metric": target_metric,
        "iterations_run": len(history) - 1,
        "best_iteration": best_iteration,
        "best_val_score": best_score,
        "pareto_frontier_iterations": [h["iteration"] for h in frontier],
        "pareto_objectives": list(PARETO_KEYS),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {out_dir / 'summary.json'}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Concatenated JSON objects; agent-written files may span several lines each."""
    txt, dec, i, out = path.read_text(), json.JSONDecoder(), 0, []
    while True:
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            return out
        r, i = dec.raw_decode(txt, i)
        out.append(r)


def _dedupe_predictions(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One record per sample: retries may append, so keep the latest with a verdict."""
    by_index: Dict[Any, Dict[str, Any]] = {}
    for r in preds:
        prev = by_index.get(r.get("index"))
        if prev is None or r.get("vulnerable") is not None or prev.get("vulnerable") is None:
            by_index[r.get("index")] = r
    return list(by_index.values())


def _prediction_candidates(out_dir: Path, session_id: str, iteration: int) -> List[Path]:
    """Every saved copy of one iteration's validation predictions: the session's
    iter_NNN/, the agent's raw opt_<id>_iter<N>/ run, and both again inside
    ../.complete-backups, ../.incomplete-backups and ../../results (as evaluation/metrics_paper.py)."""
    code_root = Path(__file__).resolve().parent.parent
    roots = [code_root] + [code_root.parent / kind / session_id
                           for kind in (".complete-backups", ".incomplete-backups")] + [code_root.parent.parent / "results" / session_id]
    rel_saved = out_dir.relative_to(code_root) / f"iter_{iteration:03d}" / "predictions.jsonl"
    paths = []
    for root in roots:
        paths.append(root / rel_saved)
        paths += sorted(root.glob(f"*-agent/outputs/opt_{session_id}_iter{iteration}/*/predictions.jsonl"))
    return [p for p in paths if p.exists()]


def _backfill_joint_metrics(history: List[Dict[str, Any]], out_dir: Path,
                            gt: List[Dict[str, Any]]) -> int:
    """Add joint_top1_* to history entries that lack it, scored from the saved
    predictions (the copy with the most verdicts). Returns how many were filled in."""
    filled = 0
    for h in history:
        if "joint_top1_rate" in h["metrics"]:
            continue
        copies = [_dedupe_predictions(_load_jsonl(p))
                  for p in _prediction_candidates(out_dir, out_dir.name, h["iteration"])]
        if not copies:
            print(f"warning: no saved predictions for iteration {h['iteration']}; joint_top1_rate left unset")
            continue
        preds = max(copies, key=lambda c: sum(r.get("vulnerable") is not None for r in c))
        h["metrics"].update(joint_top1_metrics(preds, gt))
        filled += 1
    return filled


def _finish_status(history: List[Dict[str, Any]], max_iterations: int, patience: int) -> str:
    """Empty string if the discovery loop is finished, else why not. Parse
    failures don't block: they are scored as wrong, like everywhere else."""
    last = history[-1]
    if last["iteration"] >= max_iterations:
        return ""
    # Consecutive non-improvements at the tail, as in _load_resume_state.
    no_improve = 0
    for h in reversed(history):
        if h["iteration"] == 0 or "no improvement" not in h.get("change_summary", ""):
            break
        no_improve += 1
    if no_improve >= patience:
        return ""
    return f"iteration {last['iteration']}/{max_iterations}, patience {no_improve}/{patience}"


def finalize_session(out_dir: Path, session_id: str, gt: List[Dict[str, Any]],
                     max_iterations: int, patience: int, target_metric: str,
                     config: Dict[str, Any], dry_run: bool = False) -> int:
    """Rebuild a finished session's Pareto frontier and harnesses from what is
    saved on disk. Never calls a model. Returns a process exit code."""
    history_path = out_dir / "history.json"
    if not history_path.exists():
        print(f"[finalize] {session_id}: no history.json in {out_dir}")
        return 2
    history = json.loads(history_path.read_text())
    if not history:
        print(f"[finalize] {session_id}: empty history")
        return 2
    # summary.json is written only once the loop ends, so it also means finished.
    status = "" if (out_dir / "summary.json").exists() else _finish_status(history, max_iterations, patience)
    if status:
        print(f"[finalize] {session_id}: UNFINISHED ({status})")
        return 3
    failed = {h["iteration"]: h["metrics"].get("parse_failures", 0) for h in history}
    failed = {i: n for i, n in failed.items() if n}
    if failed:
        print(f"[finalize] {session_id}: note, parse failures (scored as wrong) in iterations {failed}")

    filled = _backfill_joint_metrics(history, out_dir, gt)
    missing = [h["iteration"] for h in history if "joint_top1_rate" not in h["metrics"]]
    if missing:
        print(f"[finalize] {session_id}: MISSING predictions for iterations {missing}; "
              f"frontier not written")
        return 4
    frontier = _pareto_frontier(history)
    print(f"[finalize] {session_id}: finished, {len(history)} iteration(s), "
          f"joint_top1_rate backfilled for {filled}")
    for h in frontier:
        m = h["metrics"]
        print(f"  frontier iter {h['iteration']:3d}: pairwise {m.get('pairwise_accuracy', 0.0):.1f}, "
              f"joint1 {m.get('joint_top1_rate', 0.0):.1f}")
    print(f"FRONTIER {session_id} {','.join(str(h['iteration']) for h in frontier)}")
    if dry_run:
        print("[finalize] dry run, nothing written")
        return 0

    history_path.write_text(json.dumps(history, indent=2))

    # Keep harnesses from the old loc_f1 frontier, but out of the way.
    harness_dir = Path(__file__).resolve().parent.parent / "harness" / session_id
    stale = sorted(harness_dir.glob("*_locf1*.py")) if harness_dir.exists() else []
    if stale:
        legacy = harness_dir / "legacy_locf1"
        legacy.mkdir(exist_ok=True)
        for p in stale:
            p.rename(legacy / p.name)
        print(f"  moved {len(stale)} loc_f1 harness(es) to {legacy}")
    for p in _write_harnesses(out_dir, session_id, frontier, target_metric):
        print(f"  harness: {p}")

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        scored = [h for h in history if "score" in h]
        best = max(scored, key=lambda h: h["score"])  # first max, as the loop keeps it
        summary = {
            "session_id": session_id,
            "seed_prompt": str(config["optimization"]["seed_prompt"]),
            "target_metric": target_metric,
            "iterations_run": len(history) - 1,
            "best_iteration": best["iteration"],
            "best_val_score": best["score"],
        }
    summary["pareto_frontier_iterations"] = [h["iteration"] for h in frontier]
    summary["pareto_objectives"] = list(PARETO_KEYS)
    summary["history"] = history
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  summary: {summary_path}")
    return 0


def _pareto_frontier(history: List[Dict[str, Any]],
                      keys=PARETO_KEYS) -> List[Dict[str, Any]]:
    """Return the history entries not dominated on (keys...), all maximized.

    An entry is dominated if another entry is >= on every key and > on at
    least one. Ties on all keys keep only the first (lowest iteration).
    """
    def values(h):
        return tuple(h["metrics"].get(k, 0.0) for k in keys)

    frontier = []
    seen_values = set()
    for h in history:
        v = values(h)
        dominated = any(
            all(ov >= hv for ov, hv in zip(values(other), v)) and values(other) != v
            for other in history if other is not h
        )
        if not dominated and v not in seen_values:
            frontier.append(h)
            seen_values.add(v)
    frontier.sort(key=lambda h: h["iteration"])
    return frontier


def _write_harnesses(out_dir: Path, session_id: str,
                      frontier: List[Dict[str, Any]], target_metric: str) -> List[Path]:
    """Bake each Pareto-frontier iteration's prompt into its own harness.py."""
    harness_dir = Path(__file__).resolve().parent.parent / "harness" / session_id
    harness_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for h in frontier:
        prompt_path = out_dir / h["prompt_file"]
        prompt = prompt_path.read_text()
        pairwise = h["metrics"].get("pairwise_accuracy", 0.0)
        joint1 = h["metrics"].get("joint_top1_rate", 0.0)
        name = f"iter{h['iteration']:03d}_pairwise{pairwise:.1f}_joint1{joint1:.1f}.py"
        paths.append(_write_harness(
            harness_dir / name, prompt, session_id, h["iteration"], target_metric,
            h.get("score", 0.0),
        ))
    return paths


def _write_harness(harness_path: Path, prompt: str, session_id: str,
                    iteration: int, target_metric: str, score: float) -> Path:
    """Bake a prompt into a standalone harness.py any agent can import.

    Uses harness/default/harness.py as the base source (it's already a
    complete, runnable harness) and swaps in the new prompt.
    """
    default_path = Path(__file__).resolve().parent.parent / "harness" / "default" / "harness.py"
    base_src = default_path.read_text()
    # Use a replacement function, not a string: re.sub treats backslash
    # sequences (e.g. the \n produced by repr()) in a string replacement as
    # escapes/backreferences, which would corrupt the embedded prompt.
    harness_src, n = re.subn(
        r'(?m)^PROMPT_TEMPLATE = .*$', lambda m: f'PROMPT_TEMPLATE = {prompt!r}', base_src, count=1,
    )
    if n != 1:
        raise RuntimeError(f"Could not find PROMPT_TEMPLATE assignment in {default_path}")
    harness_path.write_text(harness_src)
    return harness_path


def _save_iteration(out_dir: Path, iteration, metrics, predictions, logs_path=None):
    """Save per-iteration metrics, predictions, and logs."""
    tag = f"iter_{iteration:03d}" if isinstance(iteration, int) else iteration
    iter_dir = out_dir / tag
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (iter_dir / "predictions.jsonl").open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    if logs_path:
        shutil.copy2(logs_path, iter_dir / "logs.jsonl")


if __name__ == "__main__":
    main()
