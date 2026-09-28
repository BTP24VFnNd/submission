#!/usr/bin/env python3
"""Per-iteration harness quality for the paper's optimizer sessions.

Reports only what the paper uses, one row per iteration:
  - detection metrics (pairwise, F1, recall, precision, accuracy)
  - harness length in tokens (and ratio to the seed harness)
  - semantic judge score (weighted 0-100) plus the three key rubric
    dimensions: SCoT scaffold, prohibited behavior, output contract

Built on the evaluator in evaluation/harness_quality/. Metrics come from
iter_XXX/metrics.json when present, otherwise from history.json; every harness
is scored even without metrics (those cells are left blank). Judge results
are cached per harness, so an interrupted run picks up where it stopped.

Run it through evaluation/harness-quality-paper.sh, which loads nvm for the judge.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.harness_quality.config import load_config
from evaluation.harness_quality.judge_client import run_judge
from evaluation.harness_quality.static_analyzer import analyze_prompt

PAPER_SESSIONS = {
    "20260716_181608": "Opus 4.6",
    "20260909_054308": "Inkling-Small",
    "20260919_204729": "DeepSeek",
    "20260922_170038": "GLM-5.3",
}
METRIC_KEYS = ("pairwise_accuracy", "f1", "recall", "precision", "accuracy")
KEY_DIMENSIONS = {
    "required_analysis_mode_scot": "scot",
    "prohibited_behavior": "prohibited",
    "output_contract_semantics": "output_contract",
}
RUBRIC = (REPO_ROOT / "evaluation/harness_quality/judge_prompt.txt").read_text(encoding="utf-8")


def load_iterations(run_dir: Path) -> list[dict]:
    """Every iter_XXX.txt harness. Metrics are attached when available, never required."""
    history_file = run_dir / "history.json"
    history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []
    history_metrics = {h["iteration"]: h.get("metrics") or {} for h in history}
    rows = []
    for prompt in sorted(run_dir.glob("iter_[0-9][0-9][0-9].txt")):
        i = int(prompt.stem[5:])
        metrics_file = run_dir / f"iter_{i:03d}" / "metrics.json"
        metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else history_metrics.get(i, {})
        rows.append({"iteration": i, "prompt": prompt, "metrics": metrics})
    return rows


def judge_config(cfg, judge: str):
    """'codex' keeps the checked-in GPT-5.4 Codex judge; anything else is an OpenRouter model id."""
    if judge == "codex":
        return cfg.judge
    os.environ["HARNESS_QUALITY_OPENROUTER_MODEL"] = judge
    return dataclasses.replace(cfg.judge, command=("python3", "scripts/harness_quality_openrouter_judge.py"),
                               model=judge, provider="openrouter")


def judge_cached(text: str, judge_cfg, cache_dir: Path) -> dict:
    key = f"{judge_cfg.model}\n{text}"  # one cache entry per (judge model, harness)
    cache = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    result = run_judge(text, RUBRIC, judge_cfg)
    if result.get("semantic_quality_score") is not None:  # don't cache failed judge calls
        cache.write_text(json.dumps(result, indent=2))
    return result


def audit_session(run_id: str, outputs_dir: Path, cfg, use_judge: bool, cache_dir: Path) -> list[dict]:
    run_dir = outputs_dir / run_id
    if not run_dir.exists():  # finished sessions live in the sibling results/ folder
        run_dir = REPO_ROOT.parent / "results" / run_id / "meta-vul/outputs" / run_id
    iterations = load_iterations(run_dir)
    if not iterations:
        print(f"  no usable iterations in {run_dir}")
        return []
    seed_text = iterations[0]["prompt"].read_bytes()
    scored = [r for r in iterations if r["metrics"].get("pairwise_accuracy") is not None]
    best = max(scored, key=lambda r: r["metrics"]["pairwise_accuracy"])["iteration"] if scored else None
    rows = []
    for n, it in enumerate(iterations, 1):
        text = it["prompt"].read_text(encoding="utf-8")
        static = analyze_prompt(text.encode(), seed_text, cfg.static)
        row = {
            "session": run_id,
            "model": PAPER_SESSIONS.get(run_id, ""),
            "iteration": it["iteration"],
            "best_pairwise": it["iteration"] == best,
            **{k: it["metrics"].get(k) for k in METRIC_KEYS},
            "tokens": static["reference_tokens"],
            "token_ratio_vs_seed": round(static["token_ratio_vs_seed"], 2),
        }
        if use_judge:
            sem = judge_cached(text, cfg.judge, cache_dir)
            ratings = sem.get("aggregated_ratings") or {}
            score = sem.get("semantic_quality_score")
            row.update({
                "judge": cfg.judge.model,
                "judge_score": round(score, 1) if score is not None else None,
                **{short: ratings.get(dim) for dim, short in KEY_DIMENSIONS.items()},
                "judge_stability": sem.get("semantic_stability"),
                "critical_failures": "; ".join(sem.get("critical_failures") or []),
                "all_ratings": ratings,
            })
        print(f"  [{n}/{len(iterations)}] iter {it['iteration']}: pairwise={row['pairwise_accuracy']} "
              f"tokens={row['tokens']}" + (f" judge={row.get('judge_score')} ({row.get('judge_stability')})" if use_judge else ""))
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], out_dir: Path, name: str, use_judge: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    columns = [k for k in rows[0] if k != "all_ratings"]
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    judge = rows[0].get("judge") if use_judge else "none (tokens only)"
    md = ["# Harness quality across iterations", "", f"Judge: {judge}", ""]
    for session in dict.fromkeys(r["session"] for r in rows):
        md += [f"## {PAPER_SESSIONS.get(session, session)} (`{session}`)", ""]
        header = "| Iter | Pairwise | F1 | Recall | Tokens | x seed |"
        sep = "| ---: | ---: | ---: | ---: | ---: | ---: |"
        if use_judge:
            header += " Judge | SCoT | Prohibited | Output | Stability |"
            sep += " ---: | ---: | ---: | ---: | --- |"
        md += [header, sep]
        for r in (r for r in rows if r["session"] == session):
            star = " ★" if r["best_pairwise"] else ""
            line = (f"| {r['iteration']}{star} | {r['pairwise_accuracy']} | {r['f1']} | {r['recall']} | "
                    f"{r['tokens']} | {r['token_ratio_vs_seed']} |")
            if use_judge:
                line += (f" {r['judge_score']} | {r['scot']} | {r['prohibited']} | "
                         f"{r['output_contract']} | {r['judge_stability']} |")
            md.append(line)
        md.append("")
    md.append("★ = best pairwise iteration. Rubric dimensions are rated 0-4; judge score is weighted 0-100.")
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="*", default=list(PAPER_SESSIONS), help="Session IDs (default: the four paper sessions).")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "meta-vul-code/meta-vul/outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evaluation/harness_quality/results")
    parser.add_argument("--no-judge", action="store_true", help="Skip the semantic judge (tokens + metrics only).")
    parser.add_argument("--judge", default="codex",
                        help="'codex' (GPT-5.4 through the Codex CLI, default) or an OpenRouter model id, "
                             "e.g. openai/gpt-oss-120b (needs OPENROUTER_API_KEY).")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # the judge command in config.yaml uses repo-relative paths
    cfg = load_config(REPO_ROOT / "evaluation/harness_quality/config.yaml")
    use_judge = not args.no_judge
    if use_judge:
        cfg = dataclasses.replace(cfg, judge=judge_config(cfg, args.judge))
    if use_judge and args.judge == "codex" and not shutil.which("codex"):
        print('error: codex not on PATH. Check that nvm is installed, or use --no-judge.', file=sys.stderr)
        return 2
    if use_judge and args.judge != "codex" and not os.environ.get("OPENROUTER_API_KEY"):
        print("error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2
    name = "harness_quality_" + (cfg.judge.model.split("/")[-1] if use_judge else "nojudge")
    cache_dir = args.output_dir / "judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_id in args.run_ids:
        print(f"{PAPER_SESSIONS.get(run_id, run_id)} ({run_id})")
        rows += audit_session(run_id, args.outputs_dir, cfg, use_judge, cache_dir)
        if rows:
            write_outputs(rows, args.output_dir, name, use_judge)  # save after each session
    if not rows:
        return 1
    print(f"\nWrote {os.path.relpath(args.output_dir, REPO_ROOT)}/{name}.{{md,csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
