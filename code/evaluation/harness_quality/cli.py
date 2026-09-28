"""Command line interface for harness-quality scorecards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import HarnessQualityConfig, load_config
from .judge_client import run_judge
from .static_analyzer import analyze_prompt


STATIC_DIAGNOSTIC_FIELDS = (
    "characters", "utf8_bytes", "lines", "blank_lines", "words",
    "token_delta_vs_seed", "token_ratio_vs_seed", "tokenizer", "template_variables",
    "atx_heading_count", "colon_label_count", "unordered_bullet_count",
    "numbered_list_item_count", "maximum_line_length",
    "trailing_whitespace_line_numbers", "tab_containing_line_numbers",
    "disallowed_control_character_positions",
    "maximum_consecutive_blank_line_run", "markdown_fence_count",
    "undefined_acronym_candidates",
    "undefined_acronym_candidate_count", "strict_json_valid",
    "pseudo_schema_parse_valid", "schema_key_count", "schema_container_key_count",
    "schema_leaf_field_count", "schema_field_paths", "schema_max_depth",
    "schema_array_count", "schema_start_offset",
    "schema_end_offset", "findings",
)

SEMANTIC_DIAGNOSTIC_FIELDS = (
    "judge_runs", "semantic_stability", "absolute_total_score_difference",
    "per_dimension_rating_differences", "global_ambiguities",
)


def _disabled_semantic_result() -> dict[str, Any]:
    return {
        "semantic_eligible": False, "semantic_quality_score": None,
        "semantic_stability": "disabled", "judge_runs": [], "aggregated_ratings": {},
        "critical_failures": [], "global_ambiguities": [],
        "absolute_total_score_difference": None, "per_dimension_rating_differences": {},
    }


def _static_eligibility_failures(static: dict[str, Any]) -> list[str]:
    """Describe the existing static gates without changing their behavior."""
    checks = (
        (not static["characters"], "empty prompt"),
        ("invalid UTF-8 input" in static["findings"], "invalid UTF-8 input"),
        (static["code_placeholder_count"] != 1, "exactly one CODE placeholder is required"),
        (bool(static["unknown_template_variables"]), "unknown template variables"),
        (bool(static["malformed_template_variables"]), "malformed template variables"),
        (not static["schema_block_found"], "missing schema block"),
        (static["unbalanced_schema_delimiters"], "unbalanced schema delimiters"),
        (bool(static["missing_required_leaf_paths"]), "missing required schema paths"),
        (bool(static["disallowed_control_character_positions"]), "disallowed control characters"),
    )
    return [message for failed, message in checks if failed]


def _build_scorecard(path: Path, static: dict[str, Any], semantic: dict[str, Any], include_diagnostics: bool) -> dict[str, Any]:
    valid_judge_run_count = sum(1 for run in semantic["judge_runs"] if run.get("valid"))
    card: dict[str, Any] = {
        "scorecard_version": "2.0",
        "prompt": {"path": str(path)},
        "scoring": {
            "template_integrity": static["components"]["template_integrity"],
            "schema_mechanics": static["components"]["schema_mechanics"],
            "format_hygiene": static["components"]["format_hygiene"],
            "token_efficiency": static["components"]["token_efficiency"],
            "duplicate_schema_paths": static["duplicate_schema_paths"],
            "duplicate_normalized_headings": static["duplicate_normalized_headings"],
            "duplicate_normalized_directive_lines": static["duplicate_normalized_directive_lines"],
            "mixed_line_endings": static["mixed_line_endings"],
            "markdown_fence_balance": static["markdown_fence_balance"],
            "reference_tokens": static["reference_tokens"],
            "seed_reference_tokens": static["seed_reference_tokens"],
            "quantitative_static_score": static["quantitative_static_score"],
            "semantic_dimension_ratings": semantic["aggregated_ratings"],
            "semantic_quality_score": semantic["semantic_quality_score"],
        },
        "eligibility": {
            "static_eligible": static["static_eligible"],
            "static_eligibility_failures": _static_eligibility_failures(static),
            "code_placeholder_count": static["code_placeholder_count"],
            "unknown_template_variables": static["unknown_template_variables"],
            "malformed_template_variables": static["malformed_template_variables"],
            "schema_block_found": static["schema_block_found"],
            "unbalanced_schema_delimiters": static["unbalanced_schema_delimiters"],
            "missing_required_leaf_paths": static["missing_required_leaf_paths"],
            "semantic_eligible": semantic["semantic_eligible"],
            "critical_failures": semantic["critical_failures"],
            "valid_judge_run_count": valid_judge_run_count,
        },
    }
    if include_diagnostics:
        card["diagnostics"] = {
            "prompt": {"raw_sha256": static["raw_sha256"], "normalized_sha256": static["normalized_sha256"]},
            "static": {field: static[field] for field in STATIC_DIAGNOSTIC_FIELDS},
            "semantic": {field: semantic[field] for field in SEMANTIC_DIAGNOSTIC_FIELDS},
        }
    return card


def scorecard(path: Path, seed: Path | None, config_path: str | Path | HarnessQualityConfig | None, *, include_diagnostics: bool = False) -> dict[str, Any]:
    """Return scoring and eligibility results, with opt-in diagnostics."""
    cfg = config_path if isinstance(config_path, HarnessQualityConfig) else load_config(config_path)
    prompt = path.read_bytes()
    seed_data = seed.read_bytes() if seed else None
    static = analyze_prompt(prompt, seed_data, cfg.static)
    rubric = Path(__file__).with_name("judge_prompt.txt").read_text(encoding="utf-8")
    semantic = run_judge(prompt.decode("utf-8", "replace"), rubric, cfg.judge) if cfg.judge.enabled else _disabled_semantic_result()
    return _build_scorecard(path, static, semantic, include_diagnostics)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _add_diagnostics_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-diagnostics", action="store_true", help="Include separate diagnostic measurements; they never affect scores or eligibility.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("score")
    one.add_argument("--prompt", required=True)
    one.add_argument("--seed")
    one.add_argument("--config")
    one.add_argument("--output", required=True)
    _add_diagnostics_flag(one)
    session = sub.add_parser("score-session")
    session.add_argument("--session-dir", required=True)
    session.add_argument("--seed")
    session.add_argument("--config")
    _add_diagnostics_flag(session)
    args = parser.parse_args(argv)
    if args.command == "score":
        _write(Path(args.output), scorecard(Path(args.prompt), Path(args.seed) if args.seed else None, args.config, include_diagnostics=args.include_diagnostics))
        return 0

    root, seed, rows, previous = Path(args.session_dir), Path(args.seed) if args.seed else None, [], None
    for prompt in sorted(root.glob("iter_*.txt")):
        card = scorecard(prompt, seed, args.config, include_diagnostics=args.include_diagnostics)
        out = prompt.with_suffix("") / "harness_quality.json" if prompt.with_suffix("").is_dir() else root / f"{prompt.stem}_harness_quality.json"
        _write(out, card)
        row = {"path": str(prompt), "scorecard": card, "delta_vs_previous": None, "delta_vs_seed": None}
        current = card["scoring"]["quantitative_static_score"]
        if previous is not None:
            row["delta_vs_previous"] = current - previous if current is not None else None
        if seed is not None and prompt != seed:
            baseline = scorecard(seed, seed, args.config)["scoring"]["quantitative_static_score"]
            row["delta_vs_seed"] = current - baseline if current is not None and baseline is not None else None
        rows.append(row)
        previous = current
    (root / "harness_quality_history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
