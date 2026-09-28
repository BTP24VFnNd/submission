"""Static and semantic scoring math."""
from __future__ import annotations
from statistics import mean
from typing import Any
from .models import DIMENSION_WEIGHTS, JudgeRun, weighted_score

def aggregate_judge_runs(runs: list[JudgeRun]) -> dict[str, Any]:
    valid = [r for r in runs if r.valid]
    if not valid:
        return {"judge_runs": [r.to_dict() for r in runs], "aggregated_ratings": {}, "semantic_quality_score": None,
                "semantic_stability": "unstable", "absolute_total_score_difference": None,
                "per_dimension_rating_differences": {}, "critical_failures": [], "global_ambiguities": []}
    ratings = {n: mean(r.ratings[n] for r in valid) for n in DIMENSION_WEIGHTS}
    scores = [weighted_score(r.ratings) for r in valid]
    diffs = {n: abs(valid[0].ratings[n] - valid[1].ratings[n]) for n in DIMENSION_WEIGHTS} if len(valid) >= 2 else {}
    total_diff = abs(scores[0] - scores[1]) if len(scores) >= 2 else None
    return {"judge_runs": [r.to_dict() for r in runs], "aggregated_ratings": ratings,
        "semantic_quality_score": weighted_score(ratings), "semantic_stability": "unstable" if (total_diff is not None and total_diff > 6) or any(v > 1 for v in diffs.values()) else "stable",
        "absolute_total_score_difference": total_diff, "per_dimension_rating_differences": diffs,
        "critical_failures": sorted({x for r in valid for x in r.critical_failures}),
        "global_ambiguities": sorted({x for r in valid for x in r.global_ambiguities})}

def semantic_eligibility(aggregation: dict[str, Any]) -> bool:
    ratings = aggregation.get("aggregated_ratings", {})
    return not aggregation.get("critical_failures") and all(ratings.get(n, -1) >= 2 for n in
        ("task_identity_primary_task", "required_analysis_mode_scot", "scope_grounding", "output_contract_semantics"))
