"""Strict local validation of semantic judge output."""
from __future__ import annotations
from typing import Any
from .models import DIMENSION_WEIGHTS, JudgeRun, weighted_score

def validate_judge_response(payload: Any, prompt: str) -> JudgeRun:
    if not isinstance(payload, dict): return JudgeRun(False, error="judge response must be a JSON object")
    ratings = payload.get("ratings")
    if not isinstance(ratings, dict): return JudgeRun(False, error="missing ratings object")
    missing = [n for n in DIMENSION_WEIGHTS if n not in ratings]
    if missing: return JudgeRun(False, error=f"missing judge dimensions: {', '.join(missing)}")
    clean = {}
    for name in DIMENSION_WEIGHTS:
        value = ratings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 4:
            return JudgeRun(False, error=f"rating outside 0-4: {name}")
        clean[name] = float(value)
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list): return JudgeRun(False, error="evidence must be an array")
    for item in evidence:
        if not isinstance(item, dict): return JudgeRun(False, error="evidence entries must be objects")
        quote = item.get("quote", item.get("quotation", item.get("text")))
        if quote is not None and (not isinstance(quote, str) or prompt.count(quote) != 1):
            return JudgeRun(False, error="fabricated or non-unique judge evidence quotation")
    failures, ambiguities = payload.get("critical_failures", []), payload.get("global_ambiguities", [])
    if not isinstance(failures, list) or not all(isinstance(x, str) for x in failures): return JudgeRun(False, error="critical_failures must be strings")
    if not isinstance(ambiguities, list) or not all(isinstance(x, str) for x in ambiguities): return JudgeRun(False, error="global_ambiguities must be strings")
    return JudgeRun(True, clean, weighted_score(clean), failures, ambiguities, evidence)
