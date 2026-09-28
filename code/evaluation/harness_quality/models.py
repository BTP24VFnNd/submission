"""Serializable models and shared semantic dimension weights."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

DIMENSION_WEIGHTS = {"role_alignment": 7, "task_identity_primary_task": 13, "input_object_boundary": 10,
    "required_analysis_mode_scot": 17, "scope_grounding": 12, "prohibited_behavior": 10,
    "output_contract_semantics": 16, "internal_consistency_precedence": 10, "organization_maintainability": 5}

@dataclass
class JudgeRun:
    valid: bool
    ratings: dict[str, float] = field(default_factory=dict)
    weighted_score: float | None = None
    critical_failures: list[str] = field(default_factory=list)
    global_ambiguities: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def weighted_score(ratings: dict[str, float]) -> float:
    return sum(DIMENSION_WEIGHTS[name] * float(ratings[name]) / 4.0 for name in DIMENSION_WEIGHTS)
