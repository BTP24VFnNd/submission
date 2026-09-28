"""Configuration for harness-quality evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import yaml

DEFAULT_ALLOWED_VARIABLES = ("CODE",)
DEFAULT_ACRONYM_ALLOWLIST = ("API", "APIs", "authn", "authz", "CWE", "JSON", "SCoT")

@dataclass(frozen=True)
class StaticConfig:
    allowed_variables: tuple[str, ...] = DEFAULT_ALLOWED_VARIABLES
    acronym_allowlist: tuple[str, ...] = DEFAULT_ACRONYM_ALLOWLIST
    tokenizer_encoding: str = "cl100k_base"
    tokenizer_package: str = "tiktoken"

@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = False
    repeats: int = 2
    command: tuple[str, ...] = ()
    response_files: tuple[str, ...] = ()
    temperature: float = 0.0
    timeout_seconds: int = 120
    model: str | None = None
    provider: str | None = None
    rubric_version: str = "1.0"

@dataclass(frozen=True)
class HarnessQualityConfig:
    enabled: bool = False
    seed_prompt: str | None = None
    output_filename: str = "harness_quality.json"
    static: StaticConfig = field(default_factory=StaticConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

def _tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None: return default
    if isinstance(value, str): return (value,)
    return tuple(str(item) for item in value)

def from_mapping(mapping: Mapping[str, Any] | None) -> HarnessQualityConfig:
    raw, static, judge = dict(mapping or {}), {}, {}
    raw = dict(raw.get("harness_quality", raw)) if "harness_quality" in raw else raw
    static, judge = dict(raw.get("static", {})), dict(raw.get("judge", {}))
    return HarnessQualityConfig(
        enabled=bool(raw.get("enabled", False)), seed_prompt=raw.get("seed_prompt"),
        output_filename=str(raw.get("output_filename", "harness_quality.json")),
        static=StaticConfig(
            _tuple(static.get("allowed_variables"), DEFAULT_ALLOWED_VARIABLES),
            _tuple(static.get("acronym_allowlist"), DEFAULT_ACRONYM_ALLOWLIST),
            str(static.get("tokenizer_encoding", "cl100k_base")), str(static.get("tokenizer_package", "tiktoken"))),
        judge=JudgeConfig(
            enabled=bool(judge.get("enabled", False)), repeats=max(0, int(judge.get("repeats", 2))),
            command=_tuple(judge.get("command"), ()), response_files=_tuple(judge.get("response_files"), ()),
            temperature=float(judge.get("temperature", 0)), timeout_seconds=int(judge.get("timeout_seconds", 120)),
            model=judge.get("model"), provider=judge.get("provider"), rubric_version=str(judge.get("rubric_version", "1.0"))))

def load_config(path: str | Path | None = None) -> HarnessQualityConfig:
    if path is None: return HarnessQualityConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(data)
