"""Provider-neutral judge clients."""
from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from .config import JudgeConfig
from .judge_validator import validate_judge_response
from .models import JudgeRun
from .scoring import aggregate_judge_runs, semantic_eligibility

class JudgeClient(Protocol):
    def judge(self, judge_prompt: str, run_number: int = 0) -> tuple[str, dict[str, Any]]: ...

class DisabledJudgeClient:
    def judge(self, judge_prompt: str, run_number: int = 0) -> tuple[str, dict[str, Any]]:
        return "", {"provider": "disabled", "error": "semantic judge disabled"}

class SubprocessJudgeClient:
    def __init__(self, command: list[str] | tuple[str, ...], config: JudgeConfig): self.command, self.config = list(command), config
    def judge(self, judge_prompt: str, run_number: int = 0) -> tuple[str, dict[str, Any]]:
        try:
            result = subprocess.run(self.command, input=judge_prompt, text=True, capture_output=True, timeout=self.config.timeout_seconds, check=False)
            return result.stdout, {"provider": self.config.provider or "subprocess", "model": self.config.model, "temperature": self.config.temperature, "returncode": result.returncode, "error": result.stderr.strip() if result.returncode else ""}
        except Exception as exc: return "", {"provider": self.config.provider or "subprocess", "error": str(exc)}

class ResponseFileJudgeClient:
    def __init__(self, response_files: list[str] | tuple[str, ...]): self.response_files = list(response_files)
    def judge(self, judge_prompt: str, run_number: int = 0) -> tuple[str, dict[str, Any]]:
        if not self.response_files: return "", {"provider": "response-file", "error": "no response file configured"}
        path = Path(self.response_files[min(run_number, len(self.response_files) - 1)])
        try: return path.read_text(encoding="utf-8"), {"provider": "response-file", "path": str(path)}
        except OSError as exc: return "", {"provider": "response-file", "error": str(exc)}

def build_judge_client(config: JudgeConfig) -> JudgeClient:
    if config.response_files: return ResponseFileJudgeClient(config.response_files)
    if config.command: return SubprocessJudgeClient(config.command, config)
    return DisabledJudgeClient()

def run_judge(prompt: str, rubric: str, config: JudgeConfig, client: JudgeClient | None = None) -> dict[str, Any]:
    client = client or build_judge_client(config); runs = []
    for number in range(config.repeats):
        rendered = rubric.replace("{{PROMPT_TO_SCORE}}", prompt)
        raw, metadata = client.judge(rendered, number)
        try: run = validate_judge_response(json.loads(raw), prompt)
        except (json.JSONDecodeError, TypeError) as exc: run = JudgeRun(False, error=f"invalid judge JSON: {exc}")
        run.metadata.update(metadata); run.metadata.update({"judge_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "rubric_version": config.rubric_version, "temperature": config.temperature, "invocation_timestamp": datetime.now(timezone.utc).isoformat()})
        runs.append(run)
    result = aggregate_judge_runs(runs); result["semantic_eligible"] = semantic_eligibility(result); return result
