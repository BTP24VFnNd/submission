#!/usr/bin/env python3
"""Run the configured Codex CLI as a harness-quality semantic judge.

The harness-quality evaluator sends the rendered rubric on stdin and expects
exactly one JSON object on stdout. Codex CLI writes its final response to a
temporary file; this adapter emits only that response after validating it is
JSON, keeping CLI progress and diagnostics off stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _json_object(text: str) -> object:
    """Parse a JSON object, tolerating a markdown fence from the model."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def main() -> int:
    rubric = sys.stdin.read()
    if not rubric.strip():
        print("judge rubric is empty", file=sys.stderr)
        return 2

    codex_bin = os.environ.get("HARNESS_QUALITY_CODEX_BIN", "codex")
    model = os.environ.get("HARNESS_QUALITY_CODEX_MODEL", "gpt-5.4")
    with tempfile.TemporaryDirectory(prefix="harness_quality_codex_") as temp_dir:
        output_path = Path(temp_dir) / "last_message.json"
        command = [
            codex_bin,
            "exec",
            "-",
            "--model",
            model,
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "-c",
            "model_reasoning_effort=low",
            "-c",
            "personality=none",
            "--output-last-message",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            input=rubric,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            print(result.stderr.strip() or f"codex exited with status {result.returncode}", file=sys.stderr)
            return result.returncode or 1
        try:
            value = _json_object(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"invalid Codex judge response: {error}", file=sys.stderr)
            return 1

    print(json.dumps(value, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
