#!/usr/bin/env python3
"""Load prompt templates and inject code."""
from pathlib import Path


def load_template(prompts_dir: Path, policy: str) -> str:
    path = prompts_dir / f"{policy}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def render(template: str, code: str) -> str:
    return template.replace("{{CODE}}", code)
