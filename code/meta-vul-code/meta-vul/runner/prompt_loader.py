#!/usr/bin/env python3
"""Load prompt templates and inject code."""
from pathlib import Path


def load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def render(template: str, code: str) -> str:
    return template.replace("{{CODE}}", code)
