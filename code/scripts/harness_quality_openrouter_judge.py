#!/usr/bin/env python3
"""Run an OpenRouter model as the harness-quality semantic judge.

Same contract as harness_quality_codex_judge.py: the rendered rubric arrives on
stdin and exactly one JSON object goes to stdout. Needs OPENROUTER_API_KEY; the
model comes from HARNESS_QUALITY_OPENROUTER_MODEL (default openai/gpt-oss-120b).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _json_object(text: str) -> dict:
    """Parse a JSON object, tolerating a markdown fence or text around it."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0]
    start, end = candidate.find("{"), candidate.rfind("}")
    value = json.loads(candidate[start:end + 1] if start != -1 else candidate)
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def main() -> int:
    rubric = sys.stdin.read()
    if not rubric.strip():
        print("judge rubric is empty", file=sys.stderr)
        return 2
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    body = json.dumps({
        "model": os.environ.get("HARNESS_QUALITY_OPENROUTER_MODEL", "openai/gpt-oss-120b"),
        "messages": [{"role": "user", "content": rubric}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=280) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"] or ""
            print(json.dumps(_json_object(content), separators=(",", ":")))
            return 0
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as exc:
            error = exc
            time.sleep(5 * (attempt + 1))
    print(f"OpenRouter judge failed: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
