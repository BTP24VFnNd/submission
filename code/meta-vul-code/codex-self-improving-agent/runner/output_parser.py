#!/usr/bin/env python3
"""Extract and validate the JSON verdict from raw model output."""
import json
import re
from typing import Any, Dict, Optional

REQUIRED_VERDICT_KEYS = {"vulnerable", "confidence", "cwe", "severity", "summary"}
REQUIRED_TOP_KEYS = {"verdict", "evidence", "notes"}


def _strip_eos_tokens(raw: str) -> str:
    """Strip common EOS/chat tokens that models append."""
    for tok in ("<|endoftext|>", "<|end|>", "<|im_end|>", "<|im_start|>"):
        raw = raw.replace(tok, "")
    return raw


def _sanitize_control_chars(raw: str) -> str:
    """Escape unescaped control characters (tabs, etc.) inside JSON strings."""
    # Replace control chars (except \n \r) that break strict JSON parsing
    def _replace(m):
        c = m.group(0)
        return {
            '\t': '\\t',
            '\b': '\\b',
            '\f': '\\f',
        }.get(c, f'\\u{ord(c):04x}')
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', _replace, raw)


def _find_verdict_object(text: str, strict: bool = True) -> Optional[Dict[str, Any]]:
    """
    Scan text for the first JSON object containing a "verdict" key
    with a boolean "vulnerable" value.
    """
    decoder = json.JSONDecoder(strict=strict)
    for m in re.finditer(r'\{', text):
        start = m.start()
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "verdict" not in obj:
            continue
        v = obj["verdict"]
        # Skip schema examples (e.g. "low|medium|high")
        if isinstance(v, str) and "|" in v:
            continue
        if isinstance(v, dict) and isinstance(v.get("vulnerable"), bool):
            return obj
    return None


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Find the first valid JSON object in the raw output that contains
    a "verdict" key with a boolean "vulnerable" value.

    Tries multiple strategies in order:
    1. Strict JSON parsing on cleaned output
    2. Non-strict JSON parsing (allows control chars like literal tabs)
    3. Sanitize control chars then strict parse
    4. Strip markdown code fences then retry
    """
    cleaned = _strip_eos_tokens(raw)

    # 1. Strict parse (original behavior)
    result = _find_verdict_object(cleaned, strict=True)
    if result:
        return result

    # 2. Non-strict parse (tolerates literal control chars in strings)
    result = _find_verdict_object(cleaned, strict=False)
    if result:
        return result

    # 3. Sanitize control chars then strict parse
    sanitized = _sanitize_control_chars(cleaned)
    result = _find_verdict_object(sanitized, strict=True)
    if result:
        return result

    # 4. Strip markdown code fences and retry
    stripped = re.sub(r'```(?:json)?\s*', '', cleaned)
    result = _find_verdict_object(stripped, strict=False)
    if result:
        return result

    # 5. Direct verdict sub-object extraction (when outer JSON is broken
    #    but the inner verdict dict is valid)
    result = _extract_verdict_direct(cleaned)
    if result:
        return result

    return None


def _extract_verdict_direct(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the verdict sub-object directly by finding "verdict": { and
    brace-matching. Useful when the outer JSON is malformed (e.g. code
    snippets with unescaped characters) but the verdict dict itself is clean.
    """
    for m in re.finditer(r'"verdict"\s*:\s*\{', text):
        vstart = m.end() - 1  # position of the opening {
        depth = 0
        for i in range(vstart, min(vstart + 5000, len(text))):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            if depth == 0:
                verdict_str = text[vstart:i + 1]
                for strict in (True, False):
                    try:
                        obj = json.loads(verdict_str, strict=strict)
                        if isinstance(obj, dict) and isinstance(obj.get("vulnerable"), bool):
                            return {"verdict": obj}
                    except json.JSONDecodeError:
                        continue
                break
    return None


def validate_schema(obj: Dict[str, Any]) -> list:
    """Return a list of schema violations (empty = valid)."""
    errors = []
    for k in REQUIRED_TOP_KEYS:
        if k not in obj:
            errors.append(f"missing top-level key: {k}")
    verdict = obj.get("verdict", {})
    if isinstance(verdict, dict):
        for k in REQUIRED_VERDICT_KEYS:
            if k not in verdict:
                errors.append(f"missing verdict key: {k}")
        if "vulnerable" in verdict and not isinstance(verdict["vulnerable"], bool):
            errors.append(f"verdict.vulnerable should be bool, got {type(verdict['vulnerable']).__name__}")
    else:
        errors.append(f"verdict should be dict, got {type(verdict).__name__}")
    return errors


def parse_output(raw: str) -> Dict[str, Any]:
    """
    Parse raw model output. Returns dict with:
      parsed: dict | None   - the extracted JSON object
      errors: list[str]     - parse/validation errors
      vulnerable: bool | None
      confidence: float | None
    """
    obj = extract_json(raw)
    if obj is None:
        return {
            "parsed": None,
            "errors": ["no valid JSON verdict found in output"],
            "vulnerable": None,
            "confidence": None,
        }
    schema_errors = validate_schema(obj)
    verdict = obj.get("verdict", {})
    return {
        "parsed": obj,
        "errors": schema_errors,
        "vulnerable": verdict.get("vulnerable") if isinstance(verdict, dict) else None,
        "confidence": verdict.get("confidence") if isinstance(verdict, dict) else None,
    }
