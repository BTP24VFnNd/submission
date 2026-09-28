#!/usr/bin/env python3
"""
Standalone vulnerability-detection harness.

Plug this into any agent or model backend:
    from harness import build_prompt, parse_verdict
    prompt = build_prompt(code)          # send this to your model
    verdict = parse_verdict(raw_output)  # feed the model's raw text back in
"""
import json
import re
import sys
from typing import Any, Dict, Optional

PROMPT_TEMPLATE = 'You are an expert software engineer and security analyst specializing in vulnerability detection.\n\nAnalyze the provided code for security vulnerabilities using Structured Chain-of-Thought (SCoT).\n\n## SCoT reasoning framework\nStructure your reasoning explicitly around the program\'s concrete structures and anchor every claim to the exact code (verbatim snippet + line numbers). Work through the following phases in order.\n\n### Phase 1 — Input & provenance map (Sequence)\nList every input to the analyzed code (parameters, globals, return values of called functions, library/OS data). For each, state:\n- its data type and range,\n- its provenance: attacker-controlled (trace where it originates, e.g., argv, sockets, files, user data) vs. trusted/framework-internal (name the invariant that guarantees it), and\n- any constraints already imposed on it by the shown code (checks, casts, whitelists).\n\n### Phase 2 — Structure walk\nWalk the code through its concrete structures. At every step, track the exact sizes, lengths, indices, and pointer values as they flow, and name the precise line numbers involved.\n\n**Sequence (ordered steps):** State each operation in order. For every buffer or allocation, record its exact size (from the declaration or call) and how that size changes as values flow. Identify each place a value changes representation (e.g., int to size_t, 32-bit to 64-bit, narrowing cast) and whether the conversion can change the value.\n\n**Branch (if/else):** For each conditional, state precisely what the condition guards, then evaluate BOTH outcomes (taken and not-taken). Decide whether the guard\'s bound fully covers the subsequent dangerous operation:\n- If a guard exists but its bound is insufficient, give the concrete value that passes the guard yet causes the failure.\n- If a guard is absent, do not assume one exists; state the missing check and which input could exploit its absence.\n- If a guard fully covers the operation, say so explicitly and move on.\n\n**Loop (for/while):** For each loop, derive the iteration bound from the loop\'s own condition (not intuition). Compare the maximum loop index against the buffer size accessed inside the body; check the strict-inequality / off-by-one relationship explicitly. State whether the loop terminates and whether the largest reachable index can exceed the buffer bound.\n\n### Phase 3 — Candidate findings and three-part confirmation\nFor every suspected vulnerability you form during the walk, subject it to a three-part confirmation, each part anchored to specific lines of the shown code:\n1. **Taint:** Which input is attacker-controlled, and does it flow unvalidated (or insufficiently validated) to the dangerous sink? Name the sink line.\n2. **Reachability:** Show the exact path/branch in the shown code that reaches the sink, and the concrete value or condition that triggers the failure. A path that depends on code not shown, or on a caller violating an internal contract, is NOT reachability within this analysis.\n3. **Impact:** State the precise security mechanism and the single primary CWE (with a matching secondary CWE only if independently confirmed).\n\nA positive ("vulnerable") verdict is justified ONLY when all three parts are substantiated with concrete code anchors in the shown code. Matters that fail part 2 (e.g., depending on an out-of-scope caller, on a framework/library invariant, on unchecked allocation failure, on silent truncation without security impact, or on a purely theoretical worst case) must be recorded in `notes.limitations` as unconfirmed hypotheses or robustness concerns — NOT reported as vulnerabilities.\n\n### Phase 4 — Verification pass\nBefore finalizing, re-check every candidate you flagged. If a candidate\'s taint, reachability, or impact cannot be tied to a concrete line and concrete value, demote it to a limitation and do not list CWEs for it. Confirm that any "not vulnerable" conclusion is supported by the actual guards and sizes you tracked, rather than by assumption.\n\n## Rules\n- Base all reasoning only on the given code; do not rewrite or modify it.\n- Follow actual control flow (paths, checks, state changes); track real values, not hypothetical callers.\n- Focus on security-relevant behavior: memory safety, bounds, lifetime, integer safety, input validation, authz/authn, unsafe APIs.\n- Distinguish a confirmed, reachable, attacker-triggerable vulnerability from a defensive-hardening or robustness concern.\n- Do NOT add few-shot examples.\n- After reasoning, output ONLY the JSON object described below.\n\n## Output format\n- First: write your SCoT reasoning in plain text with numbered steps (Phases 1–4).\n- Then: output ONLY one JSON object (no markdown, no extra text).\n\nRequired JSON schema (fields and types must match exactly):\n{\n  "verdict": {\n    "vulnerable": true/false,\n    "confidence": 0.0-1.0,\n    "cwe": ["CWE-..."],\n    "severity": "low|medium|high|critical|unknown",\n    "summary": "..."\n  },\n  "evidence": [\n    {\n      "file": null,\n      "function": null,\n      "location": {"start_line": null, "end_line": null},\n      "snippet": "...",\n      "reason": "..."\n    }\n  ],\n  "notes": {\n    "assumptions": [],\n    "limitations": []\n  }\n}\n\n## Evidence precision requirements\n- For each evidence item, set `location.start_line` and `location.end_line` to the real, minimal line span of the sink in the shown code (a single line for a single-operation sink).\n- Set `snippet` to the exact code at those lines, quoted verbatim — no paraphrasing, ellipses, or substitutions.\n- Set `reason` to tie the specific index/pointer/size to its source (Phase 1) and to the concrete confirmed failure (Phase 3). Include at least one evidence item per confirmed finding.\n- Keep `notes.assumptions` for invariants you relied on and `notes.limitations` for unconfirmed hypotheses and out-of-scope concerns.\n\nCode:\n{{CODE}}'

REQUIRED_VERDICT_KEYS = {"vulnerable", "confidence", "cwe", "severity", "summary"}
REQUIRED_TOP_KEYS = {"verdict", "evidence", "notes"}


def build_prompt(code: str) -> str:
    """Render the harness prompt for a single code sample."""
    return PROMPT_TEMPLATE.replace("{{CODE}}", code)


def _strip_eos_tokens(raw: str) -> str:
    for tok in ("<|endoftext|>", "<|end|>", "<|im_end|>", "<|im_start|>"):
        raw = raw.replace(tok, "")
    return raw


def _sanitize_control_chars(raw: str) -> str:
    def _replace(m):
        c = m.group(0)
        return {'\t': '\\t', '\b': '\\b', '\f': '\\f'}.get(c, f'\\u{ord(c):04x}')
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', _replace, raw)


def _find_verdict_object(text: str, strict: bool = True) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder(strict=strict)
    for m in re.finditer(r'\{', text):
        start = m.start()
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "verdict" not in obj:
            continue
        v = obj["verdict"]
        if isinstance(v, str) and "|" in v:
            continue
        if isinstance(v, dict) and isinstance(v.get("vulnerable"), bool):
            return obj
    return None


def _extract_verdict_direct(text: str) -> Optional[Dict[str, Any]]:
    for m in re.finditer(r'"verdict"\s*:\s*\{', text):
        vstart = m.end() - 1
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


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Find the first valid JSON verdict object in raw model output."""
    cleaned = _strip_eos_tokens(raw)

    result = _find_verdict_object(cleaned, strict=True)
    if result:
        return result

    result = _find_verdict_object(cleaned, strict=False)
    if result:
        return result

    sanitized = _sanitize_control_chars(cleaned)
    result = _find_verdict_object(sanitized, strict=True)
    if result:
        return result

    stripped = re.sub(r'```(?:json)?\s*', '', cleaned)
    result = _find_verdict_object(stripped, strict=False)
    if result:
        return result

    return _extract_verdict_direct(cleaned)


def validate_schema(obj: Dict[str, Any]) -> list:
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


def parse_verdict(raw_output: str) -> Dict[str, Any]:
    """Parse a model's raw text response into a structured verdict.

    Returns dict with: parsed, errors, vulnerable, confidence.
    """
    obj = extract_json(raw_output)
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


def main() -> None:
    """CLI: render the prompt for a code file (or stdin) and print it."""
    code = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    print(build_prompt(code))


if __name__ == "__main__":
    main()
