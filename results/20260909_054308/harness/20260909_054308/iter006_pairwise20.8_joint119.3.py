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

PROMPT_TEMPLATE = 'You are an expert software engineer and security analyst. Analyze the following code for security vulnerabilities using Structured Chain-of-Thought (SCoT). Reason with explicit program structures, anchoring every claim to narrow snippet ranges and requiring both source (attacker-controlled input/payload/index) and sink (memory access, dereference, write, authorization decision, format/string operation) quoted together with line ranges before any vulnerability conclusion.\n\nSequence — ordered data-flow from attacker-controlled sources (arguments, parsed fields, network/file payload, user value, parsed lengths) through transformations to sinks. For each step quote ≤3 lines with exact start_line/end_line; verify source is untrusted (parsed from external payload, user-controlled, or derived from unvalidated count/size); state whether validation is shown at the transition. For derived pointers (ptr = obj->field, rka = authkey->payload.data[0], idx from parsed length) quote both assignment and sink within ≤3 lines; state explicitly if assignment is unvalidated even when parent pointer is checked. When source spans parsing/derivation and sink spans access/write, quote source derivation and sink access as separate snippet pairs — never rely on a single full-function range. Confirm data-flow reaches sink without intermediate validation; identify missing validation between source transition and sink.\n\nBranch — for each if/else, state condition and which branch executes for attacker-controlled input. Confirm guard is BEFORE dangerous operation (check-before-use, never after); verify missing break/return/continue allows fallthrough past guard to sink; check operator correctness (> vs >=, off-by-one, unsigned wrap, underflow/overflow, sign-mismatch, int64-to-int8 truncation); confirm guard applies to sink\'s accessed object — explicitly state "mismatch: guard checks parent (X) but sink dereferences derived pointer (Y) at line Z" when guard object differs from sink object; quote guard and sink within ≤3 lines. For authorization/authentication branches, confirm success/failure assignment cannot be unconditionally overwritten later; quote both assignment and overwrite.\n\nLoop — for each for/while state bound (constant, parsed length, packet size, user value): state whether index/offset/count derived from untrusted input; quote index derivation and array/structure access together (≤3 lines); confirm bounds check occurs BEFORE array access/write; verify termination (break/return/continue) exits dangerous path; verify loop bounds against array/structure sizes explicitly; confirm guard applies to all paths reaching sink inside loop. When loop writes/accesses through derived pointers, confirm pointer validity is checked in loop, not only parent array/index.\n\nEvidence anchoring mandatory before any vulnerability conclusion: quote snippet at dangerous operation (≤3 lines) and input-to-sink separately; for derived pointers include assignment + sink pair; for loop access quote index derivation + access; state whether guard applies to attacker-controlled path; state mismatch explicitly; include exact dereference/assignment lines in snippet; prefer narrow ranges for high confidence, full-range only when unavoidable — state "inference: snippet spans >3 lines" in notes and reduce confidence. CWE identification: determine primary class directly from observable sink type (memory dereference/access = CWE-787/125/416; input validation failure at sink = CWE-20; authorization/auth bypass = CWE-287/862; arithmetic/integer at sink = CWE-190/191; format/string = CWE-134/78); select most specific and directly observable from sink, not from validation behavior alone; prefer observable sink over inference; list at most one CWE unless multiple clearly distinct sink types in different branches visible; do NOT list CWE from only validation behavior without observable sink. Localization: snippet must contain sink line(s); include exact dereference/assignment/write lines; use ≤3 line ranges unless source/sink span more; prefer narrow ranges for high confidence. When evidence weak (missing snippet, full-function inference, guard on different variable, only validation shown), prefer false over true, reduce confidence, state ambiguity.\n\nVerification rules (strong evidence required): only claim vulnerability when both source and sink visible, unguarded (or guard misplaced/applying to wrong object), with specific line ranges; base reasoning only on given code; follow actual control flow; if ambiguous prefer conservative verdict with lower confidence and state ambiguity/limitations; set confidence proportional to evidence specificity — narrow range + snippet at both source and sink + correct guard-check = high; inference/full-range/guard-mismatch = moderate; missing source or missing sink = low/false; when in doubt and evidence weak prefer false over true and reduce confidence.\n\nSystematic failure-pattern corrections: require source-and-sink quoted pairs; enforce guard-applies-to-sink check with explicit mismatch statement; strengthen loop bounds with quoted snippet pairs; restrict CWE to observable sink; require conservative verdict when snippet spans >3 lines or guard object differs from sink; emphasize input-to-sink data-flow and derived-pointer validation; never claim vulnerability without quoted sink at correct lines.\n\nDo NOT add bias-correction framing ("lean not vulnerable", "when in doubt say not vulnerable"). Do NOT add few-shot examples. Do NOT change JSON fields/types. Do NOT replace SCoT with different reasoning.\n\nFormat: first SCoT reasoning with numbered Sequence/Branch/Loop steps using quoted snippet pairs and line ranges; then ONLY one JSON object with no markdown/extra text.\n\nJSON: {"verdict":{"vulnerable":true/false,"confidence":0.0-1.0,"cwe":["CWE-..."],"severity":"low|medium|high|critical|unknown","summary":"..."},"evidence":[{"file":null,"function":null,"location":{"start_line":null,"end_line":null},"snippet":"...","reason":"..."}],"notes":{"assumptions":[],"limitations":[]}} Code: {{CODE}}'

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
