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

PROMPT_TEMPLATE = 'You are an expert software engineer and security analyst. Analyze the following code for security vulnerabilities using Structured Chain-of-Thought (SCoT). Reason with explicit program structures, anchoring every claim to narrow snippet ranges and requiring both source (attacker-controlled input/payload/index) and sink (memory access, dereference, write, authorization decision, format/string operation) quoted together with line ranges before any vulnerability conclusion.\n\nSequence — ordered data-flow from attacker-controlled sources through transformations to sinks. Step 1: identify attacker-controlled sources. Quote each source snippet (≤3 lines, exact start/end with line numbers), including derived pointers/indexes from untrusted counts (e.g., idx = len-1, ptr = obj->derived, offset = parsed_size). For derived-pointer sources quote both the parent-assigned derived pointer expression and its origin from untrusted parsing in the same paired snippet, anchored to the exact parsing line (e.g., idx derived from untrusted len at line N and assignment at M quoted together). Step 2: quote assignment of derived pointer/index and its sink usage together as one paired snippet with both line ranges; state explicitly "assignment unvalidated" if no guard quotes validity before sink. Prioritize derived-pointer validation at sink: quote the exact sink dereference/assignment/write line and state "derived-pointer validity at sink never quoted" when the pointer\'s validity is not explicitly validated at the access site; never infer vulnerability from parent-array or parent-object validation alone — always pair the sink line with its own validity check if present. Step 3: verify validation shown with quoted snippet at each transition; identify missing intermediate validation specifically for the derived pointer/index accessed at sink; confirm unguarded path by quoting guard-absent sink line; do not claim vulnerability without both source and sink quoted separately and both in ≤3-line pairs.\n\nBranch — for each if/else, state condition and branch controlling attacker input. Confirm guard is BEFORE dangerous operation (check-before-use) by quoting guard snippet at G and dangerous operation at S with S > G. Verify missing break/return/continue allows fallthrough past guard to sink; quote guard snippet and fallthrough path together with both line numbers. Check operator correctness (> vs >=, off-by-one, unsigned wrap, underflow/overflow, sign-mismatch, truncation) with quoted snippet including line numbers; prefer narrow ≤3-line ranges for the guard and for the operation it controls. Confirm guard applies to sink\'s accessed object: explicitly state "mismatch: guard checks parent (X) at G but sink dereferences derived pointer (Y) at S" with both line numbers as separate snippet pairs when objects differ; if derived pointer is unvalidated at sink, treat path as unguarded rather than relying on parent check. For authorization/auth, confirm success/failure assignment cannot be unconditionally overwritten; quote both assignment and overwrite together with line ranges.\n\nLoop — for each for/while, state bound (constant, parsed length, packet size, user value, array length from untrusted input). Quote index derivation expression + array/structure access together (≤3 lines including index expression and access/write line). Confirm index/offset/count derived from untrusted input; verify bounds check occurs BEFORE array access/write with quoted snippet pair (check line + access line). When loop writes through a derived pointer (e.g., loop assigns to ptr[i] where ptr derived from parent), quote derived-pointer assignment + loop access together with both ranges; confirm guard applies to derived pointer if loop accesses derived pointer, not only parent array/index — state "guard applies to parent (P) at G but loop accesses derived pointer (D) at L" if mismatch. Verify termination (break/return/continue) exits dangerous path; quote termination snippet with line number. Verify loop bounds against array/structure sizes explicitly with quoted snippet pairs covering both the bound expression and the access.\n\nEvidence anchoring mandatory: quote snippet at dangerous operation (≤3 lines with start/end line numbers) and input-to-sink separately; for derived pointers include assignment + sink pair with both ranges and require both ranges in the response; for loop access quote index derivation + access together; state whether guard applies to attacker-controlled path; state mismatch explicitly with both line numbers (guard object/line vs sink object/line); include exact dereference/assignment/write lines; prefer narrow ranges (≤3 lines) for high confidence, wider only when unavoidable — state "inference: snippet spans >3 lines; confidence reduced" and reduce confidence proportionally. CWE identification (select only from observable sink type): memory dereference/access = CWE-787/125/416; input validation failure at sink = CWE-20; authorization/auth bypass = CWE-287/862; arithmetic/integer at sink = CWE-190/191; format/string = CWE-134/78; select most specific directly observable from sink, not from validation behavior alone; prefer observable sink over inference; list at most one CWE unless multiple clearly distinct sink types in different branches visible; do NOT list CWE from only validation behavior without observable sink. Localization: snippet must contain sink line(s); include exact dereference/assignment/write lines; use ≤3 line ranges unless source/sink span more; include function name.\n\nVerification rules enforcing conservative verdict (addressing observed over-labeling/precision failure — high false-positive tendency from missing sink quotes and guard-object mismatches): only claim vulnerability when both source and sink visible as quoted pairs, unguarded or guard misplaced/applying to wrong object, with specific line ranges; base reasoning only on given code; follow actual control flow; when guard checks parent but sink uses derived pointer, state mismatch explicitly with both line numbers and treat derived pointer path as unguarded — do not claim vulnerability solely on parent check; if ambiguous (snippet >3 lines, guard object differs from sink, missing source/sink pair, full-function range used, derived-pointer validity at sink never quoted) prefer false over true and reduce confidence; set confidence proportional to evidence specificity — narrow range + snippet at both source and sink + correct guard-check = high (≥0.8); inference/full-range/guard-mismatch = moderate (0.4–0.7); missing source or missing sink = low/false (≤0.35); emphasize absence of quoted derived-pointer validity check at sink is a limitation, not evidence of vulnerability; avoid inferring vulnerability from absence of guard alone when sink is not quoted; require both source and sink quoted separately for every claim; when derived pointer access at sink is not validated by a quoted snippet at sink line, treat as unguarded only if sink is quoted, else do not claim vulnerability.\n\nFormat: first SCoT reasoning with numbered Sequence/Branch/Loop steps using quoted snippet pairs and line ranges (start_line/end_line) and explicit mismatch/guard statements; then ONLY one JSON object with no markdown/extra text.\n\nJSON: {"verdict":{"vulnerable":true/false,"confidence":0.0-1.0,"cwe":["CWE-..."],"severity":"low|medium|high|critical|unknown","summary":"..."},"evidence":[{"file":null,"function":null,"location":{"start_line":null,"end_line":null},"snippet":"...","reason":"..."}],"notes":{"assumptions":[],"limitations":[]}} Code: {{CODE}}'

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
