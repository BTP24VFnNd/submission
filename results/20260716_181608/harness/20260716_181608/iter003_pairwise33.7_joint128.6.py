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

PROMPT_TEMPLATE = 'You are an expert software security auditor performing vulnerability detection. Your goal is to determine whether the given code contains a security vulnerability.\n\nYour task: analyze the code and determine whether it is VULNERABLE or NOT VULNERABLE.\n\nUse Structured Chain-of-Thought (SCoT) reasoning with explicit program structures (Sequence, Branch, Loop) across the following phases:\n\n=== PHASE 1: CODE COMPREHENSION ===\nTrace the code using SCoT structures to build a complete model of behavior.\n\nSequence: Trace the code step by step. Identify:\n  - All inputs (parameters, global state, return values from called functions, data read from buffers/network/files)\n  - All outputs (return values, written buffers, side effects)\n  - All type conversions and arithmetic operations on sizes, indices, offsets, or lengths\n  - All resource lifecycle operations (alloc/free, open/close, lock/unlock, init/cleanup)\n\nBranch: Map every conditional path. For EACH branch, document:\n  - The exact condition expression and its operator (pay attention to > vs >=, < vs <=, == vs !=)\n  - What happens on the true path vs. the false path\n  - Whether the branch is a SAFETY GUARD (bounds check, NULL check, error check, validation)\n\nLoop: For each loop, identify:\n  - What controls iteration count or termination\n  - Whether an attacker can influence iteration count or loop body behavior\n  - What data is accessed per iteration and whether indices stay in bounds\n\n=== PHASE 2: DEFENSE INVENTORY ===\nBefore looking for vulnerabilities, catalog every defensive measure in the code:\n\nFor each safety guard found in Phase 1, record:\n  1. GUARD LOCATION: exact line/expression\n  2. GUARD TYPE: bounds check, NULL check, error return check, input validation, length limit, permission check, sanitization\n  3. PROTECTED OPERATION: what specific downstream operation this guard protects\n  4. COMPLETENESS: does the guard fully prevent the dangerous condition? Check:\n     - Does it use the correct comparison operator? (> vs >=, < vs <=)\n     - Does it check the SAME variable used in the protected operation?\n     - Does the guard cover ALL paths to the protected operation? (no alternate unguarded path?)\n     - On the error path, does execution STOP (return/break/goto cleanup) or does it fall through?\n  5. VERDICT: EFFECTIVE (fully prevents the issue) or INCOMPLETE (gap identified, state what gap)\n\n=== PHASE 3: VULNERABILITY HYPOTHESIS ===\nSystematically check each vulnerability class below. For each class, determine whether the code contains that pattern. Cite specific lines or expressions.\n\nMEMORY SAFETY:\n- Buffer overflow/out-of-bounds write (CWE-787/120/121/122): Write past buffer bounds. Check all memcpy, strcpy, strncpy, sprintf, array indexing with attacker-influenced index or size. Check if length/size parameters are validated BEFORE the copy/write.\n- Out-of-bounds read (CWE-125): Read past buffer bounds. Check array accesses, pointer arithmetic with external indices.\n- Use-after-free (CWE-416) / Double-free (CWE-415): Memory accessed or freed after it was already freed. Check all paths through error handling and cleanup.\n- Uninitialized variable use (CWE-457/908): Variable declared but not assigned on all paths before use, especially in security-critical computation.\n\nNUMERIC SAFETY:\n- Integer overflow/wraparound (CWE-190): Arithmetic on sizes or counts that wraps and is then used for allocation, comparison, or indexing.\n- Integer underflow (CWE-191): Unsigned subtraction that wraps to a large value, especially in length computations.\n- Type confusion / truncation (CWE-681/197): A value stored in a wider type is narrowed, and the truncation changes the value used in a security-critical operation.\n\nPOINTER/REFERENCE SAFETY:\n- NULL pointer dereference (CWE-476): A function returns NULL on error, and the code does not check before dereferencing.\n- Missing error check leading to unsafe state (CWE-252/390): A function returns an error code or NULL, and the code either does not check or checks but does NOT stop execution afterward.\n\nINPUT VALIDATION:\n- Command injection (CWE-78), SQL injection (CWE-89), Path traversal (CWE-22), Format string (CWE-134), Log injection (CWE-117), HTTP header injection (CWE-113), SSRF (CWE-918), XPath/LDAP/XML injection (CWE-643/90/91), Code injection (CWE-94).\n- For SSRF/injection: commented-out or disabled validation counts as missing validation.\n\nCRYPTOGRAPHY:\n- Weak key size (CWE-326): RSA < 2048 bits, DSA < 2048 bits, EC < 224 bits, symmetric < 128 bits.\n- Broken/risky algorithm (CWE-327/328): MD5, SHA1 for security purposes, DES, RC4, ECB mode.\n- Weak randomness (CWE-330/338): Using rand(), random(), or predictable seeds for security-critical values.\n- Missing verification (CWE-347): Not verifying signatures, certificates, or MACs.\n\nCONCURRENCY:\n- Race condition (CWE-362/367): Shared state modified without proper synchronization, or TOCTOU between check and use.\n\nAUTHORIZATION/ACCESS:\n- Missing authorization (CWE-862/863): Security-sensitive operation without access control check.\n\nRESOURCE MANAGEMENT:\n- Uncontrolled resource consumption / ReDoS (CWE-400/1333): Regular expressions with catastrophic backtracking, unbounded allocation controlled by attacker.\n\n=== PHASE 4: EVIDENCE VERIFICATION (MOST CRITICAL PHASE) ===\nFor each candidate vulnerability identified in Phase 3, apply ALL of the following checks. A finding must pass every check to survive.\n\nSTEP A — Construct a concrete trigger: Specify EXACT input values or program state that would trigger the vulnerability. You must name actual values for variables, buffer sizes, and indices. "An attacker-controlled value that is larger than the buffer" is NOT concrete. "Setting rsp->data[0] to 255 when the destination buffer is 64 bytes" IS concrete. If you cannot construct a concrete trigger using only information present in the code, the finding is SPECULATIVE. Mark it REJECTED.\n\nSTEP B — Verify against the defense inventory: Cross-reference your finding against EVERY guard cataloged in Phase 2.\n  - If a guard was marked EFFECTIVE in Phase 2 and it protects the exact operation you are flagging, the finding is BLOCKED. Mark it REJECTED.\n  - Only if you can demonstrate the guard is INCOMPLETE (and you identified the specific gap in Phase 2) can the finding survive.\n  - CRITICAL: Do not find a vulnerability and then ignore a guard you already documented. If you documented a guard as EFFECTIVE, you cannot later claim the guarded operation is vulnerable without first explaining why your Phase 2 assessment was wrong.\n\nSTEP C — Path reachability: Trace a complete execution path from function entry to the vulnerable operation using your concrete trigger values from Step A. The path must pass through all conditionals correctly given those specific values. If any guard on the path would catch your trigger values, the finding is BLOCKED. Mark it REJECTED.\n\nSTEP D — Root cause identification: Identify the single root cause. If multiple symptoms stem from the same root cause, report only the root cause once. Map to the most specific CWE:\n  - Missing bounds check before array access → CWE-787 (write) or CWE-125 (read)\n  - Off-by-one in bounds check → CWE-193 leading to CWE-787 or CWE-125\n  - Missing NULL check after allocation/lookup → CWE-476\n  - Missing return/break after error detection → CWE-252 or CWE-390\n  - Integer overflow used as size → CWE-190\n  - Unsigned underflow in length → CWE-191\n  - User input in format string → CWE-134\n  - Weak key size → CWE-326\n  - Disabled/commented-out input validation → the CWE for the attack that validation would prevent\n\nSTEP E — Reject common false-positive patterns: A finding is REJECTED if it matches any of these patterns:\n  - NULL pointer from standard framework/library functions where the API contract guarantees non-NULL in the calling context (e.g., internal callback parameters provided by the framework, iterator values during valid iteration)\n  - Integer overflow/underflow that requires values outside the domain enforced by input parsing, protocol constraints, or earlier validation in the same function\n  - Off-by-one claims where you have not verified the actual array allocation convention (0-indexed vs 1-indexed, whether size includes a sentinel element)\n  - Resource leaks (memory, file descriptors) that have no security impact (no DoS amplification, no information disclosure). A simple memory leak is NOT a security vulnerability unless an attacker can trigger it repeatedly to exhaust resources.\n  - Claiming a variable "might" or "could" be negative/NULL/large without demonstrating a concrete path that produces that value using the code shown\n  - Unsigned subtraction underflow where the operands are constrained by prior checks to prevent the underflow case\n  - Race conditions in code that is architecturally single-threaded or where the lock scope matches the access scope\n\n=== PHASE 5: FINAL VERDICT ===\nBased on surviving evidence from Phase 4:\n\nIf one or more findings survived ALL verification steps:\n- vulnerable = true\n- Report the CWE(s) only for verified finding(s), using the most specific CWE from Step D\n- Cite the exact code location (line numbers and code snippet) for each finding\n- Confidence reflects evidence strength: use 0.9+ only when you have a fully concrete trigger\n\nIf NO findings survived verification:\n- vulnerable = false\n- Confidence reflects analysis thoroughness\n\nRules:\n- Base analysis on the provided code. For called functions not shown, apply standard API contracts and reasonable assumptions about their behavior.\n- A vulnerability must be reachable through a concrete execution path with concrete trigger values.\n- Code that properly handles errors (checks return values AND stops execution on error) is NOT vulnerable for that error path.\n- Do NOT count the same root cause multiple times.\n- After your reasoning, you MUST output the JSON object below. Do not omit it.\n\nOutput format:\n- First: write your SCoT reasoning following Phases 1-5 above.\n- Then: output ONLY one JSON object (no markdown fences, no extra text before or after the JSON).\n\nRequired JSON schema:\n{\n  "verdict": {\n    "vulnerable": true/false,\n    "confidence": 0.0-1.0,\n    "cwe": ["CWE-..."],\n    "severity": "low|medium|high|critical|unknown",\n    "summary": "..."\n  },\n  "evidence": [\n    {\n      "file": null,\n      "function": null,\n      "location": {"start_line": null, "end_line": null},\n      "snippet": "...",\n      "reason": "..."\n    }\n  ],\n  "notes": {\n    "assumptions": [],\n    "limitations": []\n  }\n}\n\nCode:\n{{CODE}}'

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
