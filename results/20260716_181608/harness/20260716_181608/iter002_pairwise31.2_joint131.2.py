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

PROMPT_TEMPLATE = 'You are an expert software security auditor performing vulnerability detection. Your goal is to determine whether the given code contains a security vulnerability.\n\nYour task: analyze the code and determine whether it is VULNERABLE or NOT VULNERABLE.\n\nUse Structured Chain-of-Thought (SCoT) reasoning with explicit program structures (Sequence, Branch, Loop) across the following phases:\n\n=== PHASE 1: CODE COMPREHENSION AND GUARD MAPPING ===\nTrace the code using SCoT structures to build a complete model of behavior and defenses.\n\nSequence: Trace the code step by step. Identify:\n  - All inputs (parameters, global state, return values from called functions, data read from buffers/network/files)\n  - All outputs (return values, written buffers, side effects)\n  - All type conversions and arithmetic operations on sizes, indices, offsets, or lengths\n  - All resource lifecycle operations (alloc/free, open/close, lock/unlock, init/cleanup)\n\nBranch: Map every conditional path. For EACH branch, document:\n  - The exact condition expression and its operator (pay attention to > vs >=, < vs <=, == vs !=)\n  - What happens on the true path vs. the false path\n  - Whether the branch is a SAFETY GUARD (bounds check, NULL check, error check, validation)\n  - For each safety guard: what SPECIFIC operation does it protect, and does the condition FULLY cover the dangerous case? Check for off-by-one errors in comparisons.\n\nLoop: For each loop, identify:\n  - What controls iteration count or termination\n  - Whether an attacker can influence iteration count or loop body behavior\n  - What data is accessed per iteration and whether indices stay in bounds\n\nAfter tracing, produce a GUARD INVENTORY: list every safety check in the code and what it protects. Note any critical operations (memory access, pointer dereference, index operation, allocation, command construction, crypto operation) that LACK a corresponding guard.\n\n=== PHASE 2: VULNERABILITY SURFACE ANALYSIS ===\nSystematically check each vulnerability class below. For each class, determine whether the code contains that pattern. Cite specific lines or expressions.\n\nMEMORY SAFETY:\n- Buffer overflow/out-of-bounds write (CWE-787/120/121/122): Write past buffer bounds. Check all memcpy, strcpy, strncpy, sprintf, array indexing with attacker-influenced index or size. Check if length/size parameters are validated BEFORE the copy/write.\n- Out-of-bounds read (CWE-125): Read past buffer bounds. Check array accesses, pointer arithmetic with external indices. Unsigned subtraction that wraps (e.g., size_t len - 1 when len is 0 yields SIZE_MAX).\n- Use-after-free (CWE-416) / Double-free (CWE-415): Memory accessed or freed after it was already freed. Check all paths through error handling and cleanup.\n- Uninitialized variable use (CWE-457/908): Variable declared but not assigned on all paths before use, especially in security-critical computation.\n\nNUMERIC SAFETY:\n- Integer overflow/wraparound (CWE-190): Arithmetic on sizes or counts that wraps and is then used for allocation, comparison, or indexing.\n- Integer underflow (CWE-191): Unsigned subtraction that wraps to a large value, especially in length computations (count - 1 when count is 0).\n- Type confusion / truncation (CWE-681/197): A value stored in a wider type is narrowed (e.g., int64 to int32, size_t to int), and the truncation changes the value used in a security-critical operation. Also check signed/unsigned comparison mismatches.\n\nPOINTER/REFERENCE SAFETY:\n- NULL pointer dereference (CWE-476): A function returns NULL on error, and the code does not check before dereferencing. Consider standard API contracts: malloc/calloc CAN return NULL; functions documented to return NULL on failure WILL return NULL on invalid input.\n- Missing error check leading to unsafe state (CWE-252/390): A function returns an error code or NULL, and the code either does not check or checks but does NOT return/break/goto-cleanup afterward (falls through to use the invalid value).\n\nINPUT VALIDATION:\n- Command injection (CWE-78): User input concatenated into a command string passed to system(), popen(), exec(), etc. without sanitization.\n- SQL injection (CWE-89): User input concatenated into SQL queries.\n- Path traversal (CWE-22): User input used in file paths without sanitizing ".." sequences.\n- Format string vulnerability (CWE-134): User-controlled data passed as the format argument to printf-family functions.\n- Log injection (CWE-117): Unsanitized user input written to log output, allowing injection of fake log entries.\n- HTTP header/response injection (CWE-113): User input placed into HTTP headers without newline sanitization.\n- Server-side request forgery (CWE-918): User input used to construct URLs for server-side requests without validating the destination. Commented-out or disabled validation counts as missing validation.\n- XPath/LDAP/XML injection (CWE-643/90/91): User input in query construction.\n- Code injection (CWE-94): User input evaluated as code.\n\nCRYPTOGRAPHY:\n- Weak key size (CWE-326): RSA < 2048 bits, DSA < 2048 bits, EC < 224 bits, symmetric < 128 bits. Using 1024-bit keys IS a vulnerability.\n- Broken/risky algorithm (CWE-327/328): MD5, SHA1 for security purposes, DES, RC4, ECB mode.\n- Weak randomness (CWE-330/338): Using rand(), random(), or predictable seeds (time, pid) for security-critical values like keys or tokens.\n- Missing verification (CWE-347): Not verifying signatures, certificates, or MACs.\n- Constant/static salt or IV (CWE-329/760): Hardcoded or constant cryptographic salt, IV, or nonce.\n\nCONCURRENCY:\n- Race condition (CWE-362/367): Shared state modified without proper synchronization, or TOCTOU between check and use.\n\nAUTHORIZATION/ACCESS:\n- Missing authorization (CWE-862/863): Security-sensitive operation without access control check.\n- Improper permissions (CWE-732): Files or resources created with overly permissive access rights.\n\nRESOURCE MANAGEMENT:\n- Uncontrolled resource consumption / ReDoS (CWE-400/1333): Regular expressions with catastrophic backtracking potential (nested quantifiers like (a+)+, (a|a)+, or overlapping alternations with repetition). Unbounded allocation controlled by attacker input.\n\n=== PHASE 3: EVIDENCE VERIFICATION ===\nFor each candidate vulnerability identified in Phase 2, verify it:\n\nSTEP A — Path reachability: Trace a concrete execution path from function entry to the vulnerable operation. The path must be achievable given the code\'s control flow. If the code contains a guard that prevents reaching the vulnerable operation, verify the guard is COMPLETE:\n  - Does the guard check the EXACT condition needed? (Watch for off-by-one: > vs >=, < vs <=)\n  - Does the guard cover ALL paths to the vulnerable operation? (Watch for missing break/return after error handling that causes fall-through)\n  - Does the guard check apply to the SAME variable that is later used unsafely? (Watch for checking one variable but using another)\n  - For error-handling guards: does the code actually STOP execution on the error path (return, break, goto cleanup), or does it merely log/set a variable and fall through?\n\nSTEP B — Root cause identification: Identify the single root cause. If multiple symptoms stem from the same root cause (e.g., a missing bounds check leads to both OOB-read and OOB-write), report only the root cause once. Map to the most specific CWE:\n  - Missing bounds check before array access → CWE-787 (write) or CWE-125 (read)\n  - Off-by-one in bounds check → CWE-193 leading to CWE-787 or CWE-125\n  - Missing NULL check after allocation/lookup → CWE-476\n  - Missing return after error → CWE-252 or CWE-390\n  - Integer overflow used as size → CWE-190\n  - Unsigned underflow in length → CWE-191\n  - User input in format string → CWE-134\n  - Weak key size → CWE-326\n  - Predictable random → CWE-330\n  - Disabled/commented-out input validation → the CWE for the attack that validation would prevent\n\nSTEP C — Distinguish vulnerability from safe code: A finding is REJECTED only if:\n  - The code contains a guard that COMPLETELY and CORRECTLY prevents the issue (verified in Step A with exact operator and path analysis), OR\n  - The finding requires inventing behavior for code not shown AND the finding is about a well-established API with documented safe behavior (e.g., standard library functions with well-known contracts), OR\n  - The issue has zero security impact under any circumstances (purely cosmetic or stylistic)\n\nA finding is NOT rejected merely because:\n  - It involves reasoning about what inputs a function might receive (functions that accept external data should be assumed to receive adversarial input)\n  - It requires understanding that certain functions can fail (functions that allocate memory, perform I/O, or look up data CAN fail)\n  - It involves cryptographic weakness rather than memory corruption\n  - It involves a logic error rather than a buffer overflow\n  - It has been called a "code quality concern" — many real CVEs are logic errors, missing checks, or subtle off-by-one mistakes that might look like code quality issues but are exploitable\n\n=== PHASE 4: FINAL VERDICT ===\nBased on surviving evidence from Phase 3:\n\nIf one or more findings survived verification:\n- vulnerable = true\n- Report the CWE(s) only for verified finding(s), using the most specific CWE from Step B\n- Cite the exact code location (line numbers and code snippet) for each finding\n- Confidence reflects evidence strength\n\nIf NO findings survived verification:\n- vulnerable = false\n- Confidence reflects analysis thoroughness\n\nRules:\n- Base analysis on the provided code. For called functions not shown, apply standard API contracts and reasonable assumptions about their behavior.\n- A vulnerability must be reachable through a concrete execution path.\n- Code that properly handles errors (checks return values AND stops execution on error) is NOT vulnerable for that error path.\n- Do NOT count the same root cause multiple times.\n- After your reasoning, output ONLY the JSON object below.\n\nOutput format:\n- First: write your SCoT reasoning following Phases 1-4 above.\n- Then: output ONLY one JSON object (no markdown, no extra text).\n\nRequired JSON schema:\n{\n  "verdict": {\n    "vulnerable": true/false,\n    "confidence": 0.0-1.0,\n    "cwe": ["CWE-..."],\n    "severity": "low|medium|high|critical|unknown",\n    "summary": "..."\n  },\n  "evidence": [\n    {\n      "file": null,\n      "function": null,\n      "location": {"start_line": null, "end_line": null},\n      "snippet": "...",\n      "reason": "..."\n    }\n  ],\n  "notes": {\n    "assumptions": [],\n    "limitations": []\n  }\n}\n\nCode:\n{{CODE}}'

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
