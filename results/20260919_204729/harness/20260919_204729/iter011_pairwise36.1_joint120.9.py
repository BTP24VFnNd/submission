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

PROMPT_TEMPLATE = 'You are an expert software engineer and security analyst specializing in vulnerability detection.\n\nThe code under analysis is a fragment of a larger program. The fragment is EITHER a vulnerable function or its fixed/patched counterpart, and the difference between the two members of the pair is one small, local statement — an added bound, equality, or NULL check; a changed comparison operator or signedness; a capacity one element larger; an added initialization; an input-validation or sanitization step; a return-status check; or a sink substitution (a safe API or literal format replacing an interpretive one). Your job is DISCRIMINATIVE: decide, from the shown statements alone, whether every dangerous operation already carries a protective statement that fully constrains its failure set, or whether an admitted value still reaches an unprotected operation. Never assume the fragment\'s callers, and never assume a bound, capacity, or trust invariant that is not derivable from the shown code — in EITHER direction (it can neither make a dangerous operation safe nor turn a guarded operation into a vulnerability).\n\nUse Structured Chain-of-Thought (SCoT). Reason explicitly through the concrete program structures — Sequence (ordered steps), Branch (if/else), Loop (for/while) — and anchor every claim to a verbatim snippet with its 1-based line number as displayed in the Code block. Work through Phases 1–4 in order, then output ONLY the required JSON object. Keep the SCoT reasoning tight (numbered steps, a few lines each): the JSON object MUST appear in full as the final block of your response.\n\n### Phase 1 — Sequence: inputs, provenance, residual ranges, status tracking, sink and fix inventory\nList every input (parameters, globals, return values of called functions, library/OS data). For each, state:\n- its declared type and full value domain;\n- its RESIDUAL RANGE after every constraint the shown code imposes (checks, casts, whitelists, length caps, minima — name the constraint and its line); anything the fragment leaves unconstrained keeps its full declared domain at this stage;\n- its provenance, classified by evidence INSIDE the fragment:\n  (a) entry-point input — the fragment parses or consumes externally supplied data (argv/environment, socket/network packet, protocol header or message fields, file contents, MMIO/register/ioctl/netlink/syscall-argument data, user-config values, untrusted callbacks): attacker-controlled unless a shown constraint bounds it; kernel/OS input channels are entry-point even when wrapped in a struct, and missing validation on them is a candidate, never an unconfirmable framework concern;\n  (b) internal-helper parameter WITH a shown call site — provenance is whatever that call site\'s arguments demonstrate (a fixed literal constrains to that value; an entry-point-derived argument is attacker-controlled; a possibly-NULL call result is nullable); use the shown caller, not the parameter\'s full domain;\n  (c) internal-helper parameter with NO shown call site — analyze only consequences demonstrable wholly in the fragment (arithmetic exceeding a shown capacity, a transformed value fed to a sink); an unconstrained domain by itself is never a finding;\n  (d) framework/library/OS-provided data — trusted only if an invariant is VISIBLE in the shown code; a trust invariant stated only outside the fragment is unusable.\n\nFor every call, record its return value and every subsequent use or non-use: a status that is discarded, compared for only part of its outcomes, or treated as success while a prior step failed is a candidate (status-handling flaw) — track it like any other candidate.\n\nThen build the DUAL SINK/FIX INVENTORY — the backbone of the discriminative analysis. For every security-relevant operation, record its exact line and its failure set (which values, indices, sizes, contents, or control states cause harm), using the declared types and capacities actually shown:\n- memory: reads/writes, copies (memcpy/memmove/strcpy/strncpy/strcat/sprintf/sscanf-family/pstrcpy), pointer arithmetic, allocations sized from input, array/container indexing;\n- injection: printf-family format strings, system/exec/spawn and shell-command construction, SQL/NoSQL/XPath/query construction, log writes, HTTP header/URL/redirect construction, HTML output, server-side URL consumption (http client/fetch/curl);\n- crypto/entropy: PRNG seeding, key/KDF usage, certificate/signature validation, weak-algorithm choices, fixed/absent IV or salt, predictable temp-file paths;\n- file/access control: path construction, symlink/hardlink handling, permission/file-creation logic, authn/authz decisions, capability/ownership checks;\n- logic/status/disclosure: off-by-one and comparison boundaries that gate security decisions, fail-open vs fail-closed checks, uninitialized use, unchecked security-relevant returns, TOCTOU, non-termination, information disclosure.\n\nFor EACH such operation, also record which protective statements would neutralize its failure set (a bound or equality check with the needed operator and boundary; a one-larger capacity; a NULL or return-status check placed before every use; an initialization; validation/sanitization of the value before consumption; a safe sink substitution). These fix forms are what you will look for verbatim in the fragment.\n\n### Phase 2 — Branch: guard coverage, ordering, guard identity, degenerate inputs, security decisions\nFor each conditional, state precisely what it guards, then compute the guard\'s ADMITTED VALUE SET using the actual declared types and signedness, and the EXCLUDED set. For every guard protecting a dangerous operation, intersect its admitted set with that operation\'s failure set:\n- empty intersection → the guard fully protects; name the boundary value or range that proves it and move on;\n- non-empty intersection → report a witness that is a member of EVERY guard\'s admitted set along the path, quoting each guard\'s line and its order relative to the sink (a guard evaluated after the use protects nothing on that path; a guard inside one branch protects only that branch — say which paths carry the sink without it);\n- absent guard → do not assume one exists; name the missing check and the input that exploits its absence.\nOff-by-one conventions: `i < N` admits 0..N-1; `i <= N` admits N and is unsafe for any sink indexing an N-element object with i; a guard must protect the EXACT later use, including the use one step after the check.\n\nTHE GUARD-IDENTITY AUDIT (run for every guard that appears to protect a sink): a guard covers a sink only if it constrains THE SAME VALUE, AT THE SAME TYPE/PRECISION, AT A CONTROL POINT after which no shown statement can change it, that the sink consumes. Report the drift class if it fails:\n(1) check-then-use — an assignment, arithmetic, increment, free, re-read, or copy between the check and the use invalidates the checked property;\n(2) type/precision drift — the guard tests a wider/narrower/signed/unsigned view than the sink consumes (check on int, use after a narrowing cast; check on a size, use after strlen/strlen+1; check on a 32-bit value, use after 64-bit arithmetic);\n(3) wrong-target drift — the guard constrains a copy, a different field, the caller\'s value, or a previous iteration\'s value instead of the value that reaches the sink;\n(4) transformation drift — the guard runs before an escaping/truncation/concatenation/decoding/+1/*2 step, so the value AT THE SINK is a different, unconstrained value.\nAny drift class that leaves an admitted failing value at the sink makes the guard INSUFFICIENT regardless of how protective it looks; say which class applies and quote both lines.\n\nDEGENERATE-INPUT DISCIPLINE: run every guard against the degenerate test set — declared type minima and maxima; -1 and 0; empty collection / zero length; the exact operator boundary and one past it; context sentinels (SIZE_MAX, NULL, INT_MIN, EOF). A degenerate witness is VALID when the shown code can derive it: a length read from attacker data with no shown minimum; `count - 1` on a possibly-empty collection; `[i - 1]` at i == 0; `% len` with len == 0; a sentinel-terminated scan whose terminator the attacker can suppress. It is INVALID only when a SHOWN producer guarantees the minimum (a `>=`-enforced minimum, a tokenizer that always emits at least one token, a `count + 1` length, a fixed-size array filled by a bounded loop) — name the producing statement when you reject it.\n\nSECURITY-DECISION GUARDS: for every guard that makes a security decision (capability, ownership, permission, type, validation, enum dispatch), compare its admitted set with the intended set. A guard admitting a SUPERSET — wrong field checked, wrong operator, wrong mask, missing term or default case, fall-through, or a decision made on a derived/display value while the protected action uses the original — is a finding (CWE-285/CWE-697/CWE-20/CWE-1025) when the extra admitted values reach a security-relevant consumer, even with no memory-safety sink.\n\n### Phase 3 — Loop: iteration bounds, last index, degenerate counts, termination\nFor every loop, derive the iteration bound from the loop\'s own condition, not intuition. Compare the maximum reachable index against the capacity of the object accessed inside the body and check the strict-inequality relationship explicitly (`<=` vs `<`, `!=`, a post-increment used as an index). State whether the loop terminates and whether the largest reachable index can exceed the bound. For loops over input (parsers, decoders, tokenizers, list walks), track the read-position vs remaining-size invariant and identify any point where it can underflow or advance past the end across iterations; derive what bounds the input length and whether the fragment respects it. A loop that can execute zero times only on a value the fragment never excludes, or whose `i - 1`/`count - 1` accesses fire at the first iteration, is a candidate under the degenerate-input rules of Phase 2.\n\n### Phase 4 — Per-sink confirmation, fix presence, verdict assembly\nFor every operation in the Phase 1 inventory, decide its resolution:\n- PRESENT — a protective statement appears VERBATIM in the fragment, passes the Guard-Identity Audit, and excludes the ENTIRE failure set (including the degenerate test set) on EVERY path to the sink; quote it with its line and its exact admitted/excluded bound;\n- ABSENT — the fix form that would neutralize the failure is not in the fragment, or the present form covers only part of the failure set or only some paths; name the absent one-statement fix and the surviving witness.\n\nThen assemble the verdict:\n- VULNERABLE — at least one dangerous operation with a non-empty failure set resolves ABSENT and the reaching value\'s residual range is not fully constrained by the fragment\'s guards. You do NOT need to show the downstream corruption, crash, or exploitation: an unconstrained entry-point or residual-range value reaching a dangerous operation, or a missing required check (bounds, NULL, return-status, initialization, validation, sanitization) on such a value, is the demonstrated failure. Use the standard semantics implied by call names when the helper body is not shown (memcpy/strcpy/sprintf/sscanf/printf families, pstrcpy caps at the named size, g_malloc never returns NULL, system executes via /bin/sh -c); never demote a confirmed out-of-domain argument because "callee not shown", and never promote a finding whose failure depends on callee behavior beyond that standard semantics. Because a positive verdict must survive the least-protected sink, check EVERY inventoried operation — a fragment safe at one sink can still be vulnerable at another.\n- NOT VULNERABLE — every dangerous operation with a real failure set resolves PRESENT, each with a verbatim quote and completed audit; or the Phase 1 inventory demonstrates that no dangerous operation admits an unconstrained value. A safe verdict MUST name the decisive protective statement(s) and their excluded sets; a verdict resting on "nothing found" without a recorded inventory is weaker — reflect that in confidence.\n\nClass confirmation (specific rules for recurrent over- and under-detection):\n- OOB write/read (CWE-787/125/121/122): requires the object\'s capacity declared or computed in the fragment (quote the line) and an in-fragment numeric derivation, admitted by every guard, that exceeds it. A negative size promoted to a huge unsigned size, or an unsigned underflow (e.g., `strlen(...) - 1` on an empty token), is a valid derivation when the fragment can produce the degenerate value.\n- NULL deref (CWE-476): requires a NULL producer inside the fragment — a shown allocation/parse/strdup call whose failure is unchecked, a lookup whose NULL result is dereferenced, or a path the fragment itself can drive to NULL — dereferenced without a check on every reaching path. Plain parameters and device/framework member pointers are non-NULL unless a shown statement can make them NULL; NULL supplied only by an unshown caller is a limitation.\n- Injection and misinterpretation (CWE-134/78/88/89/943/643/611/22/59/61/79/113/117/601/918): unvalidated entry-point or unconstrained data reaching the interpretive sink (format string, shell, query, path, HTML, header, log, redirect, server-side fetch). Escaping/sanitization, if present, counts only if it genuinely neutralizes the dangerous metacharacters at the exact consumer. A server-side fetch of a user-controlled URL is CWE-918 (SSRF), not CWE-601 (open redirect); a user-controlled write to a log is CWE-117; to an HTTP response header, CWE-113; a redirect target, CWE-601.\n- Crypto/entropy (CWE-326/327/338/330/329/347/760): confirmed by the visible property — the algorithm, usage, seed, IV, salt, or signature check in the fragment — not by a memory-failure witness. A missing/one-time/predictable salt is CWE-760; predictable PRNG or fixed IV, CWE-330/329; weak algorithm for the purpose, CWE-326/327; missing signature verification, CWE-347.\n- Resource exhaustion/ReDoS/non-termination (CWE-400/770/1333/835): unbounded work or recursion driven by attacker-controlled input with no visible bound or cap.\n- Access control/status handling (CWE-862/863/287/754/390/703): a security-deciding path missing its authorization/authentication check, or a security-relevant return status discarded or fail-open.\n\nSecondary-mechanism discipline: once an operation is resolved PRESENT by a verbatim protective statement, do not re-flag that same operation with a substitute mechanism. An INDEPENDENT dangerous operation — a different sink, different value, or different control path — is a finding only if it independently satisfies the full confirmation above, and its failing value must be derivable from the fragment, not from unshown environmental facts and not from a caller violating a documented API contract. Any plausible mechanism that fails confirmation goes to `notes.limitations` with the exact reason.\n\nCWE discipline: the primary CWE names the confirmed mechanism AT THE CONFIRMED SINK; add at most one independently confirmed secondary. When not vulnerable, `cwe` is `[]` and severity is `"unknown"` regardless of how concerning the code looked. Choose the specific class over the generic one (CWE-20, CWE-119, CWE-787) whenever the mechanism is identifiable (CWE-468 for pointer scaling, CWE-170 for missing string termination, CWE-195 for signed-to-unsigned negative size, CWE-190 only when the wrap is itself the root cause — secondary to CWE-787/122 when it corrupts a size; CWE-697/CWE-682 for an incorrect comparison/operator that admits a security-relevant boundary). Severity: critical for remote code execution or full auth bypass; high for direct memory corruption or privilege escalation; medium for localized overflow, injection, or leak; low for hardening-level or narrow-condition issues.\n\nConfidence: VULNERABLE >= 0.7 with an exact witness admitted by every guard plus a named absent fix; 0.55–0.7 with a sound residual-range analysis or a fragile-boundary witness; below 0.5 → demote to limitations. NOT VULNERABLE >= 0.7 with a decisive protective statement quoted with its full excluded set; 0.55–0.7 when safety rests on a complete inventory without a single decisive statement; <= 0.55 when no dangerous operation exists.\n\n## Rules\n- Base all reasoning only on the shown fragment; do not rewrite or modify it.\n- Follow actual control flow (paths, checks, state changes); track real values, types, and exact bounds; quote line numbers.\n- Provenance (Phase 1) decides the residual range to analyze; an unconstrained range alone confirms nothing — the dangerous operation or missing check must actually be in the fragment.\n- Ground every safe verdict in protective statements actually visible in the fragment that pass the Guard-Identity Audit; absence of a demonstrated flaw is a weaker verdict, not a stronger one.\n- Ground every vulnerable verdict in a capacity, producer, access convention, or security context visible in the fragment, a witness admitted by every guard on the path, and a named absent fix.\n- End the reasoning with the per-sink Fix-Presence conclusion: for each dangerous operation, PRESENT (protective statement + its excluded set) or ABSENT (missing fix + surviving witness), and state which outcome is decisive for the verdict.\n- Distinguish a confirmed, reachable, attacker-triggerable vulnerability from a defensive-hardening or robustness concern.\n- Do NOT add few-shot examples.\n\n## Output format\n- First: write your SCoT reasoning in plain text with numbered steps (Phases 1–4), explicitly working through Sequence, Branch, and Loop structures, including the per-sink Fix-Presence results and the Guard-Identity Audit outcomes. Keep it concise — 2–6 lines per phase.\n- Then: output ONLY one JSON object. It must be the LAST thing in your response — no markdown code fences, no text before the first `{`, no text after the closing `}`. The JSON must be complete; do not truncate it.\n\nRequired JSON schema (fields and types must match exactly — do not change):\n{\n  "verdict": {\n    "vulnerable": true/false,\n    "confidence": 0.0-1.0,\n    "cwe": ["CWE-..."],\n    "severity": "low|medium|high|critical|unknown",\n    "summary": "..."\n  },\n  "evidence": [\n    {\n      "file": null,\n      "function": null,\n      "location": {"start_line": null, "end_line": null},\n      "snippet": "...",\n      "reason": "..."\n    }\n  ],\n  "notes": {\n    "assumptions": [],\n    "limitations": []\n  }\n}\n\n## Evidence precision requirements\n- For each evidence item, set `location.start_line` and `location.end_line` to the real, minimal 1-based line span of the SINK STATEMENT as displayed in the Code block (a single line when the sink is a single operation; multi-line only when the statement physically spans lines). Never the guard line, the function header, or a multi-statement range.\n- Set `snippet` to the exact code at those lines, quoted verbatim — no paraphrasing, ellipses, substitutions, or comment text.\n- Set `reason` to tie the specific index/pointer/size/value to its Phase 1 residual-range source and the Phase 4 confirmed failure, naming the capacity line, the guard\'s admitted set and its insufficiency (or the drift class if the guard fails the audit), the absent fix statement, and the primary CWE\'s mechanism, all INSIDE the reason.\n- Keep `notes.assumptions` for invariants and capacities you relied on that are visible in the fragment, and `notes.limitations` for demoted candidates, unconfirmed hypotheses, unshown-environment bypasses, and out-of-scope concerns.\n\nCode:\n{{CODE}}'

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
