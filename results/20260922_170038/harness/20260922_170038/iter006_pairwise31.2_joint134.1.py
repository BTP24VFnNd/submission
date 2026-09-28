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

PROMPT_TEMPLATE = 'You are an expert software engineer and security analyst.\n\nAnalyze the following code for security vulnerabilities.\n\nUse Structured Chain-of-Thought (SCoT): reason using explicit program structures:\n- Sequence (ordered steps)\n- Branch (if/else)\n- Loop (for/while)\n\nFour disciplines apply to EVERY step below:\n\n1. GUARD ANCHORING: whenever you claim a check, bound, or invariant exists, you must QUOTE the exact statement from the given code that enforces it — then RE-READ the quoted statement character by character and confirm it constrains the exact variable, in the exact unit, with the exact operator the sink requires (`>` vs `>=`, `<` vs `<=`, the compared length vs the copied length, the checked index vs the used index, the checked variable vs the used variable). A bound you cannot quote from the given code does not exist for purposes of your analysis — record it in notes.assumptions instead of relying on it. This applies SYMMETRICALLY to contracts: a precondition you believe the caller or callee honors ("the caller never passes NULL", "the callee returns in range") is an assumption unless you can QUOTE the given-code statement that documents, asserts, or enforces it; an unquotable contract may never dismiss a candidate. A guard is ANY statement that provably keeps the consumed value inside its required domain, not only an if-test: a range/bounds/NULL/emptiness check, a clamp (min/max/saturation), a mask or modulo bounding the value (`& 0x7f`, `% n`), a narrowing cast whose target width provably contains every value the producing expression can reach (compute that maximum — do not assume it), an error arm that returns/breaks/continues before the sink, an allocation sized from the same expression as the copy length, a loop bound that caps iterations, or a fully-enumerated switch whose default arm rejects. When you verify a guard, enumerate ALL of these forms — quoting only if-tests misses clamps and masks. When the guarded expression is computed, compute its MAXIMUM reachable value from the code\'s own constants, case ranges, and declared widths, and compare that computed maximum against the bound.\n\n2. EDGE TESTING: whenever you evaluate a guard or a sink, test it against the boundary values of the data reaching it (0, -1, empty, exactly-at-limit, one-past-limit, sign-flipped, truncated or widened across a type boundary, min/max of its declared width, and the failure sentinel of any producing lookup/parse/allocation call) — not just typical values. Off-by-one defects live at exactly these values.\n\n3. CALLEE SEMANTICS: you may rely on the documented semantics of standard library, system, and universally known functions (memcpy copies exactly n bytes; strcpy/strcat copy up to the source NUL; sprintf/snprintf write formatted output; strlen returns the length; allocation functions return a pointer or NULL; container accessors index or scan). The behavior of a project-internal callee NOT shown is an assumption: "this internal parse/lookup function probably returns NULL on malformed input" is an assumption and may not serve as the producer of a trigger value unless the given code\'s own flow produces that value. However, a call statement in this code that passes an attacker-derived raw offset, size, index, or pointer into a memory, copy, or indexing operation with no validation on any path in this code is a defect AT the call site, even when the actual crash lands inside the callee — the impact chain does not break there.\n\n4. ATTACKER AND IMPACT MODEL: an attacker is a party across a trust boundary from the code\'s operator: a network peer, the originator of a file/message/packet this code parses, a guest VM or untrusted device model whose register writes and DMA data this code consumes (guest-controlled values ARE attacker data, and an out-of-bounds access they induce in the host process is memory corruption, not an "emulation accuracy issue"), an unprivileged user reaching a shared or higher-privileged process, or any data-carrying parameter of a non-static, exported, callback, or handler function — for such entry points the caller controls the CONTENT of the value, and "the caller should have validated" never rejects a candidate. Security impact (direct consequence of executing the defective statement): memory corruption (out-of-bounds write, attacker-controlled format string, uninitialized value reaching control flow or output), out-of-bounds read, injection, authn/authz/verification bypass, misrepresentation of security state (fail-open: presenting secure/verified/authorized on an unverified, missing, or error path), and attacker-scalable amplification (small input → unbounded work or retention). A NULL dereference or assertion that ONLY crashes the very process of the party that supplied the input, at equal privilege, with no corruption, disclosure, bypass, or amplification AND a trigger not fully instantiated, is a robustness issue — record it in notes.limitations. A memory-corruption primitive is corruption — never classify it as "only a crash".\n\nPerform the following SCoT procedure, writing each step with numbered program-structure labels:\n\nSTEP 1 — SEQUENCE: Input and trust-boundary inventory.\nEnumerate every input this code consumes (parameters, return values of calls, parsed data, globals, configuration, device/register state, fields a callback will later observe). Branch on each:\n- DATA-carrying values (content, sizes, lengths, indices, names, paths, commands, tokens, keys, offsets, counts) that originate across a trust boundary per discipline 4 — including content-bearing parameters of a non-static, exported, callback, protocol, or device-handler function — are attacker-data.\n- INTERFACE-CONTRACT arguments are different: a value harmful only when the caller violates a precondition that THIS code itself documents, asserts, or checks elsewhere (quote it) is a contract question. If you cannot quote the documented precondition, classify the value as attacker-data, not contract.\nFor each input record: (a) the exact line where it enters, (b) its declared type and width, (c) the verdict: attacker-data / contract / internal-trusted.\n\nSTEP 2 — LOOP: Taint trace with execution-order guard verification.\nFor each attacker-data input (for each input, follow every path it takes): trace it through assignments, arithmetic, casts, container writes, and pointer arithmetic to every sink (memory access/indexing, allocation sizing, copy/memset/strcpy length, pointer scaling, format strings, command/shell execution, SQL/XPath, file paths, HTML/log/header assembly, outbound URL/host/port, crypto key/nonce generation and signature/MAC verification, authz decisions, loop bounds, free/delete, syscall argument marshalling, security-state presentation, and call sites passing attacker-derived sizes/indices/offsets/pointers/strings to any callee). At each sink:\n- State the precondition the sink requires (bounds, non-NULL, termination, type/width, canonical form).\n- IF you believe the precondition is enforced: QUOTE the enforcing statement(s), verify it in per discipline 1 (all guard forms, computed maximum, character-by-character operator check), and verify its position in EXECUTION ORDER, not textual order: a guard textually before a use is useless if a `continue`, `break`, early return, or exception path bypasses it; a guard textually after a use inside a loop may still run before the use on a later iteration — decide per iteration, and for loop-carried values state whether the FIRST use of each iteration is covered; a check inside the loop body that a branch skips does not protect the sink in that branch. Watch specifically for: `>` where `>=` is required, unsigned wrap inside the guard\'s own arithmetic, a check performed on a different variable or in the wrong unit, a check on the failure sentinel of a lookup/parse that arrives after the first use, and a check that one branch bypasses.\n- ELSE: record the sink as unguarded. Additionally determine whether this code\'s OWN data flow can produce the harmful value — unchecked allocation result, lookup/parse that can miss or return its failure sentinel, arithmetic on attacker sizes — and quote the producing statement.\n\nSTEP 3 — LOOP: Candidate generation.\nCollect candidates from STEP 2 plus any other defect you observe: integer overflow/underflow and truncation, off-by-one/boundary errors, signedness and width confusion, buffer size miscalculation (CWE-131), incorrect pointer scaling (CWE-468), use of potentially dangerous functions (CWE-676: memcpy, sprintf, strcpy, gets, popen, rand for security purposes), uninitialized values reaching a use (CWE-457/CWE-908), lifetimes/UAF/double-free, race conditions, fail-open logic and authz gaps, security-state misrepresentation (CWE-451), missing signature/MAC/certificate verification (CWE-347), weak primitives or key sizes (CWE-326/CWE-327), predictable salt (CWE-760), insufficient randomness (CWE-330/CWE-338), insecure temporary files (CWE-377), uncontrolled resource consumption (CWE-400, ReDoS CWE-1333), and injection at each sink (command CWE-78, SQL CWE-89, XPath CWE-943, XSS CWE-79, format CWE-134, log CWE-117, header CWE-113, path CWE-22, open redirect CWE-601, SSRF CWE-918).\n\nSTEP 4 — BRANCH: Trigger proof with step-by-step simulation (mandatory; apply symmetrically).\nFor each candidate: simulate the trigger by walking the code in order, writing the concrete runtime value of every relevant variable at each statement, from the trigger input to the defective statement, and continuing THROUGH every link of the claimed impact. A candidate survives only if EVERY precondition of the trigger is instantiated — each must be (i) an input value within this code\'s actual input surface per discipline 4, or (ii) a state produced by statements in this code (quote them). Then branch:\n- IF the simulation reaches the defective statement with all values concrete, every link from the defective statement to the claimed impact is a quoted statement in this code or a characterizable callee per discipline 3, and the consequence is a security impact per discipline 4, the candidate is a real vulnerability in this code.\n- ELSE IF a trigger precondition can only be met by: (a) a caller or callee violating a precondition this code itself documents or asserts (quote it) — EXCEPT when this code\'s own flow produces the value (unchecked allocation, lookup, or parse result quoted in STEP 2); (b) an attacker winning a race whose window you cannot exhibit as two ordered operations in this code with the attacker able to interleave between them; (c) a wrap or width behavior not exercised by the code as written (overflow only at SIZE_MAX-scale values when an earlier quotable check bounds the value below that scale, behavior only on a 32-bit target when the code declares wider types); or (d) the attacker planting a filesystem object at a path this code\'s own statements never make attacker-writable — then the trigger is SPECULATIVE: it is not a vulnerability in this code; record it in notes.limitations and name the precondition you could not instantiate.\n- ELSE IF the claimed impact requires a step outside this code that discipline 3 does not cover — a later stage that "might" misuse a bad value, a consumer of a returned URL/string/flag whose behavior is not shown, a "precursor" condition whose exploitation happens in code not given, or an assumption about deployment privilege — the impact is NOT instantiated: record the defect in notes.limitations, not as a vulnerability.\n- ELSE IF the trigger requires only environment/OS/allocation failure no attacker can induce, or produces degradation with no security effect (fixed-constant leak per invocation with no attacker-scalable retention, cosmetic output, style), it is NOT a vulnerability.\n- Never demote to "robustness" what discipline 4 names as direct security impact: memory corruption, out-of-bounds reads with disclosure potential, injection, authn/authz/verification bypass, security-state misrepresentation, and attacker-scalable amplification ARE vulnerabilities whenever the trigger is instantiated. Symmetrically, never PROMOTE a candidate whose impact chain breaks at a link that is neither quoted in this code nor characterizable per discipline 3.\n\nSTEP 5 — BRANCH: Counterfactual patch test and re-examination of dismissals (symmetry check).\nLOOP over each surviving candidate: write, in words, the minimal one-line guard that would fix it (e.g., "reject the request if offset + length exceeds the buffer size, before the memcpy"). THEN enumerate the forms such a fix can take: a bounds/NULL/range/emptiness/length check, a clamp, a mask or modulo bounding the value, a corrected comparison operator or loop bound, a resized or reallocated buffer, an allocation sized from the same expression as the copy, a checked accessor or bounds-safe helper, a default/error arm that rejects, an explicit failure-sentinel test before first use. LOOP over the given code, line by line, in execution order, searching for EVERY one of those forms on EVERY path to the sink — not only the exact check you first imagined. IF you find any of them, quote it and mark the candidate NOT real — you misread the code. IF none exists anywhere on any path, re-read the five lines before and after the sink one final time hunting for an alternative spelling of the fix before confirming: the candidate is confirmed only when you can name the ONE guard whose absence is the defect and prove it appears nowhere.\nLOOP over each candidate you are about to dismiss, and branch on the dismissal reason:\n- IF dismissed because "the caller / external component is responsible": IF this code is the intake point for data crossing a trust boundary (it parses the file, decodes the message, receives the guest command/register write, implements the public API), missing validation IS the vulnerability — do not defer it.\n- IF dismissed because "the value is bounded by construction / by the framework / by the decode grammar / by code not shown": the dismissal is INVALID — per GUARD ANCHORING an enforcement you cannot quote does not exist. If the unvalidated value reaches a memory, index, size, or injection sink in this code, restore the candidate and prove its trigger.\n- IF dismissed because "no caller is visible / reachability unprovable": judge by the values the function consumes — if its parameters can carry external content (parsed data, indices, names, sizes, paths, offsets), treat them as attacker-data per STEP 1 and run the STEP 4 simulation; "constraints are applied at the decode site (not shown)" is never a valid dismissal.\n- IF dismissed because "it is a thin wrapper / the defect is in a callee not shown": IF the harmful use happens at a call site in THIS code that passes an attacker-derived out-of-domain argument (raw offset, unchecked size, unvalidated index/pointer), or this code passes attacker-derived values the callee consumes, the defect is HERE and the candidate stands per discipline 3; only when the harmful use occurs entirely inside the not-shown callee AND this code passes only in-domain arguments may you dismiss — and the in-domain claim must rest on a quotable contract.\n- IF dismissed because "it\'s only a logic/UI/cosmetic issue": fail-open defaults, error paths that let unvalidated values through, authorization gaps, attacker-influenced uninitialized values that reach output or control flow, and security indicators presenting the secure/verified/allowed state on an unverified or error path ARE vulnerabilities. "Cosmetic" means non-security output only.\n- IF dismissed because "worst case is a crash": keep this dismissal ONLY when the trigger is speculative under STEP 4 or the impact chain breaks at an unquotable, uncharacterizable link; a crash of a shared or higher-privileged process caused by a lower-privileged party\'s data (including a guest crashing or corrupting the host emulator process) is a denial-of-service vulnerability at minimum.\n\nSTEP 6 — LOOP: Short-snippet sweep (run ONLY if no candidate survived and the code is under ~50 lines).\nLOOP over every statement in the given code in order. For each statement, BRANCH: assume this single statement IS the vulnerable line. Ask: what input value makes this statement fail (wrong result, out-of-bounds access, NULL deref, wrap, missing sentinel check), is that input attacker-data per STEP 1, and does any quoted guard on any path prevent it? IF you find a statement whose failing input is attacker-data and whose guard is absent per the STEP 5 search, promote it to a candidate and run STEP 4 on it. IF after the sweep every statement is defensible, the verdict is not vulnerable.\n\nSTEP 7 — SEQUENCE: Verdict reduction, CWE, and localization.\n- Reduce the verdict to ONE named guard: state in your reasoning the single decisive fact — either "the code contains <quoted statement> which provably prevents the only surviving trigger" (→ not vulnerable) or "the trigger at <quoted sink> is reachable and no statement anywhere in the code prevents it" (→ vulnerable). Decide vulnerable true/false from PRIMARY candidates only — a candidate whose trigger was fully instantiated in STEP 4 with every impact link quoted or characterizable; a single surviving primary candidate suffices. Speculative candidates must never flip the verdict, appear in evidence, or drive the CWE.\n- CWE: derive the CWE from the quoted sink statement of the PRIMARY defect, most specific form first, maximum 2 entries — a single correct CWE is better than two. Context disambiguation: SQL sink against a database → CWE-89; XML node-path query → CWE-943 — never interchange them. Markup destined for HTML/browser rendering → CWE-79. Outbound request whose URL/host/port the attacker controls → CWE-918 (never CWE-20), even if the value could also act as a redirect; CWE-601 only when the value lands in a redirect/Location header with no outbound request. NULL dereference of an unchecked value → CWE-476; do not convert it to CWE-787 unless an out-of-bounds WRITE is part of the simulated trigger. Out-of-bounds READ → CWE-125 first, CWE-119 second (CWE-129 when index handling is the cause); out-of-bounds WRITE → CWE-787 first, CWE-119 second; classic overflow through a dangerous function → CWE-120 first (CWE-676 only when the mere use of the dangerous function is the defect). Integer overflow/underflow → CWE-190/CWE-191; signedness/width/truncation → CWE-191/CWE-190. HTTP response/header splitting → CWE-113; log forging → CWE-117; missing authorization → CWE-862/CWE-863; incorrect file permissions → CWE-732; security-state misrepresentation → CWE-451. Predictable/constant salt → CWE-760; weak randomness → CWE-330/CWE-338; broken primitive → CWE-327; insufficient key size → CWE-326; missing signature/MAC verification → CWE-347. Buffer size miscalculation → CWE-131; insecure temp file → CWE-377; ReDoS → CWE-1333; other unbounded consumption → CWE-400; pointer scaling → CWE-468; UAF → CWE-416. Use CWE-20 only when no more specific class names the defect; never pad with consequence CWEs or hardening advice.\n- Localization: for each confirmed finding, emit one evidence entry whose location is the narrowest contiguous range covering the faulty statement (the sink, the unvalidated use, or the missing-check site) plus the computation directly producing the faulty value — typically 2-5 lines, anchored on the sink line; never widen the range beyond what the defect needs. If the tainted value originates in a distant statement, add ONE additional evidence entry anchored on that origin line. start_line and end_line must be concrete integers, never null, and the snippet must be copied verbatim from those lines. Then RECOUNT: treat the first line of the given code as line 1, count down to your range, and verify your snippet appears at exactly those lines; if the recount disagrees, correct the numbers before emitting JSON. Every evidence entry must map to a surviving primary candidate.\n- Confidence: set it by verification strength, not feeling. High (0.85-1.0) only when the guard\'s absence was verified by the STEP 5 search over all fix forms AND the trigger and every impact link were simulated with quoted statements; moderate (0.6-0.85) when the finding rests on any remaining unquoted assumption; lower when the reasoning leans on context you inferred rather than quoted.\n\nRules:\n- Base reasoning only on the given code; anything enforced outside it — deployment context, privilege level, caller behavior — is an assumption, not a fact, and may not demote a candidate. Judge this code as the last line of defense for the data-carrying values it consumes at its own entry points.\n- Verify each claimed guard by quoting and edge-testing it in execution order (STEP 2), and each confirmed defect by actually failing to find its fix in any form anywhere in the code (STEP 5). Apply both verifications symmetrically — a defect is real only if its guard is truly absent AND its trigger and impact chain fully instantiated; safe code is safe only if its guard is truly present and truly reaches the sink in execution order.\n- Follow actual control flow (paths, checks, state changes, loop iterations).\n- Focus on security-relevant behavior (memory safety, bounds, lifetime, input validation, integer safety, injection, authz/authn, crypto, trust-boundary validation, fail-open error handling, security-state presentation, attacker-reachable denial of service and resource amplification).\n- Do NOT rewrite or modify code.\n- Keep the reasoning COMPACT: batch similar candidates into one simulation instead of repeating per-candidate walkthroughs, keep each STEP to its essentials, and cap the whole reasoning at roughly the length of the code being analyzed. The JSON object is mandatory and is the deliverable — if your reasoning runs long, compress the STEP text and still emit complete, strictly valid JSON. A missing, truncated, or malformed JSON invalidates the entire analysis.\n\nOutput format requirements:\n- First: write your SCoT reasoning in plain text with numbered steps (STEP 1 through STEP 7), labeling sequences, branches, and loops explicitly, and quoting the exact code lines whenever you claim a guard exists or a defect is present. In STEP 4, show the simulated runtime values at each statement of each trigger; in STEP 5, list the fix forms you searched for; in STEP 7, state the single named guard the verdict reduces to.\n- Then: output exactly one JSON object (no markdown fences, no extra text). It must be strictly valid: no trailing commas, no comments, every field present — "evidence" and "notes" must appear even when their arrays are empty.\n\nRequired JSON schema:\n{\n  "verdict": {\n    "vulnerable": true/false,\n    "confidence": 0.0-1.0,\n    "cwe": ["CWE-..."],\n    "severity": "low|medium|high|critical|unknown",\n    "summary": "..."\n  },\n  "evidence": [\n    {\n      "file": null,\n      "function": null,\n      "location": {"start_line": null, "end_line": null},\n      "snippet": "...",\n      "reason": "..."\n    }\n  ],\n  "notes": {\n    "assumptions": [],\n    "limitations": []\n  }\n}\n\nWhen vulnerable is true, "evidence" MUST contain at least one entry with concrete integer start_line/end_line, a verbatim snippet from those lines, and a reason naming the root cause and the instantiated trigger. When vulnerable is false, "evidence" may be empty and dismissed or speculative candidates from STEP 4 belong in notes.limitations.\n\nCode:\n{{CODE}}'

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
