"""Pure deterministic analysis of one prompt template."""
from __future__ import annotations
import hashlib, importlib.metadata, re, unicodedata
from collections import Counter
from typing import Any
from .config import StaticConfig
from .schema_parser import extract_schema, REQUIRED_LEAF_PATHS

VAR_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
MALFORMED_RE = re.compile(r"\{\{[^\n]*?\}\}?|\{[^\n]*\}\}")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:s)?\b|\b(?:authn|authz)\b")

def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFC", prompt).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+$", "", line) for line in text.split("\n"))
    return text.rstrip("\n") + "\n"

def _tokens(text: str, config: StaticConfig) -> tuple[int, str]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(config.tokenizer_encoding)
        return len(enc.encode(text)), f"{config.tokenizer_package}=={importlib.metadata.version(config.tokenizer_package)}:{config.tokenizer_encoding}"
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE)), f"fallback:{config.tokenizer_encoding}"

def _schema_counts(result: Any) -> dict[str, Any]: return result.to_dict()

def analyze_prompt(prompt: str | bytes, seed_prompt: str | bytes | None = None, config: StaticConfig | None = None) -> dict[str, Any]:
    config = config or StaticConfig(); invalid_utf8 = False
    if isinstance(prompt, bytes):
        raw_bytes = prompt
        try: text = prompt.decode("utf-8")
        except UnicodeDecodeError: text, invalid_utf8 = prompt.decode("utf-8", "replace"), True
    else:
        text, raw_bytes = prompt, prompt.encode("utf-8")
    normalized = normalize_prompt(text)
    lines = text.splitlines(); normalized_lines = normalized.splitlines()
    variables = []
    for m in VAR_RE.finditer(text): variables.append({"name": m.group(1), "start": m.start(), "end": m.end()})
    known = [v["name"] for v in variables]
    malformed = []
    for opening in re.finditer(r"\{\{", text):
        closing = text.find("}}", opening.end())
        candidate = text[opening.start():closing + 2] if closing >= 0 else text[opening.start():]
        if closing < 0 or not VAR_RE.fullmatch(candidate): malformed.append(candidate)
    unknown = sorted(set(v for v in known if v not in config.allowed_variables))
    ref_tokens, tokenizer = _tokens(text, config); seed_tokens = None
    if seed_prompt is not None:
        seed_text = seed_prompt.decode("utf-8", "replace") if isinstance(seed_prompt, bytes) else seed_prompt
        seed_tokens, _ = _tokens(seed_text, config)
    headings = [re.sub(r"^#{1,6}\s+", "", line).strip() for line in lines if re.match(r"^#{1,6}\s+", line)]
    labels = [line.strip() for line in lines if re.match(r"^\s*[^#\-*\d].*:\s*$", line)]
    bullets = [line for line in lines if re.match(r"^\s*[-*+]\s+", line)]
    numbered = [line for line in lines if re.match(r"^\s*\d+[.)]\s+", line)]
    duplicate_headings = sorted(k for k, v in Counter(re.sub(r"\s+", " ", h).strip().lower().rstrip(".!?:;") for h in headings).items() if v > 1)
    directives = []
    for line in lines:
        x = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line); x = re.sub(r"\s+", " ", x.strip()).lower(); directives.append(re.sub(r"[.!?:;]+$", "", x))
    duplicate_directives = sorted(k for k, v in Counter(directives).items() if k and v > 1)
    control = [i for i, c in enumerate(text) if (ord(c) < 32 and c not in "\n\r\t") or ord(c) == 127]
    trailing = [i + 1 for i, line in enumerate(lines) if re.search(r"[ \t]+$", line)]
    tabs = [i + 1 for i, line in enumerate(lines) if "\t" in line]
    endings = set(re.findall(r"\r\n|\r|\n", text)); fences = len(re.findall(r"(?m)^\s*```", text)); runs = [len(x) for x in re.findall(r"(?:^|\n)(?:[ \t]*\n)+", text)]
    acronyms = sorted(set(a for a in ACRONYM_RE.findall(text) if a not in config.acronym_allowlist and not re.search(rf"\b{re.escape(a)}\s*(?:\(|:)", text)))
    schema = extract_schema(text); schema_data = _schema_counts(schema)
    measurements = {"characters": len(text), "utf8_bytes": len(raw_bytes), "lines": len(lines), "blank_lines": sum(not x.strip() for x in lines), "words": len(re.findall(r"\b\w+\b", text, re.UNICODE)), "reference_tokens": ref_tokens, "seed_reference_tokens": seed_tokens, "token_delta_vs_seed": (ref_tokens - seed_tokens if seed_tokens is not None else None), "token_ratio_vs_seed": (ref_tokens / seed_tokens if seed_tokens else None), "tokenizer": tokenizer, "template_variables": variables, "code_placeholder_count": known.count("CODE"), "unknown_template_variables": unknown, "unknown_template_variable_count": len(unknown), "malformed_template_variables": malformed, "atx_heading_count": len(headings), "colon_label_count": len(labels), "unordered_bullet_count": len(bullets), "numbered_list_item_count": len(numbered), "duplicate_normalized_headings": duplicate_headings, "duplicate_normalized_directive_lines": duplicate_directives, "maximum_line_length": max((len(x) for x in lines), default=0), "trailing_whitespace_line_numbers": trailing, "tab_containing_line_numbers": tabs, "mixed_line_endings": len(endings) > 1, "disallowed_control_character_positions": control, "maximum_consecutive_blank_line_run": max(runs, default=0), "markdown_fence_count": fences, "markdown_fence_balance": fences % 2 == 0, "undefined_acronym_candidates": acronyms, "undefined_acronym_candidate_count": len(acronyms)}
    template_points = (4.0 * bool(text) * (not invalid_utf8) + 18.0 * (measurements["code_placeholder_count"] == 1) + 10.0 * (not unknown and not malformed) + 4.0 * (not control) + 4.0)
    present_required = len(REQUIRED_LEAF_PATHS) - len(schema.missing_required_leaf_paths)
    schema_points = (5.0 * schema.schema_block_found + 5.0 * (not schema.unbalanced_schema_delimiters) + 15.0 * present_required / len(REQUIRED_LEAF_PATHS) + 5.0 * (not schema.duplicate_schema_paths and not schema.unbalanced_schema_delimiters))
    efficiency = None if seed_tokens is None else (1.0 if ref_tokens <= seed_tokens else seed_tokens / ref_tokens if ref_tokens else 1.0)
    components = {"template_integrity": template_points, "schema_mechanics": schema_points, "format_hygiene": float(6 * (not duplicate_directives) + 3 * (not duplicate_headings) + 3 * (not (trailing or tabs or len(endings) > 1)) + 3 * (fences % 2 == 0)), "token_efficiency": (15.0 * efficiency if efficiency is not None else None)}
    total = None if seed_tokens is None else round(sum(v for v in components.values() if v is not None), 4)
    gates = [not text, invalid_utf8, measurements["code_placeholder_count"] != 1, bool(unknown or malformed), not schema.schema_block_found, schema.unbalanced_schema_delimiters, bool(schema.missing_required_leaf_paths)]
    findings = []
    if invalid_utf8: findings.append("invalid UTF-8 input")
    if unknown: findings.append("unknown template variables")
    if malformed: findings.append("malformed template variables")
    if acronyms: findings.append("undefined acronym candidates are heuristic only")
    return {"raw_sha256": hashlib.sha256(raw_bytes).hexdigest(), "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(), **measurements, **schema_data, "measurements": measurements, "schema": schema_data, "components": components, "quantitative_static_score": total, "static_eligible": not any(gates), "findings": findings}
