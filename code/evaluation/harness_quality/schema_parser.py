"""Deterministic permissive scanner for JSON-like prompt schemas."""
from __future__ import annotations
import json, re
from collections import Counter
from dataclasses import dataclass
from typing import Any

REQUIRED_LEAF_PATHS = ("verdict.vulnerable", "verdict.confidence", "verdict.cwe", "verdict.severity", "verdict.summary", "evidence[].file", "evidence[].function", "evidence[].location.start_line", "evidence[].location.end_line", "evidence[].snippet", "evidence[].reason", "notes.assumptions", "notes.limitations")

@dataclass
class SchemaResult:
    schema_block_found: bool = False; schema_start_offset: int | None = None; schema_end_offset: int | None = None
    strict_json_valid: bool = False; pseudo_schema_parse_valid: bool = False; schema_key_count: int = 0
    schema_container_key_count: int = 0; schema_leaf_field_count: int = 0; schema_field_paths: list[str] | None = None
    schema_max_depth: int = 0; schema_array_count: int = 0; duplicate_schema_paths: list[str] | None = None
    missing_required_leaf_paths: list[str] | None = None; unbalanced_schema_delimiters: bool = False
    def __post_init__(self) -> None:
        self.schema_field_paths = self.schema_field_paths or []
        self.duplicate_schema_paths = self.duplicate_schema_paths or []
        self.missing_required_leaf_paths = self.missing_required_leaf_paths or list(REQUIRED_LEAF_PATHS)
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()

def _find_block(prompt: str) -> tuple[int, int | None, str | None]:
    match = re.search(r"(?im)\b(?:required\s+)?(?:json\s+)?schema\b|output\s+format", prompt); offset = match.end() if match else 0
    opening = re.search(r"[\[{]", prompt[offset:])
    if not opening: return -1, None, None
    start, stack, quote, escaped = offset + opening.start(), [], False, False
    for i in range(start, len(prompt)):
        c = prompt[i]
        if quote:
            if escaped: escaped = False
            elif c == "\\": escaped = True
            elif c == '"': quote = False
        elif c == '"': quote = True
        elif c in "[{": stack.append(c)
        elif c in "]}":
            if not stack or (stack[-1], c) not in (("[", "]"), ("{", "}")): return start, None, prompt[start:]
            stack.pop()
            if not stack: return start, i + 1, prompt[start:i + 1]
    return start, None, prompt[start:]

def _extract_paths(block: str) -> tuple[list[str], list[str], int, int, bool]:
    paths: list[str] = []; containers: set[str] = set(); arrays = 0; i = 0
    def ws() -> None:
        nonlocal i
        while i < len(block) and block[i].isspace(): i += 1
    def quoted() -> str | None:
        nonlocal i
        ws()
        if i >= len(block) or block[i] != '"': return None
        start = i; i += 1; esc = False
        while i < len(block):
            c = block[i]; i += 1
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': return json.loads(block[start:i])
        return None
    def scalar() -> None:
        nonlocal i
        quote = False; esc = False
        while i < len(block):
            c = block[i]
            if quote:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': quote = False
            elif c == '"': quote = True
            elif c in ",}]": return
            i += 1
    def value(path: str) -> None:
        nonlocal i, arrays
        ws()
        if i < len(block) and block[i] == "{": containers.add(path); i += 1; obj(path)
        elif i < len(block) and block[i] == "[":
            arrays += 1; i += 1; ws()
            while i < len(block) and block[i] != "]":
                value(path + "[]"); ws()
                if i < len(block) and block[i] == ",": i += 1; ws()
            if i < len(block) and block[i] == "]": i += 1
        else: scalar()
    def obj(prefix: str) -> None:
        nonlocal i
        ws()
        while i < len(block) and block[i] != "}":
            key = quoted(); ws()
            if key is None or i >= len(block) or block[i] != ":": i += 1; continue
            i += 1; path = f"{prefix}.{key}" if prefix else key; paths.append(path); value(path); ws()
            if i < len(block) and block[i] == ",": i += 1; ws()
        if i < len(block) and block[i] == "}": i += 1
    ws()
    if i < len(block) and block[i] == "{": i += 1; obj("")
    elif i < len(block) and block[i] == "[": i += 1; value("[]")
    counts = Counter(paths)
    return paths, sorted(k for k, v in counts.items() if v > 1), len(containers), arrays, _unbalanced(block)

def _unbalanced(block: str) -> bool:
    stack: list[str] = []; quote = False; escaped = False
    for c in block:
        if quote:
            if escaped: escaped = False
            elif c == "\\": escaped = True
            elif c == '"': quote = False
        elif c == '"': quote = True
        elif c in "[{": stack.append(c)
        elif c in "]}":
            if not stack or (stack[-1], c) not in (("[", "]"), ("{", "}")): return True
            stack.pop()
    return bool(stack or quote)

def extract_schema(prompt: str) -> SchemaResult:
    start, end, block = _find_block(prompt); result = SchemaResult(start >= 0, start if start >= 0 else None, end)
    if not block: return result
    try: json.loads(block); result.strict_json_valid = True
    except (json.JSONDecodeError, TypeError): pass
    paths, duplicates, containers, arrays, malformed = _extract_paths(block)
    result.schema_field_paths, result.duplicate_schema_paths = sorted(set(paths)), duplicates
    result.schema_key_count, result.schema_container_key_count = len(paths), containers
    result.schema_leaf_field_count = len(paths) - containers; result.schema_max_depth = max((p.count(".") + 1 for p in result.schema_field_paths), default=0)
    result.schema_array_count, result.unbalanced_schema_delimiters = arrays, end is None or malformed
    result.pseudo_schema_parse_valid = bool(paths) and not result.unbalanced_schema_delimiters
    result.missing_required_leaf_paths = [p for p in REQUIRED_LEAF_PATHS if p not in result.schema_field_paths]
    return result
