#!/usr/bin/env python3
"""
LLM-as-optimizer (agentic mode): the optimizer model (Claude or Codex,
per optimizer_model.backend) browses the optimization trajectory workspace
to identify failure patterns and proposes an improved prompt.

The optimizer gets filesystem access to:
  - trajectory/prompts/     Past iteration prompt templates
  - trajectory/metrics/     Per-iteration metrics JSONs
  - trajectory/predictions/ Past predictions with reasoning (labels stripped)
  - trajectory/history.json Full scoring history

It does NOT see ground truth labels. It must infer failure patterns from
its own reasoning and the aggregate metrics.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _claude_binary(config: Dict[str, Any]) -> str:
    configured = config.get("optimizer_model", config.get("model", {})).get("binary") \
        or os.environ.get("CLAUDE_BIN")
    binary = configured or shutil.which("claude")
    if not binary:
        raise FileNotFoundError("Claude CLI not found; install it or set CLAUDE_BIN")
    return binary


OPTIMIZER_SYSTEM = """\
You are a prompt optimization expert with access to a workspace containing
the full optimization trajectory for a vulnerability detection task.

Browse the workspace to understand what has been tried and what went wrong:
  trajectory/prompts/      — prompt templates from each iteration
  trajectory/metrics/      — per-iteration metrics (precision, recall, F1, pairwise accuracy)
  trajectory/predictions/  — past predictions with model reasoning (NO ground truth labels)
  trajectory/history.json  — scoring history across all iterations
  current_prompt.txt       — the prompt template you are improving

Your task: analyze the trajectory, identify systematic failure patterns in
the predictions, and propose an improved prompt template.

STRUCTURAL CONSTRAINT — Structured Chain-of-Thought (SCoT):
The prompt MUST retain the SCoT reasoning scaffolding. The model must reason
using explicit program structures:
- Sequence (ordered steps)
- Branch (if/else)
- Loop (for/while)
Your improvements should ENHANCE the SCoT framework, not replace it.

DO NOT:
- Remove or replace the SCoT scaffolding with a different reasoning approach.
- Add bias-correction framing like "lean toward not vulnerable" or
  "when in doubt, say not vulnerable."
- Add few-shot examples (they eat context window).
- Change the JSON output schema fields or their types.

You CAN change:
- What the model analyzes within each SCoT step.
- Analysis ordering within the SCoT structure.
- How evidence is gathered and anchored to specific code locations.
- Emphasis on specific vulnerability classes or patterns.
- Instructions for CWE identification and localization accuracy.

Focus on addressing the specific failure patterns you find in the trajectory.

The workspace is READ-ONLY. You have no filesystem write access — do not try
to create or edit files, and do not ask for write permission. Your entire
response is captured verbatim as the new prompt template.

Output ONLY the new prompt template text, nothing else. No explanations,
no preamble, no markdown fences. Begin your response with "You are ".
The template must contain a {{CODE}} placeholder.\
"""


def propose_new_prompt(
    current_prompt: str,
    metrics: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
    *,
    workspace: Optional[Path] = None,
) -> str:
    """
    Call the optimizer model to propose a new prompt template.

    In agentic mode (workspace provided), Claude browses the trajectory
    itself. In isolated mode (no workspace), falls back to bundling
    failure examples into the prompt.
    """
    if workspace:
        return _propose_agentic(current_prompt, workspace, config)
    else:
        return _propose_isolated(current_prompt, metrics, predictions, samples, config, history)


# Mirrors REQUIRED_TOP_KEYS / REQUIRED_VERDICT_KEYS in
# claude-cli-agent/runner/output_parser.py. The optimizer is instructed not to
# change the output schema; this enforces it before a run is spent.
REQUIRED_SCHEMA_FIELDS = (
    "verdict", "evidence", "notes",
    "vulnerable", "confidence", "cwe", "severity", "summary",
)


def _clean_new_prompt(raw: str) -> str:
    """Extract the bare prompt template from the optimizer's stdout.

    The agentic optimizer browses the workspace with tools and reliably emits a
    conversational lead-in before its final answer ("Now I have a thorough
    understanding... Let me craft the improved prompt template:"). Its whole
    stdout is captured verbatim as the next prompt, so that narration would
    otherwise be prepended to the template and shipped to every sample.

    Anchor on the template's opening instead of trusting the model to obey the
    "output only the template" instruction, which it does not.
    """
    text = raw.strip()

    # Drop a wrapping markdown fence if present.
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    idx = text.find("You are ")
    if idx == -1:
        first_line = text.split("\n", 1)[0][:160]
        raise ValueError(
            "Optimizer output contains no prompt template "
            f"(no 'You are ' anchor found). Leading text: {first_line!r}"
        )
    if idx > 0:
        print(f"  [warn] stripped {idx} chars of optimizer preamble: "
              f"{text[:idx].strip()[:120]!r}")
        text = text[idx:]

    _validate_new_prompt(text)
    return text


def _validate_new_prompt(new_prompt: str) -> None:
    """Reject a template that would break the evaluation run."""
    if "{{CODE}}" not in new_prompt:
        raise ValueError("Optimizer produced a prompt without {{CODE}} placeholder")

    missing = [f for f in REQUIRED_SCHEMA_FIELDS if f'"{f}"' not in new_prompt]
    if missing:
        raise ValueError(
            "Optimizer dropped required output schema field(s) from the prompt: "
            f"{', '.join(missing)}. The parser would fail on every sample."
        )


AGENTIC_USER_PROMPT = (
    "Review the optimization trajectory in this workspace. "
    "Read the history, metrics, past prompts, and predictions to identify "
    "systematic failure patterns. Then propose an improved prompt template.\n\n"
    "Output ONLY the new prompt template text."
)


def _propose_agentic(
    current_prompt: str,
    workspace: Path,
    config: Dict[str, Any],
) -> str:
    """Agentic optimizer: the optimizer model browses the workspace to find failures.

    Dispatches by optimizer_model.backend. Claude uses the Claude CLI directly
    (below); Codex delegates to codex-self-improving-agent's workspace-aware
    model client, which already runs `codex exec --cd <workspace>` under a
    read-only permission profile scoped to that workspace.
    """
    (workspace / "current_prompt.txt").write_text(current_prompt)

    optimizer_backend = config.get("optimizer_model", config["model"]).get("backend", "claude-cli")
    if optimizer_backend == "codex-cli":
        return _propose_agentic_codex(workspace, config)
    return _propose_agentic_claude(workspace, config)


def _propose_agentic_codex(workspace: Path, config: Dict[str, Any]) -> str:
    import importlib.util
    codex_agent_path = str(Path(__file__).resolve().parent.parent.parent
                           / "codex-self-improving-agent" / "runner" / "model_client.py")
    spec = importlib.util.spec_from_file_location("codex_self_improving_model_client", codex_agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    optimizer_config = {
        "model": config.get("optimizer_model", config["model"]),
        "runner": config["runner"],
        "isolation": config.get("isolation", {}),
    }
    result = mod.call_model(AGENTIC_USER_PROMPT, optimizer_config,
                             workspace=workspace, system_prompt=OPTIMIZER_SYSTEM)
    if result["error"]:
        raise RuntimeError(f"Optimizer call failed: {result['error']}")
    return _clean_new_prompt(result["raw"])


def _propose_agentic_claude(workspace: Path, config: Dict[str, Any]) -> str:
    user_prompt = AGENTIC_USER_PROMPT

    timeout_s = config["runner"]["timeout_s"]
    model = config.get("optimizer_model", config["model"]).get("name", "")

    cmd = [_claude_binary(config), "-p", "--system-prompt", OPTIMIZER_SYSTEM]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(workspace),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Optimizer timed out after {timeout_s}s")

    if result.returncode != 0:
        # The CLI reports errors (rate limits, auth, bad model) on stdout, not
        # stderr, so fall back to stdout before the bare exit code.
        err = (result.stderr.strip() or result.stdout.strip()
               or f"exit code {result.returncode}")
        raise RuntimeError(f"Optimizer call failed (exit {result.returncode}): {err[:500]}")

    return _clean_new_prompt(result.stdout)


# ---------------------------------------------------------------------------
# Isolated fallback (old behavior, no filesystem access)
# ---------------------------------------------------------------------------

ISOLATED_OPTIMIZER_SYSTEM = """\
You are a prompt optimization expert. Your job is to improve a vulnerability detection prompt template.

You will be given:
1. The current prompt template (with a {{CODE}} placeholder)
2. Performance metrics on a validation set
3. Examples of failures (false positives, false negatives, wrong CWE, missed localization)
4. History of previous prompt versions and their scores

Your task: propose an improved prompt template that will perform better on the target metric.

STRUCTURAL CONSTRAINT — Structured Chain-of-Thought (SCoT):
The prompt MUST retain the SCoT reasoning scaffolding. The model must reason using explicit program structures:
- Sequence (ordered steps)
- Branch (if/else)
- Loop (for/while)
Your improvements should ENHANCE the SCoT framework, not replace it.

DO NOT:
- Remove or replace the SCoT scaffolding with a different reasoning approach.
- Add bias-correction framing like "lean toward not vulnerable" or "when in doubt, say not vulnerable."
- Add few-shot examples (they eat context window).
- Change the JSON output schema fields or their types.

You CAN change:
- What the model analyzes within each SCoT step.
- Analysis ordering within the SCoT structure.
- How evidence is gathered and anchored to specific code locations.
- Emphasis on specific vulnerability classes or patterns.
- Instructions for CWE identification and localization accuracy.

Focus on addressing the specific failure patterns shown in the examples.

Output ONLY the new prompt template text, nothing else. No explanations, no markdown fences."""


def _propose_isolated(
    current_prompt: str,
    metrics: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    """Isolated optimizer: bundles failure examples into the prompt."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
    from model_client import call_model

    failure_sample_size = config["optimization"].get("failure_sample_size", 10)
    failures = _extract_failures(predictions, samples, config, failure_sample_size)

    optimizer_prompt = ISOLATED_OPTIMIZER_SYSTEM + "\n\n" + _build_bundled_prompt(
        current_prompt, metrics, failures, history,
    )

    optimizer_config = {
        "model": config.get("optimizer_model", config["model"]),
        "decoding": config["decoding"],
        "runner": config["runner"],
    }

    result = call_model(optimizer_prompt, optimizer_config)

    if result["error"]:
        raise RuntimeError(f"Optimizer call failed: {result['error']}")

    return _clean_new_prompt(result["raw"])


def _build_bundled_prompt(
    current_prompt: str,
    metrics: Dict[str, Any],
    failures: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> str:
    parts = []

    if history:
        parts.append("== OPTIMIZATION HISTORY ==")
        for h in history:
            parts.append(f"Iteration {h['iteration']}: {h['target_metric']}={h['score']:.2f}")
            if h.get("change_summary"):
                parts.append(f"  Change: {h['change_summary']}")
        parts.append("")

    parts.append("== CURRENT PROMPT TEMPLATE ==")
    parts.append(current_prompt)
    parts.append("")

    parts.append("== CURRENT METRICS ==")
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"  {k}: {v:.4f}")
        else:
            parts.append(f"  {k}: {v}")
    parts.append("")

    if failures:
        parts.append("== FAILURE EXAMPLES ==")
        for i, f in enumerate(failures):
            parts.append(f"\n--- Failure {i+1} ({f['failure_type']}) ---")
            parts.append(f"Label: {'vulnerable' if f['label'] == 1 else 'not vulnerable'}")
            parts.append(f"Predicted: {'vulnerable' if f['predicted'] else 'not vulnerable'}")
            if f.get("code_snippet"):
                parts.append(f"Code:\n{f['code_snippet']}")
            if f.get("model_reasoning"):
                parts.append(f"Model reasoning: {f['model_reasoning']}")
        parts.append("")

    parts.append("== TASK ==")
    parts.append("Propose an improved prompt template. Output ONLY the new template text.")

    return "\n".join(parts)


def _extract_failures(
    predictions: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    sample_size: int = 10,
) -> List[Dict[str, Any]]:
    code_field = config["dataset"]["code_field"]
    failures = []

    for pred in predictions:
        idx = pred.get("index")
        if idx is None or idx >= len(samples):
            continue

        label = pred["label"]
        predicted = pred["vulnerable"]

        if predicted is None:
            continue
        elif label == 1 and not predicted:
            failure_type = "false_negative"
        elif label == 0 and predicted:
            failure_type = "false_positive"
        else:
            continue

        verdict = pred.get("verdict_json") or {}
        summary = verdict.get("verdict", {}).get("summary", "") if isinstance(verdict, dict) else ""

        failures.append({
            "failure_type": failure_type,
            "label": label,
            "predicted": predicted,
            "code_snippet": samples[idx].get(code_field, ""),
            "model_reasoning": summary,
            "index": idx,
        })

    by_type = {}
    for f in failures:
        by_type.setdefault(f["failure_type"], []).append(f)

    sampled = []
    per_type = max(1, sample_size // max(len(by_type), 1))
    for ftype, flist in by_type.items():
        sampled.extend(flist[:per_type])

    return sampled[:sample_size]
