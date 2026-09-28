#!/usr/bin/env python3
"""
Unified model client supporting all backends:
  - codex-cli: GPT-5.4 through Codex CLI (`codex exec`)
  - claude-cli: Claude Opus through Claude CLI (`claude -p`)
  - tinker: Qwen/Kimi through Tinker API
  - openai: OpenAI Responses API
  - anthropic: Anthropic Messages API
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from tinker import ServiceClient, types  # type: ignore
except Exception:
    ServiceClient = None
    types = None


def _codex_binary(config: Dict[str, Any]) -> str:
    configured = config.get("model", {}).get("binary") or os.environ.get("CODEX_BIN")
    binary = configured or shutil.which("codex")
    if not binary:
        raise FileNotFoundError("Codex CLI not found; install it or set CODEX_BIN")
    return binary

# Shared Tinker client
_tinker_client: Optional[Any] = None
_tinker_lock = threading.Lock()

CONTEXT_WINDOWS = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 262144,
    "moonshotai/Kimi-K2-Thinking": 256000,
}


def call_model(prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single model invocation. Dispatches to the appropriate backend.
    Returns dict with keys:
      raw: str
      tokens_prompt: int | None
      tokens_completion: int | None
      latency_s: float
      error: str (empty if no error)
    """
    backend = config["model"]["backend"]
    timeout_s = config["runner"]["timeout_s"]

    t0 = time.time()

    if backend == "codex-cli":
        raw, err = _call_codex_cli(prompt, config)
    elif backend == "claude-cli":
        raw, err = _call_claude_cli(prompt, config)
    elif backend == "tinker":
        raw, err = _call_tinker(prompt, config)
    elif backend == "openai":
        raw, err = _call_openai(prompt, config)
    elif backend == "anthropic":
        raw, err = _call_anthropic(prompt, config)
    else:
        raw, err = "", f"unknown backend: {backend}"

    latency = time.time() - t0
    return {
        "raw": raw,
        "tokens_prompt": None,
        "tokens_completion": None,
        "latency_s": round(latency, 3),
        "error": err,
    }


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------

def _parse_codex_output(raw: str) -> Dict[str, Any]:
    """Extract model response and token count from Codex CLI output."""
    parts = raw.split("\n")

    separator_idx = None
    for i, line in enumerate(parts):
        if line.strip().startswith("--------"):
            separator_idx = i
            break

    if separator_idx is None:
        return {"response": raw.strip(), "tokens": None}

    codex_idx = None
    for i in range(separator_idx + 1, len(parts)):
        if parts[i].strip() == "codex":
            codex_idx = i
            break

    if codex_idx is None:
        return {"response": raw.strip(), "tokens": None}

    tokens_idx = None
    for i in range(len(parts) - 1, codex_idx, -1):
        if parts[i].strip() == "tokens used":
            tokens_idx = i
            break

    if tokens_idx is not None:
        response_lines = parts[codex_idx + 1:tokens_idx]
        tokens = None
        if tokens_idx + 1 < len(parts):
            token_str = parts[tokens_idx + 1].strip().replace(",", "")
            try:
                tokens = int(token_str)
            except ValueError:
                pass
    else:
        response_lines = parts[codex_idx + 1:]
        tokens = None

    response = "\n".join(response_lines).strip()
    return {"response": response, "tokens": tokens}


def _call_codex_cli(prompt: str, config: Dict[str, Any]):
    model = config["model"].get("name", "gpt-5.4")
    timeout_s = config["runner"]["timeout_s"]
    reasoning_effort = config["model"].get("reasoning_effort", "high")

    cmd = [_codex_binary(config), "exec", prompt, "--model", model,
           "--skip-git-repo-check",
           "-c", f"model_reasoning_effort={reasoning_effort}",
           "-c", "personality=none"]

    env = os.environ.copy()
    env["CODEX_HOME"] = env.get("CODEX_HOME", os.path.expanduser("~/.codex"))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            env=env, cwd="/tmp",
        )
        full_output = result.stdout + result.stderr
        parsed = _parse_codex_output(full_output)

        if result.returncode != 0:
            err_msg = result.stderr.strip() or f"exit code {result.returncode}"
            return parsed["response"], err_msg

        return parsed["response"], ""
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout_s}s"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Claude CLI — delegates to claude-cli-agent/runner/model_client.py
# ---------------------------------------------------------------------------

def _call_claude_cli(prompt: str, config: Dict[str, Any]):
    import importlib.util
    _claude_agent_path = str(Path(__file__).resolve().parent.parent.parent
                             / "claude-cli-agent" / "runner" / "model_client.py")
    spec = importlib.util.spec_from_file_location("claude_cli_model_client", _claude_agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.call_model(prompt, config)
    return result["raw"], result["error"]


# ---------------------------------------------------------------------------
# Tinker
# ---------------------------------------------------------------------------

def _call_tinker(prompt: str, config: Dict[str, Any]):
    if ServiceClient is None or types is None:
        return "", "tinker package not installed"

    model = config["model"]["name"]
    max_tokens = config["decoding"]["max_tokens"]
    timeout_s = config["runner"]["timeout_s"]

    try:
        global _tinker_client
        with _tinker_lock:
            if _tinker_client is None:
                _tinker_client = ServiceClient()
            client = _tinker_client

        sampling = client.create_sampling_client(base_model=model)
        tokenizer = sampling.get_tokenizer()
        prompt_tokens = tokenizer.encode(prompt)
        ctx_limit = CONTEXT_WINDOWS.get(model, 32768)
        if len(prompt_tokens) >= ctx_limit:
            return "", f"prompt ({len(prompt_tokens)} tokens) exceeds context ({ctx_limit})"
        model_input = types.ModelInput.from_ints(prompt_tokens)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=0)
        future = sampling.sample(prompt=model_input, sampling_params=params, num_samples=1)
        result = future.result(timeout=timeout_s)
        text = ""
        for seq in result.sequences:
            text += tokenizer.decode(seq.tokens)
        return text, ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, config: Dict[str, Any]):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "", "OPENAI_API_KEY not set"

    model = config["model"]["name"]
    temperature = config["decoding"]["temperature"]
    timeout_s = config["runner"]["timeout_s"]

    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = ""
            if "output" in data and data["output"]:
                for item in data["output"]:
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                text += c.get("text", "")
            return text, ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, config: Dict[str, Any]):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "", "ANTHROPIC_API_KEY not set"

    model = config["model"]["name"]
    temperature = config["decoding"]["temperature"]
    max_tokens = config["decoding"]["max_tokens"]
    timeout_s = config["runner"]["timeout_s"]

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = ""
            for c in data.get("content", []):
                if c.get("type") == "text":
                    text += c.get("text", "")
            return text, ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"
