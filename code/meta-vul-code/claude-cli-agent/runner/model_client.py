#!/usr/bin/env python3
"""Model client that calls Claude CLI in full agentic mode.

Claude gets filesystem access to a workspace containing:
- The optimization trajectory (past prompts, metrics, history)
- Past predictions and reasoning (its own previous outputs)
- Dataset metadata (but NOT ground truth labels)

Ground truth is never written to the workspace.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# --- Non-Anthropic models through a local LiteLLM proxy --------------------
# When config["model"]["provider"] is one of PROXIED_PROVIDERS, the Claude CLI
# is pointed at a local LiteLLM proxy (ANTHROPIC_BASE_URL) that speaks the
# Anthropic Messages API and forwards to the upstream named in
# .litellm/config.yaml (OpenRouter, Tinker, ...). Same CLI, same agentic
# loop, different model underneath. Mirrors codex-self-improving-agent's
# model_client; the proxy config file is shared between the two agents.
LITELLM_PROXY_PORT = int(os.environ.get("LITELLM_PROXY_PORT")
                         or os.environ.get("TINKER_PROXY_PORT") or "4444")
LITELLM_PROXY_KEY = "sk-local-proxy"
PROXIED_PROVIDERS = {"tinker", "openrouter"}
LITELLM_PROXY_CONFIG = Path(__file__).resolve().parent.parent.parent / ".litellm" / "config.yaml"
_litellm_proxy_process: Optional[subprocess.Popen] = None


def _ensure_litellm_proxy() -> str:
    """Start the local LiteLLM proxy if it isn't already running; return its root URL."""
    global _litellm_proxy_process
    root_url = f"http://localhost:{LITELLM_PROXY_PORT}"
    health_url = f"{root_url}/health/liveliness"

    try:
        urllib.request.urlopen(health_url, timeout=2)
        return root_url  # already running (started by us earlier, or externally)
    except Exception:
        pass

    litellm_bin = os.environ.get("LITELLM_BIN") or shutil.which("litellm")
    if not litellm_bin:
        raise RuntimeError(
            "litellm proxy binary not found; install it with `pip install "
            "'litellm[proxy]'`, or set LITELLM_BIN"
        )
    if not LITELLM_PROXY_CONFIG.exists():
        raise RuntimeError(f"litellm proxy config not found: {LITELLM_PROXY_CONFIG}")

    _litellm_proxy_process = subprocess.Popen(
        [litellm_bin, "--config", str(LITELLM_PROXY_CONFIG), "--port", str(LITELLM_PROXY_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return root_url
        except Exception:
            time.sleep(1)
    raise RuntimeError("litellm proxy did not become ready within 30s")


def model_env(config: Dict[str, Any]) -> Dict[str, str]:
    """Environment for a Claude CLI call that serves config["model"].

    Anthropic models: the CLI's own login is used (ANTHROPIC_API_KEY is
    dropped so a stray key in the shell cannot override it).

    Proxied models: every Anthropic endpoint the CLI touches is redirected to
    the local LiteLLM proxy. The CLI also picks "opus"/"sonnet"/"haiku"
    tiers for side tasks (title generation, quick classification); those are
    pinned to the same model so the proxy never sees an Anthropic model id
    it does not serve. Non-essential traffic (telemetry, update checks) is
    off so nothing else leaves the machine.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    model_cfg = config.get("model", {})
    if model_cfg.get("provider") not in PROXIED_PROVIDERS:
        return env

    model = model_cfg.get("name", "")
    env["ANTHROPIC_BASE_URL"] = _ensure_litellm_proxy()
    env["ANTHROPIC_AUTH_TOKEN"] = LITELLM_PROXY_KEY
    for var in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
        env[var] = model
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["DISABLE_TELEMETRY"] = "1"
    env["DISABLE_ERROR_REPORTING"] = "1"
    return env


def _claude_binary(config: Dict[str, Any]) -> str:
    configured = config.get("model", {}).get("binary") or os.environ.get("CLAUDE_BIN")
    binary = configured or shutil.which("claude")
    if not binary:
        raise FileNotFoundError("Claude CLI not found; install it or set CLAUDE_BIN")
    return binary


def cli_settings_args(config: Dict[str, Any]) -> list:
    """Extra --settings for the Claude CLI.

    model.behaves_as maps a gateway model the CLI's catalog does not know (e.g.
    "openai/gpt-oss-20b") to a Claude model it does know, through a modelPicker
    row. Without it the CLI intermittently refuses the name before any API call
    ("[claude-code:unrecognized_model]"), mostly under high concurrency. It only
    changes the CLI's own assumptions (context window, capabilities); the request
    still goes to the configured model.
    """
    model = config.get("model", {})
    name, behaves_as = model.get("name"), model.get("behaves_as")
    if not (name and behaves_as):
        return []
    row = {"model": name, "label": name, "behavesAs": behaves_as}
    return ["--settings", json.dumps({"modelPicker": {"options": [row]}})]


def call_model(
    prompt: str,
    config: Dict[str, Any],
    *,
    workspace: Optional[Path] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single model invocation using Claude CLI in agentic mode.

    Args:
        prompt: The rendered prompt (code + analysis instructions).
        config: Full config dict.
        workspace: Directory Claude can browse. If None, falls back to
                   isolated mode (cwd=/tmp, no tools).
        system_prompt: Optional system prompt with trajectory context.

    Returns dict with keys:
      raw: str
      tokens_prompt: int | None
      tokens_completion: int | None
      latency_s: float
      error: str
    """
    model = config["model"].get("name", "")
    timeout_s = config["runner"]["timeout_s"]

    cmd = [_claude_binary(config), "-p", *cli_settings_args(config)]

    if workspace:
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
    else:
        cmd.extend(["--system-prompt", "", "--tools", ""])

    if model:
        cmd.extend(["--model", model])

    try:
        env = model_env(config)
    except Exception as e:
        return {
            "raw": "",
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }

    cwd = str(workspace) if workspace else "/tmp"

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=cwd,
        )
        latency = time.time() - t0

        if result.returncode != 0:
            err_msg = result.stderr.strip() or f"exit code {result.returncode}"
            return {
                "raw": result.stdout,
                "tokens_prompt": None,
                "tokens_completion": None,
                "latency_s": round(latency, 3),
                "error": err_msg,
            }

        return {
            "raw": result.stdout,
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": round(latency, 3),
            "error": "",
        }
    except subprocess.TimeoutExpired:
        latency = time.time() - t0
        return {
            "raw": "",
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": round(latency, 3),
            "error": f"timeout after {timeout_s}s",
        }
    except Exception as e:
        latency = time.time() - t0
        return {
            "raw": "",
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": round(latency, 3),
            "error": f"{type(e).__name__}: {e}",
        }
