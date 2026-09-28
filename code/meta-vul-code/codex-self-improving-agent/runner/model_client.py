#!/usr/bin/env python3
"""Least-privilege Codex CLI model client.

Evaluator calls run in a fresh empty workspace. Optimizer calls run in the
explicit trajectory workspace supplied by the outer optimization loop. Every
call receives:

- an ephemeral Codex session;
- a temporary ``CODEX_HOME`` containing authentication plus generated config;
- a permission profile that can read only the assigned workspace and minimal
  operating-system runtime paths;
- no network, web search, user configuration, skills, MCP servers, memories,
  project instructions, approval escalation, or nested agents.

The outer Python process is not sandboxed by this module and retains the access
it needs to assemble workspaces, score predictions, and persist experiment
artifacts.
"""
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PERMISSION_PROFILE = "benchmark_readonly"

# Local LiteLLM proxy that bridges Codex CLI's Responses API calls to
# Tinker's Chat Completions endpoint (Codex only speaks Responses API;
# Tinker only speaks Chat Completions -- see .litellm/config.yaml).
# Override with LITELLM_PROXY_PORT to run a second proxy (e.g. when another
# checkout already owns 4444 with a config that lacks the model you need).
# TINKER_PROXY_PORT is the old name for the same setting and is still honored.
LITELLM_PROXY_PORT = int(os.environ.get("LITELLM_PROXY_PORT")
                         or os.environ.get("TINKER_PROXY_PORT") or "4444")
LITELLM_PROXY_KEY = "sk-local-proxy"
# Providers served by that proxy. The upstream (Tinker, OpenRouter, ...) is
# picked per model by .litellm/config.yaml; to Codex they all look the same.
PROXIED_PROVIDERS = {"tinker", "openrouter"}
LITELLM_PROXY_CONFIG = Path(__file__).resolve().parent.parent.parent / ".litellm" / "config.yaml"
_litellm_proxy_process: Optional[subprocess.Popen] = None


def _ensure_litellm_proxy() -> str:
    """Start the local LiteLLM proxy if it isn't already running; return its base_url."""
    base_url = f"http://localhost:{LITELLM_PROXY_PORT}/v1"
    health_url = f"http://localhost:{LITELLM_PROXY_PORT}/health/liveliness"

    def _healthy(timeout: float) -> bool:
        try:
            urllib.request.urlopen(health_url, timeout=timeout)
            return True
        except Exception:
            return False

    def _port_bound() -> bool:
        # A proxy that is starting up (or busy under load) owns the port before
        # it answers /health. Treat a bound port as "a proxy exists": never
        # start a second one on the same port.
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", LITELLM_PROXY_PORT)) == 0

    if _healthy(10):
        return base_url  # already running (started by us earlier, or externally)

    # Only one worker may spawn the proxy; the others wait on the lock and then
    # just poll. Without this, N concurrent workers whose 2s health probe timed
    # out (a loaded proxy answers slowly) each launched their own litellm, which
    # piled up hundreds of processes and took the machine down.
    import fcntl
    lock_path = LITELLM_PROXY_CONFIG.parent / f".proxy-{LITELLM_PROXY_PORT}.lock"
    lock_fh = open(lock_path, "w")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        if _healthy(10):
            return base_url
        if not _port_bound():
            _spawn_litellm_proxy()
        deadline = time.time() + 120
        while time.time() < deadline:
            if _healthy(5):
                return base_url
            time.sleep(1)
        raise RuntimeError("litellm proxy did not become ready within 120s")
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def _spawn_litellm_proxy() -> None:
    global _litellm_proxy_process

    litellm_bin = os.environ.get("LITELLM_BIN") or shutil.which("litellm")
    if not litellm_bin:
        raise RuntimeError(
            "litellm proxy binary not found; install it with `pip install "
            "'litellm[proxy]'`, or set LITELLM_BIN"
        )
    if not LITELLM_PROXY_CONFIG.exists():
        raise RuntimeError(f"litellm proxy config not found: {LITELLM_PROXY_CONFIG}")

    log_path = LITELLM_PROXY_CONFIG.parent / f"proxy-{LITELLM_PROXY_PORT}.log"
    log_fh = open(log_path, "ab")
    _litellm_proxy_process = subprocess.Popen(
        [litellm_bin, "--config", str(LITELLM_PROXY_CONFIG), "--port", str(LITELLM_PROXY_PORT)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _run_codex_process(
    cmd,
    *,
    prompt: str,
    timeout_s: float,
    env: Dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess:
    """Run Codex and terminate its entire process group on timeout."""
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(cwd),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=timeout_s)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate()
        raise
    return subprocess.CompletedProcess(
        cmd,
        process.returncode,
        stdout,
        stderr,
    )


def _codex_binary(config: Dict[str, Any]) -> str:
    configured = config.get("model", {}).get("binary") or os.environ.get("CODEX_BIN")
    binary = configured or shutil.which("codex")
    if not binary:
        raise FileNotFoundError("Codex CLI not found; install it or set CODEX_BIN")
    return binary


def _copy_auth(codex_home: Path) -> None:
    source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth_file = source_home / "auth.json"
    if auth_file.exists():
        shutil.copy2(auth_file, codex_home / "auth.json")


def _permission_profile_name(config: Dict[str, Any]) -> str:
    """Return a conservative TOML-safe profile name."""
    configured = config.get("isolation", {}).get(
        "permission_profile",
        DEFAULT_PERMISSION_PROFILE,
    )
    name = re.sub(r"[^A-Za-z0-9_-]", "_", str(configured)).strip("_")
    return name or DEFAULT_PERMISSION_PROFILE


def _write_permission_config(
    codex_home: Path,
    config: Dict[str, Any],
) -> str:
    """Write the complete config for a child Codex process.

    Permission profiles are intentionally used without ``sandbox_mode`` or a
    ``--sandbox`` CLI flag. Codex treats those legacy settings as overriding
    permission profiles.

    project_doc_max_bytes = 0 disables AGENTS.md discovery. Codex otherwise
    walks up from cwd looking for one; when cwd is a real path nested under
    the repo (rather than an ephemeral /tmp dir), it finds the repo-root
    AGENTS.md, which sits outside the ":workspace_roots" read scope below and
    the OS sandbox denies the read with a fatal "Operation not permitted"
    instead of a graceful skip.

    When config["model"]["provider"] is in PROXIED_PROVIDERS, model calls are routed
    through a local LiteLLM proxy (auto-started if needed) that bridges
    Codex's Responses-API calls to Tinker's Chat-Completions-only endpoint --
    Codex CLI cannot reach Tinker directly (protocol mismatch: Codex requires
    wire_api="responses", Tinker only implements chat/completions). The
    "openai" provider name is reserved by Codex and cannot be overridden, so
    a custom "litellm_proxy" provider is registered and selected via
    model_provider instead.

    For the Tinker-served Nemotron model specifically, a minimal model catalog
    (models.json) and instructions file are written so Codex does not fall
    back to its ~78K-token generic preamble, which overflows the 65K context.
    """
    profile = _permission_profile_name(config)
    is_proxied = config.get("model", {}).get("provider") in PROXIED_PROVIDERS
    _m = config.get("model", {})
    model_name = str(_m.get("name", ""))
    is_nemotron = (
        _m.get("provider") == "tinker"
        and model_name == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    )
    tinker_model_lines = ""
    if is_nemotron:
        # Codex has no built-in metadata for this Tinker model. Its generic
        # fallback injected a roughly 78K-token instruction preamble, exceeding
        # the served context before the benchmark prompt was evaluated. Supply
        # a minimal, model-specific catalog and instruction file instead. Keep
        # this scoped to Nemotron so existing Inkling runs are unchanged.
        instructions = (
            "Follow the supplied benchmark prompt exactly. Return only the requested "
            "answer; do not inspect unrelated files or use external resources.\n"
        )
        instructions_file = codex_home / "benchmark-instructions.md"
        instructions_file.write_text(instructions, encoding="utf-8")
        model_catalog = codex_home / "models.json"
        model_catalog.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "slug": model_name,
                            "display_name": model_name,
                            "description": "Isolated benchmark model",
                            "default_reasoning_level": "high",
                            "supported_reasoning_levels": [
                                {"effort": "high", "description": "Reasoning enabled"}
                            ],
                            "shell_type": "unified_exec",
                            "visibility": "hide",
                            "supported_in_api": True,
                            "priority": 1,
                            "model_messages": {
                                "instructions_template": instructions,
                                "instructions_variables": None,
                                "approvals": None,
                                "collaboration_modes": None,
                                "auto_review": None,
                                "permissions": None,
                                "multi_agent": None,
                                "token_budget": None,
                            },
                            "include_skills_usage_instructions": False,
                            "include_plugin_usage_instructions": False,
                            "include_apps_usage_instructions": False,
                            "default_reasoning_summary": "none",
                            "support_verbosity": False,
                            "apply_patch_tool_type": "freeform",
                            "web_search_tool_type": "text",
                            "truncation_policy": {"mode": "tokens", "limit": 10000},
                            "context_window": 65536,
                            "max_context_window": 65536,
                            "effective_context_window_percent": 90,
                            "experimental_supported_tools": [],
                            "input_modalities": ["text"],
                            "supports_search_tool": False,
                            "use_responses_lite": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        tinker_model_lines = f'''\
model_context_window = 65536
model_auto_compact_token_limit = 48000
model_instructions_file = "{instructions_file}"
model_catalog_json = "{model_catalog}"
'''
    # Codex >=0.150 bundles "apps" plugins (GitHub, sites, documents, ...) as
    # MCP tools: ~130 tool definitions per request. api.openai.com rejects
    # requests with >128 tools (400). OpenRouter forwards the tools array for
    # "openai/gpt-4o" to api.openai.com unchanged and the 400 comes back through
    # it, so GPT-4o runs get the trimmed toolset. Every other model keeps
    # Codex's default tools.
    is_openai_gpt4o = _m.get("name") == "openai/gpt-4o"
    trimmed_tools_block = (
        "# GPT-4o: drop the bundled app plugins (OpenAI caps tools at 128).\n"
        "apps = false\n"
        "plugins = false\n"
        "remote_plugin = false\n"
        "recommended_plugins = false\n"
        "tool_suggest = false\n"
    ) if is_openai_gpt4o else ""
    # Root-level (bare) keys must all appear before the first [section] header --
    # TOML scopes bare keys to whichever table was most recently opened, so a
    # root key placed after [permissions...] silently becomes a field of that
    # section instead of the root table.
    model_provider_line = 'model_provider = "litellm_proxy"\n' if is_proxied else ""
    config_text = f'''\
approval_policy = "never"
default_permissions = "{profile}"
personality = "none"
web_search = "disabled"
project_doc_max_bytes = 0
{tinker_model_lines}{model_provider_line}
[features]
multi_agent = false
{trimmed_tools_block}
[shell_environment_policy]
inherit = "core"
ignore_default_excludes = false

[permissions.{profile}.filesystem]
":minimal" = "read"

[permissions.{profile}.filesystem.":workspace_roots"]
"." = "read"

[permissions.{profile}.network]
enabled = false
'''
    if is_proxied:
        base_url = _ensure_litellm_proxy()
        config_text += f'''
[model_providers.litellm_proxy]
name = "LiteLLM proxy"
base_url = "{base_url}"
env_key = "LITELLM_PROXY_KEY"
wire_api = "responses"
'''
    (codex_home / "config.toml").write_text(config_text, encoding="utf-8")
    return profile


def _token_count(stderr: str) -> Optional[int]:
    match = re.search(r"tokens used\s*\n\s*([\d,]+)", stderr, flags=re.IGNORECASE)
    return int(match.group(1).replace(",", "")) if match else None


def call_model(
    prompt: str,
    config: Dict[str, Any],
    *,
    workspace: Optional[Path] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one model invocation and return the final assistant message."""
    backend = config["model"].get("backend", "codex-cli")
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt.rstrip()}\n\n{prompt.lstrip()}"

    if backend in {"openai-compatible", "vllm"}:
        return _call_openai_compatible(full_prompt, config, workspace=workspace)
    if backend != "codex-cli":
        return _error_result(time.time(), ValueError(f"unknown backend: {backend}"))

    timeout_s = config["runner"]["timeout_s"]
    model = config["model"].get("name", "gpt-5.4")
    effort = config["model"].get("reasoning_effort", "high")

    started = time.time()
    try:
        binary = _codex_binary(config)
    except Exception as exc:
        return _error_result(started, exc)

    try:
        with tempfile.TemporaryDirectory(prefix="codex_bench_home_") as home_name:
            with tempfile.TemporaryDirectory(prefix="codex_bench_work_") as work_name:
                codex_home = Path(home_name)
                _copy_auth(codex_home)
                profile = _write_permission_config(codex_home, config)
                output_file = codex_home / "last-message.txt"
                cwd = Path(workspace).resolve() if workspace else Path(work_name)

                cmd = [
                    binary,
                    "--ask-for-approval",
                    "never",
                    "exec",
                    "--ephemeral",
                    "--strict-config",
                    "--ignore-rules",
                    "--skip-git-repo-check",
                    "--cd",
                    str(cwd),
                    "--model",
                    model,
                    "-c",
                    f'model_reasoning_effort="{effort}"',
                    # Codex otherwise sends reasoning.summary="auto" on every
                    # Responses request. LiteLLM folds that into the
                    # reasoning_effort it forwards, and OpenRouter rejects it
                    # (400 "reasoning_effort: Invalid option"), so every
                    # sample fails. Disabling the summary drops the field.
                    "-c",
                    'model_reasoning_summary="none"',
                    "-c",
                    'personality="none"',
                    "--output-last-message",
                    str(output_file),
                    "-",
                ]

                env = os.environ.copy()
                env["CODEX_HOME"] = str(codex_home)
                if config.get("model", {}).get("provider") in PROXIED_PROVIDERS:
                    env["LITELLM_PROXY_KEY"] = LITELLM_PROXY_KEY
                result = _run_codex_process(
                    cmd,
                    prompt=full_prompt,
                    timeout_s=timeout_s,
                    env=env,
                    cwd=cwd,
                )
                latency = round(time.time() - started, 3)
                response = output_file.read_text(encoding="utf-8") if output_file.exists() else ""
                tokens = _token_count(result.stderr)
                error = ""
                if result.returncode != 0:
                    error = result.stderr.strip() or f"exit code {result.returncode}"
                elif not response.strip():
                    error = "Codex completed without writing a final response"

                return {
                    "raw": response.strip(),
                    "tokens_prompt": None,
                    "tokens_completion": tokens,
                    "latency_s": latency,
                    "error": error,
                    "isolation": {
                        "permission_profile": profile,
                        "workspace_kind": "optimizer" if workspace else "evaluator",
                        "network_access": False,
                        "web_search": "disabled",
                        "ephemeral": True,
                    },
                }
    except subprocess.TimeoutExpired:
        return {
            "raw": "",
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": round(time.time() - started, 3),
            "error": f"timeout after {timeout_s}s",
            "isolation": None,
        }
    except Exception as exc:
        return _error_result(started, exc)


def _openai_base_url(config: Dict[str, Any]) -> str:
    configured = config["model"].get("base_url") or os.environ.get("OPENAI_COMPAT_BASE_URL")
    base_url = configured or os.environ.get("VLLM_BASE_URL")
    if not base_url:
        raise ValueError("model.base_url, OPENAI_COMPAT_BASE_URL, or VLLM_BASE_URL is required")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def _call_openai_compatible(
    prompt: str,
    config: Dict[str, Any],
    *,
    workspace: Optional[Path] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible chat-completions endpoint such as vLLM."""
    started = time.time()
    timeout_s = float(config["runner"]["timeout_s"])
    model_cfg = config["model"]
    decoding = config.get("decoding", {})
    model = model_cfg["name"]

    try:
        base_url = _openai_base_url(config)
    except Exception as exc:
        return _error_result(started, exc)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": decoding.get("temperature", 0),
        "top_p": decoding.get("top_p", 1.0),
        "max_tokens": decoding.get("max_tokens", 4096),
    }
    chat_template_kwargs = dict(model_cfg.get("chat_template_kwargs", {}))
    if model_cfg.get("reasoning_effort"):
        chat_template_kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    for key in ("stop", "frequency_penalty", "presence_penalty"):
        if key in decoding:
            payload[key] = decoding[key]

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get(model_cfg.get('api_key_env', 'OPENAI_API_KEY'), 'EMPTY')}",
    }
    # Some public OpenAI-compatible endpoints (notably ngrok's free tier)
    # require a benign request header to suppress their HTML browser-warning
    # interstitial.  Keep this opt-in so ordinary providers are unchanged.
    headers.update({str(key): str(value) for key, value in model_cfg.get("extra_headers", {}).items()})

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("choices", [{}])[0].get("message", {})
        usage = data.get("usage", {})
        return {
            "raw": (message.get("content") or "").strip(),
            "tokens_prompt": usage.get("prompt_tokens"),
            "tokens_completion": usage.get("completion_tokens"),
            "latency_s": round(time.time() - started, 3),
            "error": "",
            "isolation": {
                "permission_profile": None,
                "workspace_kind": "optimizer" if workspace else "evaluator",
                "network_access": True,
                "web_search": "disabled",
                "ephemeral": False,
                "backend": model_cfg.get("backend"),
                "base_url": base_url,
            },
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        return {
            "raw": "",
            "tokens_prompt": None,
            "tokens_completion": None,
            "latency_s": round(time.time() - started, 3),
            "error": f"HTTPError {exc.code}: {detail}",
            "isolation": None,
        }
    except Exception as exc:
        return _error_result(started, exc)


def _error_result(started: float, exc: Exception) -> Dict[str, Any]:
    return {
        "raw": "",
        "tokens_prompt": None,
        "tokens_completion": None,
        "latency_s": round(time.time() - started, 3),
        "error": f"{type(exc).__name__}: {exc}",
        "isolation": None,
    }


def verify_permission_boundary(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify the generated profile without making a model call.

    Runs ``codex sandbox`` twice: once against an allowed file in the assigned
    workspace and once against a canary outside it. The second command must be
    denied by the operating-system sandbox.
    """
    binary = _codex_binary(config)
    with tempfile.TemporaryDirectory(prefix="codex_verify_home_") as home_name:
        with tempfile.TemporaryDirectory(prefix="codex_verify_root_") as root_name:
            root = Path(root_name)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            allowed_file = workspace / "allowed.txt"
            denied_file = outside / "denied.txt"
            allowed_value = "allowed-workspace-value"
            denied_value = "outside-canary-value"
            allowed_file.write_text(allowed_value, encoding="utf-8")
            denied_file.write_text(denied_value, encoding="utf-8")

            codex_home = Path(home_name)
            profile = _write_permission_config(codex_home, config)
            env = os.environ.copy()
            env["CODEX_HOME"] = str(codex_home)

            base_cmd = [
                binary,
                "sandbox",
                "--permission-profile",
                profile,
                "--cd",
                str(workspace),
            ]
            allowed = subprocess.run(
                [*base_cmd, "/bin/cat", str(allowed_file)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=str(workspace),
            )
            denied = subprocess.run(
                [*base_cmd, "/bin/cat", str(denied_file)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=str(workspace),
            )

            allowed_ok = allowed.returncode == 0 and allowed_value in allowed.stdout
            denied_ok = denied.returncode != 0 and denied_value not in denied.stdout
            return {
                "ok": allowed_ok and denied_ok,
                "permission_profile": profile,
                "allowed_read_ok": allowed_ok,
                "outside_read_denied": denied_ok,
                "allowed_returncode": allowed.returncode,
                "denied_returncode": denied.returncode,
                "denial_message": denied.stderr.strip()[-1000:],
            }
