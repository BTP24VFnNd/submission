#!/usr/bin/env python3
"""
Harness discovery: entrypoint for running the prompt/harness optimization loop
against a chosen agent, starting from a chosen seed reasoning policy.

This is a thin wrapper around run_optimization.py: it resolves --agent and
--seed-prompt into a concrete config, writes a derived config file, then
invokes run_optimization.py with it (passing through the remaining args).

Examples:
  python run_harness_discovery.py --agent claude-cli
  python run_harness_discovery.py --agent codex --seed-prompt vanilla --limit 10
  python run_harness_discovery.py --agent codex --model gpt-6-astra --max-iterations 5
  python run_harness_discovery.py --agent codex --optimizer-agent codex
  python run_harness_discovery.py --agent codex --model tinker/Qwen/Qwen3.5-9B-Base

By default the optimizer (the model that proposes prompt revisions) is always
Claude, independent of --agent, so results stay comparable across evaluator
agents. Pass --optimizer-agent to override that and use the same (or a
different) agent for the optimizer too.

Agentic (workspace-browsing) optimizer mode is supported for both claude-cli
and codex --optimizer-agent choices.

For --agent codex, prefixing --model with "tinker/" (e.g.
"tinker/Qwen/Qwen3.5-9B-Base") routes that call through Tinker's
OpenAI-compatible endpoint instead of OpenAI -- same Codex CLI, same sandbox,
same workspace browsing, just a different model provider underneath.
"""
import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
BASE_CONFIG = CONFIGS_DIR / "config-titanvul-cweval-generic.yaml"

# Registry of known agents: name -> (agent_dir, default model config)
AGENTS = {
    "claude-cli": {
        "dir": "claude-cli-agent",
        "model": {"name": "claude-opus-4-6", "backend": "claude-cli"},
    },
    "codex": {
        "dir": "codex-self-improving-agent",
        "model": {"name": "gpt-5.4", "backend": "codex-cli", "reasoning_effort": "high"},
    },
}

# Known model-provider prefixes: same CLI (codex or claude), same sandbox,
# same workspace browsing -- only which backend actually serves the model
# differs. "openrouter/<model>" or "tinker/<model>" routes the call through
# the local LiteLLM proxy (see each agent's model_client), which forwards to
# the upstream registered under that name in .litellm/config.yaml.
KNOWN_PROVIDERS = {"tinker", "openrouter"}

MODES = {
    "harness-discovery": "run_optimization.py",
}


def _parse_model(model: str):
    """Split "<provider>/<model-name>" into (provider, name); else (None, model)."""
    if model and "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in KNOWN_PROVIDERS:
            return prefix, rest
    return None, model


def _extract_resume_id(passthrough) -> str:
    """Pull the --resume session id out of passthrough args, if present.

    run_optimization.py's --resume takes an optional value (nargs="?",
    const="latest"), so "--resume" alone, "--resume <id>", and
    "--resume=<id>" must all be recognized.
    """
    for i, tok in enumerate(passthrough):
        if tok == "--resume":
            nxt = passthrough[i + 1] if i + 1 < len(passthrough) else None
            return nxt if nxt and not nxt.startswith("--") else "latest"
        if tok.startswith("--resume="):
            return tok.split("=", 1)[1] or "latest"
    return None


def _extract_session_id(passthrough) -> str:
    """Pull an explicit --session-id out of passthrough args, if present."""
    for i, tok in enumerate(passthrough):
        if tok == "--session-id" and i + 1 < len(passthrough):
            return passthrough[i + 1]
        if tok.startswith("--session-id="):
            return tok.split("=", 1)[1]
    return None


def _resolve_seed_prompt(agent_dir: str, seed_prompt: str) -> Path:
    """Find (or stage) the seed prompt file for a given reasoning policy name."""
    staged = BASE_DIR / "prompts" / "iterations" / f"iter_000_{seed_prompt}.txt"
    if staged.exists():
        return staged

    source = BASE_DIR.parent / agent_dir / "prompts" / f"{seed_prompt}.txt"
    if not source.exists():
        raise SystemExit(
            f"No seed prompt found for '{seed_prompt}': checked {staged} and {source}"
        )

    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, staged)
    return staged


def main():
    parser = argparse.ArgumentParser(description="Run harness discovery for a given agent")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="claude-cli",
                        help="Which agent backend to discover a harness for")
    parser.add_argument("--mode", choices=sorted(MODES), default="harness-discovery",
                        help="Discovery mode to run")
    parser.add_argument("--seed-prompt", default="scot",
                        help="Name of the seed reasoning policy (scot, vanilla, mot, tot, ...)")
    parser.add_argument("--model", default=None,
                        help="Override the agent's default model name. For --agent codex, "
                             "prefix with a known provider (e.g. 'tinker/Qwen/Qwen3.5-9B-Base') "
                             "to route that call through a different model provider.")
    parser.add_argument("--optimizer-agent", choices=sorted(AGENTS), default=None,
                        help="Which agent proposes prompt revisions (default: claude-cli, "
                             "independent of --agent). Non-claude-cli falls back to isolated "
                             "optimizer mode (no workspace browsing).")
    parser.add_argument("--optimizer-model", default=None,
                        help="Override the optimizer agent's default model name (same "
                             "provider-prefix syntax as --model)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Concurrent evaluator calls per iteration "
                             "(overrides runner.concurrency in the base config)")
    parser.add_argument("--config", default=None,
                        help="Base config YAML to derive from (default: configs/config-titanvul-cweval-generic.yaml)")

    args, passthrough = parser.parse_known_args()

    model_provider, model_name = _parse_model(args.model) if args.model else (None, None)
    optimizer_provider, optimizer_name = (
        _parse_model(args.optimizer_model) if args.optimizer_model else (None, None)
    )

    # Provider prefixes are honored by both agents: codex repoints its model
    # provider at the local LiteLLM proxy, claude-cli sets ANTHROPIC_BASE_URL
    # to it. Either way the upstream is chosen by .litellm/config.yaml.

    agent = AGENTS[args.agent]
    base_config_path = Path(args.config) if args.config else BASE_CONFIG
    config = yaml.safe_load(base_config_path.read_text())

    # A base config may leave model / optimizer_model blank (see
    # configs/config-titanvul-cweval-generic.yaml). In that case nothing is
    # defaulted from the AGENTS registry: the models must be named on the
    # command line, so a run can never silently pick one.
    missing = []
    if not (config.get("model") or {}).get("name") and not args.model:
        missing.append("--model")
    if not (config.get("optimizer_model") or {}).get("name"):
        if not args.optimizer_agent:
            missing.append("--optimizer-agent")
        if not args.optimizer_model:
            missing.append("--optimizer-model")
    if missing:
        parser.error(f"{base_config_path.name} leaves the model unset; "
                     f"also pass: {' '.join(missing)}")

    config["agent_dir"] = agent["dir"]
    config["model"] = dict(agent["model"])
    if model_name:
        config["model"]["name"] = model_name
    if model_provider:
        config["model"]["provider"] = model_provider

    if args.optimizer_agent:
        optimizer = AGENTS[args.optimizer_agent]
        config["optimizer_model"] = dict(optimizer["model"])
        if optimizer_name:
            config["optimizer_model"]["name"] = optimizer_name
        if optimizer_provider:
            config["optimizer_model"]["provider"] = optimizer_provider

    if args.concurrency:
        config.setdefault("runner", {})["concurrency"] = args.concurrency

    seed_path = _resolve_seed_prompt(agent["dir"], args.seed_prompt)
    config.setdefault("optimization", {})["seed_prompt"] = str(seed_path.relative_to(BASE_DIR))

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', config["model"]["name"])
    resume_id = _extract_resume_id(passthrough)
    # Tie a resumed run's config to the session it's resuming (stable, reused
    # across repeated --resume calls) rather than a fresh timestamp each time,
    # which would otherwise pile up disconnected configs for the same session.
    # For a fresh run, pick the session id here and hand it to the driver so
    # the config filename ends with exactly the session id it belongs to.
    session_args = []
    if resume_id:
        suffix = f"resume-{resume_id}"
    else:
        suffix = _extract_session_id(passthrough)
        if not suffix:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_args = ["--session-id", suffix]
    derived_config_path = (
        CONFIGS_DIR / f".generated-config-{args.agent}-{args.seed_prompt}-{model_slug}-{suffix}.yaml"
    )
    derived_config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    driver_script = BASE_DIR / MODES[args.mode]
    cmd = [
        sys.executable, str(driver_script),
        "--config", str(derived_config_path.relative_to(BASE_DIR)),
        *session_args,
        *passthrough,
    ]
    optimizer_name = config.get("optimizer_model", config["model"]).get("name", "")
    print(f"[harness-discovery] agent={args.agent} mode={args.mode} seed={args.seed_prompt} "
          f"model={config['model']['name']} optimizer_agent={args.optimizer_agent or 'claude-cli (default)'} "
          f"optimizer_model={optimizer_name}")
    print(f"[harness-discovery] config -> {derived_config_path}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
