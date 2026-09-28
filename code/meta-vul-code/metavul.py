#!/usr/bin/env python3
"""
metavul.py - single entrypoint for the meta-vul pipeline.

Subcommands:
  discover   Run the full harness-discovery optimization loop
  eval       Run a single evaluation pass (one policy, no optimization loop)
  held-out   Evaluate the best (or a specific) prompt from a discovery
             session on a held-out dataset

Examples:
  python3 metavul.py discover --agent codex --seed-prompt scot --new-session
  python3 metavul.py eval --agent codex --policy scot --limit 10 --new-session
  python3 metavul.py held-out --agent claude-cli --session 20260908_021704
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Agents that support the full pipeline (discovery + eval + held-out).
AGENTS = {
    "claude-cli": "claude-cli-agent",
    "codex": "codex-self-improving-agent",
}


def _run(cmd, cwd):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def cmd_discover(args, passthrough):
    script = REPO_ROOT / "meta-vul" / "run_harness_discovery.py"
    cmd = [sys.executable, str(script), "--agent", args.agent, "--seed-prompt", args.seed_prompt]
    if args.model:
        cmd += ["--model", args.model]
    if args.optimizer_agent:
        cmd += ["--optimizer-agent", args.optimizer_agent]
    if args.optimizer_model:
        cmd += ["--optimizer-model", args.optimizer_model]
    cmd += passthrough
    return _run(cmd, REPO_ROOT / "meta-vul")


def cmd_eval(args, passthrough):
    agent_dir = REPO_ROOT / AGENTS[args.agent]
    script = agent_dir / "runner" / "run_single_agent.py"
    cmd = [sys.executable, str(script), "--policy", args.policy]
    if args.config:
        cmd += ["--config", args.config]
    cmd += passthrough
    return _run(cmd, agent_dir)


def cmd_held_out(args, passthrough):
    agent_dir = REPO_ROOT / AGENTS[args.agent]
    script = agent_dir / "runner" / "run_single_agent.py"
    # run_single_agent.py requires --policy even in held-out mode, though the
    # value is ignored: the held-out prompt overrides whatever --policy loads.
    cmd = [sys.executable, str(script), "--held-out", args.session, "--policy", "held-out"]
    if args.iteration is not None:
        cmd += ["--iteration", str(args.iteration)]
    if args.model:
        if args.agent != "codex":
            print(f"error: --model is only supported for --agent codex", file=sys.stderr)
            return 2
        cmd += ["--model", args.model]
    if args.config:
        cmd += ["--config", args.config]
    cmd += passthrough
    return _run(cmd, agent_dir)


def main():
    parser = argparse.ArgumentParser(description="metavul: single entrypoint for the meta-vul pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Run the full harness-discovery optimization loop")
    p_discover.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p_discover.add_argument("--seed-prompt", default="scot", help="Seed reasoning policy (default: scot)")
    p_discover.add_argument("--model", default=None,
                            help="Override the agent's default model name. For --agent codex, "
                                 "prefix with a known provider (e.g. 'tinker/Qwen/Qwen3.5-9B-Base') "
                                 "to route through a different model provider.")
    p_discover.add_argument("--optimizer-agent", choices=sorted(AGENTS), default=None,
                            help="Which agent proposes prompt revisions (default: claude-cli)")
    p_discover.add_argument("--optimizer-model", default=None,
                            help="Override the optimizer agent's default model name (same "
                                 "provider-prefix syntax as --model)")

    p_eval = sub.add_parser("eval", help="Run a single evaluation pass (no optimization loop)")
    p_eval.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p_eval.add_argument("--policy", default="scot", help="Reasoning policy to evaluate (default: scot)")
    p_eval.add_argument("--config", default=None, help="Override the agent's default config")

    p_held = sub.add_parser("held-out", help="Evaluate a discovery session's prompt on a held-out dataset")
    p_held.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p_held.add_argument("--session", required=True, help="Discovery session id (OPT_SESSION_ID)")
    p_held.add_argument("--iteration", type=int, default=None,
                        help="Evaluate this iteration's prompt instead of the session's best")
    p_held.add_argument("--model", default=None,
                        help="Evaluate using this model instead of the one auto-detected from "
                             "the session's logs (--agent codex only; same provider-prefix "
                             "syntax as `discover --model`, e.g. 'tinker/thinkingmachines/Inkling-Small')")
    p_held.add_argument("--config", default=None, help="Override the agent's default config")

    args, passthrough = parser.parse_known_args()

    if args.command == "discover":
        code = cmd_discover(args, passthrough)
    elif args.command == "eval":
        code = cmd_eval(args, passthrough)
    else:
        code = cmd_held_out(args, passthrough)
    sys.exit(code)


if __name__ == "__main__":
    main()
