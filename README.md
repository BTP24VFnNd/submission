# MetaVul: code and results

This repository holds what is needed to reproduce the paper: the code for the four discovery sessions, their PrimeVul held-out runs, the GPT-4o transfer, and the harness-quality judge (`code/`), and the outputs of those runs (`results/`).

## Layout

```
README.md
Dockerfile, .dockerignore         clean-machine test of the setup (see Testing the setup in Docker)
results/
  <session_id>/                   one folder per session (see Sessions below)
    meta-vul/outputs/<id>/        harnesses (iter_NNN.txt), history.json, summary.json, per-iteration metrics and predictions
    *-agent/outputs/              validation (opt_*) and PrimeVul held-out (held_out_*) runs: predictions and transcripts
    harness/<id>/                 Pareto-frontier harnesses (validation pairwise accuracy vs joint1: first CWE correct + loc hit)
code/
  data/
    titanvul_cweval_optimization_val_uuid.jsonl   validation: TitanVul + CWEval, 404 functions / 202 pairs
    primevul/test_with_vul_lines_uuid.jsonl       held-out: PrimeVul paired test, 870 functions (430 pairs scored)
  meta-vul-code/
    metavul.py                    entry point: discover | eval | held-out
    .litellm/config.yaml          model routing for OpenRouter / Tinker (keys read from env vars)
    meta-vul/                     discovery loop: run_optimization.py, optimizer/, runner/, seed harness, configs
    codex-self-improving-agent/   detector through Codex CLI (runner, harness templates, config)
    claude-cli-agent/             detector through Claude Code (runner, harness templates, configs)
    harness/default/harness.py    template the Pareto-frontier harnesses are written from
    run/                          launch scripts (each launch records itself in run/discovery/ or run/held-out/)
  evaluation/
    metrics.py, metrics_paper.py  all paper metrics (pairwise accuracy, joint hit rate, ...)
    metrics-paper.sh              per-session tables / CSV used for the paper
    metrics-cross-transfer.sh     held-out results per model per iteration
    harness_quality/, harness_quality_paper.py, harness-quality-paper.sh   the judge
  scripts/harness_quality_{openrouter,codex}_judge.py   judge backends
```

## Sessions

| Session | Agent + model (detector and optimizer) | Paper role |
|---|---|---|
| 20260909_054308 | Codex CLI + Inkling-Small (Tinker) | discovery, held-out |
| 20260919_204729 | Codex CLI + DeepSeek V4 Flash (OpenRouter) | discovery, held-out, GPT-4o transfer |
| 20260716_181608 | Claude Code + Opus 4.6 | discovery, held-out |
| 20260922_170038 | Claude Code + GLM-5.3 (OpenRouter) | discovery, held-out |

## Setup

Start in the repository root. Step 2 moves into `code/`, and every later command runs from `code/` (or from a subfolder of it when a block starts with `cd`).

Setup uses [uv](https://docs.astral.sh/uv/). Versions in brackets are the ones the runs used.

**1. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Python environment** (Python 3.11, `pyyaml`, `tiktoken`; from `pyproject.toml`)

```bash
cd code
uv sync      # creates code/.venv/ (downloads Python 3.11 if needed) and installs the packages
```

The launch scripts call `python3`, so it must resolve to this environment. Either activate it once per shell:

```bash
source .venv/bin/activate    # from code/; created by uv sync above
```

or skip activation and prefix commands with `uv run`, which puts `.venv/bin` first on PATH (it works from any subfolder):

```bash
cd meta-vul-code && uv run ./run/discover.sh ...
```

**3. LiteLLM proxy** (litellm 1.100.0). Routes OpenRouter and Tinker models to both CLIs. It is installed as a separate tool so its dependencies stay out of the project environment:

```bash
uv tool install 'litellm[proxy]==1.100.0'
litellm --version            # must be on PATH, or set LITELLM_BIN=/path/to/litellm
```

**4. Agent CLIs** (Node.js 24, Codex CLI 0.153.4, Claude Code 2.1.283). Install the one(s) your runs use:

```bash
npm install -g @openai/codex@0.153.4                    # Codex CLI: Inkling-Small, DeepSeek, GPT-4o sessions
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.283   # Claude Code: Opus 4.6, GLM-5.3 sessions
```

Claude Code for Opus 4.6 uses its own login (`claude` once to sign in). For other models both CLIs talk to the local LiteLLM proxy, which the launch scripts start on `LITELLM_PROXY_PORT` when it is not running.

**5. API keys**

```bash
export OPENROUTER_API_KEY=...   # DeepSeek V4 Flash, GLM-5.3, GPT-4o, and the GPT-OSS-120B judge
export TINKER_API_KEY=...       # Inkling-Small
```

A running proxy reads keys and `.litellm/config.yaml` only at start, so restart it after changing either (`pkill -f 'litellm.*--port <port>'`).

**6. Check**

```bash
cd meta-vul-code && python3 metavul.py --help
./run/discover.sh "openrouter/deepseek/deepseek-v4-flash-0731" --agent codex --limit 4 --max-iterations 1   # smoke test: seed + 1 iteration on 4 samples (2 pairs); makes paid model calls
```

## Testing the setup in Docker

`Dockerfile` repeats Setup steps 1 to 4 on a clean Debian image (Node.js 24), so a successful build confirms the instructions. `docker-check.sh` then checks the tools, the Python environment, every entry point, both datasets (404 and 870 functions) and shell syntax, with no model calls or API keys. It also checks Codex's sandbox (`codex sandbox`), which the optimizer needs to read its workspace.

Every `docker run` below uses `--security-opt seccomp=unconfined`. On Linux, Codex runs each shell command the model issues inside `bwrap`, which has to create Linux namespaces. Docker's default seccomp profile blocks that, so without the option the sandbox fails (`bwrap: No permissions to create a new namespace`): the detector still answers, since the code sample is part of its input, but the optimizer cannot read any file and writes harnesses blind. The option turns off only Docker's syscall filter for this container; the rest of Docker's isolation stays, and Codex's own sandbox keeps the models read-only and away from the labeled data. If `codex sandbox` still fails with it, use `--privileged` instead.

Build from the repo root. The image holds `code/` and the finished runs in `results/` (without the raw logs and workspaces), so the scoring commands below work inside it as they are:

```bash
docker build -t metavul .
docker run --rm --security-opt seccomp=unconfined metavul    # offline checks; prints "all checks passed"
```

For real runs, set the key in your own terminal first, then pass it in and open a shell (`-e NAME` copies the value from that terminal; it passes nothing if the key is not set there). Mount a folder for outputs so they survive the container:

```bash
export OPENROUTER_API_KEY=...   # in your own terminal, before docker run
docker run --rm -it --security-opt seccomp=unconfined -e OPENROUTER_API_KEY -e TINKER_API_KEY \
  -v "$PWD/runs:/work/code/meta-vul-code/meta-vul/outputs" metavul bash
cd meta-vul-code && uv run ./run/discover.sh "openrouter/deepseek/deepseek-v4-flash-0731" --agent codex --limit 4 --max-iterations 1
```

If you change a key inside the container, run `pkill -f litellm` before the next run: the proxy keeps the key it started with. Claude Code with Opus 4.6 needs its own login, so run `claude` once inside the container first. The Codex and Claude runners also write detector outputs under `codex-self-improving-agent/outputs/` and `claude-cli-agent/outputs/`; mount those too to keep them.

## Discovery (validation, RQ1)

```bash
cd meta-vul-code
./run/discover.sh "openrouter/deepseek/deepseek-v4-flash-0731" --agent codex --concurrency 60
./run/discover.sh "tinker/thinkingmachines/Inkling-Small" --agent codex
./run/discover.sh "openrouter/z-ai/glm-5.3" --agent claude-cli
./run/discover.sh claude-opus-4-6 --agent claude-cli
```

Each launch records itself as `run/discovery/discover_<session_id>.sh`, which resumes that session when run again. The DeepSeek session was launched with `python3 metavul.py discover --agent codex --model openrouter/deepseek/deepseek-v4-flash-0731 --optimizer-agent codex --optimizer-model openrouter/deepseek/deepseek-v4-flash-0731`. The exact configs of the two Codex sessions are in `meta-vul/configs/.generated-config-*.yaml`. At the end, the loop writes the Pareto-frontier harnesses (validation pairwise accuracy vs joint1: first CWE correct + loc hit) to `harness/<session_id>/`. For a finished session, the same step can be rebuilt from saved predictions without model calls:

```bash
./run/finalize.sh <session_id>          # --dry-run to only report
```

## Held-out and GPT-4o transfer (RQ1 held-out, RQ2)

```bash
cd meta-vul-code
./run/held-out.sh 20260919_204729 --iteration 11                         # own evaluator
./run/held-out-ranked.sh 20260919_204729 openrouter/openai/gpt-4o        # every iteration on GPT-4o
./run/held-out-claude.sh 20260922_170038 configs/config-heldout-glm-5.3.yaml all
```

Each launch records itself in `run/held-out/`, and rerunning a record skips finished samples. The GPT-4o transfers ran through OpenRouter (`openrouter/openai/gpt-4o`).

## Scoring

The scoring scripts read the runs in `results/` directly, so the paper's numbers can be recomputed without running any model. Run them from `code/` with `uv run` (or with `.venv` activated), since they call `python3`:

```bash
uv run ./evaluation/metrics-paper.sh                                   # all sessions
uv run ./evaluation/metrics-paper.sh 20260909_054308 20260919_204729 --full
uv run ./evaluation/metrics-paper.sh 20260919_204729 --csv out.csv
```

These work the same inside the Docker shell.

Held-out scores use 430 PrimeVul pairs: 5 pairs that also appear in the validation set are left out (`HELD_EXCLUDE_PAIRS` in `metrics_paper.py`). A pair with a missing verdict counts as wrong and stays in the denominator. Set `METRICS_EXTRA_ROOTS` (colon-separated `meta-vul-code` dirs) to also read runs from other checkouts.

## Harness quality (RQ3)

```bash
uv run ./evaluation/harness-quality-paper.sh --judge openai/gpt-oss-120b    # judge used in the paper (OpenRouter)
uv run ./evaluation/harness-quality-paper.sh --no-judge                     # static metrics only
```

Rubric: `evaluation/harness_quality/judge_prompt.txt`. Two judge runs per harness. Results go to `evaluation/harness_quality/results/`, and judge calls are cached there.
