#!/bin/bash
# Offline check of a fresh setup: tools on PATH, the Python env, every entry point,
# both datasets, and shell syntax. Makes no model calls and needs no API keys.
# Run from code/ (the Docker image does this by default).
set -u
cd "$(dirname "$0")" || exit 1
fail=0
check() {  # check <label> <command...>
  if out="$("${@:2}" 2>&1)"; then echo "ok    $1${out:+  ($(printf '%s' "$out" | head -1))}"
  else echo "FAIL  $1"; printf '%s\n' "$out" | tail -5 | sed 's/^/      /'; fail=1; fi
}

check "uv"             uv --version
check "litellm"        litellm --version
check "codex"          codex --version
check "claude"         claude --version
check "python in .venv" uv run python3 -c 'import sys; assert sys.prefix.endswith(".venv"), sys.prefix; print(sys.version.split()[0])'
check "pyyaml, tiktoken" uv run python3 -c 'import yaml, tiktoken; tiktoken.get_encoding("cl100k_base"); print("imported")'

check "metavul.py"             bash -c 'cd meta-vul-code && uv run python3 metavul.py --help >/dev/null'
check "run_optimization.py"    bash -c 'cd meta-vul-code/meta-vul && uv run python3 run_optimization.py --help >/dev/null'
check "run_harness_discovery"  bash -c 'cd meta-vul-code/meta-vul && uv run python3 run_harness_discovery.py --help >/dev/null'
check "codex runner"           bash -c 'cd meta-vul-code/codex-self-improving-agent && uv run python3 runner/run_single_agent.py --help >/dev/null'
check "claude runner"          bash -c 'cd meta-vul-code/claude-cli-agent && uv run python3 runner/run_single_agent.py --help >/dev/null'
check "judge"                  uv run python3 evaluation/harness_quality_paper.py --help
check "metrics-paper.sh"       bash -c 'uv run ./evaluation/metrics-paper.sh --help >/dev/null'
check "results (4 sessions)"   bash -c 'n=$(uv run ./evaluation/metrics-paper.sh | grep -c "^session_id:"); echo "$n sessions"; [ "$n" -eq 4 ]'

check "datasets" uv run python3 - <<'PY'
import sys, yaml
from pathlib import Path
sys.path.insert(0, "meta-vul-code/meta-vul/runner"); sys.path.insert(0, "evaluation")
import eval_prompt, metrics_paper
base = Path("meta-vul-code/meta-vul").resolve()
rel = yaml.safe_load(open(base / "configs/config-titanvul-cweval-generic.yaml"))["dataset"]["path"]
val = base / rel if (base / rel).exists() else base.parent / rel
n_val, n_held = len(eval_prompt.load_dataset(val)), len(metrics_paper.load_gt(metrics_paper.HELD_GT))
assert (n_val, n_held) == (404, 870), (n_val, n_held)
print(f"validation {n_val}, held-out {n_held}")
PY

# Codex's OS sandbox must work: the optimizer reads its workspace (current prompt,
# history, predictions) through it. Containers often block it, and then the optimizer
# writes prompts blind. Reads a file inside the workspace (must work) and one outside
# it (must be denied). No model call.
check "codex sandbox" bash -c 'cd meta-vul-code/codex-self-improving-agent && uv run python3 - <<"PY"
import sys, yaml
sys.path.insert(0, "runner")
import model_client as m
r = m.verify_permission_boundary(yaml.safe_load(open("configs/config-test.yaml")))
print("allowed_read_ok=%s outside_read_denied=%s" % (r["allowed_read_ok"], r["outside_read_denied"]))
if not r["ok"]:
    print(r["denial_message"] or "(no message)")
    sys.exit(1)
PY'

check "shell syntax" bash -c 'for f in meta-vul-code/run/*.sh evaluation/*.sh; do bash -n "$f" || exit 1; done'

echo
[ $fail = 0 ] && echo "all checks passed" || echo "some checks FAILED"
exit $fail
