#!/bin/bash
# Held-out (PrimeVul) evaluation of a Claude CLI discovery session, with the evaluator model
# taken from a config file. The Claude CLI counterpart of run/held-out.sh (which is Codex-only:
# metavul.py accepts --model only for --agent codex, so here the model comes from --config).
#
# Usage: ./run/held-out-claude.sh <session_id> <config> [all | <iteration>...] [extra runner flags]
#   ./run/held-out-claude.sh 20260922_170038 configs/config-heldout-gpt-oss-120b.yaml
#       every iteration of the GLM-5.3 session (0..8), cross-transfer to GPT-OSS-120B
#   ./run/held-out-claude.sh 20260922_170038 configs/config-heldout-gpt-oss-120b.yaml 4
#       only iteration 4 (the validation-selected harness)
#   ./run/held-out-claude.sh 20260922_170038 configs/config-heldout-gpt-oss-120b.yaml 0 4 6 --limit 5
#       iterations 0, 4, 6 on the first 5 samples (smoke test)
# <config> is relative to claude-cli-agent/.
#
# Needs: OPENROUTER_API_KEY; the claude CLI on PATH (or CLAUDE_BIN=/path/to/claude);
# a litellm with proxy extras (litellm on PATH, or LITELLM_BIN).
# The LiteLLM proxy starts on LITELLM_PROXY_PORT (default 4453) if it is not already running.
# A proxy only reads its config and API keys when it starts: if one is already up on that
# port with an old key or an old .litellm/config.yaml, stop it first:
#   pkill -f 'litellm.*--port 4453'
#
# If the session's files are only in ../results/<session_id>/, they are copied back
# to meta-vul/outputs/<session_id>/ (the runner reads the prompts from there).
# Each launch is recorded in run/held-out/ like run/held-out.sh does.
# Results: ../evaluation/metrics-paper.sh <session_id>
set -u
RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$RUNS_DIR/.." && pwd)"
TOP_DIR="$(cd "$REPO_DIR/.." && pwd)"

if [ $# -lt 2 ]; then
  sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 1
fi
SESSION_ID="$1"; CONFIG="$2"; shift 2

# Iterations first ("all" or numbers), then optional runner flags; everything from the first
# --flag onward is passed through unchanged (e.g. --limit 5, --concurrency 30, --dry-run).
ITERS=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    all) shift ;;
    --*) EXTRA=("$@"); break ;;
    *) ITERS+=("$1"); shift ;;
  esac
done

cd "$REPO_DIR" || exit 1
[ -f "claude-cli-agent/$CONFIG" ] || { echo "error: config not found: claude-cli-agent/$CONFIG" >&2; exit 1; }

# Session files: restore from the completed backup if they are not in meta-vul/outputs.
SESS_DIR="meta-vul/outputs/$SESSION_ID"
BACKUP="$TOP_DIR/../results/$SESSION_ID/meta-vul/outputs/$SESSION_ID"
if [ ! -f "$SESS_DIR/summary.json" ]; then
  if [ -f "$BACKUP/summary.json" ]; then
    echo "[run] copying session files back from ../results/$SESSION_ID"
    mkdir -p meta-vul/outputs && cp -R "$BACKUP" meta-vul/outputs/ || exit 1
  else
    echo "error: no $SESS_DIR/summary.json and no backup at $BACKUP" >&2; exit 1
  fi
fi
if [ ${#ITERS[@]} -eq 0 ]; then
  ITERS=($(ls "$SESS_DIR" | sed -n 's/^iter_0*\([0-9][0-9]*\)\.txt$/\1/p' | sort -n | uniq))
fi
echo "[run] session $SESSION_ID, iterations: ${ITERS[*]}"

# Environment.
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=sk-or-v1-...)}"
export LITELLM_PROXY_PORT="${LITELLM_PROXY_PORT:-4453}"
if [ -z "${CLAUDE_BIN:-}" ] && ! command -v claude >/dev/null 2>&1; then
  echo "error: claude CLI not found; install it or set CLAUDE_BIN=/path/to/claude" >&2; exit 1
fi
MODEL_ID="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['name'])" "claude-cli-agent/$CONFIG")"
SERVED="$(curl -s -m 3 -H 'Authorization: Bearer sk-local-proxy' "http://localhost:$LITELLM_PROXY_PORT/v1/models" 2>/dev/null)"
if [ -n "$SERVED" ] && ! printf '%s' "$SERVED" | grep -qF "\"id\":\"$MODEL_ID\""; then
  echo "error: proxy on port $LITELLM_PROXY_PORT does not serve '$MODEL_ID'; stop it (pkill -f 'litellm.*--port $LITELLM_PROXY_PORT') and rerun" >&2
  exit 1
fi
echo "[run] model $MODEL_ID, proxy port $LITELLM_PROXY_PORT"

mkdir -p "$RUNS_DIR/held-out"
MODEL_TAG="$(printf '%s' "$MODEL_ID" | tr '/:' '__')"
status=0
for it in "${ITERS[@]}"; do
  CMD=(python3 metavul.py held-out --agent claude-cli --session "$SESSION_ID" --iteration "$it" --config "$CONFIG")
  case " ${EXTRA[*]:-} " in *" --dry-run "*) ;; *)
    RECORD="$RUNS_DIR/held-out/held-out_${SESSION_ID}_claude-cli_${MODEL_TAG}_iter${it}.sh"
    {
      echo '#!/bin/bash'
      echo "# Held-out eval of session $SESSION_ID iteration $it, launched $(date +%Y%m%d_%H%M%S) by run/held-out-claude.sh."
      echo 'cd "$(dirname "$0")/../.." || exit 1'
      echo "export LITELLM_PROXY_PORT=$LITELLM_PROXY_PORT"
      [ -n "${LITELLM_BIN:-}" ] && echo "export LITELLM_BIN=$(printf '%q' "$LITELLM_BIN")"
      [ -n "${CLAUDE_BIN:-}" ] && echo "export CLAUDE_BIN=$(printf '%q' "$CLAUDE_BIN")"
      printf '%q ' "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"}; echo
    } > "$RECORD"; chmod +x "$RECORD"
  ;; esac
  echo "[run] iteration $it"
  "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"} || status=$?
done
exit $status
