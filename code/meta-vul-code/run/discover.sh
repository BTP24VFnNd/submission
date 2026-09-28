#!/bin/bash
# Usage: ./run/discover.sh [model] [--agent codex|claude-cli] [extra discover flags]
#   ./run/discover.sh                                          agent and model set below
#   ./run/discover.sh --limit 10                               smoke test
#   ./run/discover.sh "openrouter/qwen/qwen3.8-flash"           that model for both roles
#   ./run/discover.sh claude-opus-4-6                          Anthropic model, CLI login
#   ./run/discover.sh "openrouter/deepseek/deepseek-v4-flash-0731" --agent codex --concurrency 60
# Both roles (evaluator and optimizer) go through the chosen CLI.
#
# Model routing is the same for both agents:
#   - "openrouter/<slug>" or "tinker/<name>" goes through this checkout's LiteLLM
#     proxy (.litellm/config.yaml). The model must be registered there, and the
#     proxy on LITELLM_PROXY_PORT must have started after that entry was added
#     (a proxy reads the config only at start). OpenRouter needs
#     OPENROUTER_API_KEY, Tinker needs TINKER_API_KEY.
#   - a plain Anthropic id ("claude-opus-4-6") is only valid with --agent claude-cli
#     and calls Anthropic directly with the CLI's own login. Pin a full id, never an
#     alias like "opus" (aliases drift to the newest model of that tier).
#
# Each launch picks the session id up front and records itself as
# run/discovery/discover_<session_id>.sh, so every session in meta-vul/outputs/ has a matching
# script here. Running that recorded script resumes its session.

AGENT="claude-cli"
EVALUATOR_MODEL="openrouter/z-ai/glm-5.3"
OPTIMIZER_MODEL="openrouter/z-ai/glm-5.3"
# Others served by .litellm/config.yaml:
#   openrouter/qwen/qwen3.8-flash
#   openrouter/deepseek/deepseek-v4-flash-0731
#   openrouter/z-ai/glm-5.3-flash
#   openrouter/nvidia/nemotron-3.5-lightning
#   tinker/zai-org/GLM-5.3:peft:262144
#   tinker/thinkingmachines/Inkling-Small
#   tinker/thinkingmachines/Inkling

# Proxy port for provider-prefixed models. 4445/4446 belong to the codex
# sessions and predate the Nemotron entry, so a fresh port is the default;
# LITELLM_PROXY_PORT in the environment overrides.
DEFAULT_PROXY_PORT=4449

# Concurrent evaluator calls per iteration. A --concurrency N typed on the
# command line overrides this.
CONCURRENCY=60

if [ -n "$1" ] && [ "${1#--}" = "$1" ]; then
  EVALUATOR_MODEL="$1"; OPTIMIZER_MODEL="$1"; shift
fi

# Pull --agent out of the remaining flags; everything else passes through.
REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --agent)   AGENT="$2"; shift 2 ;;
    --agent=*) AGENT="${1#--agent=}"; shift ;;
    *)         REST+=("$1"); shift ;;
  esac
done
set -- "${REST[@]}"
case "$AGENT" in codex|claude-cli) ;; *) echo "error: --agent must be codex or claude-cli" >&2; exit 1 ;; esac

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUNS_DIR/.." || exit 1

case "$EVALUATOR_MODEL" in
  openrouter/*|tinker/*)
    # This checkout's own LiteLLM proxy, kept off 4444 so it never collides with
    # a proxy started from another checkout.
    export LITELLM_PROXY_PORT="${LITELLM_PROXY_PORT:-${TINKER_PROXY_PORT:-$DEFAULT_PROXY_PORT}}"
    # litellm needs the proxy extras (pip install 'litellm[proxy]'); set LITELLM_BIN to
    # use one that is not first on PATH.
    case "$EVALUATOR_MODEL" in
      openrouter/*) [ -n "$OPENROUTER_API_KEY" ] || { echo "error: OPENROUTER_API_KEY is not set" >&2; exit 1; } ;;
      tinker/*)     [ -n "$TINKER_API_KEY" ]     || { echo "error: TINKER_API_KEY is not set" >&2; exit 1; } ;;
    esac
    # If a proxy is already up on that port, it must serve the model.
    MODEL_ID="${EVALUATOR_MODEL#*/}"
    SERVED="$(curl -s -m 3 -H 'Authorization: Bearer sk-local-proxy' \
      "http://localhost:$LITELLM_PROXY_PORT/v1/models" 2>/dev/null)"
    if [ -n "$SERVED" ] && ! printf '%s' "$SERVED" | grep -qF "\"id\":\"$MODEL_ID\""; then
      echo "error: proxy on port $LITELLM_PROXY_PORT does not serve '$MODEL_ID'." >&2
      echo "       It serves: $(printf '%s' "$SERVED" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')" >&2
      echo "       Restart it (pkill -f 'litellm.*--port $LITELLM_PROXY_PORT') so it reloads .litellm/config.yaml," >&2
      echo "       or launch with LITELLM_PROXY_PORT=<a free port> so a fresh proxy is started." >&2
      exit 1
    fi
    echo "[run] agent $AGENT, model $EVALUATOR_MODEL via proxy port $LITELLM_PROXY_PORT"
    ;;
  *)
    if [ "$AGENT" != "claude-cli" ]; then
      echo "error: '$EVALUATOR_MODEL' has no provider prefix; codex needs openrouter/... or tinker/..." >&2
      exit 1
    fi
    unset LITELLM_PROXY_PORT
    echo "[run] agent $AGENT, model $EVALUATOR_MODEL via Anthropic (CLI login)"
    ;;
esac

CMD=(python3 metavul.py discover --agent "$AGENT" --model "$EVALUATOR_MODEL"
     --optimizer-agent "$AGENT" --optimizer-model "$OPTIMIZER_MODEL"
     --concurrency "$CONCURRENCY")

# Dry runs and resumes don't start a new session, so they leave no record.
case " $* " in
  *" --dry-run "*|*" --resume"*|*" --session-id"*) exec "${CMD[@]}" "$@" ;;
esac

SESSION_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNS_DIR/discovery"
RECORD="$RUNS_DIR/discovery/discover_$SESSION_ID.sh"
{
  echo '#!/bin/bash'
  echo "# Session $SESSION_ID ($AGENT, $EVALUATOR_MODEL), launched by run/discover.sh. Launch command:"
  echo "#   $(printf '%q ' "${CMD[@]}" "$@" --session-id "$SESSION_ID")"
  echo "# Running this script resumes that session."
  echo 'cd "$(dirname "$0")/../.." || exit 1'
  [ -n "$LITELLM_PROXY_PORT" ] && echo "export LITELLM_PROXY_PORT=$LITELLM_PROXY_PORT"
  [ -n "$LITELLM_BIN" ] && echo "export LITELLM_BIN=$(printf '%q' "$LITELLM_BIN")"
  echo "$(printf '%q ' "${CMD[@]}" "$@" --resume "$SESSION_ID")"
} > "$RECORD"
chmod +x "$RECORD"
echo "[run] session $SESSION_ID recorded at $RECORD"

exec "${CMD[@]}" "$@" --session-id "$SESSION_ID"
