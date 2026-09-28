#!/bin/bash
# Usage: ./run/held-out.sh <session_id> [model] [extra held-out flags]
#   ./run/held-out.sh 20260919_204729                        best iteration, session's own model
#   ./run/held-out.sh 20260919_204729 --iteration 3          a specific iteration
#   ./run/held-out.sh 20260919_204729 "tinker/thinkingmachines/Inkling"   cross-model transfer
#   ./run/held-out.sh 20260919_204729 --concurrency 60
# Evaluates the session's prompt on the PrimeVul held-out test set through the Codex CLI.
# Results: ../evaluation/metrics.sh <session_id>
#
# The evaluator model is inferred from the session: the held-out run uses the
# same model that was used for validation during discovery (read from
# meta-vul/configs/.generated-config-*-<session_id>.yaml). Passing a model as
# the second argument overrides that and makes the run a cross-model transfer:
# same optimized prompt, different evaluator model.
#
# Each launch records the exact command as
# run/held-out/held-out_<session_id>_<model>[_iter<N>].sh, alongside run/discovery/.
# Rerunning with the same arguments overwrites that record; running the record
# reruns the evaluation (skip_existing means finished samples are not redone).

CONCURRENCY=60

RUNS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$RUNS_DIR/.." && pwd)"

if [ -z "$1" ] || [ "${1#--}" != "$1" ]; then
  echo "usage: $0 <session_id> [model] [extra held-out flags]" >&2
  echo "sessions:" >&2; ls "$REPO_DIR/meta-vul/outputs" >&2
  exit 1
fi
SESSION_ID="$1"; shift

MODEL=""
if [ -n "$1" ] && [ "${1#--}" = "$1" ]; then
  MODEL="$1"; shift
fi

if [ -z "$MODEL" ]; then
  GEN_CFG="$(ls "$REPO_DIR"/meta-vul/configs/.generated-config-*-"$SESSION_ID".yaml 2>/dev/null | head -1)"
  if [ -z "$GEN_CFG" ]; then
    echo "error: no generated config for session $SESSION_ID in meta-vul/configs/; pass a model explicitly" >&2
    exit 1
  fi
  # model.provider + model.name from the config -> "<provider>/<name>", as --model expects.
  MODEL="$(python3 - "$GEN_CFG" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))["model"]
p = m.get("provider"); n = m["name"]
print(f"{p}/{n}" if p else n)
PY
)" || exit 1
  echo "[run] model from $(basename "$GEN_CFG"): $MODEL"
fi

# Proxy port: reuse the one the discovery session ran on (recorded in
# run/discovery/discover_<session_id>.sh), so the held-out run talks to a
# proxy that serves the same model list. Each proxy loads .litellm/config.yaml
# only at start, so a proxy on another port may predate the session's model
# entries and reject its model name. LITELLM_PROXY_PORT in the environment
# still overrides.
if [ -z "$LITELLM_PROXY_PORT" ]; then
  REC="$RUNS_DIR/discovery/discover_$SESSION_ID.sh"
  if [ -f "$REC" ]; then
    LITELLM_PROXY_PORT="$(sed -n 's/^export LITELLM_PROXY_PORT=//p' "$REC" | head -1)"
  fi
fi
export LITELLM_PROXY_PORT="${LITELLM_PROXY_PORT:-${TINKER_PROXY_PORT:-4445}}"

# Preflight: if a proxy is already up on that port, it must serve the model.
# (If none is up, the runner starts one from the current .litellm/config.yaml.)
MODEL_ID="${MODEL#*/}"   # strip the provider prefix (tinker/, openrouter/)
SERVED="$(curl -s -m 3 -H 'Authorization: Bearer sk-local-proxy' \
  "http://localhost:$LITELLM_PROXY_PORT/v1/models" 2>/dev/null)"
if [ -n "$SERVED" ] && ! printf '%s' "$SERVED" | grep -qF "\"id\":\"$MODEL_ID\""; then
  echo "error: proxy on port $LITELLM_PROXY_PORT does not serve '$MODEL_ID'." >&2
  echo "       It serves: $(printf '%s' "$SERVED" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')" >&2
  echo "       Either restart that proxy (pkill -f 'litellm.*--port $LITELLM_PROXY_PORT') so it reloads" >&2
  echo "       .litellm/config.yaml, or launch with LITELLM_PROXY_PORT=<port of a proxy that serves it>." >&2
  exit 1
fi
echo "[run] proxy port $LITELLM_PROXY_PORT"

cd "$REPO_DIR" || exit 1

# An explicit --concurrency in the extra flags wins (it comes later on the command line).
echo "[run] concurrency $CONCURRENCY"

CMD=(python3 metavul.py held-out --agent codex --session "$SESSION_ID" --model "$MODEL"
     --concurrency "$CONCURRENCY")

# Dry runs leave no record.
case " $* " in
  *" --dry-run "*) exec "${CMD[@]}" "$@" ;;
esac

# Record name: session + model tag (+ iteration if given). Slashes/colons in
# the model name become underscores.
MODEL_TAG="$(printf '%s' "$MODEL" | tr '/:' '__')"
ITER_TAG=""
prev=""
for a in "$@"; do
  [ "$prev" = "--iteration" ] && ITER_TAG="_iter$a"
  case "$a" in --iteration=*) ITER_TAG="_iter${a#--iteration=}";; esac
  prev="$a"
done
mkdir -p "$RUNS_DIR/held-out"
RECORD="$RUNS_DIR/held-out/held-out_${SESSION_ID}_${MODEL_TAG}${ITER_TAG}.sh"
{
  echo '#!/bin/bash'
  echo "# Held-out eval of session $SESSION_ID, launched $(date +%Y%m%d_%H%M%S) by run/held-out.sh. Launch command:"
  echo "#   $(printf '%q ' "${CMD[@]}" "$@")"
  echo "# Running this script reruns that evaluation."
  echo 'cd "$(dirname "$0")/../.." || exit 1'
  echo "export LITELLM_PROXY_PORT=$LITELLM_PROXY_PORT"
  [ -n "$LITELLM_BIN" ] && echo "export LITELLM_BIN=$(printf '%q' "$LITELLM_BIN")"
  echo "$(printf '%q ' "${CMD[@]}" "$@")"
} > "$RECORD"
chmod +x "$RECORD"
echo "[run] held-out launch recorded at $RECORD"

exec "${CMD[@]}" "$@"
