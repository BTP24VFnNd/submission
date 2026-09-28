#!/bin/bash
# Held-out eval of session 20260919_204729, launched 20260924_161958 by run/held-out.sh. Launch command:
#   python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/z-ai/glm-4.7-flash --concurrency 60 --iteration 8 --concurrency 100 
# Running this script reruns that evaluation.
cd "$(dirname "$0")/../.." || exit 1
export LITELLM_PROXY_PORT=4454
python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/z-ai/glm-4.7-flash --concurrency 60 --iteration 8 --concurrency 100 
