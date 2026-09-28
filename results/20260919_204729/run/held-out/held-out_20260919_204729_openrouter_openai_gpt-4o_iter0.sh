#!/bin/bash
# Held-out eval of session 20260919_204729, launched 20260922_131001 by run/held-out.sh. Launch command:
#   python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/openai/gpt-4o --concurrency 60 --iteration 0 
# Running this script reruns that evaluation.
cd "$(dirname "$0")/../.." || exit 1
export LITELLM_PROXY_PORT=4451
python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/openai/gpt-4o --concurrency 60 --iteration 0 
