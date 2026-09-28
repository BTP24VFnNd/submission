#!/bin/bash
# Held-out eval of session 20260919_204729, launched 20260922_032839 by run/held-out.sh. Launch command:
#   python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/deepseek/deepseek-v4-flash-0731 --concurrency 30 --iteration 1 --concurrency 60 
# Running this script reruns that evaluation.
cd "$(dirname "$0")/../.." || exit 1
export LITELLM_PROXY_PORT=4446
python3 metavul.py held-out --agent codex --session 20260919_204729 --model openrouter/deepseek/deepseek-v4-flash-0731 --concurrency 30 --iteration 1 --concurrency 60 
