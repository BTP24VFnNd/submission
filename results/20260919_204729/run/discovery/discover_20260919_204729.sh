#!/bin/bash
# Session 20260919_204729, launched by run/discover.sh. Launch command:
#   python3 metavul.py discover --agent codex --model openrouter/deepseek/deepseek-v4-flash-0731 --optimizer-agent codex --optimizer-model openrouter/deepseek/deepseek-v4-flash-0731 --session-id 20260919_204729 
# Running this script resumes that session.
cd "$(dirname "$0")/../.." || exit 1
export LITELLM_PROXY_PORT=4446
python3 metavul.py discover --agent codex --model openrouter/deepseek/deepseek-v4-flash-0731 --optimizer-agent codex --optimizer-model openrouter/deepseek/deepseek-v4-flash-0731 --resume 20260919_204729 
