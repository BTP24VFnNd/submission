#!/bin/bash
# Session 20260927_195846 (codex, openrouter/deepseek/deepseek-v4-flash-0731), launched by run/discover.sh. Launch command:
#   python3 metavul.py discover --agent codex --model openrouter/deepseek/deepseek-v4-flash-0731 --optimizer-agent codex --optimizer-model openrouter/deepseek/deepseek-v4-flash-0731 --concurrency 60 --limit 4 --max-iterations 1 --session-id 20260927_195846 
# Running this script resumes that session.
cd "$(dirname "$0")/../.." || exit 1
export LITELLM_PROXY_PORT=4449
python3 metavul.py discover --agent codex --model openrouter/deepseek/deepseek-v4-flash-0731 --optimizer-agent codex --optimizer-model openrouter/deepseek/deepseek-v4-flash-0731 --concurrency 60 --limit 4 --max-iterations 1 --resume 20260927_195846 
