#!/bin/bash
# Harness quality across optimizer iterations, one row per iteration: pairwise,
# F1, recall, precision, accuracy, harness tokens (and ratio to seed), and the
# GPT-5.4 semantic judge score with the SCoT, prohibited-behavior and
# output-contract ratings. Built on the evaluator in evaluation/harness_quality/.
#
# Usage: ./evaluation/harness-quality-paper.sh                    the four paper sessions
#        ./evaluation/harness-quality-paper.sh 20260716_181608    one or more sessions
#        ./evaluation/harness-quality-paper.sh --no-judge         tokens + metrics only (fast)
#        ./evaluation/harness-quality-paper.sh --judge openai/gpt-oss-120b   OpenRouter judge
#
# Default judge is GPT-5.4 through the Codex CLI. --judge <OpenRouter model id>
# uses OpenRouter instead (needs OPENROUTER_API_KEY).
#
# Writes evaluation/harness_quality/results/harness_quality_<judge>.{md,csv,json}. Judge results
# are cached in evaluation/harness_quality/results/judge_cache/, so a rerun skips
# harnesses that were already judged. Every validation iteration with a harness and
# metrics is included, so rerun it as sessions finish more iterations.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The judge runs through the Codex CLI, which is installed under nvm.
if [[ " $* " != *" --no-judge "* && " $* " != *" --judge "* ]] && ! command -v codex >/dev/null; then
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi
exec python3 "$ROOT/evaluation/harness_quality_paper.py" "$@"
