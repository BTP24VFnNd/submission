#!/bin/bash
# Paper metrics, one block per session: the validation run (dataset, optimizer
# model and agent, results), then one PrimeVul held-out block per model (dataset,
# held-out model and agent, results). Default columns: n, pairwise, acc, prec,
# recall, F1, CWE accuracy, loc F1, loc hit rate.
#
# Each session ends with its validation Pareto frontier: pairwise accuracy vs
# joint1 (first CWE correct + loc hit).
#
# Usage: ./evaluation/metrics-paper.sh                      all sessions
#        ./evaluation/metrics-paper.sh 20260919_204729      one or more sessions
#        ./evaluation/metrics-paper.sh --csv results/paper_metrics.csv
#        ./evaluation/metrics-paper.sh --min-coverage 95    only runs with at least 95% of samples
#        ./evaluation/metrics-paper.sh --full               every metric column
#        ./evaluation/metrics-paper.sh --all                also show smoke-test sessions
#        ./evaluation/metrics-paper.sh --include-contaminated  held-out on all 435 PrimeVul pairs
#
# Held-out results leave out 5 PrimeVul pairs that also appear in the TitanVul
# validation sample (430 pairs); see HELD_EXCLUDE_PAIRS in metrics_paper.py.
#
# Set METRICS_EXTRA_ROOTS (colon-separated meta-vul-code dirs) to also read
# runs from other checkouts.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/evaluation/metrics_paper.py" "$@"
