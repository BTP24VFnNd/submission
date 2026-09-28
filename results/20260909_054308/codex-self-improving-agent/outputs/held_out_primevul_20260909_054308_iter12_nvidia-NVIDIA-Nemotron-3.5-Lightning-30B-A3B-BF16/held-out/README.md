# Nemotron iteration 12 rerun

Inkling-Small → Nemotron-3.5-Lightning-30B-A3B-BF16, session `20260909_054308`, PrimeVul test. Completed September 24, 2026 (UTC).

| Metric | Result |
|---|---:|
| Pairwise accuracy | 20.92% (91/435) |
| Localization precision | 15.89 |
| Localization recall | 17.79 |
| Localization F1 | 16.78 |
| Unique samples | 870 |
| Unparseable verdicts | 2 |
| Final request errors | 0 |

This is a fresh rerun to recover the missing raw predictions and localization score. It replaces the old summary-only pairwise result of 18.62%; both new scores come from these predictions. The prompt was selected on validation, not held-out results. The old summary and aggregate remain in [`results/archive/nemotron_iter12_original`](../../../results/archive/nemotron_iter12_original/).

## Files

- `predictions.jsonl`: original predictions, unchanged.
- `logs.jsonl.gz`: original prompts, responses, attempts and isolation metadata; gzip-compressed without changing the contents.
- `provenance.json`: settings, source and artifact hashes, timestamps, and preflight history. No credentials are included.

Codex CLI 0.155.1 through Tinker; high reasoning, concurrency 8, 900-second timeout, three retries. The one-sample smoke result was reused. Two unparseable verdicts are retained: pairwise scoring includes all 435 pairs; classification metrics use 868 parseable verdicts. Localization follows `evaluation/metrics.py` and uses true-positive vulnerable samples, not all 870 functions.

## Recompute

From the repository root (no inference or credentials needed):

```sh
python3 evaluation/metrics.py \
  --predictions meta-vul-code/codex-self-improving-agent/outputs/held_out_primevul_20260909_054308_iter12_nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/held-out/predictions.jsonl \
  --ground-truth data/primevul/test_with_vul_lines_uuid.jsonl
```

The current metrics implementation has the same SHA-256 as the one used for the rerun. Dataset and prompt hashes also match. `evaluation/metrics.sh 20260909_054308` discovers these predictions automatically.

To inspect logs without replacing any files:

```sh
gzip -dc meta-vul-code/codex-self-improving-agent/outputs/held_out_primevul_20260909_054308_iter12_nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/held-out/logs.jsonl.gz | less
```
