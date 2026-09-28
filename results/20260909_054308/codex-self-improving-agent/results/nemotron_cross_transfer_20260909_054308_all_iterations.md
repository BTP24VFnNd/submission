# Nemotron cross-transfer: all iterations

- Optimization session: `20260909_054308`
- Source model: `thinkingmachines/Inkling-Small`
- Target model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Held-out set: PrimeVul test, 870 samples / 435 pairs
- Target reasoning effort: high
- Timeout / retries: 900 seconds / 3
- Concurrency: 30 for iterations 0–11 and 13–15; 8 for the iteration 12 rerun

| Iteration | Validation pairwise | Held-out pairwise | Correct pairs | Accuracy | Precision | Recall | F1 | Parse failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.44% | 12.87% | 56/435 | 53.22% | 51.90% | 87.82% | 65.24% | 0 |
| 1 | 11.39% | 8.97% | 39/435 | 51.84% | 51.09% | 85.98% | 64.10% | 0 |
| 2 | 15.84% | 17.01% | 74/435 | 53.45% | 52.48% | 72.87% | 61.02% | 0 |
| 3 | 12.87% | 13.10% | 57/435 | 51.84% | 51.19% | 78.85% | 62.08% | 0 |
| 4 | 17.33% | 20.00% | 87/435 | 54.94% | 53.90% | 68.28% | 60.24% | 0 |
| 5 | 14.85% | 20.46% | 89/435 | 53.33% | 52.98% | 59.31% | 55.97% | 0 |
| 6 | 20.79% | 19.08% | 83/435 | 54.25% | 53.29% | 68.97% | 60.12% | 0 |
| 7 | 15.84% | 19.77% | 86/435 | 53.10% | 52.53% | 64.37% | 57.85% | 0 |
| 8 | 23.27% | 14.71% | 64/435 | 50.80% | 50.64% | 63.91% | 56.50% | 0 |
| 9 | 20.79% | 17.01% | 74/435 | 52.30% | 52.24% | 53.56% | 52.89% | 0 |
| 10 | 20.79% | 19.08% | 83/435 | 53.22% | 53.37% | 51.03% | 52.17% | 0 |
| 11 | 24.26% | 19.31% | 84/435 | 53.45% | 54.05% | 45.98% | 49.69% | 0 |
| 12 **(selected; rerun)** | 27.72% | 20.92% | 91/435 | 53.46% | 53.83% | 50.11% | 51.90% | 2 |
| 13 | 23.27% | 17.93% | 78/435 | 53.22% | 53.45% | 49.89% | 51.61% | 0 |
| 14 | 24.75% | 20.46% | 89/435 | 53.33% | 53.35% | 53.10% | 53.23% | 0 |
| 15 | 21.78% | 17.01% | 74/435 | 50.00% | 50.00% | 45.29% | 47.53% | 0 |

## Summary

- The validation-selected prompt was iteration 12 (27.72% validation pairwise accuracy).
- The iteration 12 rerun reached 20.92% held-out pairwise accuracy (91/435), versus 12.87% (56/435) for the seed (+8.05 points). It is now the highest held-out score in this table; selection still used validation only.
- Iteration 12 localization F1 is 16.78, computed from the same fresh predictions as its pairwise score.
- The rerun contains 870 unique records: 868 boolean verdicts and two unparseable verdicts, with zero final request errors. Pairwise scoring retains all 435 pairs; classification metrics use the 868 parseable verdicts. Other iterations are unchanged.
- The old iteration 12 result (18.62%, 81/435) and previous aggregate are preserved in [archive/nemotron_iter12_original](archive/nemotron_iter12_original/). The rerun replaces missing raw data; it was not chosen by comparing repeated held-out scores. The existing `.complete-backups` snapshot is unchanged.

## Iteration 12 artifacts

[Predictions, compressed logs, provenance and scoring command](../outputs/held_out_primevul_20260909_054308_iter12_nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/held-out/README.md).

## Isolation canary

- The OS sandbox allowed reads inside the assigned workspace and denied the outside-workspace read.
- The model output exposed none of the outside marker, fake `ANTHROPIC_API_KEY` marker, or local URL marker.
