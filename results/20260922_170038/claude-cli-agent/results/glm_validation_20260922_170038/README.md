# GLM-5.3 validation

Session `20260922_170038`: Claude CLI with OpenRouter GLM-5.3 for evaluation and optimization. TitanVul + CWEval, 404 functions / 202 pairs. Eight refinements plus seed; stopped after four iterations without a pairwise improvement (maximum configured: 15).

| Iteration | Pairwise accuracy | Correct pairs | Localization F1 | Unparseable |
|---:|---:|---:|---:|---:|
| 0 | 19.80% | 40/202 | 32.32 | 0 |
| 1 | 25.74% | 52/202 | 35.46 | 2 |
| 2 | 27.23% | 55/202 | 33.53 | 2 |
| 3 | 28.71% | 58/202 | 35.73 | 1 |
| 4 | 31.19% | 63/202 | 32.57 | 0 |
| 5 | 26.24% | 53/202 | 32.29 | 16 |
| 6 | 31.19% | 63/202 | 37.43 | 4 |
| 7 | 27.72% | 56/202 | 34.93 | 18 |
| 8 | 28.71% | 58/202 | 30.73 | 1 |

`best_prompt.txt` is iteration 4, the first highest pairwise score. Iteration 6 has the same pairwise score and higher localization F1; it is the recorded Pareto frontier. Pairwise gain over the repaired seed is **11.39 percentage points**. These are validation results, not held-out results.

## Files and verification

[Session artifacts](../../../meta-vul/outputs/20260922_170038/): all nine prompts, `best_prompt.txt`, history, summary, checkpoint predictions and metrics. Per-checkpoint `logs.jsonl.gz` contains the original logs compressed losslessly. Predictions are the canonical deduplicated checkpoints; logs retain the raw attempts.

From the repository root, with Python 3 and no inference credentials:

```sh
python3 meta-vul-code/claude-cli-agent/results/glm_validation_20260922_170038/verify.py
```

This checks hashes, all 404 indices/UUIDs/labels per checkpoint, model identity and recorded metrics. It expands the 202 paired dataset rows with the optimizer's loader; passing the paired file directly to the generic metrics CLI does not reproduce CWE/localization metrics.

`run-config.yaml` is the saved configuration snapshot (paths retain their original config-directory meaning). Discovery used concurrency 60, timeout 900 seconds and three retries. `provenance.json` records hashes and the original base commit; `runtime-source.patch.gz` preserves the three-file retry/resume changes used by the run without applying them to this branch. `completion-audit.json` is the original post-run audit.

## Limitations

- Seed and iteration 1 were resumed after an interrupted run; successful samples were not rerun. Seed pairwise accuracy was originally 18.81% and became 19.80% after recovery. Iteration 1 was generated using the original seed feedback, not the repaired seed metrics.
- Across the nine checkpoints, 44 verdicts are unparseable and 129 parseable verdicts have schema warnings. They are retained, not dropped to improve scores. Pairwise denominators remain 202; classification metrics exclude unparseable verdicts.
- Optimizer transcripts recorded permission/approval and other tool errors; counts are preserved in the audit. The audit also records model and thinking evidence. Canonical evaluator logs lack token usage, so this PR makes no token-cost claim and does not include the separate full Claude transcript archive.
- No held-out results, experiment source changes or paper edits are included.
