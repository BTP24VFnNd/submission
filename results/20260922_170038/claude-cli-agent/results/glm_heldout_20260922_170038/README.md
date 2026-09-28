# GLM-5.3 held-out: iterations 0, 1, 2, 4, 6 and 8

Session `20260922_170038`, Claude CLI / OpenRouter GLM-5.3, PrimeVul test (870 functions / 435 pairs).

| Iteration | Pairwise accuracy | Correct pairs | Localization F1 | Unparseable | Final request errors |
|---:|---:|---:|---:|---:|---:|
| 0 (seed) | 18.16% | 79/435 | 22.95 | 30 | 0 |
| 1 | 25.52% | 111/435 | 26.97 | 4 | 3 |
| 2 | 22.53% | 98/435 | 25.10 | 6 | 4 |
| 4 (validation-selected) | 21.61% | 94/435 | 23.95 | 9 | 7 |
| 6 (validation Pareto frontier) | 24.37% | 106/435 | 24.73 | 16 | 7 |
| 8 | 22.76% | 99/435 | 27.67 | 2 | 1 |

Iteration 4 is the validation-selected `best_prompt.txt`; iteration 6 is the validation Pareto prompt. Held-out scores do not change prompt selection. Iterations 3, 5 and 7 have no results in this PR.

## Files and scoring

`summary.json` contains full metrics, artifact hashes, completion times and recovery notes. Raw predictions, losslessly compressed logs, and exact prompts are under `../../outputs/held_out_primevul_20260922_170038_iter<N>_z-ai/glm-5.3/held-out/`.

From the repository root:

```sh
python3 meta-vul-code/claude-cli-agent/results/glm_heldout_20260922_170038/verify.py
```

This verifies the raw files and reproduces all metrics using the current scorer. Iterations 1, 2, 4 and 8 contain retry records: deduplicate by sample index, preferring a non-null verdict and the latest valid record, before scoring. Do not score the raw appended file as independent samples. The model-name slash creates a nested output directory; use the explicit paths in `summary.json`, not a shallow glob.

## Recovery and limitations

- Iteration 8 has 952 raw prediction records covering all 870 samples, with 868 valid verdicts and two nulls. Logs retain 80 historical request errors; one remains in the latest per-sample attempts. After a credit failure at concurrency 60, it resumed at 30, skipping valid records and retrying absent/null records. All attempts are preserved.
- Pairwise scoring retains all 435 pairs. Classification metrics use parseable verdicts; localization uses true-positive vulnerable samples under the repository scorer.
- These are full-dataset scores, before the paper's exclusion of five overlapping pairs (430-pair scoring).
- Iteration 2 has 1,039 raw prediction records covering all 870 samples, with 864 valid verdicts and six nulls. Logs retain 160 historical request errors; four remain in the latest per-sample attempts. Resumes skipped valid records and retried absent/null records, using concurrency 60, 30, 15, 8, 2 and 1 before finishing at 60. All attempts are preserved.
- Iteration 1 has 993 raw prediction records. Resumes skipped valid verdicts and retried absent/null records. Logs retain 108 historical request errors; three remain in the latest per-sample attempts.
- Iteration 4 has 898 raw records. Nine null verdicts were retained after bounded recovery: five output-cap failures, two timeout failures and two format failures. Logs retain 11 historical request errors.
- Iteration 6 retained 869 saved records, including 16 nulls, then evaluated only missing index 802. Its seven request failures remain in the results. Seed has 30 unparseable verdicts and no request failures.
- The saved config uses concurrency 60. Iteration 1 resumes reduced it to 8 and then 2; iteration 6's missing-only recovery used one sample. Model, prompt, timeout (900 seconds), retries (3) and output cap (32,000 tokens) were not changed. The config's `max_tokens: 4096` is not the effective Claude CLI cap.
- These are completed checkpoints with retained failures, not error-free runs. Canonical token-usage fields are incomplete; no token-cost claims are made.

## Work split

The run completed **iterations 2 and 8** and stopped. Iterations **3, 5 and 7** remain outside this PR and can be evaluated independently using saved validation prompts and matching settings. The prompts and validation history are in `../glm_validation_20260922_170038/` and `../../../meta-vul/outputs/20260922_170038/`. Do not run new optimization or select prompts from held-out scores.

`run-config.yaml` is the original held-out config snapshot; relative paths retain their original config-directory meaning. No experiment code or paper changes are included.
