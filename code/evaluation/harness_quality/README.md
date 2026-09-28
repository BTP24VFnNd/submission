# Harness-quality Evaluation

This package evaluates harnesses as intrinsic artifacts. Quantitative
mechanics and qualitative semantic judging remain separate from each other and
from downstream model performance.

Choose the guide that matches the task:

- [Quantitative harness evaluation](quantitative_evaluation.md) covers
  deterministic template, schema, format, token, eligibility, and scorecard
  behavior.
- [Qualitative harness evaluation](qualitative_evaluation.md) covers the
  semantic rubric, judge adapters, validation, aggregation, and eligibility.

## Quick Start

Run quantitative-only scoring by omitting `--config`:

```bash
python3 -m evaluation.harness_quality.cli score \
  --prompt path/to/harness.txt \
  --seed path/to/seed_harness.txt \
  --output harness_quality.json
```

Add the checked-in configuration for semantic judging as well:

```bash
python3 -m evaluation.harness_quality.cli score \
  --prompt path/to/harness.txt \
  --seed path/to/seed_harness.txt \
  --config evaluation/harness_quality/config.yaml \
  --output harness_quality.json
```

For an immediate directory of `iter_*.txt` harnesses without optimizer history
or metrics:

```bash
python3 -m evaluation.harness_quality.cli score-session \
  --session-dir path/to/harnesses \
  --seed path/to/harnesses/iter_000.txt
```
