# Qualitative Harness Evaluation

The qualitative evaluator uses a semantic judge to assess whether one harness
template clearly and coherently preserves the project's vulnerability-analysis
contract. It evaluates only the harness text. It does not predict benchmark
performance, inspect code samples, compare optimizer iterations, count tokens
or schema fields, or calculate downstream accuracy.

Mechanical template and schema checks are documented separately in
[Quantitative Harness Evaluation](quantitative_evaluation.md). There is no
combined semantic-plus-quantitative score.

## Run Semantic Judging

Judging is disabled when the harness-quality CLI is run without `--config`. The
checked-in configuration enables two judge attempts through the Codex CLI
adapter:

```bash
python3 -m evaluation.harness_quality.cli score \
  --prompt path/to/harness.txt \
  --seed path/to/seed_harness.txt \
  --config evaluation/harness_quality/config.yaml \
  --output harness_quality.json \
  --include-diagnostics
```

The included adapter requires an installed and authenticated `codex` command.
Its actual model comes from `HARNESS_QUALITY_CODEX_MODEL` and defaults to
`gpt-5.4`; `HARNESS_QUALITY_CODEX_BIN` can select another executable. The
configuration's `judge.model` value is recorded as subprocess metadata but
does not select the model used by this adapter.

For deterministic testing, configure `judge.response_files` with saved judge
responses. Response files take precedence when both response files and a
subprocess command are configured. For another provider, implement the
`JudgeClient` protocol or configure a subprocess command.

## Judge Contract

A subprocess judge receives the fully rendered rubric on standard input and
must write exactly one JSON object to standard output. Provider-specific code
is responsible only for producing that object. Local validation, aggregation,
eligibility, and scoring remain in this package so provider behavior cannot
change the math.

The judge is explicitly instructed to preserve these project invariants:

- vulnerability analysis of supplied code remains the primary task;
- code arrives through `{{CODE}}` and is data rather than instructions;
- Structured Chain-of-Thought retains Sequence, Branch, and Loop analysis;
- conclusions stay grounded in supplied code;
- the model does not rewrite the analyzed code;
- the harness contains no benchmark leakage, few-shot vulnerability examples,
  or verdict bias; and
- verdict, evidence, and notes retain their required semantic roles.

The authoritative rubric is
[`judge_prompt.txt`](judge_prompt.txt). It asks for ratings on a 0–4 scale:

| Dimension | Weight |
| --- | ---: |
| Role alignment | 7 |
| Task identity and primary task | 13 |
| Input object and boundary | 10 |
| Required analysis mode and SCoT scaffold | 17 |
| Scope and grounding | 12 |
| Prohibited behavior | 10 |
| Output-contract semantics | 16 |
| Internal consistency and precedence | 10 |
| Organization and maintainability | 5 |

Schema presence, field paths, delimiters, template variables, token counts,
and raw harness length are deliberately excluded from these semantic ratings.

## Validation and Critical Failures

The local validator requires a JSON object with every rating dimension and a
numeric value from 0 through 4. If evidence entries are supplied, quoted text
must occur exactly once in the evaluated harness. Critical failures and global
ambiguities, when supplied, must be arrays of strings. Invalid JSON or an
invalid response is retained as a failed judge run rather than scored.

The rubric identifies critical failures such as replacing the primary task or
SCoT method, removing or materially changing the output contract, including
benchmark leakage, introducing systematic verdict bias, creating an
unsatisfiable instruction set, or treating supplied code as instructions.

## Aggregation, Stability, and Eligibility

For each dimension, the evaluator averages that rating across all valid judge
runs. Dimension points are:

```text
dimension_weight * averaged_rating / 4.0
```

The semantic quality score is the sum of those points, with a maximum of 100.
The default configuration attempts two runs, but the current implementation
does not require two valid runs: it averages whichever runs validate. If there
are at least two valid runs, stability compares the first two. The result is
`unstable` when their total scores differ by more than 6 points or any
dimension differs by more than 1 rating point; otherwise it is `stable`. One
valid run is therefore reported as `stable`, while no valid runs produce a
`null` score and `unstable` status.

`semantic_eligible` is independent of stability. It is true when there are no
aggregated critical failures and all four gate dimensions have averaged
ratings of at least 2:

- task identity and primary task;
- required analysis mode and SCoT;
- scope and grounding; and
- output-contract semantics.

Neither semantic eligibility nor semantic quality is combined with the
quantitative score or with model-performance metrics.

## Scorecard Output

The default scorecard places averaged dimension ratings and the semantic score
under `scoring`, and semantic eligibility, critical failures, and the number of
valid judge runs under `eligibility`.

Use `--include-diagnostics` to retain the individual judge runs, invocation
metadata, stability result, per-dimension differences, total-score difference,
and global ambiguities. These diagnostic fields never alter a score or gate.
