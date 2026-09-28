# Quantitative Harness Evaluation

The quantitative evaluator scores a harness as an intrinsic artifact.
It reads the candidate harness and, when supplied, a seed harness for token-length
comparison. It does not read datasets, code samples, predictions, labels,
downstream performance metrics, `logs.jsonl`, evaluator responses, or optimizer
history.

## Run the Evaluator

Score one harness from the repository root:

```bash
python3 -m evaluation.harness_quality.cli score \
  --prompt path/to/harness.txt \
  --seed path/to/seed_harness.txt \
  --output harness_quality.json
```

Omit `--seed` when no token-efficiency baseline is available. The evaluator
will still report the first three component scores, but `token_efficiency` and
`quantitative_static_score` will be `null`.

Score the immediate `iter_*.txt` children of a harness directory:

```bash
python3 -m evaluation.harness_quality.cli score-session \
  --session-dir path/to/harnesses \
  --seed path/to/harnesses/iter_000.txt
```

`score-session` does not recurse and does not require `history.json` or
`metrics.json`. For each harness, it writes either
`<harness-stem>/harness_quality.json` when the matching stem is a directory, or
`<harness-stem>_harness_quality.json` beside the harness. It also writes
`harness_quality_history.jsonl` in the session directory, including static-score
deltas from the preceding harness and the seed when those scores are available.

Add `--config evaluation/harness_quality/config.yaml` to either command when
semantic judging is also wanted. Quantitative analysis itself does not require
a configuration file.

## Deterministic Measurements

The evaluator records both a raw SHA-256 hash and a normalized SHA-256 hash.
Hash normalization applies NFC Unicode normalization, converts line endings to
LF, removes trailing horizontal whitespace, preserves internal blank lines,
and leaves exactly one final newline.

Reference token counts normally use `tiktoken` with the configured
`cl100k_base` encoding and identify the installed package version in the
diagnostics. They are reproducible comparison measurements, not claims about a
provider's billing tokens. If `tiktoken` or the configured encoding cannot be
loaded, the current implementation uses a regex-based fallback and labels the
tokenizer `fallback:<encoding>`; compare token counts only when the recorded
tokenizer identifiers match.

The remaining measurements cover template variables, schema structure,
headings and directive lines, whitespace and line endings, Markdown fences,
control characters, and other format diagnostics. Undefined acronyms are a
lexical heuristic: candidates are reported in diagnostics but never contribute
points or an eligibility gate.

## Score Composition

The quantitative score is the sum of four independent components, for a
maximum of 100 points:

| Component | Maximum | Current point schedule |
| --- | ---: | --- |
| Template integrity | 40 | 4 for nonempty valid UTF-8; 18 for exactly one `{{CODE}}`; 10 for no unknown or malformed template variables; 4 for no disallowed control characters; 4 base points |
| Schema mechanics | 30 | 5 for finding a schema block; 5 for balanced delimiters; up to 15 in proportion to required leaf paths present; 5 for no duplicate paths and balanced delimiters |
| Format hygiene | 15 | 6 for no duplicate normalized directive lines; 3 for no duplicate normalized headings; 3 for clean trailing whitespace, tabs, and line endings; 3 for balanced Markdown fences |
| Token efficiency | 15 | `15 * efficiency`, using the seed comparison below |

The required schema leaf paths are:

```text
verdict.vulnerable
verdict.confidence
verdict.cwe
verdict.severity
verdict.summary
evidence[].file
evidence[].function
evidence[].location.start_line
evidence[].location.end_line
evidence[].snippet
evidence[].reason
notes.assumptions
notes.limitations
```

Schema mechanics measure presence, paths, nesting, duplicates, and delimiter
balance with a permissive JSON-like scanner. Strict JSON validity is recorded
as a diagnostic, but strict JSON is not itself a scoring or eligibility
requirement. Semantic clarity and completeness of the output contract belong
to the [qualitative evaluation](qualitative_evaluation.md).

Token efficiency is:

```text
1.0                          when candidate_tokens <= seed_tokens
seed_tokens / candidate_tokens otherwise
```

Consequently, the total score is `null` without a seed even though the other
component points are still available. The quantitative score is never combined
with the semantic score or downstream model-performance metrics.

## Static Eligibility

`static_eligible` is separate from the numeric score. It currently requires:

- a nonempty, valid UTF-8 harness;
- exactly one `{{CODE}}` placeholder;
- no unknown or malformed template variables;
- a discovered schema block with balanced delimiters; and
- every required schema leaf path.

One implementation detail is easy to misread in a scorecard: disallowed
control characters reduce template-integrity points and may appear in
`static_eligibility_failures`, but they do not currently flip
`static_eligible` by themselves. Consumers that need a hard control-character
gate should inspect that failure list or the diagnostics explicitly.

## Scorecard Output

By default, a version 2.0 scorecard contains:

- `prompt.path`;
- `scoring`, including the four components, selected format findings,
  reference-token counts, and the total quantitative score; and
- `eligibility`, including static gate results and any semantic results if a
  judge was configured.

Use `--include-diagnostics` to add raw and normalized harness hashes, all static
measurements, parser details, findings, judge runs, and semantic stability:

```bash
python3 -m evaluation.harness_quality.cli score \
  --prompt path/to/harness.txt \
  --seed path/to/seed_harness.txt \
  --output harness_quality.json \
  --include-diagnostics
```

Diagnostics explain a result but never change its score or eligibility gate.
