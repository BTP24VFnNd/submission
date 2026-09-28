# Pilot study: choosing the seed harness

This is a preliminary pilot study to determine the most effective seed harness for the discovery sessions, that is, which hand-written reasoning strategy makes a good baseline to start optimizing from.

## Strategies

All four harnesses share the same role, rules and JSON output schema. They differ only in how the model is told to reason before giving its verdict.

- **Vanilla**: no reasoning scaffold. The model analyzes the code and outputs the verdict directly.
- **SCoT** (Structured Chain of Thought): the model reasons in numbered steps organized by program structure (sequence, branch, loop), following the actual control flow and focusing on security-relevant behavior such as bounds, lifetimes, input validation and integer safety.
- **MoT** (Modularized Chain of Thought): the model decomposes the function hierarchically into high-level security concerns, intermediate sub-steps and detailed operations (H/I/D levels), and analyzes each module.
- **ToT** (Tree of Thoughts): the model proposes up to 3 vulnerability hypotheses, expands the most promising one with evidence for and against, prunes the weak ones and concludes from the best-supported path.

## Setup

- Models: Opus 4.6 through Claude Code (`claude -p`); Qwen3-235B-A22B-Instruct-2507 and Kimi-K2-Thinking at temperature 0
- Data: over 700 C/C++ functions
- Each strategy is a standalone detector, with no optimization
- For the two open models, verdicts that failed to parse were rerun with the same harness, so every function has a verdict
- Scored with `code/evaluation/metrics.py`. A missing verdict counts as wrong.

## Results

### Opus 4.6

| Metric | Vanilla | SCoT | MoT | ToT |
|---|---:|---:|---:|---:|
| Pairwise (%) | **11.26** | 10.57 | 10.11 | 8.05 |
| Accuracy | 0.537 | **0.543** | 0.536 | 0.531 |
| Precision | **0.527** | 0.525 | 0.521 | 0.517 |
| Recall | 0.720 | 0.910 | 0.901 | **0.940** |
| F1 | 0.608 | 0.666 | 0.660 | **0.667** |
| CWE accuracy (%) | 25.88 | 29.04 | **29.59** | 28.85 |
| Loc F1 (%) | 23.74 | 24.06 | 24.19 | **25.23** |
| Loc hit rate (%) | 44.73 | 45.96 | 46.43 | **49.88** |

### Qwen3-235B-A22B-Instruct-2507

| Metric | Vanilla | SCoT | MoT | ToT |
|---|---:|---:|---:|---:|
| Pairwise (%) | 5.06 | **14.02** | 8.97 | 8.74 |
| Accuracy | 0.508 | **0.532** | 0.520 | 0.516 |
| Precision | 0.509 | **0.521** | 0.511 | 0.509 |
| Recall | 0.448 | 0.809 | 0.881 | **0.920** |
| F1 | 0.477 | 0.634 | 0.647 | **0.655** |
| CWE accuracy (%) | **32.31** | 20.45 | 26.37 | 17.00 |
| Loc F1 (%) | 20.57 | **20.65** | 19.36 | 19.10 |
| Loc hit rate (%) | 43.59 | 38.64 | **44.65** | 34.50 |

### Kimi-K2-Thinking

| Metric | Vanilla | SCoT | MoT | ToT |
|---|---:|---:|---:|---:|
| Pairwise (%) | 2.07 | **11.03** | 2.53 | **11.03** |
| Accuracy | 0.502 | **0.528** | 0.507 | 0.523 |
| Precision | 0.501 | **0.516** | 0.504 | 0.513 |
| Recall | 0.975 | 0.906 | **0.982** | 0.922 |
| F1 | 0.662 | 0.657 | **0.666** | 0.659 |
| CWE accuracy (%) | **27.36** | 19.29 | 27.17 | 20.45 |
| Loc F1 (%) | 20.03 | 20.90 | **21.21** | 21.05 |
| Loc hit rate (%) | 49.53 | 43.65 | **50.12** | 40.15 |

## Findings

- All strategies are weak at telling a vulnerable function from its patch: pairwise accuracy ranges from 2% to 14%, and every model leans toward calling functions vulnerable (recall well above precision).
- SCoT has the best or tied-best pairwise accuracy on both open models (14.02 on Qwen3-235B, well ahead of the next best at 8.97; 11.03 on Kimi-K2-Thinking, tied with ToT), and is within one point of the best on Opus 4.6.
- SCoT also has the best accuracy and precision on all three models except Opus 4.6 precision, where it is 0.002 behind Vanilla.
- The other strategies are less stable across models. Vanilla leads pairwise on Opus 4.6 but is near the bottom on both open models; MoT drops to 2.53 on Kimi-K2-Thinking; ToT is last on Opus 4.6.
- High recall alone does not help: the highest-recall strategy on each model (ToT on Opus 4.6 and Qwen3-235B, MoT on Kimi-K2-Thinking) flags most patched functions too, which keeps its pairwise accuracy low.

## Conclusion

SCoT is the only strategy that is at or near the top in pairwise accuracy on every model tested, and it has the best accuracy on all three, so it is the most reliable baseline. It is also a single linear pass over the code's control flow, which gives the optimizer a clear structure to edit, while ToT's branching hypotheses and MoT's three-level decomposition are longer and harder to revise step by step. We therefore chose SCoT as the seed for every session in this repository. The Opus 4.6 SCoT column above is the Opus 4.6 seed baseline used in the paper (pairwise 10.57).
