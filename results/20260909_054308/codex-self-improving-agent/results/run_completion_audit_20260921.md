# Run completion audit

## Codex CLI

| Run | Status |
|---|---|
| GPT-5.4-mini validation (`20260726_002902`) | Complete: seed + 5 iterations, 404 samples each, 0 parse failures |
| GPT-5.4-mini PrimeVul | Complete for selected iteration 1: 870 predictions |
| Inkling-Small validation (`20260909_054308`) | Complete: iterations 0-15 |
| Inkling-Small PrimeVul | Complete: iterations 0-15 |
| Nemotron cross-transfer | Complete: iterations 0-15 |
| GLM-5.3 | Smoke only; optimizer failed the schema check twice |

## Claude CLI

No full benchmark was launched from this setup. The saved OpenRouter run is a smoke test.

## Nemotron files

- Raw predictions added for iterations 0-11 and 13-15.
- Each file contains 870 unique boolean predictions and matches the committed summary.
- Iteration 12 has a committed compact result, but its raw folder is unavailable.
- Raw logs were excluded (904 MB).
