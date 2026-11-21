# Evaluation results (2026-11-18, `openai/gpt-oss-120b` on Groq free tier)

Raw JSONL output of `python -m evals.evaluate_research`. Each line is one case with its checks,
timing, model turns, tool executions and the full agent trace.

| File | Command | Cases |
|---|---|---|
| `run1_full.jsonl` | `python -m evals.evaluate_research` | all 14 (6 stopped by HTTP 429) |
| `run2_rerun_pause60.jsonl` | `--only R01 R05 R07 R10 R11 R13 --pause 60` | the 6 rate-limited cases |
| `run3_rerun_pause90.jsonl` | `--only R01 R10 --pause 90` | the 2 still rate-limited |
| `manual_rag_before_2021.jsonl` | `python -m research.api.scout "<R02 request>"` (2026-11-19) | one manual run: the model parsed "before 2021" as `< 2020` (see the main README) |

`manual_rag_before_2021.jsonl` is a single-run trace from the CLI (run, tool and summary records), not an evaluator row.

The README's results table uses, for each case, the first run that was not stopped by a rate limit.
