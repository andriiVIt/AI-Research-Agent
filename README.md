# Research Scout

An AI research assistant that searches papers on OpenAlex, applies year and citation constraints, and returns a short, evidence-based recommendation. Example requests:

- "Find a research paper about LLM agents for software engineering published after 2022 with at least 100 citations. Explain why it is relevant and provide the source of the citation count."
- "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
- "Find a recent paper about AI agents using tools and explain whether it would be useful for someone building autonomous software agents."

Topic, year, citation count and operators (`>`, `>=`, `<`, `<=`, `==`) can be combined freely; "recent" means the current year or the two before it.

## Setup

Use Python 3.11 or newer.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Copy `.env.example` for the variable names. Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` for an OpenAI-compatible Chat Completions provider with a tool-calling model, and `OPENALEX_API_KEY` (free at https://openalex.org/settings/api). Keep keys out of the repository.

## Usage

```bash
python -m research.api.scout --demo --log demo_scout.jsonl                # offline, fixture data, no keys
python -m research.api.scout "Find a paper about ..." --log scout.jsonl   # live
python -m unittest -v                                                     # offline tests (mocked model and APIs)
python -m evals.evaluate_research                                         # live eval cases in evals/research_cases.py
```

The CLI is also installed as `research-scout`. Each run prints the tool calls and the answer, and writes a JSONL trace.

## How it works

- Tools: `search_papers` → `get_paper` → (`cross_check_citations`) → `choose_paper` (or `report_no_match`), then the model replies with plain prose.
- The model passes the constraints as flat `search_papers` arguments (`year_op`, `year`, `citations_op`, `citations`, `recent`). Python builds typed `Constraints` from them and turns those into OpenAlex filters (`>= 100` becomes `cited_by_count:>99`).
- `ResearchSession` (`src/research/deps.py`) is the trusted boundary. It enforces a budget of 7 tool calls, 9 model turns and 120 s, keeps only papers that OpenAlex actually returned, removes results that violate a constraint, and refuses later searches that relax the numeric constraints. `choose_paper` rejects a paper that is unknown, unread, or violates a constraint.
- An output validator checks the prose reply. It must have the requested sentence count (default 2-8) and must not just repeat the title. A rejected reply is retried up to twice.
- Python renders the final answer. Title, year, citation count, the OpenAlex API URL and retrieval date (plus Crossref's count when there is a DOI) come from source data. Only the explanation comes from the model.
- If nothing qualifies, the agent returns `no_match` instead of inventing a paper.
- Provider notes: one fresh retry on Groq's intermittent `tool_use_failed`, with the same session and budget. `LLM_MAX_TOKENS` (default 800) stays under Groq's free-tier limit of 1000 output tokens per minute. Long prose is requested as the plain reply, never inside tool-call JSON, because the tested model filled such fields with just the paper title.

## Layout

```
pyproject.toml              # package metadata; `pip install -e .`
requirements.txt            # full pinned dependency set (also used by the Dockerfile)
.env.example                # variable names; keep real keys out of Git
src/research/
  config.py                 # Settings: model, base URL, API keys, limits
  provider.py               # opens the OpenAI-compatible PydanticAI model
  schemas/paper.py          # Comparison, Constraints, Paper, CitationEvidence, Recommendation
  services/openalex.py      # OpenAlex search, filters, abstract reconstruction (no LLM)
  services/crossref.py      # Crossref citation cross-check (no LLM)
  deps.py                   # ResearchSession: budget, trace, candidates, constraint oracle
  tools/search.py           # the agent's tools
  agents/scout.py           # instructions, output validator, run loop, rendering
  fixtures.py               # offline fixture data and scripted (non-LLM) model
  api/scout.py              # CLI
tests/test_research.py      # offline tests
evals/                      # research_cases.py, evaluate_research.py (live)
```

## Documentation

- [Agents](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Function tools](https://pydantic.dev/docs/ai/tools-toolsets/tools/)
- [Testing](https://pydantic.dev/docs/ai/guides/testing/)
- [Usage limits](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits)
- [OpenAlex API](https://docs.openalex.org/)
