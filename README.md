# Research Scout

An AI research assistant that searches papers on OpenAlex, applies year and citation constraints, and returns a short, evidence-based recommendation. Example requests:

- "Find a research paper about LLM agents for software engineering published after 2022 with at least 100 citations. Explain why it is relevant and provide the source of the citation count."
- "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
- "Find a recent paper about AI agents using tools and explain whether it would be useful for someone building autonomous software agents."

Topic, year, citation count and operators (`>`, `>=`, `<`, `<=`, `==`) can be combined freely; "recent" means the current year or the two before it.

## Setup

Use Python 3.11 or newer (the evaluation below was run with Python 3.14).

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### API keys

Copy the template to a local `.env` file (it is git-ignored) and fill in your keys:

```bash
cp .env.example .env            # macOS/Linux
Copy-Item .env.example .env     # Windows PowerShell
```

Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` for an OpenAI-compatible Chat Completions provider with a tool-calling model, and `OPENALEX_API_KEY` (free at https://openalex.org/settings/api). The `.env` file in the working directory is loaded automatically; variables already set in your shell take precedence. Keep `.env.example` as the committed template and never commit `.env`.

## Usage

```bash
python -m research.api.scout --demo --log demo_scout.jsonl                # offline, fixture data, no keys
python -m research.api.scout "Find a paper about ..." --log scout.jsonl   # live
python -m unittest -v                                                     # offline tests (mocked model and APIs)
python -m evals.evaluate_research --pause 60                              # live eval cases in evals/research_cases.py
```

Example of a live answer (abridged):

```
[answered]
Recommended paper: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)
Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin | Venue: arXiv (Cornell University)
Link: https://doi.org/10.48550/arxiv.2005.11401
Citations: 10277 according to OpenAlex (https://api.openalex.org/works/W3027879771, retrieved 2026-11-18)
Constraints verified against source data: year < 2021 -> 2020 OK; citations > 500 -> 10277 OK

Summary:
The paper introduces Retrieval-Augmented Generation (RAG), a framework that couples a pre-trained sequence-to-sequence language model with a dense vector index of Wikipedia ...

Uncertainty: The explanation is based on the title, abstract and metadata, not the full text. Citation counts change over time and differ between indexes.
```

The CLI is also installed as `research-scout`. Each run prints the tool calls and the answer, and writes a JSONL trace.

Docker (optional):

```bash
docker build -t research-scout .
docker run --rm research-scout                                      # offline demo
docker run --rm --env-file .env research-scout "Find a paper about ..." --log /tmp/run.jsonl
```

## Agent workflow

Each step is done either by the LLM or by deterministic Python code:

| # | Step | Done by | Where |
|---|---|---|---|
| 1 | Parse the request: extract the topic, the year constraint (`>`, `>=`, `<`, `<=`, `==`, or "recent"), the citation constraint, and what kind of answer is wanted | **LLM** (as `search_papers` tool arguments) | `agents/scout.py` instructions |
| 2 | Validate the constraints into a typed `Constraints` model; read the requested sentence count from the request text | **Code** | `schemas/paper.py` |
| 3 | Search OpenAlex with the constraints as API filters | **Code** (tool) | `services/openalex.py` |
| 4 | Filter the candidates again against the constraints; drop any result that violates them | **Code** | `deps.py` `check_paper` |
| 5 | Choose the most relevant candidate and read its abstract (`get_paper`); optionally cross-check citations with Crossref | **LLM** chooses, **code** fetches | `tools/search.py`, `services/crossref.py` |
| 6 | Search again with other topic wording if nothing fits; the year and citation constraints cannot change | **LLM** decides, **code** enforces | `deps.py` `search` |
| 7 | Select the paper (`choose_paper`) or declare `report_no_match` | **LLM** proposes, **code** verifies (paper was returned and read, constraints hold) | `deps.py` `choose` |
| 8 | Write the explanation: relevance, summary, or usefulness verdict | **LLM** | plain-text reply |
| 9 | Validate the reply (sentence count, not just the title); ask the model to retry if it fails | **Code** | `agents/scout.py` `verify_answer` |
| 10 | Render the final answer: title, authors, year, citations, citation source, link, checked constraints, uncertainty notes | **Code** | `agents/scout.py` `render` |

**Structured output.** The constraints (`Constraints`, `Comparison`), the paper metadata (`Paper`), the citation evidence (`CitationEvidence`) and the final answer (`Recommendation` or `NoMatch`) are all Pydantic models (`src/research/schemas/paper.py`). The final `Recommendation` is assembled by Python from the verified `choose_paper` call and the model's prose. We do not ask the model to fill one large JSON object, because the tested model shortened long text inside tool-call or JSON fields (for example, it wrote only the paper title instead of a summary). Each run also writes a machine-readable JSONL trace with a `summary` record.

## How it works

- Tools: `search_papers` → `get_paper` → (`cross_check_citations`) → `choose_paper` (or `report_no_match`), then the model replies with plain prose.
- The model passes the constraints as flat `search_papers` arguments (`year_op`, `year`, `citations_op`, `citations`, `recent`). Python builds typed `Constraints` from them and turns those into OpenAlex filters (`>= 100` becomes `cited_by_count:>99`).
- `ResearchSession` (`src/research/deps.py`) is the trusted boundary. It enforces a budget of 7 tool calls, 9 model turns and 120 s, keeps only papers that OpenAlex actually returned, removes results that violate a constraint, and refuses later searches that relax the numeric constraints. `choose_paper` rejects a paper that is unknown, unread, or violates a constraint.
- An output validator checks the prose reply. It must have the requested sentence count (default 2-8) and must not just repeat the title. A rejected reply is retried up to twice.
- Python renders the final answer. Title, year, citation count, the OpenAlex API URL and retrieval date (plus Crossref's count when there is a DOI) come from source data. Only the explanation comes from the model.
- If nothing qualifies, the agent returns `no_match` instead of inventing a paper.
- Provider notes: one fresh retry on Groq's intermittent `tool_use_failed`, with the same session and budget. `LLM_MAX_TOKENS` (default 800) caps each reply, which keeps runs within the free tier's tokens-per-minute limit (see [LLM choice](#llm-choice)). Long prose is requested as the plain reply, never inside tool-call JSON, because the tested model filled such fields with just the paper title.

## LLM choice

**Provider: [Groq](https://console.groq.com) (free tier). Model: `openai/gpt-oss-120b`.**

Why Groq:

- It has a free tier with no credit card, which matters for a student project that runs a 14-case evaluation several times.
- It exposes an OpenAI-compatible Chat Completions API, so PydanticAI's `OpenAIChatModel` works through `LLM_BASE_URL` alone (`src/research/provider.py`). Switching to another OpenAI-compatible provider only changes `.env`.
- Inference is fast, which keeps multi-turn agent runs and repeated evaluations practical.

Why `openai/gpt-oss-120b`:

- **Tool use is required.** The agent cannot work without tool calling. Groq's model page lists *Tool Use*, *JSON Object Mode* and *JSON Schema Mode* for this model.
- **It is the strongest model available to our key.** We listed the models our key can access through `GET /openai/v1/models`. The candidates with tool support were `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`. We picked the largest one, because the agent has to follow a multi-step tool protocol and map wording like "at least" and "more than" to the correct operators.
- **Its context window is large enough.** The model has a 131,072-token context, and our runs use a small fraction of it: search results are trimmed to 300-character abstract snippets and full abstracts to 1,500 characters.

Alternatives we tried or ruled out:

| Model | Result |
|---|---|
| `llama-3.3-70b-versatile` | It was the original default and is listed in Groq's docs, but our key received `404 model_not_found`. We could not use it. |
| `openai/gpt-oss-20b` | Has the same rate limits and is cheaper, but it is a smaller model with weaker instruction-following. We kept it as a fallback. |
| `qwen/qwen3.8-27b` | It is a preview model, and Groq says preview models "may be discontinued at short notice". We also observed intermittent `tool_use_failed` errors with Qwen on Groq for nested tool arguments, which is why `search_papers` takes flat scalar arguments. |

Free-tier limits for `openai/gpt-oss-120b` (from https://console.groq.com/docs/rate-limits, checked 2026-11-18; check your own account's limits page, as they change):

| Requests/min | Requests/day | Tokens/min | Tokens/day |
|---|---|---|---|
| 30 | 1,000 | 8,000 | 200,000 |

These limits shaped the design:

- Every run is capped at 9 model turns and 7 tool calls (`src/research/agents/scout.py`), so one request can never use more than a handful of the per-minute requests.
- `LLM_MAX_TOKENS` (default 800) caps each reply.
- The evaluation runs its cases one after another. A case that fails with HTTP 429 (rate limit) can be rerun with `--only R05 R06`.

To use a different model or provider, change `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env`. No code changes are needed, as long as the provider is OpenAI-compatible and the model supports tool calling.

## Evaluation

### How AI agents are evaluated, and what we took from it

- **Score the outcome against ground truth that the agent cannot influence.** Agent benchmarks such as AgentBench (Liu et al., 2023, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)) and τ-bench (Yao et al., 2024, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) judge an agent by whether its final state or answer is correct, checked programmatically, rather than by how convincing its text sounds.
  - *Our version:* every case in `evals/research_cases.py` declares its expected constraints independently of the agent. The grader re-checks the chosen paper's year and citation count against the metadata Python fetched from OpenAlex. It never trusts the constraints the model chose to search with, or the model's own text.
- **Look at the trajectory, not only the answer.** The survey by Yehudai et al. (2025, [arXiv:2503.16416](https://arxiv.org/abs/2503.16416)) treats tool use and planning as core agent capabilities to evaluate, alongside final-task success.
  - *Our version:* each run writes a JSONL trace of every tool call, its arguments and whether Python accepted it. We report model turns and tool executions per case, and we inspect the traces for failures.
- **Check factual claims one by one against a source.** FActScore (Min et al., 2023, [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)) measures hallucination by splitting generated text into atomic facts and verifying each against a knowledge source.
  - *Our version:* the metadata (title, authors, year, citations, link) is rendered by Python from source data, so the model cannot state it. Only the prose explanation comes from the model, and we check its claims manually against the paper's abstract.
- **Be careful with LLM-as-a-judge.** Zheng et al. (2023, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) show that LLM judges have position, verbosity and self-enhancement biases.
  - *Our version:* everything that can be checked deterministically is checked by code. Only the subjective criteria (relevance and usefulness of the explanation) are graded manually, not by another LLM.
- **Reliability over repeated trials.** τ-bench introduces pass^k, because agents that succeed once often fail on reruns. This is noted as a limitation: with the free-tier rate limits, we ran each case once, not k times.

### Test set

The test set has 14 prompts in `evals/research_cases.py`. They cover four categories and six constraint types:

| ID | Category | Constraints | Expected |
|---|---|---|---|
| R01 | narrow | year > 2022, citations ≥ 100 | answered (relevance) |
| R02 | narrow | year < 2021, citations > 500 | answered (summary, 5-7 sentences) |
| R03 | broad | recent | answered (usefulness verdict) |
| R04 | narrow | year ≥ 2021, citations ≥ 200 | answered (relevance) |
| R05 | broad | year == 2018 | answered (summary, exactly 3 sentences) |
| R06 | narrow | citations < 50 | answered (relevance) |
| R07 | narrow | year < 2023 | answered (relevance) |
| R08 | no match | year > 2025, citations > 100000 | no_match |
| R09 | ambiguous | none ("something good about AI agents") | answered (relevance) |
| R10 | ambiguous | none ("a well-known paper on transformers") | answered (relevance) |
| R11 | narrow | year > 2020, citations ≥ 1000 | answered (relevance) |
| R12 | narrow | year == 2017 | answered (relevance) |
| R13 | no match | year < 2015, citations ≥ 500 (jailbreaking ChatGPT) | no_match |
| R14 | broad | year > 2020, citations ≤ 20 | answered (relevance) |

### Grading criteria

| Assignment criterion | How it is checked | Automatic? |
|---|---|---|
| Found a relevant paper | `status` equals the expected status (`answered` or `no_match`), then relevance of the paper to the topic is checked manually | partly |
| Respected the year constraint | the case's declared year comparison holds for the paper's OpenAlex year | yes |
| Respected the citation constraint | the case's declared citation comparison holds for the OpenAlex `cited_by_count` | yes |
| Provided a valid source | the citation evidence has an OpenAlex API URL, and its count equals the source metadata | yes |
| Avoided hallucinated information | metadata is rendered from source data by design; the explanation's claims are compared manually with the abstract | partly |
| Gave a useful explanation | answer kind matches the request (relevance, summary or usefulness with a verdict), the sentence count matches when the request gives one, and usefulness is judged manually | partly |

A case passes the automatic grade only if every applicable automatic check is true.

Run it with:

```bash
python -m evals.evaluate_research --pause 60                  # all cases; pause avoids free-tier HTTP 429
python -m evals.evaluate_research --only R01 R05 --pause 60   # selected cases
```

### Results

Model: `openai/gpt-oss-120b` on Groq's free tier. Run on 2026-11-18. Each case was run once.

**Infrastructure note.** In the first full run, 6 of 14 cases (R01, R05, R07, R10, R11, R13) stopped with `ModelRateLimitError` (HTTP 429), because the cases ran back-to-back and exhausted the free tier's tokens-per-minute limit. These failures came from the provider, not from the agent. We added `--pause` to the evaluator and reran only those cases: four passed with `--pause 60`, and R01 and R10 hit 429 again and passed on a third run with `--pause 90`. The table shows the first run that was not stopped by a rate limit. The other eight cases are from the first full run.

#### Automatic checks

**12/14 cases pass every automatic check.** Counting the rate-limited first run as-is, the score was 6/14.

| ID | Status | Year | Citations | Valid source | Answer kind | Sentences | Model turns | Tool calls | Paper chosen |
|---|---|---|---|---|---|---|---|---|---|
| R01 | answered ✅ | ✅ | ✅ | ✅ | ✅ | – | 5 | 4 | Demystifying LLM-Based Software Engineering Agents (2025, 1,052 cit.) |
| R02 | answered ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 4 | 3 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020, 10,277) |
| R03 | answered ✅ | ✅ | – | ✅ | ✅ (useful) | – | 4 | 3 | AgentTrust: Runtime Safety Evaluation and Interception for AI Agent Tool Use (2026, 9) |
| R04 | answered ✅ | ✅ | ✅ | ✅ | ❌ summary | – | 4 | 3 | MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation (2023, 622) |
| R05 | answered ✅ | ✅ | – | ✅ | ✅ | ✅ | 4 | 3 | Attention-based Graph Neural Network for Semi-supervised Learning (2018, 492) |
| R06 | answered ✅ | – | ✅ | ✅ | ✅ | – | 4 | 3 | A Systematic Literature Review of Prompt Injection Attacks in LLM-Integrated Systems (2026, 7) |
| R07 | answered ✅ | ✅ | – | ✅ | ✅ | – | 4 | 3 | Temporal Fusion Transformers for interpretable multi-horizon time series forecasting (2021, 5,049) |
| R08 | no_match ✅ | – | – | – | – | – | 4 | 3 | (none, correctly) |
| R09 | answered ✅ | – | – | ✅ | ❌ summary | – | 4 | 3 | Empowering biomedical discovery with AI agents (2024, 558) |
| R10 | answered ✅ | – | – | ✅ | ✅ | – | 6 | 4 | Attention Is All You Need (2017, 151,508) |
| R11 | answered ✅ | ✅ | ✅ | ✅ | ✅ | – | 4 | 3 | LoRA: Low-Rank Adaptation of Large Language Models (2021, 22,790) |
| R12 | answered ✅ | ✅ | – | ✅ | ✅ | – | 4 | 3 | Federated Multi-Task Learning (2017, 1,526) |
| R13 | no_match ✅ | – | – | – | – | – | 3 | 2 | (none, correctly) |
| R14 | answered ✅ | ✅ | ✅ | ✅ | ✅ | – | 4 | 3 | Machine Learning (2024, 10) |

"–" means the criterion does not apply to that case. Citation counts are OpenAlex values on the run date.

#### Manual checks

For every answered case, we compared each claim in the model's explanation with the abstract that `get_paper` returned, and judged whether the paper fits the request.

| ID | Relevant paper? | Hallucination check (explanation vs. abstract) | Useful explanation? |
|---|---|---|---|
| R01 | ✅ | ⚠️ minor: "the highest performance on SWE-bench Lite" drops the abstract's qualifier "compared with all existing open-source software agents at the time of submission" | ✅ names OpenAlex as the citation source, as asked |
| R02 | ✅ the original RAG paper | ✅ faithful | ✅ |
| R03 | ✅ | ⚠️ two distortions: "all driven by an LLM-as-Judge" (the abstract uses it only for ambiguous inputs) and "over 95 % verdict accuracy" (the reported values are 95.0 % and 96.7 %) | ✅ the verdict is justified |
| R04 | ✅ | ⚠️ minor: "reproducible" is not in the abstract | ⚠️ gives a summary instead of the requested relevance explanation |
| R05 | ✅ | ⚠️ minor: merges the paper's linear-model finding with the proposed attention model | ✅ exactly 3 sentences |
| R06 | ✅ | ✅ faithful | ✅ |
| R07 | ✅ | ✅ faithful | ✅ |
| R09 | ⚠️ acceptable but narrow (biomedical agents) for a vague request | ✅ faithful | ⚠️ summary instead of relevance |
| R10 | ✅ an ideal pick for an ambiguous request | ⚠️ minor: "computer vision and many other domains" comes from the model's own knowledge, not the abstract (it is true, but not grounded) | ✅ |
| R11 | ✅ | ✅ faithful | ✅ |
| R12 | ⚠️ partial: about federated learning on devices in general, not mobile devices specifically | ⚠️ "includes mobile phones and tablets" is not in the abstract | ⚠️ |
| R14 | ⚠️ meets the constraints, but it is a generic overview in a low-profile journal | ✅ faithful | ✅ for such a broad request |

#### Summary

| Metric | Result |
|---|---|
| Correct status (answered or no_match) | 14/14 |
| Year constraint respected (answered cases with a year or "recent" constraint) | 9/9 |
| Citation constraint respected (answered cases with a citation constraint) | 6/6 |
| Valid, verifiable citation source | 12/12 |
| Invented papers, titles, years, citation counts or links | **0** |
| Explanations fully faithful to the abstract | 6/12 (4 minor issues, 2 notable distortions) |
| Clearly relevant paper (manual) | 9/12 (3 partial: R09, R12, R14) |
| Correct answer kind | 10/12 |
| Average model turns / tool calls per case | 4.1 / 3.1 (no case answered without tools) |

The raw results of these runs, with full agent traces, are in [`evals/results/`](evals/results/). New runs write to `research_eval_results.jsonl` in the working directory, which is git-ignored.

#### Failure found in manual testing (outside the evaluation set)

On 2026-11-19 we ran the R02 request by hand again: *"Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."* The full trace is in [`evals/results/manual_rag_before_2021.jsonl`](evals/results/manual_rag_before_2021.jsonl).

| Turn | What the model did | What Python did |
|---|---|---|
| 1 | Its own reasoning said "before 2021", but it called `search_papers` with `year_op "<", year 2020` | Accepted the search. These became the **locked** constraints for the run. The OpenAlex filter was `publication_year:<2020`, which excludes the original RAG paper (2020) |
| 2 | Said the results looked irrelevant and searched again with other topic wording, still `< 2020` | Accepted (same constraints) |
| 3 | Tried to correct itself to `< 2021` | **Rejected**: the year constraint must stay `< 2020` |
| 4 | Tried `<= 2020`, which means the same as `< 2021` for whole years | **Rejected**: the lock compares operator and value literally, not by meaning |
| 5 | Gave up and searched with `< 2020` again | Accepted. Groq then stopped the run with HTTP 429 (`ModelRateLimitError`) |

Two problems combined here:

1. **The model misparsed the year.** The system prompt literally contains the example `"before 2021" -> "<"`, and the model still wrote 2020. The same request was parsed correctly in R02 and in our first live test, so this happens only sometimes. One run per case did not reveal it.
2. **Our safeguard locked in the mistake.** Rejecting changes to the numeric constraints is meant to stop the model from quietly relaxing the request (for example, from "≥ 1000 citations" down to "≥ 100"). But Python locks the constraints from the model's *first* search and never checks them against the request text. So when the first search was wrong, the lock blocked both correct fixes.

In this run, the error made the search **too narrow**: it removed every 2020 paper, including the original RAG paper, which the user would most likely want. Any paper the agent returned would still have satisfied "before 2021". The same mechanism is more dangerous in the other direction. Had the model written `< 2022`, a 2021 paper would have passed every Python check, because Python checks against the model's constraints, and only the evaluation's independent oracle would have caught it. We decided not to change the agent this close to the deadline, so the evaluation above still describes the tested code. The fix we would make is listed under *What we would improve*.

## Reflection

**What worked well**

- **Python owns every fact, so metadata never contained hallucinations.** The model never states a title, year, citation count or link. Python renders them from OpenAlex data and re-checks every constraint before the answer is shown. Across 14 cases there were 0 invented papers or numbers, and 0 constraint violations among the answered cases.
- **Constraint parsing was mostly reliable.** In all 14 evaluation cases, the model mapped "after", "before", "in", "at least", "more than", "fewer than", "at most" and "recent" to the correct operators and values. The OpenAlex filter does the numeric filtering, so the LLM never has to compare numbers itself. A later manual run showed this is not guaranteed (see below).
- **"No match" works.** Both impossible requests (R08, R13) ended in `no_match` instead of a near-miss paper. R13's explanation even gives the right reason: ChatGPT did not exist before 2015.
- **Ambiguous requests were handled sensibly.** For "a well-known paper on transformers" (R10), the agent searched twice and picked *Attention Is All You Need*.

**What failed or was unreliable**

- **A misparsed year that our own safeguard then locked in.** This is the most instructive failure we saw. In a manual run, the model turned "before 2021" into `< 2020`, and Python's rule that the numeric constraints may not change between searches rejected both of the model's correct attempts to fix it. A safeguard that trusts the model's first output only enforces *consistency*, not *correctness*. Details are in [Failure found in manual testing](#failure-found-in-manual-testing-outside-the-evaluation-set).
- **Free-tier rate limits were the biggest practical problem.** 6 of 14 cases failed with HTTP 429 in the first run. Even with a 60 s pause between cases, two cases failed again, because one long case can exhaust the tokens-per-minute budget on its own. Runs that hit the limit took around 40-50 s instead of around 4 s, because of client-side retries.
- **The model chose the wrong answer kind.** In R04 and R09, the model chose `summary` although the request asked for neither a summary nor a usefulness verdict.
- **Small distortions in the prose.** 6 of 12 explanations contained claims that were not fully supported by the abstract. Two were notable (R03, R12), for example dropping a qualifier or inferring "mobile phones" from "devices". The deterministic checks cannot catch this kind of error.
- **Relevance on broad requests.** For broad or vague topics (R09, R12, R14), the agent accepts the first result that passes the filters, not necessarily the best one. "Federated learning on mobile devices" returned a general federated-learning paper.
- **Instructions were not always followed.** The instructions ask the model to call `cross_check_citations` whenever citations are mentioned, but it did so only in R01 out of the 8 cases with a citation constraint. It also wrote citation counts into its prose in R04, R11 and R14 despite being told not to. All of those numbers were correct, because they were copied from the tool output.
- **The model was not deterministic.** In one run of R13 it called `report_no_match` without searching (Python rejected the call); in the rerun it searched first. A single run per case says little about reliability (see pass^k above).

**How often the agent needed tool calls**

Every case used tools: 2-4 tool calls (3.1 on average) and 3-6 model turns (4.1 on average). The typical sequence was `search_papers` → `get_paper` → `choose_paper`. R10 needed a second search, and R01 added the Crossref cross-check. No case reached the 7-tool or 9-turn budget.

**Did the LLM hallucinate?**

The model never hallucinated facts that the user sees as data, because none of those come from the model. The year error in the manual run was a parsing mistake, not an invented fact. Here it only narrowed the search, but a mistake in the other direction would let a non-matching paper pass Python's checks. In the free-text explanation it did produce smaller unsupported claims: overstatements, dropped qualifiers and background knowledge not found in the abstract (R03, R12, and minor cases R01, R04, R05, R10). It invented no papers, numbers or citations.

**How we prevented incorrect answers**

- **Separation of concerns.** The LLM parses the request, picks a paper and writes prose. Deterministic code fetches data, filters by constraints, verifies the chosen paper and renders all metadata.
- **Every model action passes through a trusted session (`ResearchSession`).** It accepts only paper IDs that OpenAlex actually returned. It refuses `choose_paper` for a paper that was not read or that violates a constraint, and it blocks later searches that relax the numeric constraints. Its limitation is that it checks constraints against the model's first search, not against the request itself.
- **An output validator** rejects replies with the wrong sentence count or replies that only repeat the title.
- **Hard budgets** (7 tool calls, 9 model turns, 120 s) stop runaway loops.
- **The evaluation's oracle** is independent of the agent: expected constraints are declared per case and checked against source data.

**What we would improve with more time**

1. **Check the constraints against the request text.** Parse explicit phrases such as "before 2021", "in 2018" and "at least 100 citations" in Python, as the sentence count already is. Reject a `search_papers` call whose arguments disagree, before anything is locked, and tell the model the expected values. Compare constraints by meaning, so that `< 2021` and `<= 2020` are treated as equal.
2. **Retry rate limits properly.** Honour `Retry-After` on HTTP 429 inside the agent, and add a token-aware delay to the evaluator, so free-tier runs are not lost.
3. **Rank before choosing.** Fetch more candidates and let the model, or an embedding similarity score, rank them against the full request before choosing, so that "mobile devices" is not dropped from the topic.
4. **Answer kind in Python.** Determine the answer kind from the request in code, as the sentence count already is, instead of trusting the model.
5. **Check explanations automatically.** Use a FActScore-style claim check: split the explanation into claims and verify each against the abstract with a second, independent pass, and flag unsupported sentences.
6. **Measure reliability.** Run each case k times and report pass^k; the manual run above shows that a case can pass once and fail on a rerun. Also compare `openai/gpt-oss-20b` and other providers on the same set.

## Group member contributions

<!-- TODO: replace with real names and contributions before submission. Every member must also have meaningful commits. -->

| Member | GitHub | Contributions |
|---|---|---|
| Andrii Savchenko | @andriiVIt | TODO |
| TODO | TODO | TODO |
| TODO | TODO | TODO |

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
