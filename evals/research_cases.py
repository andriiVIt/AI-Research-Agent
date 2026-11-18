"""Research Scout eval cases. Expected constraints are declared here, independently of the agent,
so the evaluator never trusts the constraints the model chose to search with.

Every case has one main `category` (CATEGORIES) and a `constraint_types` list (CONSTRAINT_TYPES)
describing which constraints the request states, so results can be grouped systematically.
"""

CATEGORIES = ("broad", "narrow", "ambiguous", "no_match")
CONSTRAINT_TYPES = ("year_after", "year_before", "year_exact", "recent",
                    "min_citations", "max_citations")


def cmp(op, value):
    return {"op": op, "value": value}


CASES = [
    {"id": "R01", "category": "narrow", "constraint_types": ["year_after", "min_citations"],
     "expect": "answered", "kind": "relevance",
     "request": ("Find a research paper about LLM agents for software engineering that was published "
                 "after 2022 and has at least 100 citations. Explain why the paper is relevant and "
                 "provide the source of the citation count."),
     "year": cmp(">", 2022), "citations": cmp(">=", 100)},
    {"id": "R02", "category": "narrow", "constraint_types": ["year_before", "min_citations"],
     "expect": "answered", "kind": "summary", "sentences": (5, 7),
     "request": ("Find a paper about retrieval-augmented generation published before 2021 with more "
                 "than 500 citations. Summarize its contribution in 5-7 sentences."),
     "year": cmp("<", 2021), "citations": cmp(">", 500)},
    {"id": "R03", "category": "broad", "constraint_types": ["recent"],
     "expect": "answered", "kind": "usefulness", "recent": True,
     "request": ("Find a recent paper about AI agents using tools and explain whether it would be "
                 "useful for someone building autonomous software agents.")},
    {"id": "R04", "category": "narrow", "constraint_types": ["year_after", "min_citations"],
     "expect": "answered", "kind": "relevance",
     "request": "Find a paper on code generation benchmarks published in 2021 or later with at least 200 citations.",
     "year": cmp(">=", 2021), "citations": cmp(">=", 200)},
    {"id": "R05", "category": "broad", "constraint_types": ["year_exact"],
     "expect": "answered", "kind": "summary", "sentences": (3, 3),
     "request": "Find a paper about graph neural networks published in 2018 and summarize it in 3 sentences.",
     "year": cmp("==", 2018)},
    {"id": "R06", "category": "narrow", "constraint_types": ["max_citations"],
     "expect": "answered", "kind": "relevance",
     "request": ("Find a paper about prompt injection attacks on LLM applications with fewer than "
                 "50 citations. Explain why it is relevant."),
     "citations": cmp("<", 50)},
    {"id": "R07", "category": "narrow", "constraint_types": ["year_before"],
     "expect": "answered", "kind": "relevance",
     "request": "Find a paper about transformer models for time series forecasting published before 2023.",
     "year": cmp("<", 2023)},
    {"id": "R08", "category": "no_match", "constraint_types": ["year_after", "min_citations"],
     "expect": "no_match",
     "request": ("Find a paper about LLM agents published after 2025 with more than 100000 citations."),
     "year": cmp(">", 2025), "citations": cmp(">", 100000)},
    # Ambiguous: no explicit numbers. Only the status and answer kind are checked automatically;
    # relevance to the vague intent is graded manually.
    {"id": "R09", "category": "ambiguous", "constraint_types": [],
     "expect": "answered", "kind": "relevance",
     "request": "Find me something good about AI agents."},
    {"id": "R10", "category": "ambiguous", "constraint_types": [],
     "expect": "answered", "kind": "relevance",
     "request": "I need a well-known paper on transformers."},
    {"id": "R11", "category": "narrow", "constraint_types": ["year_after", "min_citations"],
     "expect": "answered", "kind": "relevance",
     "request": ("Find a paper about low-rank adaptation (LoRA) for fine-tuning large language "
                 "models published after 2020 with at least 1000 citations. Explain why it is relevant."),
     "year": cmp(">", 2020), "citations": cmp(">=", 1000)},
    {"id": "R12", "category": "narrow", "constraint_types": ["year_exact"],
     "expect": "answered", "kind": "relevance",
     "request": "Find a paper about federated learning on mobile devices published in 2017.",
     "year": cmp("==", 2017)},
    {"id": "R13", "category": "no_match", "constraint_types": ["year_before", "min_citations"],
     "expect": "no_match",
     "request": ("Find a paper about jailbreaking ChatGPT published before 2015 with at least "
                 "500 citations."),
     "year": cmp("<", 2015), "citations": cmp(">=", 500)},
    {"id": "R14", "category": "broad", "constraint_types": ["year_after", "max_citations"],
     "expect": "answered", "kind": "relevance",
     "request": "Find a paper about machine learning published after 2020 with at most 20 citations.",
     "year": cmp(">", 2020), "citations": cmp("<=", 20)},
]
