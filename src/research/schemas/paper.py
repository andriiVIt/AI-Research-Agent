"""Typed constraints, paper metadata and the agent's final answer."""
import operator
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Op = Literal[">", ">=", "<", "<=", "=="]
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
RECENT_YEARS = 2  # "recent" means published in the current year or the two before it.


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Op = Field(description="Comparison operator.")
    value: int = Field(ge=0, description="Year or citation count to compare with.")

    def holds(self, actual):
        return actual is not None and OPS[self.op](actual, self.value)

    def __str__(self):
        return f"{self.op} {self.value}"


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=2, max_length=200, description=(
        "Search phrase for the subject only, e.g. 'LLM agents software engineering'. "
        "No years, numbers or instructions."))
    year: Comparison | None = Field(default=None, description=(
        "Publication-year constraint. 'after 2022' -> {op: '>', value: 2022}; "
        "'in or after 2022' / 'since 2022' -> '>='; 'before 2021' -> '<'; 'in 2020' -> '=='."))
    citations: Comparison | None = Field(default=None, description=(
        "Citation-count constraint. 'at least 100' -> {op: '>=', value: 100}; "
        "'more than 500' -> '>'; 'fewer than 50' -> '<'."))
    recent: bool = Field(default=False, description=(
        "True when the request asks for a recent/latest paper and gives no explicit year."))

    @classmethod
    def from_flat(cls, *, topic, year_op=None, year=None, citations_op=None, citations=None,
                  recent=False):
        """Build from scalar tool arguments; an operator and its value must come together."""
        def pair(name, op, value):
            if (op is None) != (value is None):
                raise ValueError(f"give both {name}_op and {name}, or neither")
            return None if op is None else {"op": op, "value": value}
        return cls.model_validate({"topic": topic, "recent": recent,
                                   "year": pair("year", year_op, year),
                                   "citations": pair("citations", citations_op, citations)})

    def effective_year(self, today: date):
        if self.year is not None:
            return self.year
        if self.recent:
            return Comparison(op=">=", value=today.year - RECENT_YEARS)
        return None

    def numeric_key(self):
        """The part of the request the agent may never relax between searches."""
        return (self.year and self.year.model_dump(), self.citations and self.citations.model_dump(),
                self.recent)


class Paper(BaseModel):
    openalex_id: str
    title: str
    year: int | None
    cited_by_count: int
    doi: str | None = None
    landing_url: str | None = None
    venue: str | None = None
    authors: list[str] = []
    abstract: str = ""
    retrieved_at: str

    def compact(self):
        return {"paper_id": self.openalex_id, "title": self.title, "year": self.year,
                "citations": self.cited_by_count, "venue": self.venue,
                "abstract_snippet": self.abstract[:300]}

    @property
    def link(self):
        return f"https://doi.org/{self.doi}" if self.doi else self.landing_url


class CitationEvidence(BaseModel):
    source: str = "OpenAlex"
    count: int
    url: str
    retrieved_at: str
    crossref_count: int | None = None
    crossref_url: str | None = None


class Recommendation(BaseModel):
    """Assembled by Python: the verified choose_paper selection plus the model's prose reply."""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    answer_kind: Literal["relevance", "summary", "usefulness"]
    verdict: Literal["useful", "partly useful", "not useful"] | None = None
    explanation: str


class Pick(BaseModel):
    """One shortlisted paper in comparison mode, numbered in the order it was chosen."""
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    verdict: Literal["useful", "partly useful", "not useful"] | None = None


class Shortlist(BaseModel):
    """Comparison mode (--top N): the verified shortlist plus the model's comparative prose.
    The prose refers to the papers as "Paper 1", "Paper 2", ... in shortlist order."""
    model_config = ConfigDict(extra="forbid")
    picks: list[Pick]
    answer_kind: Literal["relevance", "summary", "usefulness"]
    requested: int
    explanation: str

    @property
    def paper_ids(self):
        return [pick.paper_id for pick in self.picks]


class NoMatch(BaseModel):
    reason: str


class ClaimCheck(BaseModel):
    """One factual claim from the explanation, judged against the title, abstract and metadata."""
    claim: str = Field(description="One factual claim from the explanation, quoted or closely "
                                   "paraphrased.")
    status: Literal["supported", "overstated", "unsupported"] = Field(description=(
        "supported: the source states it; overstated: the source says something weaker; "
        "unsupported: not in the source, even if true in general."))
    fix: str | None = Field(default=None, description=(
        "For a claim that is not supported: how to reword it so the source supports it, or "
        "'remove'. Null for supported claims."))


class Critique(BaseModel):
    """The reflection agent's review of the writer's explanation (claims and request fit only)."""
    claims: list[ClaimCheck] = Field(description="Every factual claim in the explanation.")
    answers_the_request: bool = Field(description=(
        "True when the explanation gives what the request asks for: why the paper is relevant, "
        "a summary of its contribution, or whether it would be useful."))
    request_feedback: str | None = Field(default=None, description=(
        "When answers_the_request is false: one sentence on what the request asks for instead."))

    @property
    def problems(self) -> list[str]:
        found = [f'"{c.claim}" is {c.status}: {c.fix or "remove it"}'
                 for c in self.claims if c.status != "supported"]
        if not self.answers_the_request:
            found.append(self.request_feedback or "the reply does not answer what the request asks")
        return found


SENTENCE_RANGE = re.compile(r"(\d+)\s*(?:-|–|—|to)\s*(\d+)\s+sentences?", re.I)
SENTENCE_EXACT = re.compile(r"\b(?:in|with|using)?\s*(\d+)\s+sentences?", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def requested_sentences(request: str):
    """Python reads the sentence limit from the request so the model cannot reinterpret it."""
    match = SENTENCE_RANGE.search(request)
    if match:
        low, high = sorted((int(match[1]), int(match[2])))
        return low, high
    match = SENTENCE_EXACT.search(request)
    if match:
        return int(match[1]), int(match[1])
    return None


def count_sentences(text: str):
    text = text.strip()
    return len([part for part in SENTENCE_SPLIT.split(text) if part.strip()]) if text else 0
