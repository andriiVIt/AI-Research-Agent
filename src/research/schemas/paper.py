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


class NoMatch(BaseModel):
    reason: str


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
