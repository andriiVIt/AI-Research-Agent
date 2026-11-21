"""Trusted boundary between tool calls and bibliographic sources.

The session records every tool call, enforces the budget, keeps the papers that sources
actually returned, and re-checks constraints against source metadata (never model text).
"""
import copy
import time
from dataclasses import dataclass, field

from research.schemas import CitationEvidence, Constraints, Paper
from research.services.crossref import CrossrefClient, work_url
from research.services.openalex import OpenAlexClient, SourceError, evidence_url


class BudgetReached(Exception):
    """A local limit stopped a tool before any further source request."""


def check_paper(paper: Paper, constraints: Constraints, today):
    """Return [(label, actual, ok)] for each numeric constraint."""
    checks = []
    year = constraints.effective_year(today)
    if year is not None:
        label = f"year {year}" + (" (recent)" if constraints.year is None else "")
        checks.append((label, paper.year, year.holds(paper.year)))
    if constraints.citations is not None:
        checks.append((f"citations {constraints.citations}", paper.cited_by_count,
                       constraints.citations.holds(paper.cited_by_count)))
    return checks


@dataclass
class ResearchSession:
    openalex: OpenAlexClient
    crossref: CrossrefClient | None = None
    trace: list = field(default_factory=list)
    calls: int = 0
    tool_limit: int = 6
    deadline: float = field(default_factory=lambda: time.monotonic() + 120)
    constraints: Constraints | None = None
    searches: int = 0
    candidates: dict = field(default_factory=dict)
    detailed: set = field(default_factory=set)
    crossref_counts: dict = field(default_factory=dict)
    sentence_range: tuple | None = None
    selection: dict | None = None
    no_match_reason: str | None = None

    @property
    def today(self):
        return self.openalex.today

    def begin_call(self, name, arguments):
        if time.monotonic() >= self.deadline:
            raise BudgetReached("Time limit reached.")
        if self.calls >= self.tool_limit:
            raise BudgetReached("Tool execution limit reached.")
        self.calls += 1
        record = {"event": "tool", "number": self.calls, "tool": name,
                  "arguments": copy.deepcopy(arguments)}
        self.trace.append(record)
        return record

    def _finish(self, record, result, *, accepted=True):
        result = {"accepted": accepted, **result,
                  "tool_calls_remaining": max(0, self.tool_limit - self.calls)}
        record.update(validation="accepted" if accepted else "rejected", result=copy.deepcopy(result))
        return result

    def reject(self, name, arguments, feedback):
        """Record a tool call whose arguments failed validation before any source request."""
        record = self.begin_call(name, arguments)
        return self._finish(record, {"feedback": feedback}, accepted=False)

    async def search(self, constraints: Constraints):
        record = self.begin_call("search_papers", constraints.model_dump())
        if self.constraints is not None and constraints.numeric_key() != self.constraints.numeric_key():
            return self._finish(record, {"feedback": (
                "Rejected: year, citation and recency constraints come from the request and must "
                f"stay exactly {self.constraints.model_dump(exclude={'topic'})}. Only the topic "
                "wording may change.")}, accepted=False)
        try:
            papers, work_filter = await self.openalex.search(constraints)
        except SourceError as error:
            return self._finish(record, {"feedback": str(error)}, accepted=False)
        self.constraints = self.constraints or constraints
        self.searches += 1
        kept, dropped = [], 0
        for paper in papers:
            # Defence in depth: the source filter should already exclude these.
            if all(ok for _, _, ok in check_paper(paper, constraints, self.today)):
                self.candidates[paper.openalex_id] = paper
                kept.append(paper.compact())
            else:
                dropped += 1
        feedback = (f"{len(kept)} paper(s) satisfy the constraints." if kept else
                    "No papers matched. You may retry once with different topic wording "
                    "(same numeric constraints) or call report_no_match.")
        if dropped:
            feedback += f" {dropped} results that violated the constraints were removed."
        return self._finish(record, {"feedback": feedback, "applied_filter": work_filter,
                                     "results": kept})

    async def paper(self, paper_id: str):
        record = self.begin_call("get_paper", {"paper_id": paper_id})
        if paper_id not in self.candidates:
            return self._finish(record, {"feedback": "Unknown paper_id. Use an id returned by "
                                                     "search_papers."}, accepted=False)
        try:
            paper = await self.openalex.get(paper_id)
        except SourceError as error:
            return self._finish(record, {"feedback": str(error)}, accepted=False)
        self.candidates[paper_id] = paper
        self.detailed.add(paper_id)
        return self._finish(record, {"feedback": "Full metadata from OpenAlex.",
                                     "paper": paper.model_dump(exclude={"retrieved_at"})})

    async def cross_check(self, paper_id: str):
        record = self.begin_call("cross_check_citations", {"paper_id": paper_id})
        paper = self.candidates.get(paper_id)
        if paper is None:
            return self._finish(record, {"feedback": "Unknown paper_id."}, accepted=False)
        if not paper.doi or self.crossref is None:
            return self._finish(record, {"feedback": "No DOI or Crossref unavailable; OpenAlex "
                                                     "count only.", "openalex": paper.cited_by_count})
        try:
            count = await self.crossref.citation_count(paper.doi)
        except SourceError as error:
            return self._finish(record, {"feedback": str(error), "openalex": paper.cited_by_count},
                                accepted=False)
        self.crossref_counts[paper_id] = count
        return self._finish(record, {"feedback": "Counts from two indexes (they may legitimately "
                                                 "differ); OpenAlex is the primary source.",
                                     "openalex": paper.cited_by_count, "crossref": count})

    def choose(self, paper_id, answer_kind, verdict, reply_hint):
        record = self.begin_call("choose_paper", {"paper_id": paper_id, "answer_kind": answer_kind,
                                                  "verdict": verdict})
        problems = self.verify(paper_id)
        if paper_id in self.candidates and paper_id not in self.detailed:
            problems.append(f"call get_paper on {paper_id} first")
        if answer_kind == "usefulness" and verdict is None:
            problems.append("give a verdict for a usefulness answer")
        if problems:
            return self._finish(record, {"feedback": "Rejected: " + "; ".join(problems) + "."},
                                accepted=False)
        self.selection = {"paper_id": paper_id, "answer_kind": answer_kind, "verdict": verdict}
        self.no_match_reason = None
        return self._finish(record, {"feedback": (
            f"Selected and verified. Now reply with {reply_hint} plain prose sentences that answer "
            "the request. No title heading, citation counts or URLs.")})

    def report_no_match(self, reason):
        record = self.begin_call("report_no_match", {"reason": reason})
        if self.searches == 0:
            return self._finish(record, {"feedback": "Rejected: call search_papers first."},
                                accepted=False)
        self.no_match_reason, self.selection = reason, None
        return self._finish(record, {"feedback": "Recorded. Reply with one or two sentences "
                                                 "explaining that no paper qualified."})

    def checks(self, paper_id):
        paper = self.candidates.get(paper_id)
        if paper is None or self.constraints is None:
            return []
        return check_paper(paper, self.constraints, self.today)

    def verify(self, paper_id):
        """The oracle: violations found in source metadata, empty when the paper qualifies."""
        if paper_id not in self.candidates:
            return [f"{paper_id!r} was not returned by any search_papers call"]
        return [f"{label} fails (actual: {actual})" for label, actual, ok in self.checks(paper_id)
                if not ok]

    def evidence(self, paper_id):
        paper = self.candidates[paper_id]
        crossref = self.crossref_counts.get(paper_id)
        return CitationEvidence(count=paper.cited_by_count, url=evidence_url(paper_id),
                                retrieved_at=paper.retrieved_at, crossref_count=crossref,
                                crossref_url=work_url(paper.doi) if crossref is not None else None)
