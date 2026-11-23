"""The agent's five sequential tools. Each delegates to the trusted ResearchSession.

search_papers takes flat scalar arguments: some providers (e.g. Groq-hosted Qwen) intermittently
fail to generate nested-object tool arguments. Python assembles and validates Constraints.
"""
from typing import Literal

from pydantic_ai import Agent, RunContext

from research.deps import ResearchSession
from research.schemas import Constraints
from research.schemas.paper import Op


def register_tools(agent: Agent, reply_hint=lambda session: "2-8"):
    @agent.tool(sequential=True, retries=0)
    async def search_papers(ctx: RunContext[ResearchSession], topic: str,
                            year_op: Op | None = None, year: int | None = None,
                            citations_op: Op | None = None, citations: int | None = None,
                            recent: bool = False) -> dict:
        """Search OpenAlex for papers on a topic that satisfy the year/citation constraints.

        Args:
            topic: Subject words only, e.g. "LLM agents software engineering". No years or numbers.
            year_op: ">" for "after 2022", "<" for "before 2021", ">=" for "since/in or after", "==" for "in 2020". Null if no year is stated.
            year: The year in the request, or null.
            citations_op: ">=" for "at least N", ">" for "more than N", "<" for "fewer than N". Null if no citation count is stated.
            citations: The citation count in the request, or null.
            recent: True for "recent"/"latest"/"new" when no year is given.
        """
        arguments = {"topic": topic, "year_op": year_op, "year": year,
                     "citations_op": citations_op, "citations": citations, "recent": recent}
        try:
            constraints = Constraints.from_flat(**arguments)
        except ValueError as error:
            return ctx.deps.reject("search_papers", arguments, f"Invalid constraints: {error}")
        return await ctx.deps.search(constraints)

    @agent.tool(sequential=True, retries=0)
    async def get_paper(ctx: RunContext[ResearchSession], paper_id: str) -> dict:
        """Fetch full metadata and abstract for a paper_id returned by search_papers.
        Required before answering about that paper."""
        return await ctx.deps.paper(paper_id)

    @agent.tool(sequential=True, retries=0)
    async def cross_check_citations(ctx: RunContext[ResearchSession], paper_id: str) -> dict:
        """Compare the OpenAlex citation count with Crossref's count for the paper's DOI."""
        return await ctx.deps.cross_check(paper_id)

    @agent.tool(sequential=True, retries=0)
    def choose_paper(ctx: RunContext[ResearchSession], paper_id: str,
                     answer_kind: Literal["relevance", "summary", "usefulness"],
                     verdict: Literal["useful", "partly useful", "not useful"] | None = None) -> dict:
        """Select the paper you will answer about. Python verifies it before you write the reply.
        In comparison mode, call it once per paper: each call adds the next numbered paper.

        Args:
            paper_id: A paper_id you already read with get_paper.
            answer_kind: "summary" if asked to summarise, "usefulness" if asked whether the paper would be useful, otherwise "relevance".
            verdict: Required when answer_kind is "usefulness"; otherwise null.
        """
        return ctx.deps.choose(paper_id, answer_kind, verdict, reply_hint(ctx.deps))

    @agent.tool(sequential=True, retries=0)
    def report_no_match(ctx: RunContext[ResearchSession], reason: str) -> dict:
        """Declare that no searched paper satisfies every constraint.

        Args:
            reason: One sentence on what was searched and why nothing qualified.
        """
        return ctx.deps.report_no_match(reason)
