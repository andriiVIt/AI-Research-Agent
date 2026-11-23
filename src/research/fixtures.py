"""Offline fixtures: fictional OpenAlex/Crossref responses and a scripted (non-LLM) model.

Used by `--demo` and the unit tests. Every title says "(fixture)" so the data can never be
mistaken for a real paper or a real citation count.
"""
import json
from urllib.parse import unquote

import httpx2
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

DEMO_REQUEST = ("Find a research paper about LLM agents for software engineering that was published "
                "after 2022 and has at least 100 citations. Explain why the paper is relevant and "
                "provide the source of the citation count.")


def _invert(text):
    index = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def fixture_work(openalex_id, title, year, citations, doi, abstract):
    return {"id": f"https://openalex.org/{openalex_id}", "doi": f"https://doi.org/{doi}",
            "display_name": title, "publication_year": year, "cited_by_count": citations,
            "primary_location": {"landing_page_url": f"https://example.org/{openalex_id}",
                                 "source": {"display_name": "Fixture Proceedings"}},
            "authorships": [{"author": {"display_name": "A. Example"}},
                            {"author": {"display_name": "B. Sample"}}],
            "abstract_inverted_index": _invert(abstract)}


FIXTURE_WORKS = [
    fixture_work("W900001", "Example A: Tool-Using LLM Agents for Repository-Level Program Repair (fixture)",
                 2024, 312, "10.0000/fixture.a",
                 "We present an agent that lets a large language model navigate a code repository, "
                 "edit files and run tests through a small set of tools. On a benchmark of real "
                 "GitHub issues the agent resolves more issues than retrieval-only baselines."),
    fixture_work("W900002", "Example B: Retrieval-Augmented Generation for Open-Domain QA (fixture)",
                 2020, 950, "10.0000/fixture.b",
                 "We combine a pretrained sequence-to-sequence model with a dense retriever over "
                 "Wikipedia and train both end to end for knowledge-intensive tasks."),
    fixture_work("W900003", "Example C: A Small Study of Coding Assistants (fixture)",
                 2023, 40, "10.0000/fixture.c",
                 "A survey of developers using coding assistants."),
]
CROSSREF_COUNTS = {"10.0000/fixture.a": 287, "10.0000/fixture.b": 901, "10.0000/fixture.c": 35}


def fixture_transport(works=None, seen=None, crossref=None):
    """MockTransport serving fixture works; every request is appended to `seen` if given.
    Search ignores filters on purpose, so Python's own constraint checks are exercised."""
    works = FIXTURE_WORKS if works is None else works
    crossref = CROSSREF_COUNTS if crossref is None else crossref
    by_id = {work["id"].rsplit("/", 1)[-1]: work for work in works}

    def handler(request):
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if request.url.host == "api.crossref.org":
            doi = unquote(path.removeprefix("/works/"))
            if doi not in crossref:
                return httpx2.Response(404, json={"status": "error"})
            return httpx2.Response(200, json={"message": {"is-referenced-by-count": crossref[doi]}})
        if path == "/works":
            return httpx2.Response(200, json={"meta": {"count": len(works)}, "results": works})
        work = by_id.get(path.removeprefix("/works/"))
        return httpx2.Response(200, json=work) if work else httpx2.Response(404, json={})

    return httpx2.MockTransport(handler)


def scripted_step(step, info):
    """One step as a model response: (tool, args) is a tool call, ("text", str) a plain reply,
    ("output", args) a structured final output (as the critic agent returns), and
    ("raise", exception) a provider failure."""
    tool, args = step
    if tool == "raise":
        raise args
    if tool == "text":
        return ModelResponse(parts=[TextPart(args)])
    if tool == "output":
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])
    return ModelResponse(parts=[ToolCallPart(tool, args)])


def scripted_model(steps, *, name="scripted-offline"):
    """A FunctionModel (not an LLM) that replays scripted steps in order."""
    calls = iter(steps)
    seen_prompts = []

    def respond(messages, info):
        seen_prompts.append(json.dumps([str(m) for m in messages])[:60000])
        return scripted_step(next(calls), info)

    model = FunctionModel(respond, model_name=name)
    model.seen_prompts = seen_prompts
    return model


DEMO_STEPS = [
    ("search_papers", {"topic": "LLM agents for software engineering", "year_op": ">",
                       "year": 2022, "citations_op": ">=", "citations": 100}),
    ("get_paper", {"paper_id": "W900001"}),
    ("cross_check_citations", {"paper_id": "W900001"}),
    ("choose_paper", {"paper_id": "W900001", "answer_kind": "relevance"}),
    # First draft: "state-of-the-art" is not in the fixture abstract, so the critic flags it.
    ("text", "The paper builds an LLM agent that edits repositories and runs tests through tools, "
             "which is exactly the software engineering agent setting asked about. It reaches "
             "state-of-the-art results on real GitHub issues. This judgement is based on the "
             "abstract only, and the data is a fixture."),
    # Rewrite after the critique.
    ("text", "The paper builds an LLM agent that edits repositories and runs tests through tools, "
             "which is exactly the software engineering agent setting asked about. On a benchmark "
             "of real GitHub issues it resolves more issues than retrieval-only baselines. This "
             "judgement is based on the abstract only, and the data is a fixture."),
]

DEMO_CRITIC_STEPS = [
    ("output", {"answers_the_request": True, "claims": [
        {"claim": "The paper builds an LLM agent that edits repositories and runs tests through tools",
         "status": "supported"},
        {"claim": "It reaches state-of-the-art results on real GitHub issues", "status": "overstated",
         "fix": "say it resolves more issues than retrieval-only baselines, as the abstract does"}]}),
    ("output", {"answers_the_request": True, "claims": [
        {"claim": "The paper builds an LLM agent that edits repositories and runs tests through tools",
         "status": "supported"},
        {"claim": "it resolves more issues than retrieval-only baselines", "status": "supported"}]}),
]


def demo_model():
    return scripted_model(DEMO_STEPS, name="scripted-offline-demo")


def demo_critic_model():
    """A scripted reflection agent: flags one overstated claim, then passes the rewrite."""
    return scripted_model(DEMO_CRITIC_STEPS, name="scripted-offline-critic")
