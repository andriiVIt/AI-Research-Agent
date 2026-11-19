"""Research Scout tests. No API keys or network: scripted models and mocked transports."""
import asyncio
import json
import os
import time
import unittest
from datetime import date

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
import httpx2  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from research.agents.scout import execute, run  # noqa: E402
from research.deps import BudgetReached, ResearchSession  # noqa: E402
from research.fixtures import (FIXTURE_WORKS, fixture_transport, fixture_work,  # noqa: E402
                               scripted_model, scripted_step)
from research.schemas import (Comparison, Constraints, count_sentences,  # noqa: E402
                              requested_sentences)
from research.services.crossref import CrossrefClient  # noqa: E402
from research.services.openalex import (OpenAlexClient, build_filter,  # noqa: E402
                                        reconstruct_abstract, range_filter)

TODAY = date(2026, 11, 16)
SECRET = "secret-openalex-key-123"


def constraints(topic="LLM agents software engineering", year=None, citations=None, recent=False):
    return {"topic": topic, "recent": recent,
            "year": None if year is None else {"op": year[0], "value": year[1]},
            "citations": None if citations is None else {"op": citations[0], "value": citations[1]}}


def search(topic="LLM agents software engineering", year=None, citations=None, recent=False):
    """A search_papers step with the flat scalar arguments the tool takes."""
    args = {"topic": topic, "recent": recent}
    if year:
        args.update(year_op=year[0], year=year[1])
    if citations:
        args.update(citations_op=citations[0], citations=citations[1])
    return ("search_papers", args)


REPLY = ("It studies tool-using LLM agents that repair code in repositories. "
         "It evaluates on real issues.")


def choose(paper_id="W900001", kind="relevance", verdict=None):
    return ("choose_paper", {"paper_id": paper_id, "answer_kind": kind, "verdict": verdict})


def reply(text=REPLY):
    return ("text", text)


def make_session(works=None, seen=None):
    http = httpx2.AsyncClient(transport=fixture_transport(works, seen))
    return ResearchSession(openalex=OpenAlexClient(http, SECRET, today=TODAY),
                           crossref=CrossrefClient(http))


def run_steps(steps, request, works=None, seen=None):
    async def go():
        session = make_session(works, seen)
        summary = await execute(scripted_model(steps), session, request)
        return session, summary
    return asyncio.run(go())


class TestConstraintsAndFilters(unittest.TestCase):
    def test_operator_normalisation(self):
        cases = {(">", 2022): "publication_year:>2022", (">=", 2022): "publication_year:>2021",
                 ("<", 2021): "publication_year:<2021", ("<=", 2021): "publication_year:<2022",
                 ("==", 2020): "publication_year:2020"}
        for (op, value), expected in cases.items():
            self.assertEqual(range_filter("publication_year", Comparison(op=op, value=value)), expected)

    def test_build_filter_combines_year_and_citations(self):
        c = Constraints.model_validate(constraints(year=(">", 2022), citations=(">=", 100)))
        self.assertEqual(build_filter(c, TODAY),
                         "type:article|preprint|review,publication_year:>2022,cited_by_count:>99")
        c = Constraints.model_validate(constraints(citations=(">", 500)))
        self.assertEqual(build_filter(c, TODAY), "type:article|preprint|review,cited_by_count:>500")

    def test_recent_means_last_two_years_unless_year_given(self):
        recent = Constraints.model_validate(constraints(recent=True))
        self.assertIn("publication_year:>2023", build_filter(recent, TODAY))
        explicit = Constraints.model_validate(constraints(year=(">=", 2020), recent=True))
        self.assertIn("publication_year:>2019", build_filter(explicit, TODAY))

    def test_constraints_reject_bad_input(self):
        for bad in [dict(constraints(), extra=1), constraints(year=("!=", 2020)),
                    constraints(citations=(">", -5)), constraints(topic="")]:
            with self.assertRaises(ValueError):
                Constraints.model_validate(bad)

    def test_abstract_reconstruction_orders_words(self):
        self.assertEqual(reconstruct_abstract({"world": [1], "hello": [0, 2]}), "hello world hello")
        self.assertEqual(reconstruct_abstract(None), "")

    def test_sentence_requests(self):
        self.assertEqual(requested_sentences("Summarize it in 5-7 sentences."), (5, 7))
        self.assertEqual(requested_sentences("Summarize in 7 to 5 sentences"), (5, 7))
        self.assertEqual(requested_sentences("Explain in 3 sentences."), (3, 3))
        self.assertIsNone(requested_sentences("Explain why it is relevant."))
        self.assertEqual(count_sentences("One. Two! Three? 4 is four."), 4)


class TestSources(unittest.TestCase):
    def test_openalex_request_and_parsing(self):
        seen = []
        async def go():
            async with httpx2.AsyncClient(transport=fixture_transport(seen=seen)) as http:
                client = OpenAlexClient(http, SECRET, today=TODAY)
                c = Constraints.model_validate(constraints(year=(">", 2022), citations=(">=", 100)))
                return await client.search(c)
        papers, work_filter = asyncio.run(go())
        params = seen[0].url.params
        self.assertEqual(params["search"], "LLM agents software engineering")
        self.assertEqual(params["filter"], work_filter)
        self.assertEqual(params["api_key"], SECRET)
        self.assertEqual(papers[0].openalex_id, "W900001")
        self.assertEqual(papers[0].doi, "10.0000/fixture.a")
        self.assertTrue(papers[0].abstract.startswith("We present an agent"))

    def test_source_errors_never_leak_key(self):
        def handler(request):
            raise httpx2.ConnectError(f"cannot reach {request.url}")
        async def go():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
                session = ResearchSession(openalex=OpenAlexClient(http, SECRET, today=TODAY))
                return await session.search(Constraints.model_validate(constraints()))
        result = asyncio.run(go())
        self.assertFalse(result["accepted"])
        self.assertNotIn(SECRET, json.dumps(result))


class TestSession(unittest.TestCase):
    def test_search_drops_violations_and_registers_candidates(self):
        async def go():
            session = make_session()
            c = Constraints.model_validate(constraints(year=(">", 2022), citations=(">=", 100)))
            return session, await session.search(c)
        session, result = asyncio.run(go())
        self.assertEqual([r["paper_id"] for r in result["results"]], ["W900001"])
        self.assertEqual(set(session.candidates), {"W900001"})
        self.assertEqual(session.verify("W900001"), [])
        self.assertTrue(session.verify("W900002"))

    def test_oracle_rejects_paper_that_violates_constraints(self):
        async def go():
            session = make_session()
            await session.search(Constraints.model_validate(constraints(year=(">", 2022))))
            return session
        session = asyncio.run(go())
        old = session.candidates["W900001"].model_copy(update={"year": 2021})
        session.candidates["W900001"] = old
        self.assertEqual(session.verify("W900001"), ["year > 2022 fails (actual: 2021)"])

    def test_numeric_constraints_cannot_be_relaxed(self):
        async def go():
            session = make_session()
            await session.search(Constraints.model_validate(constraints(citations=(">=", 100))))
            return await session.search(Constraints.model_validate(
                constraints(topic="coding agents", citations=(">=", 10))))
        result = asyncio.run(go())
        self.assertFalse(result["accepted"])
        self.assertIn("must stay exactly", result["feedback"])

    def test_budget_and_deadline(self):
        session = make_session()
        session.tool_limit = 1
        asyncio.run(session.paper("W1"))
        with self.assertRaises(BudgetReached):
            asyncio.run(session.paper("W1"))
        session = make_session()
        session.deadline = time.monotonic() - 1
        with self.assertRaises(BudgetReached):
            asyncio.run(session.paper("W1"))


SEARCH_AFTER_2022 = search(year=(">", 2022), citations=(">=", 100))
AGENTS_REQUEST = ("Find a paper about LLM agents for software engineering published after 2022 with "
                  "at least 100 citations. Explain why it is relevant and give the citation source.")
RAG_REQUEST = ("Find a paper about retrieval-augmented generation published before 2021 with more "
               "than 500 citations. Summarize its contribution in 5-7 sentences.")
RAG_SEARCH = search("retrieval-augmented generation", year=("<", 2021), citations=(">", 500))
SIX_SENTENCES = " ".join(f"Sentence {n} describes the retrieval and generation model." for n in range(1, 7))
READ_A = ("get_paper", {"paper_id": "W900001"})


class TestAgent(unittest.TestCase):
    def test_scenario_relevance_with_citation_source(self):
        session, summary = run_steps([SEARCH_AFTER_2022, READ_A,
            ("cross_check_citations", {"paper_id": "W900001"}), choose(), reply()], AGENTS_REQUEST)
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(summary["violations"], [])
        self.assertEqual(summary["answer"]["explanation"], REPLY)
        self.assertEqual(summary["citation_evidence"]["count"], 312)
        self.assertEqual(summary["citation_evidence"]["crossref_count"], 287)
        self.assertIn("https://api.openalex.org/works/W900001", summary["rendered"])
        self.assertIn("Uncertainty: The explanation is based on the title, abstract and metadata",
                      summary["rendered"])
        self.assertIn("OpenAlex reports 312 and Crossref 287", summary["rendered"])
        self.assertNotIn(SECRET, json.dumps(session.trace) + json.dumps(summary))

    def test_scenario_summary_enforces_sentence_count(self):
        session, summary = run_steps([RAG_SEARCH, ("get_paper", {"paper_id": "W900002"}),
            choose("W900002", "summary"), reply("It retrieves passages. Then it generates."),
            reply(SIX_SENTENCES)], RAG_REQUEST)
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(summary["model_turns"], 5)
        self.assertEqual(summary["paper"]["year"], 2020)
        self.assertIn("reply with 5-7", json.dumps(session.trace[-1]["result"]))  # choose_paper hint

    def test_repeated_invalid_reply_is_rejected(self):
        _, summary = run_steps([RAG_SEARCH, ("get_paper", {"paper_id": "W900002"}),
                                choose("W900002", "summary"), reply("Too short."),
                                reply("Too short."), reply("Too short.")], RAG_REQUEST)
        self.assertEqual(summary["status"], "rejected")

    def test_invented_or_unread_paper_cannot_be_chosen(self):
        session, summary = run_steps([SEARCH_AFTER_2022, choose("W999999"), choose(), READ_A,
                                      choose(), reply()], AGENTS_REQUEST)
        self.assertEqual([r["validation"] for r in session.trace],
                         ["accepted", "rejected", "rejected", "accepted", "accepted"])
        self.assertIn("not returned by any search", session.trace[1]["result"]["feedback"])
        self.assertIn("call get_paper", session.trace[2]["result"]["feedback"])
        self.assertEqual(summary["status"], "answered")

    def test_reply_without_choosing_is_retried(self):
        _, summary = run_steps([SEARCH_AFTER_2022, READ_A, reply(), choose(), reply()], AGENTS_REQUEST)
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(summary["model_turns"], 5)

    def test_scenario_usefulness_recent(self):
        recent = fixture_work("W900004", "Example D: Agents That Call APIs (fixture)", 2025, 80,
                              "10.0000/fixture.d", "Agents learn when and how to call external tools.")
        session, summary = run_steps([search("AI agents tool use", recent=True),
            ("get_paper", {"paper_id": "W900004"}), choose("W900004", "usefulness"),
            choose("W900004", "usefulness", "useful"),
            reply("It shows how agents decide when to call tools. That is a core design choice.")],
            "Find a recent paper about AI agents using tools and explain whether it would be useful "
            "for someone building autonomous software agents.", works=FIXTURE_WORKS + [recent])
        self.assertIn("give a verdict", session.trace[2]["result"]["feedback"])
        self.assertEqual(summary["status"], "answered")
        self.assertIn("Usefulness verdict: useful", summary["rendered"])
        self.assertEqual(summary["checks"][0]["constraint"], "year >= 2024 (recent)")

    def test_no_match(self):
        session, summary = run_steps([("report_no_match", {"reason": "too early"}),
            search(year=(">", 2025), citations=(">", 100000)),
            ("report_no_match", {"reason": "No paper after 2025 has that many citations."}),
            reply("No paper published after 2025 has more than 100000 citations.")],
            "Find an LLM agents paper after 2025 with more than 100000 citations.")
        self.assertEqual(session.trace[0]["validation"], "rejected")  # must search first
        self.assertEqual(summary["status"], "no_match")
        self.assertIn("No qualifying paper", summary["rendered"])

    def test_title_only_reply_is_rejected(self):
        title = FIXTURE_WORKS[0]["display_name"]
        _, summary = run_steps([SEARCH_AFTER_2022, READ_A, choose(), reply(title), reply()],
                               AGENTS_REQUEST)
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(summary["answer"]["explanation"], REPLY)

    def test_op_without_value_is_rejected_without_source_request(self):
        seen = []
        session, summary = run_steps([("search_papers", {"topic": "LLM agents", "year_op": ">"}),
            SEARCH_AFTER_2022, READ_A, choose(), reply()], AGENTS_REQUEST, seen=seen)
        self.assertEqual(session.trace[0]["validation"], "rejected")
        self.assertIn("year_op", session.trace[0]["result"]["feedback"])
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(len([r for r in seen if r.url.path == "/works"]), 1)

    def test_provider_tool_generation_failure_is_retried_once(self):
        steps = iter([SEARCH_AFTER_2022, READ_A, choose(), reply()])
        failures = [ModelHTTPError(400, "mock", {"code": "tool_use_failed"})]
        def respond(messages, info):
            if failures:
                raise failures.pop()
            return scripted_step(next(steps), info)
        summary = asyncio.run(execute(FunctionModel(respond), make_session(), AGENTS_REQUEST))
        self.assertEqual(summary["status"], "answered")
        self.assertEqual(summary["provider_retries"], 1)
        self.assertEqual(summary["model_turns"], 4)

    def test_other_provider_errors_are_not_retried(self):
        calls = []
        def respond(messages, info):
            calls.append(1)
            raise ModelHTTPError(429, "mock", {"code": "rate_limited"})
        summary = asyncio.run(execute(FunctionModel(respond), make_session(), "anything"))
        self.assertEqual(summary["status"], "error")
        self.assertEqual(len(calls), 1)

    def test_demo_run_records(self):
        records = asyncio.run(run(demo=True))
        self.assertEqual(records[0]["mode"], "scripted_offline")
        self.assertEqual([r["tool"] for r in records if r["event"] == "tool"],
                         ["search_papers", "get_paper", "cross_check_citations", "choose_paper"])
        self.assertEqual(records[-1]["status"], "answered")


if __name__ == "__main__":
    unittest.main()
