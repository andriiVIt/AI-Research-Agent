"""Live evaluation of Research Scout (needs LLM_* and OPENALEX_API_KEY).

The oracle below re-checks each answer against the case's declared constraints, using the paper
metadata Python fetched from OpenAlex, not the model's text and not the agent's own checks.
"""
import argparse
import asyncio
import json
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
from evals.research_cases import CASES  # noqa: E402
from research.agents.scout import run  # noqa: E402
from research.schemas import Comparison, count_sentences  # noqa: E402
from research.schemas.paper import RECENT_YEARS  # noqa: E402


def grade(case, summary, today):
    status = summary["status"]
    checks = {"status": status == case["expect"]}
    if case["expect"] == "answered" and status == "answered":
        paper, answer = summary["paper"], summary["answer"]
        year = case.get("year")
        if case.get("recent"):
            year = {"op": ">=", "value": today.year - RECENT_YEARS}
        if year:
            checks["year"] = Comparison(**year).holds(paper["year"])
        if case.get("citations"):
            checks["citations"] = Comparison(**case["citations"]).holds(paper["cited_by_count"])
        evidence = summary.get("citation_evidence") or {}
        checks["citation_source"] = bool(evidence.get("url")) and evidence.get("count") == paper["cited_by_count"]
        checks["answer_kind"] = answer["answer_kind"] == case["kind"]
        if case.get("sentences"):
            low, high = case["sentences"]
            checks["sentences"] = low <= count_sentences(answer["explanation"]) <= high
        if case["kind"] == "usefulness":
            checks["verdict"] = answer.get("verdict") is not None
    return checks


async def evaluate(cases, output):
    today = date.today()
    rows = []
    for case in cases:
        started = time.monotonic()
        records = await run(case["request"])
        summary = records[-1]
        checks = grade(case, summary, today)
        row = {"id": case["id"], "passed": all(checks.values()), "checks": checks,
               "status": summary["status"], "seconds": round(time.monotonic() - started, 1),
               "model_turns": summary["model_turns"], "tool_executions": summary["tool_executions"],
               "model_attempts": records[0]["attempts"],
               "paper": (summary.get("paper") or {}).get("title"), "records": records}
        rows.append(row)
        print(f"{case['id']} {'PASS' if row['passed'] else 'FAIL'} {summary['status']:<13} "
              f"{row['seconds']:>5}s turns={row['model_turns']} tools={row['tool_executions']} "
              f"{ {k: v for k, v in checks.items() if not v} or ''} {row['paper'] or ''}")
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    passed = sum(row["passed"] for row in rows)
    print(f"\n{passed}/{len(rows)} cases passed. Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Case ids to run, e.g. R01 R02")
    parser.add_argument("--output", default="research_eval_results.jsonl")
    args = parser.parse_args()
    cases = [case for case in CASES if not args.only or case["id"] in args.only]
    asyncio.run(evaluate(cases, Path(args.output)))


if __name__ == "__main__":
    main()
