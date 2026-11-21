"""Ask Research Scout for a paper and write a JSONL trace.

Example:
  python -m research.api.scout "Find a paper about retrieval-augmented generation published
  before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
from research.agents.scout import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", help="The research request in plain language")
    parser.add_argument("--demo", action="store_true",
                        help="Scripted offline run on fixture data (no model, no network)")
    parser.add_argument("--log", default="scout_trace.jsonl")
    parser.add_argument("--no-reflection", action="store_true",
                        help="Skip the reflection agent that critiques the explanation")
    args = parser.parse_args()
    if args.demo and args.request:
        parser.error("--demo uses a fixed request; omit the request text.")
    path = Path(args.log)
    if path.exists():
        parser.error("Log exists; choose a new --log filename.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        records = asyncio.run(run(args.request, demo=args.demo, reflection=not args.no_reflection))
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                    encoding="utf-8")
    for record in records:
        if record["event"] == "tool":
            print(f"{record['number']}. {record['tool']} {json.dumps(record['arguments'])}\n"
                  f"   {record['result']['feedback']}")
        elif record["event"] == "critique":
            verdict = (f"skipped ({record['skipped']})" if "skipped" in record else
                       "passed" if record["passed"] else "; ".join(record["problems"]))
            action = " -> rewrite requested" if record.get("revision_requested") else ""
            print(f"   review {record['number']}: {verdict}{action}")
    summary = records[-1]
    print(f"\n[{summary['status']}]\n{summary['rendered']}")
    print(f"\nTrace: {path}; mode: {records[0]['mode']}")


if __name__ == "__main__":
    main()
