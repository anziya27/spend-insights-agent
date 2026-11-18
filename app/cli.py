"""Command-line interface.

Examples:
    python -m app.cli ask "How much did I spend on groceries in October 2026?"
    python -m app.cli ask "Anything suspicious on my card recently?" --customer C001 --trace
    python -m app.cli report --customer C001
    python -m app.cli anomalies --customer C001
"""

from __future__ import annotations

import argparse
import json
import sys

from app.agent import AgentError, SpendAgent
from app.anomalies import detect_for_customer
from app.db import connect_ro, latest_date
from app.report import build_weekly_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spend-insights")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask the agent a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--customer", default="C001")
    p_ask.add_argument("--trace", action="store_true", help="Print every tool call the agent made")

    p_rep = sub.add_parser("report", help="Print the weekly Slack report")
    p_rep.add_argument("--customer", default="C001")
    p_rep.add_argument("--no-llm", action="store_true")

    p_an = sub.add_parser("anomalies", help="List flagged transactions")
    p_an.add_argument("--customer", default="C001")
    p_an.add_argument("--start", default="2026-05-01")

    args = parser.parse_args(argv)

    if args.command == "ask":
        try:
            result = SpendAgent().ask(args.customer, args.question)
        except AgentError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(result.answer)
        if args.trace:
            print(f"\n--- trace: {result.steps} model steps, {result.latency_ms} ms ---")
            for t in result.tool_calls:
                print(f"[step {t.step}] {t.tool}({json.dumps(t.args)}) -> {t.result_preview[:160]}")
    elif args.command == "report":
        print(build_weekly_report(args.customer, use_llm=not args.no_llm)["slack_text"])
    elif args.command == "anomalies":
        conn = connect_ro()
        for f in detect_for_customer(conn, args.customer, args.start, latest_date(conn, args.customer)):
            print(f"{f['ts']}  {f['merchant']:<22} EUR {f['amount_eur']:>9.2f}  {'; '.join(f['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
