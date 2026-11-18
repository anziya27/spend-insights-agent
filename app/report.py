"""Weekly spending report, built for a Slack message (sent by the n8n workflow).

The numbers are always computed in code. The LLM only writes the short
narrative on top of those numbers, and if it is unavailable or fails, the
report falls back to a template, so the workflow never breaks because of the
model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from app import config
from app.db import connect_ro, customer_exists, latest_date
from app.tools import CustomerTools

log = logging.getLogger(__name__)

NARRATIVE_PROMPT = """Write a 2-3 sentence weekly spending summary for a bank customer, in plain,
friendly English. Use only the numbers in the JSON below; do not add any others.
Mention the total, the biggest change vs last week, and any flagged transactions.

JSON:
{facts}"""


def _d(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt(x: float) -> str:
    return f"EUR {x:,.2f}"


def build_facts(customer_id: str, as_of: str | None = None) -> dict:
    conn = connect_ro()
    if not customer_exists(conn, customer_id):
        raise KeyError(customer_id)
    tools = CustomerTools(conn, customer_id)
    end = _d(as_of) if as_of else _d(latest_date(conn, customer_id))
    week_start = end - timedelta(days=6)
    prev_end, prev_start = week_start - timedelta(days=1), week_start - timedelta(days=7)
    iso = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731

    this_week = tools.spending_by_category(iso(week_start), iso(end))
    last_week = tools.spending_by_category(iso(prev_start), iso(prev_end))
    prev_by_cat = {c["category"]: c["total_eur"] for c in last_week["categories"]}

    changes = sorted(
        (
            {"category": c["category"], "this_week_eur": c["total_eur"],
             "last_week_eur": prev_by_cat.get(c["category"], 0.0),
             "change_eur": round(c["total_eur"] - prev_by_cat.get(c["category"], 0.0), 2)}
            for c in this_week["categories"]
        ),
        key=lambda c: abs(c["change_eur"]), reverse=True,
    )
    flags = tools.find_anomalies(iso(end - timedelta(days=29)), iso(end))["flagged"]

    return {
        "customer_id": customer_id,
        "week": {"start": iso(week_start), "end": iso(end)},
        "total_eur": this_week["total_eur"],
        "previous_week_total_eur": last_week["total_eur"],
        "change_vs_previous_week_eur": round(this_week["total_eur"] - last_week["total_eur"], 2),
        "biggest_category_changes": changes[:3],
        "top_merchants": tools.top_merchants(iso(week_start), iso(end), limit=3)["merchants"],
        "flagged_last_30_days": _group_flags(flags),
    }


def _group_flags(flags: list[dict]) -> list[dict]:
    """Collapse a burst of rapid repeat charges into one line; keep other flags as they are."""
    grouped: list[dict] = []
    for f in flags:
        last = grouped[-1] if grouped else None
        if (last and "rapid_repeat" in f["rules"] and last.get("rule") == "rapid_repeat"
                and last["merchant"] == f["merchant"] and last["ts"][:10] == f["ts"][:10]):
            last["amount_eur"] = round(last["amount_eur"] + f["amount_eur"], 2)
            last["charges"] += 1
            last["reason"] = f"{last['charges']} charges at {f['merchant']} within minutes (possible card testing)"
            continue
        grouped.append({
            "ts": f["ts"], "merchant": f["merchant"], "amount_eur": f["amount_eur"], "charges": 1,
            "rule": f["rules"][0], "reason": f["reasons"][0],
        })
    return grouped


def template_narrative(facts: dict) -> str:
    diff = facts["change_vs_previous_week_eur"]
    direction = "more" if diff >= 0 else "less"
    text = (f"You spent {_fmt(facts['total_eur'])} this week, {_fmt(abs(diff))} {direction} "
            f"than the week before.")
    if facts["biggest_category_changes"]:
        c = facts["biggest_category_changes"][0]
        text += f" The biggest change was {c['category']} ({_fmt(c['this_week_eur'])} vs {_fmt(c['last_week_eur'])})."
    n = sum(f["charges"] for f in facts["flagged_last_30_days"])
    if n:
        text += f" {n} transaction(s) in the last 30 days look unusual. Please review them."
    return text


def llm_narrative(facts: dict, client=None) -> str:
    from google import genai

    client = client or genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL, contents=NARRATIVE_PROMPT.format(facts=json.dumps(facts, indent=2)),
    )
    return (resp.text or "").strip()


def slack_text(facts: dict, narrative: str) -> str:
    lines = [f"*Weekly spending: {facts['week']['start']} to {facts['week']['end']}* ({facts['customer_id']})",
             narrative, ""]
    if facts["top_merchants"]:
        lines.append("*Top merchants*")
        lines += [f"- {m['merchant']}: {_fmt(m['total_eur'])}" for m in facts["top_merchants"]]
    if facts["flagged_last_30_days"]:
        lines += ["", ":rotating_light: *Flagged in the last 30 days*"]
        lines += [f"- {f['ts'][:16]} {f['merchant']} {_fmt(f['amount_eur'])}: {f['reason']}"
                  for f in facts["flagged_last_30_days"]]
    return "\n".join(lines)


def build_weekly_report(customer_id: str, as_of: str | None = None, use_llm: bool = True) -> dict:
    facts = build_facts(customer_id, as_of)
    narrative, source = template_narrative(facts), "template"
    if use_llm and config.llm_enabled():
        try:
            narrative, source = llm_narrative(facts), "llm"
        except Exception as exc:  # the report must still go out if the model fails
            log.warning("LLM narrative failed, using template: %s", exc)
    return {
        **facts,
        "anomaly_count": sum(f["charges"] for f in facts["flagged_last_30_days"]),
        "narrative": narrative,
        "narrative_source": source,
        "slack_text": slack_text(facts, narrative),
    }
