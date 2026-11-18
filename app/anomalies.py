"""Rule-based anomaly detection over a customer's card transactions.

Deliberately simple and explainable: every flag carries the rule that fired and
a human-readable reason, which matters in financial services where a flagged
customer or an analyst needs to understand *why*.

Rules
-----
1. amount_outlier       amount is > Z_THRESHOLD standard deviations above the
                        customer's own history in that category
2. new_country_high     first-ever transaction in a country, for >= NEW_COUNTRY_MIN EUR
3. rapid_repeat         >= RAPID_COUNT charges at one merchant within RAPID_WINDOW
                        (a classic card-testing pattern)
4. night_high_value     online charge >= NIGHT_MIN EUR between 00:00 and 05:00
"""

from __future__ import annotations

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

Z_THRESHOLD = 3.5
MIN_HISTORY = 5
OUTLIER_MIN_EUR = 50.0
NEW_COUNTRY_MIN = 250.0
RAPID_COUNT = 3
RAPID_WINDOW = timedelta(minutes=10)
NIGHT_MIN = 200.0
NIGHT_HOURS = range(0, 5)

TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FMT)


def detect(rows: list[dict], start_date: str, end_date: str) -> list[dict]:
    """Return flagged transactions dated between start_date and end_date (inclusive).

    `rows` is the customer's full history up to end_date, sorted by time, so the
    rules can compare each transaction against what came before it.
    """
    flags: dict[str, dict] = {}

    def flag(txn: dict, rule: str, reason: str) -> None:
        entry = flags.setdefault(txn["txn_id"], {
            "txn_id": txn["txn_id"], "ts": txn["ts"], "merchant": txn["merchant"],
            "category": txn["category"], "amount_eur": txn["amount_eur"],
            "country": txn["country"], "channel": txn["channel"], "rules": [], "reasons": [],
        })
        if rule in entry["rules"]:
            entry["reasons"][entry["rules"].index(rule)] = reason  # keep the latest (fullest) reason
        else:
            entry["rules"].append(rule)
            entry["reasons"].append(reason)

    def in_window(txn: dict) -> bool:
        return start_date <= txn["ts"][:10] <= end_date

    history_by_cat: dict[str, list[float]] = defaultdict(list)
    seen_countries: set[str] = set()
    recent_by_merchant: dict[str, list[dict]] = defaultdict(list)

    for txn in rows:
        amount, ts = txn["amount_eur"], _parse(txn["ts"])

        if in_window(txn):
            hist = history_by_cat[txn["category"]]
            if len(hist) >= MIN_HISTORY and amount >= OUTLIER_MIN_EUR:
                mean, sd = statistics.fmean(hist), statistics.pstdev(hist)
                if sd > 0 and (amount - mean) / sd > Z_THRESHOLD:
                    flag(txn, "amount_outlier",
                         f"EUR {amount:.2f} is {(amount - mean) / sd:.1f} std devs above usual "
                         f"{txn['category']} spend (avg EUR {mean:.2f})")

            if seen_countries and txn["country"] not in seen_countries and amount >= NEW_COUNTRY_MIN:
                flag(txn, "new_country_high", f"First ever transaction in {txn['country']}, for EUR {amount:.2f}")

            if txn["channel"] == "online" and ts.hour in NIGHT_HOURS and amount >= NIGHT_MIN:
                flag(txn, "night_high_value", f"EUR {amount:.2f} online at {ts:%H:%M}")

        burst = [t for t in recent_by_merchant[txn["merchant"]] if ts - _parse(t["ts"]) <= RAPID_WINDOW]
        burst.append(txn)
        recent_by_merchant[txn["merchant"]] = burst
        if len(burst) >= RAPID_COUNT:
            for t in burst:
                if in_window(t):
                    flag(t, "rapid_repeat",
                         f"{len(burst)} charges at {txn['merchant']} within {int(RAPID_WINDOW.total_seconds() // 60)} minutes")

        history_by_cat[txn["category"]].append(amount)
        seen_countries.add(txn["country"])

    return sorted(flags.values(), key=lambda f: f["ts"])


def detect_for_customer(conn: sqlite3.Connection, customer_id: str, start_date: str, end_date: str) -> list[dict]:
    rows = conn.execute(
        """SELECT txn_id, ts, merchant, category, amount_eur, country, channel
           FROM transactions WHERE customer_id = ? AND substr(ts, 1, 10) <= ?
           ORDER BY ts""",
        (customer_id, end_date),
    ).fetchall()
    return detect([dict(r) for r in rows], start_date, end_date)
