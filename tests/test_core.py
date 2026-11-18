"""Unit tests. None of these call a real LLM, so they run in CI without an API key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app import anomalies
from app.agent import SpendAgent
from app.api import app
from app.db import connect_ro
from app.report import build_weekly_report
from app.tools import CustomerTools


@pytest.fixture(scope="module")
def tools() -> CustomerTools:
    return CustomerTools(connect_ro(), "C001")


# --- tools -------------------------------------------------------------------

def test_category_totals_match_raw_sql(tools):
    result = tools.spending_by_category("2026-09-01", "2026-09-30")
    raw = tools.conn.execute(
        "SELECT ROUND(SUM(amount_eur), 2) FROM transactions WHERE customer_id='C001' AND substr(ts,1,7)='2026-09'"
    ).fetchone()[0]
    assert result["total_eur"] == pytest.approx(raw, abs=0.05)


def test_tools_are_scoped_to_one_customer(tools):
    rows = tools.find_transactions("2026-05-01", "2026-10-31", limit=50)["transactions"]
    ids = {r["txn_id"] for r in rows}
    owners = {r[0] for r in tools.conn.execute(
        f"SELECT customer_id FROM transactions WHERE txn_id IN ({','.join('?' * len(ids))})", tuple(ids))}
    assert owners == {"C001"}


def test_bad_dates_return_an_error_not_an_exception(tools):
    assert "error" in tools.call("spending_by_category", {"start_date": "last month", "end_date": "2026-10-31"})
    assert "error" in tools.call("spending_by_category", {"start_date": "2026-10-31", "end_date": "2026-10-01"})
    assert "error" in tools.call("drop_table", {})


def test_database_is_read_only():
    conn = connect_ro()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM transactions")


# --- anomaly rules -------------------------------------------------------------

def _txn(i, ts, merchant="Shop", amount=20.0, category="shopping", country="ES", channel="card_present"):
    return {"txn_id": f"X{i}", "ts": ts, "merchant": merchant, "category": category,
            "amount_eur": amount, "country": country, "channel": channel}


def test_rapid_repeat_flags_card_testing():
    rows = [_txn(i, f"2026-10-01 10:0{i}:00", merchant="QuickPay", amount=1.0) for i in range(4)]
    flagged = anomalies.detect(rows, "2026-10-01", "2026-10-01")
    assert len(flagged) == 4 and all("rapid_repeat" in f["rules"] for f in flagged)


def test_night_and_new_country_rules():
    rows = [_txn(0, "2026-10-01 12:00:00"),
            _txn(1, "2026-10-02 03:10:00", amount=500, channel="online"),
            _txn(2, "2026-10-03 12:00:00", amount=1200, country="NG", channel="online")]
    rules = {f["txn_id"]: f["rules"] for f in anomalies.detect(rows, "2026-10-01", "2026-10-31")}
    assert "night_high_value" in rules["X1"]
    assert "new_country_high" in rules["X2"]
    assert "X0" not in rules


def test_detector_finds_every_injected_anomaly_for_demo_customer():
    conn = connect_ro()
    flagged = {f["txn_id"] for f in anomalies.detect_for_customer(conn, "C001", "2026-05-01", "2026-10-31")}
    truth = {r[0] for r in conn.execute(
        "SELECT txn_id FROM transactions WHERE customer_id='C001' AND is_injected_anomaly=1")}
    assert truth <= flagged


# --- agent loop (with a fake model) -------------------------------------------

class FakeModels:
    """Asks for one tool call, then answers using the tool's result."""

    def __init__(self):
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.calls == 1:
            call = types.FunctionCall(name="spending_by_category",
                                      args={"start_date": "2026-09-01", "end_date": "2026-09-30"})
            content = types.Content(role="model", parts=[types.Part(function_call=call)])
            return SimpleNamespace(function_calls=[call], candidates=[SimpleNamespace(content=content)], text=None)
        result = contents[-1].parts[0].function_response.response["result"]
        return SimpleNamespace(function_calls=None, candidates=[], text=f"You spent EUR {result['total_eur']}.")


def test_agent_loop_calls_tools_and_records_trace(tools):
    agent = SpendAgent(client=SimpleNamespace(models=FakeModels()), model="fake")
    result = agent.ask("C001", "How much did I spend in September?")
    expected = tools.spending_by_category("2026-09-01", "2026-09-30")["total_eur"]
    assert str(expected) in result.answer
    assert [t.tool for t in result.tool_calls] == ["spending_by_category"]
    assert result.steps == 2


# --- report and API -------------------------------------------------------------

def test_weekly_report_without_llm():
    report = build_weekly_report("C001", use_llm=False)
    assert report["narrative_source"] == "template"
    assert report["anomaly_count"] >= 1
    assert "Weekly spending" in report["slack_text"]


def test_api_endpoints():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/reports/weekly/C001?use_llm=false").status_code == 200
    assert client.get("/anomalies/C001").json()["flagged_count"] >= 1
    assert client.get("/reports/weekly/NOPE").status_code == 404
