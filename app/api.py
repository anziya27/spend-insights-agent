"""HTTP API used by n8n (and handy for demos at http://localhost:8000/docs).

Run:  uvicorn app.api:app --reload
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.agent import AgentError, SpendAgent
from app.anomalies import detect_for_customer
from app.db import connect_ro, customer_exists, latest_date, list_customers
from app.report import build_weekly_report

logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Spend Insights Agent",
    description="An LLM agent that explains card spending and flags suspicious transactions (synthetic data).",
    version="1.0.0",
)
agent = SpendAgent()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.SERVICE_API_KEY and x_api_key != config.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def check_customer(customer_id: str) -> None:
    if not customer_exists(connect_ro(), customer_id):
        raise HTTPException(status_code=404, detail=f"Unknown customer {customer_id}")


class AskRequest(BaseModel):
    customer_id: str = Field(examples=["C001"])
    question: str = Field(min_length=3, max_length=1000, examples=["How much did I spend on groceries in October?"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_enabled": config.llm_enabled(), "model": config.GEMINI_MODEL}


@app.get("/customers", dependencies=[Depends(require_api_key)])
def customers() -> list[dict]:
    return list_customers(connect_ro())


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(req: AskRequest) -> dict:
    check_customer(req.customer_id)
    try:
        return agent.ask(req.customer_id, req.question).to_dict()
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/reports/weekly/{customer_id}", dependencies=[Depends(require_api_key)])
def weekly_report(customer_id: str, as_of: str | None = None, use_llm: bool = True) -> dict:
    check_customer(customer_id)
    return build_weekly_report(customer_id, as_of=as_of, use_llm=use_llm)


@app.get("/anomalies/{customer_id}", dependencies=[Depends(require_api_key)])
def anomalies(customer_id: str, start_date: str = "2026-05-01", end_date: str | None = None) -> dict:
    check_customer(customer_id)
    conn = connect_ro()
    end_date = end_date or latest_date(conn, customer_id)
    flagged = detect_for_customer(conn, customer_id, start_date, end_date)
    return {"customer_id": customer_id, "start_date": start_date, "end_date": end_date,
            "flagged_count": len(flagged), "flagged": flagged}
