"""The tools the agent can call.

Design choice: the model never writes SQL. It can only call these narrow,
typed functions, and every function is locked to one customer_id that is set
by the application, not by the model. That gives least-privilege access: a
prompt such as "show me another customer's data" has no tool that could
answer it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Callable

from app import anomalies


class ToolError(ValueError):
    """Raised for bad tool arguments; the message is returned to the model."""


def _date(value: str, field: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        raise ToolError(f"{field} must be a date in YYYY-MM-DD format, got {value!r}") from None


def _range(start_date: str, end_date: str) -> tuple[str, str]:
    start, end = _date(start_date, "start_date"), _date(end_date, "end_date")
    if start > end:
        raise ToolError("start_date must be on or before end_date")
    return start, end


# JSON schemas describing each tool to the LLM.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "spending_by_category",
        "description": "Total spend and number of transactions per category between two dates (inclusive).",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "monthly_spending",
        "description": "Total spend per calendar month across all available history, optionally for one category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category name, or 'all' for every category."},
            },
        },
    },
    {
        "name": "top_merchants",
        "description": "Merchants ranked by total spend between two dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "description": "How many merchants to return (1-20)."},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "find_transactions",
        "description": "List individual transactions between two dates, with optional filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "category": {"type": "string", "description": "Exact category name to filter on."},
                "merchant_contains": {"type": "string", "description": "Case-insensitive part of a merchant name."},
                "min_amount": {"type": "number", "description": "Minimum amount in EUR."},
                "limit": {"type": "integer", "description": "Maximum rows to return (1-50)."},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "find_anomalies",
        "description": (
            "Run the fraud and anomaly rules over transactions between two dates. "
            "Returns each flagged transaction with the rules that fired and a plain-English reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "list_categories",
        "description": "The spending categories that exist in this customer's history.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class CustomerTools:
    """Tool implementations, scoped to a single customer."""

    def __init__(self, conn: sqlite3.Connection, customer_id: str):
        self.conn = conn
        self.customer_id = customer_id
        self._registry: dict[str, Callable[..., Any]] = {
            "spending_by_category": self.spending_by_category,
            "monthly_spending": self.monthly_spending,
            "top_merchants": self.top_merchants,
            "find_transactions": self.find_transactions,
            "find_anomalies": self.find_anomalies,
            "list_categories": self.list_categories,
        }

    # -- dispatch -----------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Run a tool by name. Errors are returned as data so the model can recover."""
        fn = self._registry.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name!r}"}
        try:
            return fn(**args)
        except ToolError as exc:
            return {"error": str(exc)}
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}

    def _rows(self, sql: str, params: tuple) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # -- tools --------------------------------------------------------------
    def spending_by_category(self, start_date: str, end_date: str) -> dict:
        start, end = _range(start_date, end_date)
        rows = self._rows(
            """SELECT category, ROUND(SUM(amount_eur), 2) AS total_eur, COUNT(*) AS transactions
               FROM transactions
               WHERE customer_id = ? AND substr(ts, 1, 10) BETWEEN ? AND ?
               GROUP BY category ORDER BY total_eur DESC""",
            (self.customer_id, start, end),
        )
        return {
            "start_date": start, "end_date": end,
            "total_eur": round(sum(r["total_eur"] for r in rows), 2),
            "categories": rows,
        }

    def monthly_spending(self, category: str = "all") -> dict:
        params: tuple = (self.customer_id,)
        where = ""
        if category and category.lower() != "all":
            where, params = " AND category = ?", (self.customer_id, category.lower())
        rows = self._rows(
            f"""SELECT substr(ts, 1, 7) AS month, ROUND(SUM(amount_eur), 2) AS total_eur, COUNT(*) AS transactions
                FROM transactions WHERE customer_id = ?{where}
                GROUP BY month ORDER BY month""",
            params,
        )
        return {"category": category or "all", "months": rows}

    def top_merchants(self, start_date: str, end_date: str, limit: int = 5) -> dict:
        start, end = _range(start_date, end_date)
        limit = max(1, min(int(limit), 20))
        rows = self._rows(
            """SELECT merchant, category, ROUND(SUM(amount_eur), 2) AS total_eur, COUNT(*) AS transactions
               FROM transactions
               WHERE customer_id = ? AND substr(ts, 1, 10) BETWEEN ? AND ?
               GROUP BY merchant ORDER BY total_eur DESC LIMIT ?""",
            (self.customer_id, start, end, limit),
        )
        return {"start_date": start, "end_date": end, "merchants": rows}

    def find_transactions(
        self,
        start_date: str,
        end_date: str,
        category: str = "",
        merchant_contains: str = "",
        min_amount: float = 0.0,
        limit: int = 20,
    ) -> dict:
        start, end = _range(start_date, end_date)
        limit = max(1, min(int(limit), 50))
        sql = """SELECT txn_id, ts, merchant, category, amount_eur, country, channel
                 FROM transactions
                 WHERE customer_id = ? AND substr(ts, 1, 10) BETWEEN ? AND ? AND amount_eur >= ?"""
        params: list[Any] = [self.customer_id, start, end, float(min_amount or 0)]
        if category:
            sql += " AND category = ?"
            params.append(category.lower())
        if merchant_contains:
            sql += " AND LOWER(merchant) LIKE ?"
            params.append(f"%{merchant_contains.lower()}%")
        count = self.conn.execute(f"SELECT COUNT(*) FROM ({sql})", params).fetchone()[0]
        rows = self._rows(sql + " ORDER BY ts DESC LIMIT ?", tuple(params + [limit]))
        return {"matching_transactions": count, "returned": len(rows), "transactions": rows}

    def find_anomalies(self, start_date: str, end_date: str) -> dict:
        start, end = _range(start_date, end_date)
        flagged = anomalies.detect_for_customer(self.conn, self.customer_id, start, end)
        return {"start_date": start, "end_date": end, "flagged_count": len(flagged), "flagged": flagged}

    def list_categories(self) -> dict:
        rows = self._rows(
            "SELECT DISTINCT category FROM transactions WHERE customer_id = ? ORDER BY category",
            (self.customer_id,),
        )
        return {"categories": [r["category"] for r in rows]}
