"""Database access. The agent only ever gets a read-only connection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app import config


def ensure_db(path: Path | None = None) -> Path:
    path = Path(path or config.DB_PATH)
    if not path.exists():
        from data.generate_data import build

        build(path)
    return path


def connect_ro(path: Path | None = None) -> sqlite3.Connection:
    """Open the database in read-only mode, so no tool can modify data."""
    path = ensure_db(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def customer_exists(conn: sqlite3.Connection, customer_id: str) -> bool:
    return conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)).fetchone() is not None


def list_customers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT customer_id, first_name FROM customers ORDER BY customer_id").fetchall()
    return [dict(r) for r in rows]


def latest_date(conn: sqlite3.Connection, customer_id: str) -> str:
    row = conn.execute("SELECT MAX(substr(ts, 1, 10)) FROM transactions WHERE customer_id = ?", (customer_id,)).fetchone()
    return row[0]
