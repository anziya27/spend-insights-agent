"""Runtime settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order if the main model is overloaded or unavailable (comma-separated).
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
DB_PATH = Path(os.getenv("SPEND_DB_PATH", ROOT / "data" / "transactions.db"))

# Optional shared secret for the HTTP API (sent by n8n as the X-API-Key header).
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "").strip()

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))


def llm_enabled() -> bool:
    return bool(GEMINI_API_KEY)
