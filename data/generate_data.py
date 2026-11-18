"""Generate a synthetic card-transaction dataset for the Spend Insights Agent.

Everything here is fake: customers, merchants and amounts are generated from a
fixed random seed so results are reproducible. A small number of fraud-like
patterns are injected and labelled, which lets us measure how well the anomaly
detector works (see evals/anomaly_eval.py).

Usage:
    python -m data.generate_data            # writes data/transactions.db + .csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
START = datetime(2026, 5, 1)
END = datetime(2026, 10, 31, 23, 59)
N_CUSTOMERS = 25
DATA_DIR = Path(__file__).resolve().parent

FIRST_NAMES = [
    "Lucia", "Marc", "Amina", "Jordi", "Sofia", "Pablo", "Grace", "Nuria", "Daniel", "Wanjiru",
    "Elena", "Hugo", "Fatima", "Pau", "Chloe", "Ivan", "Zara", "Oriol", "Maya", "Tomas",
    "Aisha", "Laia", "Kofi", "Irene", "Leo",
]

# category -> (merchants, mean amount EUR, std dev, expected transactions per month)
CATEGORIES: dict[str, tuple[list[str], float, float, float]] = {
    "groceries": (["Mercadona", "Carrefour", "Lidl", "Aldi", "Bonpreu"], 42.0, 18.0, 9),
    "restaurants": (["Honest Greens", "Five Guys", "La Tagliatella", "Sushi Shop", "Cafe del Barri"], 24.0, 11.0, 6),
    "transport": (["Uber", "Cabify", "TMB Metro", "Renfe", "Bolt"], 11.0, 6.0, 10),
    "shopping": (["Zara", "Amazon", "Decathlon", "IKEA", "El Corte Ingles"], 55.0, 30.0, 3),
    "entertainment": (["Cinesa", "Ticketmaster", "Steam", "Bowling Barcelona"], 22.0, 10.0, 2),
    "health": (["Farmacia Central", "DKV Clinic", "Optica Universitaria"], 28.0, 14.0, 1),
}

# Fixed monthly bills: (merchant, category, amount, day of month)
SUBSCRIPTIONS = [
    ("Spotify", "subscriptions", 10.99, 3),
    ("Netflix", "subscriptions", 13.99, 9),
    ("iCloud", "subscriptions", 2.99, 15),
    ("Endesa", "utilities", 64.0, 5),
    ("Movistar", "utilities", 45.0, 12),
]

TRIP_COUNTRIES = ["PT", "FR", "IT"]
TRIP_MERCHANTS = ["Cafe Lisboa", "Brasserie du Port", "Trattoria Roma", "Metro Ticket", "Local Bakery"]

COLUMNS = [
    "txn_id", "customer_id", "ts", "merchant", "category", "amount_eur",
    "country", "channel", "is_injected_anomaly", "anomaly_type",
]


def _clamp_amount(x: float) -> float:
    return round(max(1.0, x), 2)


def _random_daytime(rng: random.Random, day: datetime) -> datetime:
    return day.replace(hour=rng.randint(8, 22), minute=rng.randint(0, 59), second=rng.randint(0, 59))


def _month_starts() -> list[datetime]:
    months, d = [], START
    while d <= END:
        months.append(d)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def _days_in_month(month_start: datetime) -> int:
    nxt = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (nxt - month_start).days


def _normal_transactions(rng: random.Random, customer_id: str) -> list[dict]:
    rows: list[dict] = []
    spend_level = rng.uniform(0.6, 1.8)  # some customers spend more than others
    has_netflix = rng.random() < 0.7

    for month in _month_starts():
        n_days = _days_in_month(month)

        for category, (merchants, mean, sd, per_month) in CATEGORIES.items():
            n = max(0, int(rng.gauss(per_month, per_month * 0.25)))
            favourite = rng.choice(merchants)
            for _ in range(n):
                day = month + timedelta(days=rng.randrange(n_days))
                merchant = favourite if rng.random() < 0.5 else rng.choice(merchants)
                channel = "online" if merchant in {"Amazon", "Steam", "Ticketmaster"} else "card_present"
                rows.append({
                    "customer_id": customer_id,
                    "ts": _random_daytime(rng, day),
                    "merchant": merchant,
                    "category": category,
                    "amount_eur": _clamp_amount(rng.gauss(mean * spend_level, sd * spend_level)),
                    "country": "ES",
                    "channel": channel,
                })

        for merchant, category, amount, dom in SUBSCRIPTIONS:
            if merchant == "Netflix" and not has_netflix:
                continue
            ts = month.replace(day=dom, hour=6, minute=0, second=0)
            if category == "utilities":
                amount = _clamp_amount(rng.gauss(amount, amount * 0.15))
            rows.append({
                "customer_id": customer_id, "ts": ts, "merchant": merchant, "category": category,
                "amount_eur": amount, "country": "ES", "channel": "online",
            })

    # Roughly half the customers take a short trip abroad: flights booked from
    # Spain, then small everyday spending in the destination country.
    if rng.random() < 0.5:
        trip_start = START + timedelta(days=rng.randrange(30, 150))
        country = rng.choice(TRIP_COUNTRIES)
        rows.append({
            "customer_id": customer_id, "ts": _random_daytime(rng, trip_start - timedelta(days=20)),
            "merchant": rng.choice(["Vueling", "Ryanair"]), "category": "travel",
            "amount_eur": _clamp_amount(rng.gauss(160, 40)), "country": "ES", "channel": "online",
        })
        for d in range(rng.randint(3, 5)):
            for _ in range(rng.randint(1, 3)):
                rows.append({
                    "customer_id": customer_id, "ts": _random_daytime(rng, trip_start + timedelta(days=d)),
                    "merchant": rng.choice(TRIP_MERCHANTS), "category": "restaurants",
                    "amount_eur": _clamp_amount(rng.gauss(22, 10)), "country": country, "channel": "card_present",
                })
    return rows


def _inject_anomalies(rng: random.Random, customer_id: str, force_all: bool) -> list[dict]:
    """Fraud-like patterns, labelled so the detector can be scored."""
    rows: list[dict] = []

    def add(ts, merchant, category, amount, country, channel, kind):
        rows.append({
            "customer_id": customer_id, "ts": ts, "merchant": merchant, "category": category,
            "amount_eur": round(amount, 2), "country": country, "channel": channel,
            "is_injected_anomaly": 1, "anomaly_type": kind,
        })

    # 1) Large online purchase from a country the customer has never used.
    if force_all or rng.random() < 0.35:
        day = datetime(2026, 9, 1) + timedelta(days=rng.randrange(55))
        add(_random_daytime(rng, day), rng.choice(["MegaStore Online Ltd", "TechHub Direct"]),
            "electronics", rng.uniform(900, 2500), rng.choice(["NG", "US", "CN"]), "online", "large_foreign_purchase")

    # 2) Card testing: a burst of tiny charges at one merchant within minutes.
    if force_all or rng.random() < 0.3:
        start = _random_daytime(rng, datetime(2026, 10, 1) + timedelta(days=rng.randrange(28)))
        for i in range(rng.randint(4, 6)):
            add(start + timedelta(minutes=i, seconds=rng.randint(0, 50)), "QuickPay*Digital", "online_services",
                rng.uniform(0.5, 2.0), "GB", "online", "card_testing")

    # 3) High-value online purchase in the middle of the night.
    if force_all or rng.random() < 0.3:
        day = datetime(2026, 10, 1) + timedelta(days=rng.randrange(30))
        add(day.replace(hour=rng.randint(1, 4), minute=rng.randint(0, 59)), "LuxuryGoods Online",
            "shopping", rng.uniform(300, 800), "ES", "online", "night_high_value")
    return rows


def generate(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(1, N_CUSTOMERS + 1):
        cid = f"C{i:03d}"
        rows.extend(_normal_transactions(rng, cid))
        # C001 is the demo customer and always gets every anomaly type.
        rows.extend(_inject_anomalies(rng, cid, force_all=(cid == "C001")))

    rows.sort(key=lambda r: (r["customer_id"], r["ts"]))
    for n, r in enumerate(rows, start=1):
        r["txn_id"] = f"T{n:06d}"
        r["ts"] = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
        r.setdefault("is_injected_anomaly", 0)
        r.setdefault("anomaly_type", "")
    return rows


def build(db_path: Path | None = None, csv_path: Path | None = None, seed: int = SEED) -> Path:
    db_path = Path(db_path or DATA_DIR / "transactions.db")
    csv_path = Path(csv_path or db_path.with_suffix(".csv"))
    rows = generate(seed)

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            first_name  TEXT NOT NULL,
            home_country TEXT NOT NULL
        );
        CREATE TABLE transactions (
            txn_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            ts TEXT NOT NULL,
            merchant TEXT NOT NULL,
            category TEXT NOT NULL,
            amount_eur REAL NOT NULL,
            country TEXT NOT NULL,
            channel TEXT NOT NULL,
            is_injected_anomaly INTEGER NOT NULL DEFAULT 0,
            anomaly_type TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX idx_txn_customer_ts ON transactions(customer_id, ts);
        """
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, 'ES')",
        [(f"C{i:03d}", FIRST_NAMES[i - 1]) for i in range(1, N_CUSTOMERS + 1)],
    )
    conn.executemany(
        f"INSERT INTO transactions ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
        [tuple(r[c] for c in COLUMNS) for r in rows],
    )
    conn.commit()
    conn.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return db_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DATA_DIR / "transactions.db")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    path = build(args.db, seed=args.seed)
    print(f"Wrote {path} and {path.with_suffix('.csv')}")
