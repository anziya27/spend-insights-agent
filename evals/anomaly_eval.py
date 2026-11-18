"""Score the anomaly detector against the labelled synthetic fraud patterns.

No API key needed: this evaluates the deterministic rules only.

Usage:
    python -m evals.anomaly_eval
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.anomalies import detect_for_customer
from app.db import connect_ro

RESULTS_DIR = Path(__file__).resolve().parent / "results"
START, END = "2026-05-01", "2026-10-31"


def evaluate() -> dict:
    conn = connect_ro()
    tp = fp = fn = 0
    recall_by_type: Counter = Counter()
    total_by_type: Counter = Counter()
    fp_rules: Counter = Counter()

    for (cid,) in conn.execute("SELECT customer_id FROM customers ORDER BY customer_id"):
        flagged = {f["txn_id"]: f for f in detect_for_customer(conn, cid, START, END)}
        truth = {r["txn_id"]: r["anomaly_type"] for r in conn.execute(
            "SELECT txn_id, anomaly_type FROM transactions WHERE customer_id=? AND is_injected_anomaly=1", (cid,))}
        for txn_id, kind in truth.items():
            total_by_type[kind] += 1
            if txn_id in flagged:
                tp += 1
                recall_by_type[kind] += 1
            else:
                fn += 1
        for txn_id, f in flagged.items():
            if txn_id not in truth:
                fp += 1
                fp_rules.update(f["rules"])

    total_txns = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "transactions_scanned": total_txns,
        "labelled_anomalies": tp + fn,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "false_positive_rate": round(fp / max(1, total_txns - (tp + fn)), 4),
        "recall_by_type": {k: round(recall_by_type[k] / v, 3) for k, v in sorted(total_by_type.items())},
        "false_positives_by_rule": dict(fp_rules),
    }


def main() -> int:
    result = evaluate()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "anomaly_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
