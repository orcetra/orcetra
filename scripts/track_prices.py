#!/usr/bin/env python3
"""Track market price changes over time for pending predictions.

Snapshots current prices for all pending predictions.
Run via cron alongside daily_pipeline.sh.
"""
import json
import os
from datetime import datetime

import httpx

ROOT = os.path.join(os.path.dirname(__file__), "..")
BATCH_PATH = os.path.join(ROOT, "results", "batch_predictions.json")
PRICE_HIST = os.path.join(ROOT, "results", "price_snapshots.jsonl")


def main():
    with open(BATCH_PATH) as f:
        batch = json.load(f)

    preds = batch["predictions"]
    checked_path = os.path.join(ROOT, "results", "check_log.json")
    checked_ids = set()
    if os.path.exists(checked_path):
        with open(checked_path) as f:
            checked_ids = {c["condition_id"] for c in json.load(f)["checks"]}

    # Only track pending (unresolved) predictions
    pending = {k: v for k, v in preds.items() if k not in checked_ids}

    now = datetime.utcnow().isoformat()
    snapshot = {
        "timestamp": now,
        "count": len(pending),
        "prices": {}
    }

    # Fetch current prices from Gamma API (paginated)
    price_map = {}
    try:
        offset = 0
        while offset < 5000:
            resp = httpx.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": 100, "offset": offset, "active": True},
                timeout=30,
            )
            if resp.status_code != 200:
                break
            markets = resp.json()
            if not markets:
                break
            for m in markets:
                cid = m.get("conditionId") or m.get("condition_id")
                if cid and cid in pending:
                    price_map[cid] = m.get("outcomePrices", m.get("yes_price"))
            offset += 100
            if len(price_map) >= len(pending):
                break
    except Exception as e:
        print(f"Failed to fetch prices: {e}")

    updated = 0
    for cid, pred in pending.items():
        current_price = None

        # Try from API
        if cid in price_map:
            p = price_map[cid]
            if isinstance(p, str):
                try:
                    prices = json.loads(p)
                    current_price = float(prices[0]) if prices else None
                except (json.JSONDecodeError, IndexError):
                    pass
            elif isinstance(p, (int, float)):
                current_price = float(p)

        if current_price is not None:
            snapshot["prices"][cid] = {
                "q": pred["question"][:100],
                "tag": pred.get("tag", "?"),
                "price": round(current_price, 4),
                "our_pred": pred.get("our_prediction"),
                "original_market": pred.get("market_price"),
            }
            updated += 1

    # Append to JSONL (one snapshot per line)
    with open(PRICE_HIST, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    print(f"📈 Price snapshot: {updated} prices tracked at {now}")


if __name__ == "__main__":
    main()
