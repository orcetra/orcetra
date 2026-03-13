#!/usr/bin/env python3
"""Batch live tracker: predict ALL active markets, check daily for resolution.
Goal: 100s of predictions over 2 weeks for statistical significance."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console

from murmur.prices import get_price_context
from murmur.models import Event, OutcomeToken

console = Console()
RESULTS_DIR = Path("results")
BATCH_FILE = RESULTS_DIR / "batch_predictions.json"


def _parse_json(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return default or []
    return value if value else (default or [])


def load_batch():
    if BATCH_FILE.exists():
        with open(BATCH_FILE) as f:
            return json.load(f)
    return {"predictions": {}, "stats": {}, "version": 2}


def save_batch(data):
    RESULTS_DIR.mkdir(exist_ok=True)
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(BATCH_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


async def fetch_all_active_markets(tags=None, max_events=200):
    """Fetch active markets across categories."""
    if tags is None:
        tags = ["sports", "commodities", "economy", "golf"]

    all_markets = []
    seen_conditions = set()

    async with httpx.AsyncClient(timeout=30) as client:
        for tag in tags:
            try:
                r = await client.get("https://gamma-api.polymarket.com/events", params={
                    "limit": 100, "active": True, "tag_slug": tag,
                    "order": "startDate", "ascending": False,
                })
                events = r.json()
            except:
                continue

            for e in events[:max_events]:
                for m in e.get("markets", []):
                    if m.get("closed"):
                        continue

                    cid = m.get("conditionId", "")
                    if cid in seen_conditions:
                        continue
                    seen_conditions.add(cid)

                    prices = _parse_json(m.get("outcomePrices"), [])
                    token_ids = _parse_json(m.get("clobTokenIds"), [])
                    if not prices or not token_ids:
                        continue

                    yes_p = float(prices[0])
                    # Skip extreme odds (no edge) and 50/50 (no price discovery)
                    if yes_p > 0.92 or yes_p < 0.08:
                        continue

                    all_markets.append({
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "condition_id": cid,
                        "token_id": token_ids[0],
                        "yes_price": yes_p,
                        "volume": float(m.get("volume", 0) or 0),
                        "end_date": m.get("endDateIso") or m.get("endDate", ""),
                        "event_title": e.get("title", ""),
                        "description": m.get("description", "")[:300],
                        "tag": tag,
                    })

    return all_markets


def rule_predict(title, description, market_price, volume):
    """Fast rule-based prediction (no LLM call = can do 100s instantly).
    Uses evolved strategy + real-time price context."""
    p = market_price

    # Get real-time price context
    price_ctx = get_price_context(title)

    if price_ctx:
        # We have real data — extract price and compute distance to threshold
        import re
        # e.g. "Will Crude Oil hit $110?" + "CURRENT CRUDE OIL PRICE: $98.43"
        current_match = re.search(r'\$([0-9,.]+)', price_ctx)
        threshold_match = re.search(r'\$([0-9,.]+)', title)

        if current_match and threshold_match:
            try:
                current = float(current_match.group(1).replace(',', ''))
                threshold = float(threshold_match.group(1).replace(',', ''))

                # Distance-based prediction
                if "hit" in title.lower() or "reach" in title.lower() or "above" in title.lower():
                    ratio = current / threshold if threshold > 0 else 1.0
                    if ratio >= 1.0:
                        p = 0.92  # Already past threshold
                    elif ratio >= 0.98:
                        p = 0.85  # Very close
                    elif ratio >= 0.95:
                        p = 0.70
                    elif ratio >= 0.90:
                        p = 0.55
                    elif ratio >= 0.80:
                        p = 0.35
                    else:
                        p = 0.15
                elif "dip" in title.lower() or "below" in title.lower():
                    ratio = threshold / current if current > 0 else 1.0
                    if ratio >= 1.0:
                        p = 0.92
                    elif ratio >= 0.95:
                        p = 0.70
                    elif ratio >= 0.90:
                        p = 0.45
                    else:
                        p = 0.20
            except (ValueError, ZeroDivisionError):
                pass

    # Market anchor blend (40% our signal, 60% market)
    p = market_price * 0.55 + p * 0.45
    p = max(0.01, min(0.99, p))

    # Sports: mostly trust market (we have no edge on sports)
    title_lower = title.lower()
    is_sports = any(kw in title_lower for kw in [
        "win on", "o/u", "handicap", "vs.", "set 1", "match o/u",
        "nba", "premier", "la liga", "champions", "total sets",
        "total kills", "series:", "game 1"
    ])
    if is_sports:
        # Slight contrarian: markets on sports slightly favor favorites
        if market_price > 0.65:
            p = market_price - 0.03
        elif market_price < 0.35:
            p = market_price + 0.03
        else:
            p = market_price
        p = max(0.01, min(0.99, p))

    confidence = 0.5
    if price_ctx:
        confidence = 0.7
    if is_sports:
        confidence = 0.4  # Low confidence on sports

    return {
        "probability": round(p, 4),
        "confidence": round(confidence, 2),
        "has_price_data": price_ctx is not None,
        "is_sports": is_sports,
    }


def predict_batch():
    """Make predictions on ALL available markets."""
    console.print("[bold]🔮 Murmur Batch Tracker — Mass Prediction[/bold]\n")

    markets = asyncio.run(fetch_all_active_markets())
    console.print(f"Found {len(markets)} active markets\n")

    batch = load_batch()
    existing = set(batch["predictions"].keys())
    new_count = 0
    skipped = 0

    by_tag = {}
    t0 = time.time()

    for m in markets:
        key = m["condition_id"] or m["slug"]
        if key in existing:
            skipped += 1
            continue

        result = rule_predict(
            m["question"], m.get("description", ""),
            m["yes_price"], m["volume"]
        )

        batch["predictions"][key] = {
            "question": m["question"],
            "slug": m["slug"],
            "condition_id": m["condition_id"],
            "token_id": m["token_id"],
            "tag": m["tag"],
            "market_price": m["yes_price"],
            "our_prediction": result["probability"],
            "confidence": result["confidence"],
            "has_price_data": result["has_price_data"],
            "is_sports": result["is_sports"],
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "end_date": m["end_date"],
            "volume": m["volume"],
            "resolved": False,
            "actual": None,
        }
        new_count += 1
        tag = m["tag"]
        by_tag[tag] = by_tag.get(tag, 0) + 1

    dt = time.time() - t0
    save_batch(batch)

    console.print(f"✅ {new_count} new predictions in {dt:.1f}s (skipped {skipped} existing)")
    console.print(f"   Total tracked: {len(batch['predictions'])}")
    for tag, n in sorted(by_tag.items(), key=lambda x: -x[1]):
        console.print(f"   {tag:>15s}: {n}")

    # Show signal distribution
    active = [p for p in batch["predictions"].values() if not p["resolved"]]
    buy_yes = sum(1 for p in active if p["our_prediction"] - p["market_price"] > 0.05)
    buy_no = sum(1 for p in active if p["market_price"] - p["our_prediction"] > 0.05)
    hold = len(active) - buy_yes - buy_no
    console.print(f"\n   Signals: ↑BUY YES: {buy_yes} | ↓BUY NO: {buy_no} | ≈HOLD: {hold}")


def check_batch():
    """Check all predictions for resolution."""
    console.print("[bold]📊 Murmur Batch Tracker — Checking Results[/bold]\n")

    batch = load_batch()
    unresolved = {k: v for k, v in batch["predictions"].items() if not v.get("resolved")}
    console.print(f"Checking {len(unresolved)} unresolved predictions...")

    resolved_count = 0
    checked = 0

    # Check in batches via slug
    for key, pred in unresolved.items():
        slug = pred.get("slug")
        if not slug:
            continue

        try:
            r = httpx.get("https://gamma-api.polymarket.com/markets", params={
                "slug": slug,
            }, timeout=10)
            data = r.json()
            if not data:
                continue

            m = data[0] if isinstance(data, list) else data
            if not m.get("closed"):
                checked += 1
                continue

            prices = _parse_json(m.get("outcomePrices"), [])
            if not prices:
                continue

            yes_final = float(prices[0])
            actual = 1.0 if yes_final > 0.5 else 0.0

            pred["resolved"] = True
            pred["actual"] = actual
            pred["final_yes_price"] = yes_final
            pred["resolved_at"] = datetime.now(timezone.utc).isoformat()
            pred["our_brier"] = round((pred["our_prediction"] - actual) ** 2, 4)
            pred["mkt_brier"] = round((pred["market_price"] - actual) ** 2, 4)
            pred["beat_market"] = pred["our_brier"] < pred["mkt_brier"]

            resolved_count += 1
            checked += 1

        except Exception:
            pass

        time.sleep(0.15)  # Rate limit

        if checked % 50 == 0:
            console.print(f"  Checked {checked}... ({resolved_count} resolved)")

    save_batch(batch)

    # Summary
    all_resolved = [p for p in batch["predictions"].values() if p.get("resolved")]
    console.print(f"\n{'='*60}")
    console.print(f"[bold]BATCH RESULTS[/bold]")
    console.print(f"  Total tracked:  {len(batch['predictions'])}")
    console.print(f"  Resolved:       {len(all_resolved)}")
    console.print(f"  Pending:        {len(batch['predictions']) - len(all_resolved)}")

    if all_resolved:
        avg_ours = sum(p["our_brier"] for p in all_resolved) / len(all_resolved)
        avg_mkt = sum(p["mkt_brier"] for p in all_resolved) / len(all_resolved)
        wins = sum(1 for p in all_resolved if p["beat_market"])
        ties = sum(1 for p in all_resolved if p["our_brier"] == p["mkt_brier"])

        console.print(f"\n  Our Brier:     {avg_ours:.4f}")
        console.print(f"  Market Brier:  {avg_mkt:.4f}")
        console.print(f"  Beat Rate:     {wins}/{len(all_resolved)} ({wins/len(all_resolved):.0%})")
        console.print(f"  Ties:          {ties}")

        if avg_ours < avg_mkt:
            console.print(f"\n  [green bold]🎯 BEATING THE MARKET![/green bold]")
        else:
            console.print(f"\n  [yellow]Market leads by {avg_ours-avg_mkt:.4f}[/yellow]")

        # Breakdown by category
        by_tag = {}
        for p in all_resolved:
            tag = p.get("tag", "unknown")
            if tag not in by_tag:
                by_tag[tag] = {"n": 0, "wins": 0, "brier": 0, "mkt_brier": 0}
            by_tag[tag]["n"] += 1
            by_tag[tag]["wins"] += 1 if p["beat_market"] else 0
            by_tag[tag]["brier"] += p["our_brier"]
            by_tag[tag]["mkt_brier"] += p["mkt_brier"]

        if by_tag:
            console.print(f"\n  By category:")
            for tag, s in sorted(by_tag.items()):
                avg_b = s["brier"] / s["n"]
                avg_m = s["mkt_brier"] / s["n"]
                console.print(
                    f"    {tag:>12s}: {s['wins']}/{s['n']} beat | "
                    f"B={avg_b:.4f} vs M={avg_m:.4f}"
                )


def status_command():
    """Quick status."""
    batch = load_batch()
    total = len(batch["predictions"])
    resolved = sum(1 for p in batch["predictions"].values() if p.get("resolved"))
    console.print(f"Batch: {total} total, {resolved} resolved, {total-resolved} pending")
    console.print(f"Last updated: {batch.get('last_updated', 'never')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Murmur Batch Tracker")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("predict", help="Mass predict all active markets")
    sub.add_parser("check", help="Check for resolved predictions")
    sub.add_parser("status", help="Quick status")

    args = parser.parse_args()
    if args.command == "predict":
        predict_batch()
    elif args.command == "check":
        check_batch()
    elif args.command == "status":
        status_command()
    else:
        parser.print_help()
