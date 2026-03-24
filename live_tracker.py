#!/usr/bin/env python3
"""Live prediction tracker: record predictions on active events, check results later."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from orcetra.strategy import LLMStrategy
from orcetra.models import Event, OutcomeToken

console = Console()
TRACKER_FILE = Path("results/live_predictions.json")
RESULTS_DIR = Path("results")


def _parse_json(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return default or []
    return value if value else (default or [])


async def fetch_active_markets(min_volume=100_000, max_days=21):
    """Fetch active markets from top events, filtered by volume and end date."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get("https://gamma-api.polymarket.com/events", params={
            "limit": 40, "active": True, "closed": False,
            "order": "volume", "ascending": False,
        })
        events = r.json()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_days)
    markets = []

    for e in events:
        for m in e.get("markets", []):
            end = m.get("endDateIso") or m.get("endDate")
            if not end:
                continue
            try:
                if "T" in str(end):
                    dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
                else:
                    dt = datetime.strptime(str(end), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except:
                continue

            if dt < now or dt > cutoff:
                continue

            prices = _parse_json(m.get("outcomePrices"), ["0.5", "0.5"])
            token_ids = _parse_json(m.get("clobTokenIds"), [])
            if not prices or not token_ids:
                continue

            yes_p = float(prices[0])
            vol = float(m.get("volume", 0) or 0)
            if vol < min_volume:
                continue
            # Skip extreme odds
            if yes_p > 0.92 or yes_p < 0.08:
                continue

            days_left = (dt - now).total_seconds() / 86400

            markets.append({
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "condition_id": m.get("conditionId", ""),
                "token_id": token_ids[0],
                "yes_price": yes_p,
                "volume": vol,
                "days_left": days_left,
                "end_date": str(end)[:10],
                "event_title": e.get("title", ""),
                "description": m.get("description", "")[:500],
                "closed": m.get("closed", False),
            })

    markets.sort(key=lambda x: x["volume"], reverse=True)
    return markets


def load_predictions():
    """Load existing predictions from file."""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {"predictions": [], "version": 1}


def save_predictions(data):
    """Save predictions to file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def predict_command(n_markets=15):
    """Make predictions on active markets and record them."""
    console.print("[bold]🔮 Orcetra Live Tracker — Recording Predictions[/bold]\n")

    # Fetch markets
    markets = asyncio.run(fetch_active_markets(min_volume=100_000, max_days=21))
    if not markets:
        # Try lower volume threshold
        markets = asyncio.run(fetch_active_markets(min_volume=50_000, max_days=21))
    if not markets:
        markets = asyncio.run(fetch_active_markets(min_volume=10_000, max_days=21))

    console.print(f"Found {len(markets)} active markets\n")
    markets = markets[:n_markets]

    strategy = LLMStrategy()
    tracker = load_predictions()
    existing_slugs = {p["slug"] for p in tracker["predictions"] if not p.get("resolved")}

    new_count = 0
    for i, m in enumerate(markets):
        if m["slug"] in existing_slugs:
            console.print(f"  {i+1:2d}. [dim]Already tracked: {m['question'][:50]}[/dim]")
            continue

        event = Event(
            id=m["condition_id"] or m["token_id"][:20],
            slug=m["slug"],
            title=m["question"],
            description=m["description"],
            tokens=[OutcomeToken(token_id=m["token_id"], outcome="Yes", price=m["yes_price"])],
            volume=m["volume"],
        )

        t0 = time.time()
        try:
            pred = strategy.predict(event, [], [], None)
            p = max(0.01, min(0.99, pred.probability))
            reasoning = pred.reasoning
        except Exception as e:
            p = m["yes_price"]
            reasoning = f"Error: {str(e)[:100]}"

        dt = time.time() - t0

        record = {
            "slug": m["slug"],
            "question": m["question"],
            "event": m["event_title"],
            "market_price_at_prediction": m["yes_price"],
            "our_prediction": round(p, 4),
            "reasoning": reasoning[:300],
            "confidence": round(getattr(pred, "confidence", 0.5), 2) if "pred" in dir() else 0.5,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "end_date": m["end_date"],
            "volume": m["volume"],
            "token_id": m["token_id"],
            "condition_id": m["condition_id"],
            "resolved": False,
            "actual_outcome": None,
        }
        tracker["predictions"].append(record)
        new_count += 1

        # Direction signal
        diff = p - m["yes_price"]
        if abs(diff) < 0.03:
            signal = "≈ HOLD"
        elif diff > 0:
            signal = f"↑ BUY YES (+{diff:.2f})"
        else:
            signal = f"↓ BUY NO ({diff:.2f})"

        console.print(
            f"  {i+1:2d}. {m['question'][:50]:50s} "
            f"| mkt={m['yes_price']:.2f} → pred={p:.2f} "
            f"| {signal:20s} | {m['days_left']:.0f}d | {dt:.1f}s"
        )

    save_predictions(tracker)
    console.print(f"\n✅ Recorded {new_count} new predictions (total: {len(tracker['predictions'])})")
    console.print(f"   Saved to {TRACKER_FILE}")

    # Show portfolio summary
    show_portfolio(tracker)


def check_command():
    """Check outcomes of predicted events."""
    console.print("[bold]📊 Orcetra Live Tracker — Checking Results[/bold]\n")

    tracker = load_predictions()
    unresolved = [p for p in tracker["predictions"] if not p.get("resolved")]

    if not unresolved:
        console.print("No unresolved predictions to check.")
        return

    console.print(f"Checking {len(unresolved)} unresolved predictions...\n")

    # Fetch current market data
    resolved_count = 0
    for pred in unresolved:
        try:
            r = httpx.get(f"https://gamma-api.polymarket.com/markets", params={
                "slug": pred["slug"],
            })
            data = r.json()
            if not data:
                continue

            m = data[0] if isinstance(data, list) else data
            prices = _parse_json(m.get("outcomePrices"), [])
            closed = m.get("closed", False)

            if closed and prices:
                yes_final = float(prices[0])
                actual = 1.0 if yes_final > 0.5 else 0.0

                pred["resolved"] = True
                pred["actual_outcome"] = actual
                pred["final_yes_price"] = yes_final
                pred["resolved_at"] = datetime.now(timezone.utc).isoformat()

                # Brier scores
                our_brier = (pred["our_prediction"] - actual) ** 2
                mkt_brier = (pred["market_price_at_prediction"] - actual) ** 2
                pred["our_brier"] = round(our_brier, 4)
                pred["mkt_brier"] = round(mkt_brier, 4)
                pred["beat_market"] = our_brier < mkt_brier

                icon = "✅" if pred["beat_market"] else "❌"
                result = "YES" if actual == 1 else "NO"
                console.print(
                    f"  {icon} {pred['question'][:50]:50s} "
                    f"| Result={result} | ours={our_brier:.4f} mkt={mkt_brier:.4f}"
                )
                resolved_count += 1
            else:
                # Update current price
                if prices:
                    current_yes = float(prices[0])
                    pred["last_checked_price"] = current_yes
                    pred["last_checked"] = datetime.now(timezone.utc).isoformat()
                    
                    diff = current_yes - pred["market_price_at_prediction"]
                    direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                    console.print(
                        f"  ⏳ {pred['question'][:50]:50s} "
                        f"| was={pred['market_price_at_prediction']:.2f} now={current_yes:.2f} {direction}"
                    )

        except Exception as e:
            console.print(f"  ⚠️ Error checking {pred['question'][:40]}: {str(e)[:50]}")

        time.sleep(0.2)  # Rate limit

    save_predictions(tracker)

    # Show summary
    all_resolved = [p for p in tracker["predictions"] if p.get("resolved")]
    if all_resolved:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]LIVE RESULTS ({len(all_resolved)} resolved)[/bold]")
        avg_ours = sum(p["our_brier"] for p in all_resolved) / len(all_resolved)
        avg_mkt = sum(p["mkt_brier"] for p in all_resolved) / len(all_resolved)
        wins = sum(1 for p in all_resolved if p["beat_market"])
        console.print(f"  Our Brier:     {avg_ours:.4f}")
        console.print(f"  Market Brier:  {avg_mkt:.4f}")
        console.print(f"  Beat Rate:     {wins}/{len(all_resolved)}")

        if avg_ours < avg_mkt:
            console.print(f"  [green bold]🎯 BEATING THE MARKET![/green bold]")
    else:
        console.print(f"\n  No resolved predictions yet. Check back after events settle.")


def show_portfolio(tracker):
    """Show current portfolio / signal summary."""
    active = [p for p in tracker["predictions"] if not p.get("resolved")]
    if not active:
        return

    console.print(f"\n[bold]📈 Active Predictions ({len(active)})[/bold]")

    buy_yes = []
    buy_no = []
    hold = []

    for p in active:
        diff = p["our_prediction"] - p["market_price_at_prediction"]
        if diff > 0.05:
            buy_yes.append((p, diff))
        elif diff < -0.05:
            buy_no.append((p, diff))
        else:
            hold.append(p)

    if buy_yes:
        console.print(f"\n  [green]BUY YES (we think market underprices):[/green]")
        for p, diff in sorted(buy_yes, key=lambda x: -x[1]):
            console.print(f"    ↑ {p['question'][:55]:55s} mkt={p['market_price_at_prediction']:.2f} → us={p['our_prediction']:.2f} (+{diff:.2f})")

    if buy_no:
        console.print(f"\n  [red]BUY NO (we think market overprices):[/red]")
        for p, diff in sorted(buy_no, key=lambda x: x[1]):
            console.print(f"    ↓ {p['question'][:55]:55s} mkt={p['market_price_at_prediction']:.2f} → us={p['our_prediction']:.2f} ({diff:.2f})")

    if hold:
        console.print(f"\n  [dim]HOLD ({len(hold)} markets within ±5% of market)[/dim]")


def status_command():
    """Show current tracker status."""
    tracker = load_predictions()
    total = len(tracker["predictions"])
    resolved = sum(1 for p in tracker["predictions"] if p.get("resolved"))
    active = total - resolved

    console.print(f"[bold]Orcetra Live Tracker Status[/bold]")
    console.print(f"  Total predictions: {total}")
    console.print(f"  Active: {active}")
    console.print(f"  Resolved: {resolved}")

    if resolved > 0:
        all_resolved = [p for p in tracker["predictions"] if p.get("resolved")]
        avg_ours = sum(p["our_brier"] for p in all_resolved) / len(all_resolved)
        avg_mkt = sum(p["mkt_brier"] for p in all_resolved) / len(all_resolved)
        wins = sum(1 for p in all_resolved if p["beat_market"])
        console.print(f"  Our Brier:     {avg_ours:.4f}")
        console.print(f"  Market Brier:  {avg_mkt:.4f}")
        console.print(f"  Beat Rate:     {wins}/{resolved}")

    show_portfolio(tracker)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Orcetra Live Prediction Tracker")
    sub = parser.add_subparsers(dest="command")

    p_predict = sub.add_parser("predict", help="Make predictions on active markets")
    p_predict.add_argument("-n", "--markets", type=int, default=15)

    p_check = sub.add_parser("check", help="Check outcomes of predictions")
    p_status = sub.add_parser("status", help="Show tracker status")

    args = parser.parse_args()
    if args.command == "predict":
        predict_command(args.markets)
    elif args.command == "check":
        check_command()
    elif args.command == "status":
        status_command()
    else:
        parser.print_help()
