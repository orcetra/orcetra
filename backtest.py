#!/usr/bin/env python3
"""Fair backtest: predict events using data available N days before resolution."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.table import Table

from orcetra.fetcher import PolymarketFetcher
from orcetra.strategy import LLMStrategy
from orcetra.models import Event, OutcomeToken, Prediction

console = Console()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


async def get_high_volume_resolved_markets(limit=50, min_volume=100_000):
    """Get resolved markets with decent volume and token IDs."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{GAMMA_API}/markets", params={
            "limit": limit, "closed": "true",
            "order": "volumeClob", "ascending": "false",
        })
        data = r.json()

    markets = []
    for m in data:
        vol = float(m.get("volumeClob", 0) or 0)
        if vol < min_volume:
            continue

        outcomes = _parse_json(m.get("outcomes"), ["Yes", "No"])
        prices = _parse_json(m.get("outcomePrices"), ["0.5", "0.5"])
        token_ids = _parse_json(m.get("clobTokenIds"), [])
        end_date = m.get("endDateIso") or m.get("endDate")

        if not token_ids or not token_ids[0] or not end_date:
            continue

        # YES token price → actual outcome
        yes_price = float(prices[0]) if prices else 0.5
        actual = 1.0 if yes_price > 0.5 else 0.0

        markets.append({
            "question": m.get("question", ""),
            "description": m.get("description", "")[:500],
            "token_id": token_ids[0],
            "end_date": end_date,
            "actual": actual,
            "final_yes_price": yes_price,
            "volume": vol,
            "outcomes": outcomes,
            "slug": m.get("slug", ""),
        })

    return markets


async def get_price_at_days_before(token_id: str, end_date_str: str, days_before: int = 7):
    """Get the market price N days before event resolution."""
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
    except ValueError:
        return None

    end_ts = int(dt.timestamp())
    start_ts = end_ts - days_before * 86400
    target_ts = start_ts  # We want the price at this point

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{CLOB_API}/prices-history", params={
                "market": token_id,
                "interval": "1d",
                "fidelity": days_before,
                "startTs": start_ts,
                "endTs": end_ts,
            })
            if r.status_code != 200:
                return None
            data = r.json()
            history = data.get("history", [])
            if history:
                # Return first price point (= price N days before end)
                return float(history[0]["p"])
        except Exception:
            pass

    return None


def _parse_json(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default or []
    return value if value else (default or [])


async def run_backtest(n_events=20, days_before=7):
    """Run fair backtest: predict using info available N days before resolution."""
    console.print(f"[bold]Orcetra Backtest — predicting {days_before} days before resolution[/bold]\n")

    # 1. Get resolved markets
    console.print("Fetching resolved markets...", end=" ")
    markets = await get_high_volume_resolved_markets(limit=100, min_volume=500_000)
    console.print(f"found {len(markets)} high-volume markets")

    if len(markets) < 5:
        console.print("[red]Not enough markets. Try lower min_volume.[/red]")
        return

    # 2. Get historical prices (market consensus N days before)
    console.print(f"Fetching prices from {days_before} days before resolution...")
    valid_markets = []
    for m in markets[:n_events * 2]:  # fetch extra in case some fail
        price = await get_price_at_days_before(m["token_id"], m["end_date"], days_before)
        if price is not None:
            m["market_price_before"] = price
            valid_markets.append(m)
            if len(valid_markets) >= n_events:
                break
        await asyncio.sleep(0.1)  # rate limit

    console.print(f"Got {len(valid_markets)} markets with price history\n")

    if len(valid_markets) < 3:
        console.print("[red]Not enough markets with price history.[/red]")
        return

    # 3. Run predictions
    strategy = LLMStrategy()
    results = []

    for i, m in enumerate(valid_markets):
        # Build a minimal Event for the strategy
        event = Event(
            id=m["token_id"][:20],
            slug=m["slug"],
            title=m["question"],
            description=m["description"],
            tokens=[OutcomeToken(token_id=m["token_id"], outcome="Yes", price=m["market_price_before"])],
            volume=m["volume"],
        )

        t0 = time.time()
        try:
            pred = strategy.predict(event, [], [], None)
            p = max(0.01, min(0.99, pred.probability))
        except Exception as e:
            p = m["market_price_before"]

        dt = time.time() - t0
        actual = m["actual"]
        mkt = max(0.01, min(0.99, m["market_price_before"]))

        brier_pred = (p - actual) ** 2
        brier_mkt = (mkt - actual) ** 2
        beat = brier_pred < brier_mkt

        results.append({
            "question": m["question"],
            "actual": actual,
            "market": mkt,
            "prediction": p,
            "brier_pred": brier_pred,
            "brier_mkt": brier_mkt,
            "beat": beat,
            "time": dt,
        })

        icon = "✅" if beat else "❌"
        console.print(
            f"  {i+1:2d}. {m['question'][:45]:45s} "
            f"| Y={'YES' if actual==1 else 'NO ':3s} "
            f"| mkt={mkt:.2f} | pred={p:.2f} "
            f"| B={brier_pred:.3f} vs {brier_mkt:.3f} {icon}"
        )

    # 4. Summary
    n = len(results)
    wins = sum(1 for r in results if r["beat"])
    avg_brier = sum(r["brier_pred"] for r in results) / n
    avg_mkt_brier = sum(r["brier_mkt"] for r in results) / n

    console.print(f"\n{'='*60}")
    console.print(f"[bold]RESULTS ({n} events, {days_before}-day lookahead)[/bold]")
    console.print(f"{'='*60}")
    console.print(f"  Orcetra Brier Score:  {avg_brier:.4f}")
    console.print(f"  Market Brier Score:  {avg_mkt_brier:.4f}")
    console.print(f"  Beat Market Rate:    {wins}/{n} ({wins/n:.0%})")

    if avg_brier < avg_mkt_brier:
        console.print(f"\n  [green bold]🎯 Orcetra BEATS the market![/green bold]")
    else:
        gap = avg_brier - avg_mkt_brier
        console.print(f"\n  [yellow]Market wins by {gap:.4f} Brier points[/yellow]")
        console.print(f"  [dim]This is expected for baseline — evolution loop will improve[/dim]")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/backtest.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "days_before": days_before,
            "n_events": n,
            "avg_brier": avg_brier,
            "avg_market_brier": avg_mkt_brier,
            "beat_rate": wins / n,
            "events": results,
        }, f, indent=2, default=str)

    console.print(f"\n  Results saved to results/backtest.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--events", type=int, default=15, help="Number of events")
    parser.add_argument("-d", "--days", type=int, default=7, help="Days before resolution")
    args = parser.parse_args()

    asyncio.run(run_backtest(n_events=args.events, days_before=args.days))
