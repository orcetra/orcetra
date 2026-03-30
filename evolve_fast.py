#!/usr/bin/env python3
"""Fast evolution loop using cached backtest data."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from orcetra.polymarket.strategy import LLMStrategy, load_strategy_from_code, DEFAULT_ANALYSIS_CODE, STRATEGY_TEMPLATE
from orcetra.polymarket.models import Event, OutcomeToken, Prediction
from backtest import get_high_volume_resolved_markets, get_price_at_days_before

console = Console()
RESULTS_DIR = Path("results")


async def prepare_backtest_data(n_events=15, days_before=7):
    """Fetch and cache backtest data (do this once)."""
    cache_file = RESULTS_DIR / "backtest_cache.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            if len(data) >= n_events:
                console.print(f"Using cached data ({len(data)} events)")
                return data

    console.print("Fetching backtest data (one-time)...")
    markets = await get_high_volume_resolved_markets(limit=100, min_volume=500_000)
    
    valid = []
    for m in markets[:n_events * 2]:
        price = await get_price_at_days_before(m["token_id"], m["end_date"], days_before)
        if price is not None:
            m["market_price_before"] = price
            valid.append(m)
            if len(valid) >= n_events:
                break
        await asyncio.sleep(0.1)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(valid, f, indent=2)
    
    console.print(f"Cached {len(valid)} events")
    return valid


def evaluate_strategy(strategy, cached_events):
    """Evaluate a strategy against cached events. Returns (brier, beat_rate, details)."""
    results = []
    total_brier = 0
    total_mkt_brier = 0
    wins = 0

    for m in cached_events:
        event = Event(
            id=m["token_id"][:20],
            slug=m.get("slug", ""),
            title=m["question"],
            description=m.get("description", "")[:300],
            tokens=[OutcomeToken(
                token_id=m["token_id"],
                outcome="Yes",
                price=m["market_price_before"]
            )],
            volume=m["volume"],
        )

        try:
            pred = strategy.predict(event, [], [], None)
            p = max(0.01, min(0.99, pred.probability))
        except:
            p = m["market_price_before"]

        actual = m["actual"]
        mkt = max(0.01, min(0.99, m["market_price_before"]))

        brier_p = (p - actual) ** 2
        brier_m = (mkt - actual) ** 2
        beat = brier_p < brier_m
        if beat:
            wins += 1
        total_brier += brier_p
        total_mkt_brier += brier_m

        results.append({
            "question": m["question"][:45],
            "actual": actual,
            "market": mkt,
            "pred": p,
            "brier_pred": brier_p,
            "brier_mkt": brier_m,
            "beat": beat,
        })

    n = len(results)
    return {
        "brier": total_brier / n,
        "market_brier": total_mkt_brier / n,
        "beat_rate": wins / n,
        "wins": wins,
        "n": n,
        "details": results,
    }


def run_evolution(rounds=20, n_events=15, days_before=7):
    """Run evolution loop with cached backtest data."""
    # 1. Prepare data
    cached = asyncio.run(prepare_backtest_data(n_events, days_before))
    
    # 2. Baseline
    console.print("\n[bold]Evaluating baseline strategy...[/bold]")
    baseline = LLMStrategy()
    base_result = evaluate_strategy(baseline, cached)
    console.print(f"  Baseline Brier: {base_result['brier']:.4f} | Market: {base_result['market_brier']:.4f} | Beat: {base_result['wins']}/{base_result['n']}")

    # 3. Evolution
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        console.print("[red]GROQ_API_KEY not set[/red]")
        return

    from groq import Groq
    client = Groq(api_key=groq_key)

    best_brier = base_result["brier"]
    best_code = None
    tried = set()
    
    # Get current strategy source
    current_code = STRATEGY_TEMPLATE.format(version=1, analysis_code=DEFAULT_ANALYSIS_CODE)

    console.print(f"\n[bold]Starting evolution ({rounds} rounds)...[/bold]\n")

    for r in range(1, rounds + 1):
        t0 = time.time()

        # Build failure analysis
        failures = [d for d in base_result["details"] if not d["beat"]]
        failure_summary = "\n".join(
            f"  - {d['question']}: predicted {d['pred']:.2f}, market {d['market']:.2f}, actual {'YES' if d['actual']==1 else 'NO'}"
            for d in failures[:5]
        )

        prompt = f"""You are evolving a Polymarket prediction strategy. The strategy analyzes events and outputs probability predictions.

CURRENT STRATEGY CODE (the analyze() function):
```python
{current_code}
```

CURRENT PERFORMANCE:
- Brier Score: {best_brier:.4f} (lower is better)
- Market baseline: {base_result['market_brier']:.4f}
- Beat market: {base_result['wins']}/{base_result['n']}

FAILURES (events where market beat us):
{failure_summary}

CHANGES ALREADY TRIED (don't repeat these):
{chr(10).join(f'  - {c}' for c in list(tried)[-10:])}

RULES:
1. Make EXACTLY ONE change to the strategy
2. The strategy takes: event title, description, current market price, volume
3. Output must be valid Python that returns a dict with 'probability' (0-1), 'confidence' (0-1), 'reasoning' (str)
4. Key insight: when market price is extreme (>0.90 or <0.10), don't deviate much — the market is usually right on obvious calls
5. Focus on the events where we lost the most Brier score

Return ONLY a JSON object:
{{"change_description": "brief description of what you changed", "code": "the complete analyze() function body"}}"""

        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2000,
            )
            content = resp.choices[0].message.content

            # Parse response
            import re
            # Find JSON
            start = content.find('{')
            if start < 0:
                console.print(f"  Round {r}: no JSON in response, skip")
                continue
            
            depth = 0
            for i, ch in enumerate(content[start:], start):
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(content[start:i+1])
                    except:
                        data = None
                    break
            
            if not data or "code" not in data:
                console.print(f"  Round {r}: invalid JSON, skip")
                continue

            change = data.get("change_description", "unknown")
            new_code = data["code"]

            if change in tried:
                console.print(f"  Round {r}: duplicate change, skip")
                continue
            tried.add(change)

            # Try loading the new strategy
            full_code = STRATEGY_TEMPLATE.format(version=r+1, analysis_code=new_code)
            new_strategy = load_strategy_from_code(full_code)
            if new_strategy is None:
                console.print(f"  Round {r}: code doesn't compile, skip")
                continue

            # Evaluate
            new_result = evaluate_strategy(new_strategy, cached)
            dt = time.time() - t0
            improvement = best_brier - new_result["brier"]
            
            icon = "🟢" if improvement > 0 else "🔴"
            console.print(
                f"  {icon} Round {r:2d} | Brier={new_result['brier']:.4f} "
                f"| Δ={improvement:+.4f} | Beat={new_result['wins']}/{new_result['n']} "
                f"| {change[:50]} | {dt:.0f}s"
            )

            if improvement > 0:
                best_brier = new_result["brier"]
                best_code = full_code
                current_code = new_code
                base_result = new_result

        except Exception as e:
            console.print(f"  Round {r}: error — {str(e)[:80]}")
            continue

    # Summary
    console.print(f"\n{'='*60}")
    console.print(f"[bold]EVOLUTION COMPLETE[/bold]")
    console.print(f"  Starting Brier: {evaluate_strategy(LLMStrategy(), cached)['brier']:.4f}")
    console.print(f"  Final Brier:    {best_brier:.4f}")
    console.print(f"  Market Brier:   {base_result['market_brier']:.4f}")
    
    if best_brier < base_result['market_brier']:
        console.print(f"  [green bold]🎯 BEATS THE MARKET![/green bold]")
    
    if best_code:
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "evolved_strategy.py", "w") as f:
            f.write(best_code)
        console.print(f"  Best strategy saved to results/evolved_strategy.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=20)
    parser.add_argument("-n", "--events", type=int, default=15)
    parser.add_argument("-d", "--days", type=int, default=7)
    args = parser.parse_args()
    
    run_evolution(rounds=args.rounds, n_events=args.events, days_before=args.days)
