#!/usr/bin/env python3
"""Evolution v2: evolve a RULE-BASED strategy (no LLM per prediction = fast eval)."""

import asyncio
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from _deprecated_murmur.models import Event, OutcomeToken, Prediction
from backtest import get_high_volume_resolved_markets, get_price_at_days_before

console = Console()
RESULTS_DIR = Path("results")


# === EVOLVABLE RULE-BASED STRATEGY ===
# This is what the evolution loop modifies

DEFAULT_STRATEGY = '''
def predict(title, description, market_price, volume):
    """Predict probability based on available signals."""
    p = market_price  # Start from market consensus
    
    # Confidence-based adjustment: extreme markets are usually right
    if market_price > 0.90 or market_price < 0.10:
        # Trust the market on obvious calls
        p = market_price
    elif market_price > 0.70:
        # Slight contrarian: markets slightly overconfident on likely events
        p = market_price - 0.02
    elif market_price < 0.30:
        p = market_price + 0.02
    else:
        # Middle range: market is uncertain, we're uncertain too
        p = market_price
    
    # Volume signal: high volume = more informed market
    if volume > 100_000_000:  # >$100M volume
        # Very liquid market, trust it more
        p = market_price * 0.95 + p * 0.05
    
    return {
        "probability": max(0.01, min(0.99, p)),
        "confidence": 0.6,
        "reasoning": f"Market-informed prediction. Market={market_price:.2f}, Volume=${volume/1e6:.0f}M"
    }
'''


async def prepare_data(n_events=15, days_before=7):
    """Fetch and cache backtest data."""
    cache_file = RESULTS_DIR / "backtest_cache.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            if len(data) >= n_events:
                return data[:n_events]

    markets = await get_high_volume_resolved_markets(limit=100, min_volume=500_000)
    valid = []
    for m in markets[:n_events * 3]:
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
    return valid


def eval_strategy(code: str, cached_events: list) -> dict:
    """Evaluate a rule-based strategy (no API calls = instant)."""
    # Compile strategy
    namespace = {"math": math}
    try:
        exec(code, namespace)
    except Exception as e:
        return {"brier": 999, "error": str(e)}
    
    predict_fn = namespace.get("predict")
    if not predict_fn:
        return {"brier": 999, "error": "no predict function"}

    total_brier = 0
    total_mkt_brier = 0
    wins = 0
    details = []

    for m in cached_events:
        mkt = max(0.01, min(0.99, m["market_price_before"]))
        actual = m["actual"]
        
        try:
            result = predict_fn(
                m["question"],
                m.get("description", ""),
                mkt,
                m["volume"]
            )
            p = max(0.01, min(0.99, float(result["probability"])))
        except Exception as e:
            p = mkt

        brier_p = (p - actual) ** 2
        brier_m = (mkt - actual) ** 2
        beat = brier_p < brier_m
        if beat: wins += 1
        total_brier += brier_p
        total_mkt_brier += brier_m
        
        details.append({
            "q": m["question"][:45],
            "actual": actual,
            "mkt": mkt,
            "pred": p,
            "bp": brier_p,
            "bm": brier_m,
            "beat": beat,
        })

    n = len(cached_events)
    return {
        "brier": total_brier / n,
        "mkt_brier": total_mkt_brier / n,
        "beat_rate": wins / n,
        "wins": wins,
        "n": n,
        "details": details,
    }


def run_evolution(rounds=30, n_events=15, days_before=7):
    """Evolve rule-based strategy."""
    cached = asyncio.run(prepare_data(n_events, days_before))
    console.print(f"Loaded {len(cached)} events for backtest\n")

    # Baseline
    base = eval_strategy(DEFAULT_STRATEGY, cached)
    console.print(f"[bold]Baseline:[/bold] Brier={base['brier']:.4f} | Market={base['mkt_brier']:.4f} | Beat={base['wins']}/{base['n']}\n")

    # Setup Groq
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    best_brier = base["brier"]
    best_code = DEFAULT_STRATEGY
    current_code = DEFAULT_STRATEGY
    current_result = base
    tried = set()
    improvements = []

    console.print(f"[bold]Evolving ({rounds} rounds)...[/bold]\n")

    for r in range(1, rounds + 1):
        t0 = time.time()

        # Failure analysis
        failures = [d for d in current_result["details"] if not d["beat"]]
        fail_str = "\n".join(
            f"  {d['q']}: pred={d['pred']:.3f} mkt={d['mkt']:.3f} actual={'Y' if d['actual']==1 else 'N'} "
            f"(our_brier={d['bp']:.4f} mkt_brier={d['bm']:.4f})"
            for d in sorted(failures, key=lambda x: x["bp"]-x["bm"], reverse=True)[:5]
        )

        prompt = f"""You are evolving a Python function that predicts Polymarket event outcomes.

CURRENT CODE:
```python
{current_code}
```

PERFORMANCE: Brier={best_brier:.4f}, Market={current_result['mkt_brier']:.4f}, Beat={current_result['wins']}/{current_result['n']}

TOP FAILURES (where market beat us most):
{fail_str}

TRIED CHANGES (don't repeat):
{chr(10).join(f'  - {c}' for c in list(tried)[-8:])}

RULES:
1. Change EXACTLY ONE thing
2. Function signature: predict(title, description, market_price, volume) -> dict with probability, confidence, reasoning
3. Key insight: when market_price is extreme (>0.90 or <0.10), staying close to market is usually optimal
4. For mid-range predictions, look for keywords in title/description that signal direction
5. Return the COMPLETE function code, not just the changed part

Respond with EXACTLY this format (no other text):
CHANGE: one-line description
```python
def predict(title, description, market_price, volume):
    ... your complete code ...
    return {{"probability": p, "confidence": c, "reasoning": r}}
```"""

        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=1500,
            )
            content = resp.choices[0].message.content.strip()

            # Extract change description
            change = "unknown"
            for line in content.split("\n"):
                if line.startswith("CHANGE:"):
                    change = line.replace("CHANGE:", "").strip()[:80]
                    break

            # Extract code
            import re
            code_match = re.search(r'```python\s*(def predict\([\s\S]*?)```', content)
            if not code_match:
                code_match = re.search(r'(def predict\([\s\S]*?return\s+\{[^}]+\})', content)
            
            if not code_match:
                console.print(f"  Round {r:2d}: no code found, skip")
                continue

            new_code = code_match.group(1)
            
            if change in tried:
                console.print(f"  Round {r:2d}: duplicate, skip")
                continue
            tried.add(change)

            # Evaluate (instant — no API calls!)
            new_result = eval_strategy(new_code, cached)
            dt = time.time() - t0

            if "error" in new_result:
                console.print(f"  Round {r:2d}: code error — {new_result['error'][:60]}")
                continue

            improvement = best_brier - new_result["brier"]
            icon = "🟢" if improvement > 0 else "🔴"
            
            console.print(
                f"  {icon} R{r:2d} | B={new_result['brier']:.4f} | Δ={improvement:+.4f} "
                f"| Beat={new_result['wins']}/{new_result['n']} | {change[:55]} | {dt:.1f}s"
            )

            if improvement > 0:
                best_brier = new_result["brier"]
                best_code = new_code
                current_code = new_code
                current_result = new_result
                improvements.append({"round": r, "brier": best_brier, "change": change})

        except Exception as e:
            console.print(f"  Round {r:2d}: error — {str(e)[:60]}")
            time.sleep(2)  # rate limit backoff
            continue

    # Summary
    console.print(f"\n{'='*60}")
    console.print(f"[bold]EVOLUTION COMPLETE[/bold]")
    console.print(f"  Start Brier:  {base['brier']:.4f}")
    console.print(f"  Final Brier:  {best_brier:.4f}")
    console.print(f"  Market Brier: {current_result['mkt_brier']:.4f}")
    console.print(f"  Improvement:  {base['brier'] - best_brier:+.4f}")
    
    if improvements:
        console.print(f"\n  Key improvements:")
        for imp in improvements:
            console.print(f"    Round {imp['round']}: {imp['brier']:.4f} — {imp['change']}")
    
    if best_brier < current_result['mkt_brier']:
        console.print(f"\n  [green bold]🎯 BEATS THE MARKET! ({best_brier:.4f} < {current_result['mkt_brier']:.4f})[/green bold]")
    else:
        gap = best_brier - current_result['mkt_brier']
        console.print(f"\n  Gap to market: {gap:.4f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "evolved_rules.py", "w") as f:
        f.write(best_code)
    with open(RESULTS_DIR / "evolution_v2_log.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_brier": base["brier"],
            "final_brier": best_brier,
            "market_brier": current_result["mkt_brier"],
            "rounds": rounds,
            "improvements": improvements,
            "best_code": best_code,
        }, f, indent=2)
    
    console.print(f"  Saved to results/evolved_rules.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=30)
    parser.add_argument("-n", "--events", type=int, default=15)
    parser.add_argument("-d", "--days", type=int, default=7)
    args = parser.parse_args()
    
    run_evolution(rounds=args.rounds, n_events=args.events, days_before=args.days)
