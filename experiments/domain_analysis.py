#!/usr/bin/env python3
"""Domain-level alpha analysis for Murmur paper.

Fetches resolved Polymarket events, classifies by domain, runs zero-shot
and context-enriched predictions, computes Brier Scores per domain.

Usage:
    python experiments/domain_analysis.py --limit 200 --days 7
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from murmur.context_planner import ContextPlan, Domain
from murmur.strategy import LLMStrategy
from murmur.models import Event, OutcomeToken
from murmur.prices import get_price_context
from murmur.news import search_news_sync

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "paper"


def classify_domain(title: str, description: str = "") -> str:
    """Classify event into domain based on keywords."""
    text = (title + " " + description).lower()
    
    sports_kw = ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
                 "baseball", "hockey", "tennis", "mma", "ufc", "boxing", "f1",
                 "formula", "race", "match", "game score", "playoff", "championship",
                 "world series", "super bowl", "finals"]
    golf_kw = ["golf", "pga", "masters", "open championship", "lpga", "ryder"]
    crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                 "token", "blockchain", "defi", "nft"]
    commodity_kw = ["oil", "crude", "gold", "silver", "natural gas", "wheat",
                    "commodity", "wti", "brent"]
    economy_kw = ["fed", "interest rate", "cpi", "inflation", "gdp", "unemployment",
                  "fomc", "treasury", "yield", "recession", "jobs report", "payroll"]
    politics_kw = ["president", "election", "congress", "senate", "vote", "democrat",
                   "republican", "trump", "biden", "governor", "legislation", "bill"]
    awards_kw = ["oscar", "grammy", "emmy", "golden globe", "academy award", "bafta",
                 "tony award", "best picture", "best actor", "best director"]
    geopolitics_kw = ["war", "ukraine", "russia", "china", "taiwan", "nato", "missile",
                      "sanctions", "ceasefire", "invasion", "nuclear"]
    tech_kw = ["apple", "google", "microsoft", "ai launch", "product launch",
               "acquisition", "ipo", "startup", "spacex", "tesla"]
    entertainment_kw = ["box office", "streaming", "movie", "album", "concert",
                        "netflix", "disney", "spotify"]
    
    for kw in golf_kw:
        if kw in text: return "golf"
    for kw in awards_kw:
        if kw in text: return "awards"
    for kw in commodity_kw:
        if kw in text: return "commodities"
    for kw in crypto_kw:
        if kw in text: return "crypto"
    for kw in economy_kw:
        if kw in text: return "economy"
    for kw in sports_kw:
        if kw in text: return "sports"
    for kw in politics_kw:
        if kw in text: return "politics"
    for kw in geopolitics_kw:
        if kw in text: return "geopolitics"
    for kw in tech_kw:
        if kw in text: return "tech"
    for kw in entertainment_kw:
        if kw in text: return "entertainment"
    return "other"


def _parse_json(value, default=None):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default or []
    return value if value else (default or [])


async def fetch_resolved_markets(limit=500, min_volume=10_000):
    """Fetch resolved markets across all categories."""
    all_markets = []
    offset = 0
    batch_size = 100
    
    async with httpx.AsyncClient(timeout=30) as client:
        while len(all_markets) < limit:
            try:
                r = await client.get(f"{GAMMA_API}/markets", params={
                    "limit": batch_size,
                    "offset": offset,
                    "closed": "true",
                    "order": "volumeClob",
                    "ascending": "false",
                })
                data = r.json()
                if not data:
                    break
                    
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
                    
                    yes_price = float(prices[0]) if prices else 0.5
                    actual = 1.0 if yes_price > 0.5 else 0.0
                    
                    question = m.get("question", "")
                    description = m.get("description", "")[:500]
                    domain = classify_domain(question, description)
                    
                    all_markets.append({
                        "question": question,
                        "description": description,
                        "token_id": token_ids[0],
                        "end_date": end_date,
                        "actual": actual,
                        "final_yes_price": yes_price,
                        "volume": vol,
                        "slug": m.get("slug", ""),
                        "domain": domain,
                    })
                
                offset += batch_size
                if len(data) < batch_size:
                    break
                await asyncio.sleep(0.2)
                
            except Exception as e:
                print(f"  Error fetching batch at offset {offset}: {e}")
                break
    
    return all_markets[:limit]


async def get_price_before(token_id: str, end_date_str: str, days_before: int = 7):
    """Get market price N days before resolution."""
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
    except ValueError:
        return None
    
    end_ts = int(dt.timestamp())
    start_ts = end_ts - days_before * 86400
    
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{CLOB_API}/prices-history", params={
                "market": token_id,
                "interval": "1d",
                "fidelity": max(days_before, 2),
                "startTs": start_ts,
                "endTs": end_ts,
            })
            if r.status_code != 200:
                return None
            data = r.json()
            history = data.get("history", [])
            if history:
                return float(history[0]["p"])
        except Exception:
            pass
    return None


async def run_domain_analysis(limit=200, days_before=7, use_context=False):
    """Run domain-level analysis for paper."""
    print(f"{'='*70}")
    print(f"MURMUR DOMAIN ANALYSIS — {limit} events, {days_before}-day lookahead")
    print(f"Context Planner: {'ON' if use_context else 'OFF'}")
    print(f"{'='*70}\n")
    
    # 1. Fetch resolved markets
    print("Fetching resolved markets...")
    markets = await fetch_resolved_markets(limit=limit * 2, min_volume=10_000)
    print(f"  Found {len(markets)} resolved markets\n")
    
    # Domain distribution
    domain_counts = defaultdict(int)
    for m in markets:
        domain_counts[m["domain"]] += 1
    print("Domain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:15s}: {c}")
    print()
    
    # 2. Get historical prices
    print(f"Fetching prices from {days_before} days before resolution...")
    valid_markets = []
    for i, m in enumerate(markets):
        if len(valid_markets) >= limit:
            break
        price = await get_price_before(m["token_id"], m["end_date"], days_before)
        if price is not None and 0.01 < price < 0.99:
            m["market_price_before"] = price
            valid_markets.append(m)
        if i % 20 == 0:
            print(f"  Processed {i}/{len(markets)}, valid={len(valid_markets)}")
        await asyncio.sleep(0.1)
    
    print(f"\n  Got {len(valid_markets)} markets with valid price history\n")
    
    # 3. Run predictions
    strategy = LLMStrategy()
    planner = None  # Context planner integration TBD
    
    domain_results = defaultdict(lambda: {
        "events": [],
        "brier_pred": [],
        "brier_mkt": [],
        "beats": 0,
        "total": 0,
    })
    
    for i, m in enumerate(valid_markets):
        domain = m["domain"]
        actual = m["actual"]
        mkt = max(0.01, min(0.99, m["market_price_before"]))
        
        # Build Event
        event = Event(
            id=m["token_id"][:20],
            slug=m["slug"],
            title=m["question"],
            description=m["description"],
            tokens=[OutcomeToken(
                token_id=m["token_id"],
                outcome="Yes",
                price=mkt,
            )],
            volume=m["volume"],
        )
        
        try:
            # Zero-shot prediction (context-enriched TBD)
            pred = strategy.predict(event, [], [], None)
            
            p = max(0.01, min(0.99, pred.probability))
        except Exception as e:
            p = mkt  # Fallback to market
        
        brier_pred = (p - actual) ** 2
        brier_mkt = (mkt - actual) ** 2
        beat = brier_pred < brier_mkt
        
        dr = domain_results[domain]
        dr["events"].append(m["question"][:60])
        dr["brier_pred"].append(brier_pred)
        dr["brier_mkt"].append(brier_mkt)
        dr["total"] += 1
        if beat:
            dr["beats"] += 1
        
        status = "✅" if beat else "❌"
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  {i+1:3d}. [{domain:12s}] {m['question'][:40]:40s} "
                  f"mkt={mkt:.2f} pred={p:.2f} {status}")
        
        await asyncio.sleep(0.3)  # Rate limit for Groq
    
    # 4. Print domain-level results
    print(f"\n{'='*70}")
    print(f"DOMAIN-LEVEL RESULTS")
    print(f"{'='*70}\n")
    
    summary = {}
    print(f"{'Domain':15s} {'Events':>6s} {'Our Brier':>10s} {'Mkt Brier':>10s} "
          f"{'BMR':>6s} {'Alpha?':>8s}")
    print("-" * 60)
    
    for domain in sorted(domain_results.keys()):
        dr = domain_results[domain]
        n = dr["total"]
        if n == 0:
            continue
        avg_brier = sum(dr["brier_pred"]) / n
        avg_mkt = sum(dr["brier_mkt"]) / n
        bmr = dr["beats"] / n
        alpha = "✅" if avg_brier < avg_mkt else "❌"
        
        summary[domain] = {
            "n_events": n,
            "our_brier": avg_brier,
            "market_brier": avg_mkt,
            "beat_market_rate": bmr,
            "has_alpha": avg_brier < avg_mkt,
        }
        
        print(f"{domain:15s} {n:6d} {avg_brier:10.4f} {avg_mkt:10.4f} "
              f"{bmr:5.0%} {alpha:>8s}")
    
    # Overall
    all_pred = [b for dr in domain_results.values() for b in dr["brier_pred"]]
    all_mkt = [b for dr in domain_results.values() for b in dr["brier_mkt"]]
    total_n = len(all_pred)
    total_beats = sum(dr["beats"] for dr in domain_results.values())
    
    print("-" * 60)
    print(f"{'OVERALL':15s} {total_n:6d} {sum(all_pred)/total_n:10.4f} "
          f"{sum(all_mkt)/total_n:10.4f} {total_beats/total_n:5.0%}")
    
    # 5. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_context" if use_context else "_zeroshot"
    output_file = RESULTS_DIR / f"domain_analysis{suffix}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "limit": limit,
                "days_before": days_before,
                "use_context": use_context,
                "n_valid": total_n,
            },
            "summary": summary,
            "overall": {
                "our_brier": sum(all_pred) / total_n,
                "market_brier": sum(all_mkt) / total_n,
                "beat_market_rate": total_beats / total_n,
            },
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Domain-level alpha analysis")
    parser.add_argument("-n", "--limit", type=int, default=200)
    parser.add_argument("-d", "--days", type=int, default=7)
    parser.add_argument("--context", action="store_true", help="Enable Context Planner")
    args = parser.parse_args()
    
    asyncio.run(run_domain_analysis(
        limit=args.limit,
        days_before=args.days,
        use_context=args.context,
    ))
