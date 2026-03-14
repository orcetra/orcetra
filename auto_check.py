#!/usr/bin/env python3
"""Murmur Auto-Check: Verify predictions against resolved markets.
Run daily via cron or manually: python auto_check.py"""

import json, httpx, time, os, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
BATCH_FILE = RESULTS_DIR / "batch_predictions.json"
CHECK_LOG = RESULTS_DIR / "check_log.json"

# Categories we DON'T bet on (negative EV from Phase 1)
NO_BET_CATEGORIES = {"sports"}  # Soccer kept separate; tennis borderline
# Categories we DO bet on
BET_CATEGORIES = {"economy", "commodities", "crypto", "other", "politics", 
                  "golf", "soccer", "entertainment", "powell_words"}


def load_predictions():
    if not BATCH_FILE.exists():
        print("No batch predictions found. Run batch_tracker.py predict first.")
        return {}
    return json.load(open(BATCH_FILE))


def load_check_log():
    if CHECK_LOG.exists():
        return json.load(open(CHECK_LOG))
    return {"checks": [], "summary": {}}


def save_check_log(log):
    with open(CHECK_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


def fetch_resolved():
    """Fetch recently resolved events from Gamma API."""
    resolved = {}
    for offset in range(0, 500, 100):
        r = httpx.get("https://gamma-api.polymarket.com/events", params={
            "limit": 100, "closed": True,
            "offset": offset,
            "order": "updatedAt", "ascending": False,
        }, timeout=15)
        events = r.json()
        if not events:
            break
        for e in events:
            for m in e.get("markets", []):
                cid = m.get("conditionId", "")
                if not cid or not m.get("closed"):
                    continue
                outcomes = json.loads(m.get("outcomes", "[]")) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
                prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
                if len(outcomes) != 2 or not prices:
                    continue
                yes_final = float(prices[0])
                actual = 1.0 if yes_final > 0.5 else 0.0
                resolved[cid] = {
                    "actual": actual,
                    "final_price": yes_final,
                    "question": m.get("question", ""),
                }
        time.sleep(0.15)
    return resolved


def check_predictions():
    batch = load_predictions()
    preds = batch.get("predictions", {})
    if not preds:
        print("No predictions to check.")
        return

    print("🔍 Fetching resolved markets...")
    resolved = fetch_resolved()
    print(f"   Found {len(resolved)} resolved markets")

    log = load_check_log()
    already_checked = {c["condition_id"] for c in log["checks"]}

    new_checks = 0
    wins = 0
    losses = 0
    results_by_cat = {}

    for slug, pred in preds.items():
        cid = pred.get("condition_id", "")
        if not cid or cid in already_checked:
            continue
        if cid not in resolved:
            continue

        actual = resolved[cid]["actual"]
        our_pred = pred["our_prediction"]
        market_price = pred["market_price"]
        tag = pred.get("tag", "other")
        signal = pred.get("signal", "HOLD")

        # Brier scores
        our_brier = (our_pred - actual) ** 2
        mkt_brier = (market_price - actual) ** 2
        beat_market = our_brier < mkt_brier

        # Simulated P&L (only for BET categories with signal != HOLD)
        pnl = 0.0
        bet_made = False
        if tag not in NO_BET_CATEGORIES and signal != "HOLD":
            bet_made = True
            if signal == "BUY NO":
                no_cost = 1 - market_price
                pnl = (1.0 - no_cost) if actual == 0.0 else -no_cost
            elif signal == "BUY YES":
                yes_cost = market_price
                pnl = (1.0 - yes_cost) if actual == 1.0 else -yes_cost
            
            if pnl > 0:
                wins += 1
            else:
                losses += 1

        check = {
            "condition_id": cid,
            "question": pred["question"][:80],
            "tag": tag,
            "signal": signal,
            "market_price": market_price,
            "our_prediction": our_pred,
            "actual": actual,
            "our_brier": round(our_brier, 4),
            "mkt_brier": round(mkt_brier, 4),
            "beat_market": beat_market,
            "bet_made": bet_made,
            "pnl": round(pnl, 4),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        log["checks"].append(check)
        new_checks += 1

        # Accumulate by category
        if tag not in results_by_cat:
            results_by_cat[tag] = {"n": 0, "beat": 0, "bets": 0, "wins": 0, "pnl": 0}
        results_by_cat[tag]["n"] += 1
        if beat_market:
            results_by_cat[tag]["beat"] += 1
        if bet_made:
            results_by_cat[tag]["bets"] += 1
            if pnl > 0:
                results_by_cat[tag]["wins"] += 1
            results_by_cat[tag]["pnl"] += pnl

    # Update summary
    all_checks = log["checks"]
    total = len(all_checks)
    total_beat = sum(1 for c in all_checks if c["beat_market"])
    total_bets = sum(1 for c in all_checks if c["bet_made"])
    total_wins = sum(1 for c in all_checks if c["bet_made"] and c["pnl"] > 0)
    total_pnl = sum(c["pnl"] for c in all_checks if c["bet_made"])
    
    log["summary"] = {
        "total_checked": total,
        "beat_market_rate": round(total_beat / total, 4) if total else 0,
        "total_bets": total_bets,
        "win_rate": round(total_wins / total_bets, 4) if total_bets else 0,
        "total_pnl_per_dollar": round(total_pnl, 4),
        "last_check": datetime.now(timezone.utc).isoformat(),
    }

    save_check_log(log)

    # Print report
    print(f"\n{'='*60}")
    print(f"📊 MURMUR CHECK REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  New resolved:    {new_checks}")
    print(f"  Total checked:   {total}")
    print(f"  Beat market:     {total_beat}/{total} ({total_beat/total:.0%})" if total else "")
    
    if total_bets:
        print(f"\n  💰 BETTING SIMULATION ($2/bet)")
        print(f"  Bets placed:     {total_bets}")
        print(f"  Won:             {total_wins} ({total_wins/total_bets:.0%})")
        print(f"  Simulated P&L:   ${total_pnl * 2:.2f}")
    
    if results_by_cat:
        print(f"\n  By category (new):")
        for cat, s in sorted(results_by_cat.items(), key=lambda x: -x[1]["n"]):
            beat_pct = f"{s['beat']/s['n']:.0%}" if s["n"] else "n/a"
            line = f"    {cat:>12s}: {s['n']:3d} checked, {beat_pct} beat mkt"
            if s["bets"]:
                line += f", {s['wins']}/{s['bets']} bets won, ${s['pnl']*2:+.2f}"
            print(line)

    # Return summary for Telegram notification
    return {
        "new": new_checks,
        "total": total,
        "beat_rate": round(total_beat / total, 2) if total else 0,
        "win_rate": round(total_wins / total_bets, 2) if total_bets else 0,
        "pnl": round(total_pnl * 2, 2),
    }


if __name__ == "__main__":
    check_predictions()
