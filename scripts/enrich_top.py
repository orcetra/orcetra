#!/usr/bin/env python3
"""Context-enriched re-prediction for high-value markets.

Picks the top N markets by volume/spread where Orcetra has low confidence,
runs deep LLM+news analysis, and updates the prediction.

Run: python3 scripts/enrich_top.py [--top 20] [--min-volume 10000]
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orcetra.fetcher import PolymarketFetcher
from orcetra.news import NewsCollector
from orcetra.strategy import LLMStrategy

ROOT = os.path.join(os.path.dirname(__file__), "..")
BATCH_PATH = os.path.join(ROOT, "results", "batch_predictions.json")
ENRICH_LOG = os.path.join(ROOT, "results", "enrichment_log.json")


def load_batch():
    with open(BATCH_PATH) as f:
        return json.load(f)


def save_batch(batch):
    with open(BATCH_PATH, "w") as f:
        json.dump(batch, f, indent=2)


def load_enrich_log():
    if os.path.exists(ENRICH_LOG):
        with open(ENRICH_LOG) as f:
            return json.load(f)
    return {"enrichments": []}


def save_enrich_log(log):
    with open(ENRICH_LOG, "w") as f:
        json.dump(log, f, indent=2)


def select_candidates(batch, top_n=20, min_volume=5000):
    """Select markets that would benefit most from deep analysis."""
    preds = batch["predictions"]
    candidates = []

    for cid, p in preds.items():
        # Skip already resolved
        if p.get("resolved") or p.get("actual") is not None:
            continue
        # Skip sports (low alpha from context)
        if p.get("is_sports"):
            continue
        # Skip already enriched
        if p.get("enriched"):
            continue

        volume = p.get("volume", 0)
        confidence = p.get("confidence", 0.5)
        tag = p.get("tag", "unknown")

        # Priority score: high volume + low confidence + non-sports tag
        tag_bonus = {
            "economy": 2.0, "politics": 1.8, "science": 1.5,
            "crypto": 1.3, "entertainment": 1.5, "commodities": 1.5
        }.get(tag, 1.0)

        score = (volume / 1000) * tag_bonus * (1.0 - confidence)
        candidates.append((cid, p, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n]


async def enrich_prediction(cid, pred, fetcher, news_collector, strategy):
    """Run deep LLM+news analysis on a single market."""
    question = pred["question"]
    slug = pred.get("slug", "")
    market_price = pred.get("market_price", 0.5)

    # 1. Search for relevant news
    try:
        news = await news_collector.search_news(question, days_back=3)
    except Exception as e:
        print(f"  News search failed: {e}")
        news = []

    # 2. Build minimal event object for LLM strategy
    from orcetra.models import Event, Token
    event = Event(
        id=cid,
        slug=slug,
        title=question,
        description=pred.get("description", question),
        end_date=pred.get("end_date"),
        volume=pred.get("volume", 0),
        liquidity=0,
        tokens=[Token(
            token_id=pred.get("token_id", ""),
            outcome="Yes",
            price=market_price,
        )],
        active=True,
    )

    # 3. Run LLM prediction
    try:
        result = strategy.predict(event, [], news, None)
        return {
            "enriched_prob": round(result.probability, 4),
            "original_prob": pred.get("our_prediction", pred.get("prediction")),
            "market_price": market_price,
            "confidence": round(result.confidence, 2) if hasattr(result, 'confidence') else 0.7,
            "news_count": len(news),
            "reasoning": result.reasoning[:500] if hasattr(result, 'reasoning') and result.reasoning else "",
            "enriched_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        print(f"  LLM prediction failed: {e}")
        return None


async def main(top_n=20, min_volume=5000):
    batch = load_batch()
    candidates = select_candidates(batch, top_n, min_volume)

    if not candidates:
        print("No candidates for enrichment.")
        return

    print(f"🔬 Enriching top {len(candidates)} markets...\n")

    fetcher = PolymarketFetcher()
    news_collector = NewsCollector()
    strategy = LLMStrategy()
    enrich_log = load_enrich_log()

    enriched_count = 0
    updated_count = 0

    for i, (cid, pred, score) in enumerate(candidates):
        tag = pred.get("tag", "?")
        q = pred["question"][:80]
        print(f"[{i+1}/{len(candidates)}] [{tag}] {q}")

        result = await enrich_prediction(cid, pred, fetcher, news_collector, strategy)

        if result:
            enriched_count += 1
            old_prob = result["original_prob"]
            new_prob = result["enriched_prob"]
            shift = abs(new_prob - old_prob) if old_prob else 0

            print(f"  Original: {old_prob:.1%} → Enriched: {new_prob:.1%} (shift: {shift:.1%}, news: {result['news_count']})")

            # Update batch prediction
            batch["predictions"][cid]["enriched"] = True
            batch["predictions"][cid]["enriched_prob"] = new_prob
            batch["predictions"][cid]["enriched_at"] = result["enriched_at"]
            batch["predictions"][cid]["news_count"] = result["news_count"]

            # If significant shift (>5%), update the main prediction
            if shift > 0.05:
                batch["predictions"][cid]["our_prediction"] = new_prob
                batch["predictions"][cid]["prediction_source"] = "enriched"
                updated_count += 1
                print(f"  ⚡ Prediction UPDATED (shift > 5%)")

            enrich_log["enrichments"].append({
                "condition_id": cid,
                "question": pred["question"],
                "tag": tag,
                **result,
            })

            # Rate limit: 2s between LLM calls
            time.sleep(2)
        else:
            print(f"  ⚠️ Skipped (enrichment failed)")

    save_batch(batch)
    save_enrich_log(enrich_log)

    print(f"\n✅ Done: {enriched_count} enriched, {updated_count} predictions updated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-volume", type=int, default=5000)
    args = parser.parse_args()
    asyncio.run(main(args.top, args.min_volume))
