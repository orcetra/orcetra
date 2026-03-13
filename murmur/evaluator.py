"""Backtest & scoring system for prediction strategies."""

import math
from collections import defaultdict
from typing import Optional

from .fetcher import PolymarketFetcher
from .models import BacktestResult, Event, EventResult, Prediction
from .news import NewsCollector
from .strategy import PredictionStrategy


class StrategyEvaluator:
    """Evaluates prediction strategies against resolved events."""

    def __init__(self):
        self.fetcher = PolymarketFetcher()
        self.news_collector = NewsCollector()

    async def close(self):
        await self.fetcher.close()
        await self.news_collector.close()

    async def evaluate_strategy(
        self,
        strategy: PredictionStrategy,
        resolved_events: list[Event],
        fetch_context: bool = True,
    ) -> BacktestResult:
        """Evaluate a strategy against resolved events."""
        results = []
        total_brier = 0.0
        total_market_brier = 0.0
        total_log_loss = 0.0
        beat_market_count = 0

        # Calibration tracking
        calibration_data = defaultdict(lambda: {"predictions": [], "outcomes": []})

        for event in resolved_events:
            if event.outcome_price is None:
                continue

            actual_outcome = event.outcome_price  # 0 or 1

            # Get market's prediction (last known price before resolution)
            market_prob = 0.5
            if event.tokens:
                # Use the YES token price as market prediction
                for token in event.tokens:
                    if token.outcome.lower() in ["yes", "true", "1"]:
                        market_prob = token.price
                        break
                else:
                    market_prob = event.tokens[0].price

            # Get strategy prediction
            try:
                if fetch_context:
                    # Fetch additional context
                    price_history = []
                    orderbook = None
                    news_signals = []

                    if event.tokens:
                        token_id = event.tokens[0].token_id
                        if token_id:
                            price_history = await self.fetcher.fetch_price_history(token_id)
                            orderbook = await self.fetcher.fetch_orderbook(token_id)

                    # Search for news (limited to avoid rate limits in backtesting)
                    if event.title:
                        news_signals = await self.news_collector.search_news(
                            event.title[:100], days_back=7
                        )
                        if news_signals:
                            news_signals = await self.news_collector.summarize_signals(
                                news_signals, event.title
                            )

                    prediction = strategy.predict(
                        event, price_history, news_signals, orderbook
                    )
                else:
                    # Minimal context prediction
                    prediction = strategy.predict(event, [], [], None)

            except Exception as e:
                print(f"Error predicting event {event.id}: {e}")
                # Use market price as fallback
                prediction = Prediction(
                    probability=market_prob,
                    confidence=0.3,
                    reasoning=f"Error: {str(e)}",
                    trajectory_7d=[market_prob] * 7,
                )

            # Calculate metrics
            pred_prob = max(0.01, min(0.99, prediction.probability))
            market_prob = max(0.01, min(0.99, market_prob))

            # Brier score: (prediction - outcome)^2
            brier = (pred_prob - actual_outcome) ** 2
            market_brier = (market_prob - actual_outcome) ** 2

            # Log loss: -[y*log(p) + (1-y)*log(1-p)]
            log_loss = -(
                actual_outcome * math.log(pred_prob)
                + (1 - actual_outcome) * math.log(1 - pred_prob)
            )

            beat_market = brier < market_brier

            result = EventResult(
                event_id=event.id,
                event_title=event.title,
                predicted_prob=pred_prob,
                market_prob=market_prob,
                actual_outcome=actual_outcome,
                brier_score=brier,
                market_brier_score=market_brier,
                beat_market=beat_market,
                reasoning=prediction.reasoning[:200],
            )
            results.append(result)

            total_brier += brier
            total_market_brier += market_brier
            total_log_loss += log_loss

            if beat_market:
                beat_market_count += 1

            # Track calibration by confidence bucket
            bucket = self._get_confidence_bucket(prediction.confidence)
            calibration_data[bucket]["predictions"].append(pred_prob)
            calibration_data[bucket]["outcomes"].append(actual_outcome)

        # Calculate aggregate metrics
        n = len(results)
        if n == 0:
            return BacktestResult(
                total_events=0,
                brier_score=1.0,
                market_brier_score=1.0,
                log_loss=float("inf"),
                accuracy_50=0.0,
                accuracy_70=0.0,
                calibration_buckets={},
                beat_market_rate=0.0,
                per_event_results=[],
            )

        avg_brier = total_brier / n
        avg_market_brier = total_market_brier / n
        avg_log_loss = total_log_loss / n
        beat_market_rate = beat_market_count / n

        # Accuracy at thresholds
        correct_50 = sum(
            1
            for r in results
            if (r.predicted_prob > 0.5) == (r.actual_outcome > 0.5)
        )
        correct_70 = sum(
            1
            for r in results
            if abs(r.predicted_prob - 0.5) > 0.2
            and (r.predicted_prob > 0.5) == (r.actual_outcome > 0.5)
        )
        high_conf_count = sum(1 for r in results if abs(r.predicted_prob - 0.5) > 0.2)

        accuracy_50 = correct_50 / n
        accuracy_70 = correct_70 / high_conf_count if high_conf_count > 0 else 0.0

        # Calculate calibration metrics
        calibration_buckets = {}
        for bucket, data in calibration_data.items():
            if data["predictions"]:
                avg_pred = sum(data["predictions"]) / len(data["predictions"])
                avg_outcome = sum(data["outcomes"]) / len(data["outcomes"])
                calibration_buckets[bucket] = {
                    "avg_prediction": avg_pred,
                    "actual_rate": avg_outcome,
                    "count": len(data["predictions"]),
                    "calibration_error": abs(avg_pred - avg_outcome),
                }

        return BacktestResult(
            total_events=n,
            brier_score=avg_brier,
            market_brier_score=avg_market_brier,
            log_loss=avg_log_loss,
            accuracy_50=accuracy_50,
            accuracy_70=accuracy_70,
            calibration_buckets=calibration_buckets,
            beat_market_rate=beat_market_rate,
            per_event_results=results,
        )

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Map confidence to a bucket for calibration analysis."""
        if confidence < 0.3:
            return "0-30%"
        elif confidence < 0.5:
            return "30-50%"
        elif confidence < 0.7:
            return "50-70%"
        elif confidence < 0.9:
            return "70-90%"
        else:
            return "90-100%"


def evaluate_strategy_sync(
    strategy: PredictionStrategy,
    resolved_events: list[Event],
    fetch_context: bool = False,  # Default to False for speed in backtesting
) -> BacktestResult:
    """Synchronous wrapper for evaluate_strategy."""
    import asyncio

    evaluator = StrategyEvaluator()
    try:
        return asyncio.run(
            evaluator.evaluate_strategy(strategy, resolved_events, fetch_context)
        )
    finally:
        asyncio.run(evaluator.close())


def quick_evaluate(
    strategy: PredictionStrategy, resolved_events: list[Event]
) -> BacktestResult:
    """Quick evaluation without fetching additional context (faster for evolution)."""
    return evaluate_strategy_sync(strategy, resolved_events, fetch_context=False)


def format_backtest_results(result: BacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        f"Total Events: {result.total_events}",
        "",
        "SCORING METRICS:",
        f"  Brier Score:        {result.brier_score:.4f} (lower is better)",
        f"  Market Brier Score: {result.market_brier_score:.4f}",
        f"  Log Loss:           {result.log_loss:.4f}",
        "",
        "PERFORMANCE:",
        f"  Beat Market Rate:   {result.beat_market_rate:.1%}",
        f"  Accuracy (>50%):    {result.accuracy_50:.1%}",
        f"  Accuracy (>70%):    {result.accuracy_70:.1%}",
        "",
        "CALIBRATION:",
    ]

    for bucket, data in sorted(result.calibration_buckets.items()):
        lines.append(
            f"  {bucket}: predicted={data['avg_prediction']:.2f}, "
            f"actual={data['actual_rate']:.2f}, n={data['count']}, "
            f"error={data['calibration_error']:.3f}"
        )

    if result.beat_market_rate > 0.5:
        lines.append("")
        lines.append("STATUS: Strategy BEATS market baseline!")
    else:
        lines.append("")
        lines.append("STATUS: Strategy underperforms market baseline")

    lines.append("=" * 60)

    return "\n".join(lines)
