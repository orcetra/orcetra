"""Base prediction strategy using LLM-based analysis."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

from groq import Groq

from .models import Event, NewsSignal, OrderBook, PricePoint, Prediction


class PredictionStrategy(ABC):
    """Base class for prediction strategies."""

    @abstractmethod
    def predict(
        self,
        event_data: Event,
        price_history: list[PricePoint],
        news_signals: list[NewsSignal],
        orderbook: Optional[OrderBook],
    ) -> Prediction:
        """Generate a prediction for the event."""
        pass

    def get_source_code(self) -> str:
        """Return the source code of this strategy for evolution."""
        import inspect
        return inspect.getsource(self.__class__)


class LLMStrategy(PredictionStrategy):
    """Default LLM-based prediction strategy."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def predict(
        self,
        event_data: Event,
        price_history: list[PricePoint],
        news_signals: list[NewsSignal],
        orderbook: Optional[OrderBook],
    ) -> Prediction:
        """Generate prediction using LLM analysis."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        # Build context for LLM
        context = self._build_context(event_data, price_history, news_signals, orderbook)
        prompt = self._build_prompt(context)

        try:
            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            prediction = self._parse_response(content, event_data)

            # Add market price for reference
            if event_data.tokens:
                prediction.market_price = event_data.tokens[0].price

            return prediction

        except Exception as e:
            print(f"Error in LLM prediction: {e}")
            # Fallback to market price
            market_prob = event_data.tokens[0].price if event_data.tokens else 0.5
            return Prediction(
                probability=market_prob,
                confidence=0.3,
                reasoning=f"Fallback to market price due to error: {str(e)}",
                trajectory_7d=[market_prob] * 7,
                market_price=market_prob,
            )

    def _get_system_prompt(self) -> str:
        return """You are an expert prediction market analyst. Your task is to analyze events and predict probabilities.

You will be given:
1. Event details (title, description, end date)
2. Current market prices
3. Price history (recent trends)
4. News signals with sentiment
5. Order book data (liquidity/depth)

Your goal is to predict:
1. The probability of the YES outcome (0-1)
2. Your confidence in this prediction (0-1)
3. Clear reasoning for your prediction
4. A 7-day trajectory of expected price movement

Focus on:
- News sentiment and relevance
- Price momentum and trends
- Market liquidity and depth
- Event timeline and deadlines
- Historical patterns

Be calibrated: if you're uncertain, reflect that in both probability and confidence.
Look for signals the market might be missing."""

    def _build_context(
        self,
        event: Event,
        price_history: list[PricePoint],
        news_signals: list[NewsSignal],
        orderbook: Optional[OrderBook],
    ) -> dict:
        """Build context dictionary for the prompt."""
        context = {
            "event": {
                "title": event.title,
                "description": event.description[:500] if event.description else "",
                "end_date": event.end_date.isoformat() if event.end_date else "Unknown",
                "current_prices": {t.outcome: t.price for t in event.tokens},
                "volume": event.volume,
                "liquidity": event.liquidity,
            },
            "price_history": [],
            "news_signals": [],
            "orderbook": None,
        }

        # Add recent price history (last 10 points)
        for point in price_history[-10:]:
            context["price_history"].append({
                "date": point.timestamp.strftime("%Y-%m-%d"),
                "price": round(point.price, 3),
            })

        # Add news signals (top 10 by relevance)
        sorted_signals = sorted(news_signals, key=lambda x: x.relevance, reverse=True)
        for signal in sorted_signals[:10]:
            context["news_signals"].append({
                "title": signal.title,
                "source": signal.source,
                "sentiment": signal.sentiment or "neutral",
                "relevance": signal.relevance,
                "snippet": signal.snippet[:200] if signal.snippet else "",
            })

        # Add orderbook summary
        if orderbook:
            context["orderbook"] = {
                "best_bid": orderbook.best_bid,
                "best_ask": orderbook.best_ask,
                "spread": orderbook.spread,
                "bid_depth": sum(b.size for b in orderbook.bids[:5]),
                "ask_depth": sum(a.size for a in orderbook.asks[:5]),
            }

        return context

    def _build_prompt(self, context: dict) -> str:
        """Build the analysis prompt."""
        return f"""Analyze this prediction market event and provide your prediction:

EVENT:
Title: {context['event']['title']}
Description: {context['event']['description']}
End Date: {context['event']['end_date']}
Current Prices: {json.dumps(context['event']['current_prices'])}
Volume: ${context['event']['volume']:,.0f}
Liquidity: ${context['event']['liquidity']:,.0f}

PRICE HISTORY (recent):
{json.dumps(context['price_history'], indent=2)}

NEWS SIGNALS:
{json.dumps(context['news_signals'], indent=2)}

ORDER BOOK:
{json.dumps(context['orderbook'], indent=2) if context['orderbook'] else 'Not available'}

Provide your prediction in this exact JSON format:
{{
    "probability": <float 0-1>,
    "confidence": <float 0-1>,
    "reasoning": "<string explaining your analysis>",
    "trajectory_7d": [<7 floats, daily predicted prices>]
}}

Focus on finding alpha - where might the market be wrong?"""

    def _parse_response(self, content: str, event: Event) -> Prediction:
        """Parse LLM response into Prediction object."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)

        if json_match:
            try:
                data = json.loads(json_match.group())

                probability = float(data.get("probability", 0.5))
                probability = max(0.01, min(0.99, probability))  # Clip to valid range

                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                trajectory = data.get("trajectory_7d", [])
                if not trajectory or len(trajectory) != 7:
                    trajectory = [probability] * 7
                trajectory = [max(0.01, min(0.99, float(p))) for p in trajectory]

                return Prediction(
                    probability=probability,
                    confidence=confidence,
                    reasoning=data.get("reasoning", "No reasoning provided"),
                    trajectory_7d=trajectory,
                )

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Error parsing LLM response: {e}")

        # Fallback
        market_prob = event.tokens[0].price if event.tokens else 0.5
        return Prediction(
            probability=market_prob,
            confidence=0.3,
            reasoning=f"Could not parse LLM response. Raw: {content[:200]}",
            trajectory_7d=[market_prob] * 7,
        )


# Strategy code template for evolution
STRATEGY_TEMPLATE = '''
class EvolvedStrategy(PredictionStrategy):
    """Evolved prediction strategy - version {version}."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def predict(
        self,
        event_data: Event,
        price_history: list[PricePoint],
        news_signals: list[NewsSignal],
        orderbook: Optional[OrderBook],
    ) -> Prediction:
        """Generate prediction using evolved analysis logic."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        # === EVOLVED ANALYSIS LOGIC START ===
        {analysis_code}
        # === EVOLVED ANALYSIS LOGIC END ===

        # Build prompt with analysis
        prompt = self._build_evolved_prompt(event_data, analysis_results)

        try:
            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {{"role": "system", "content": self._get_system_prompt()}},
                    {{"role": "user", "content": prompt}},
                ],
                temperature=0.2,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            return self._parse_response(content, event_data)

        except Exception as e:
            market_prob = event_data.tokens[0].price if event_data.tokens else 0.5
            return Prediction(
                probability=market_prob,
                confidence=0.3,
                reasoning=f"Error: {{str(e)}}",
                trajectory_7d=[market_prob] * 7,
                market_price=market_prob,
            )

    def _get_system_prompt(self) -> str:
        return """You are an expert prediction market analyst. Predict probabilities accurately.
Output JSON: {{"probability": float, "confidence": float, "reasoning": str, "trajectory_7d": [7 floats]}}"""

    def _build_evolved_prompt(self, event: Event, analysis: dict) -> str:
        return f"""Event: {{event.title}}
Current market: {{event.tokens[0].price if event.tokens else 0.5:.2f}}
Analysis: {{json.dumps(analysis)}}

Predict probability (0-1), confidence (0-1), reasoning, 7-day trajectory.
Output valid JSON only."""

    def _parse_response(self, content: str, event: Event) -> Prediction:
        import re
        json_match = re.search(r'\\{{[\\s\\S]*\\}}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                prob = max(0.01, min(0.99, float(data.get("probability", 0.5))))
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                traj = data.get("trajectory_7d", [prob]*7)
                traj = [max(0.01, min(0.99, float(p))) for p in traj][:7]
                if len(traj) < 7:
                    traj.extend([prob] * (7 - len(traj)))
                return Prediction(
                    probability=prob,
                    confidence=conf,
                    reasoning=data.get("reasoning", ""),
                    trajectory_7d=traj,
                    market_price=event.tokens[0].price if event.tokens else None,
                )
            except:
                pass
        mp = event.tokens[0].price if event.tokens else 0.5
        return Prediction(probability=mp, confidence=0.3, reasoning="Parse error", trajectory_7d=[mp]*7, market_price=mp)
'''

# Default analysis code for initial strategy
DEFAULT_ANALYSIS_CODE = '''
        # Compute basic signals
        analysis_results = {}

        # Price momentum
        if price_history and len(price_history) >= 2:
            recent_prices = [p.price for p in price_history[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 0.01)
            analysis_results["momentum"] = momentum
            analysis_results["price_trend"] = "up" if momentum > 0.05 else "down" if momentum < -0.05 else "flat"
        else:
            analysis_results["momentum"] = 0
            analysis_results["price_trend"] = "unknown"

        # News sentiment aggregate
        if news_signals:
            positive = sum(1 for n in news_signals if n.sentiment == "positive")
            negative = sum(1 for n in news_signals if n.sentiment == "negative")
            total = len(news_signals)
            analysis_results["news_sentiment_score"] = (positive - negative) / max(total, 1)
            analysis_results["news_volume"] = total
        else:
            analysis_results["news_sentiment_score"] = 0
            analysis_results["news_volume"] = 0

        # Orderbook imbalance
        if orderbook and orderbook.bids and orderbook.asks:
            bid_vol = sum(b.size for b in orderbook.bids[:5])
            ask_vol = sum(a.size for a in orderbook.asks[:5])
            analysis_results["orderbook_imbalance"] = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1)
        else:
            analysis_results["orderbook_imbalance"] = 0

        # Market price
        analysis_results["market_price"] = event_data.tokens[0].price if event_data.tokens else 0.5
'''


def load_strategy_from_code(code: str) -> Optional[PredictionStrategy]:
    """Dynamically load a strategy from code string."""
    try:
        # Create a namespace with required imports
        namespace = {
            "PredictionStrategy": PredictionStrategy,
            "Event": Event,
            "PricePoint": PricePoint,
            "NewsSignal": NewsSignal,
            "OrderBook": OrderBook,
            "Prediction": Prediction,
            "Groq": Groq,
            "json": json,
            "os": os,
            "re": re,
            "Optional": Optional,
        }

        exec(code, namespace)

        # Find the strategy class
        for name, obj in namespace.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, PredictionStrategy)
                and obj is not PredictionStrategy
            ):
                return obj()

        return None

    except Exception as e:
        print(f"Error loading strategy: {e}")
        return None


def get_default_strategy() -> PredictionStrategy:
    """Get the default LLM strategy."""
    return LLMStrategy()
