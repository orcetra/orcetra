"""News & signal collector using Brave Search API."""

import os
from datetime import datetime, timedelta
from typing import Optional

import httpx
from groq import Groq

from .models import NewsSignal

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"
DEFAULT_TIMEOUT = 30.0


class NewsCollector:
    """Collects and processes news signals for prediction markets."""

    def __init__(self, brave_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search_news(self, query: str, days_back: int = 7) -> list[NewsSignal]:
        """Search recent news via Brave Search API."""
        if not self.brave_api_key:
            print("Warning: BRAVE_API_KEY not set, returning empty news")
            return []

        client = await self._get_client()

        try:
            # Use Brave News Search endpoint
            response = await client.get(
                f"{BRAVE_API_BASE}/news/search",
                params={
                    "q": query,
                    "count": 20,
                    "freshness": f"pd" if days_back <= 1 else f"pw" if days_back <= 7 else "pm",
                },
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.brave_api_key,
                }
            )
            response.raise_for_status()
            data = response.json()

            signals = []
            results = data.get("results", [])

            for result in results:
                # Parse date
                date = None
                if result.get("age"):
                    # Brave returns relative time like "2 hours ago"
                    date = self._parse_relative_date(result["age"])
                elif result.get("page_age"):
                    try:
                        date = datetime.fromisoformat(result["page_age"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                signal = NewsSignal(
                    title=result.get("title", ""),
                    snippet=result.get("description", ""),
                    source=result.get("meta_url", {}).get("hostname", result.get("url", "")),
                    url=result.get("url", ""),
                    date=date,
                )
                signals.append(signal)

            return signals

        except httpx.HTTPStatusError as e:
            print(f"Brave API error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error searching news: {e}")
            return []

    def _parse_relative_date(self, age_str: str) -> Optional[datetime]:
        """Parse relative date strings like '2 hours ago'."""
        now = datetime.utcnow()
        age_str = age_str.lower()

        try:
            if "hour" in age_str:
                hours = int(age_str.split()[0])
                return now - timedelta(hours=hours)
            elif "day" in age_str:
                days = int(age_str.split()[0])
                return now - timedelta(days=days)
            elif "minute" in age_str:
                minutes = int(age_str.split()[0])
                return now - timedelta(minutes=minutes)
            elif "week" in age_str:
                weeks = int(age_str.split()[0])
                return now - timedelta(weeks=weeks)
        except (ValueError, IndexError):
            pass

        return None

    async def summarize_signals(
        self, news_items: list[NewsSignal], event_context: str
    ) -> list[NewsSignal]:
        """Use LLM to summarize key signals and add sentiment analysis."""
        if not news_items:
            return []

        if not self.groq_api_key:
            print("Warning: GROQ_API_KEY not set, returning unsummarized signals")
            return news_items

        # Prepare news context for LLM
        news_text = "\n".join([
            f"- [{item.source}] {item.title}: {item.snippet}"
            for item in news_items[:15]  # Limit to avoid token limits
        ])

        prompt = f"""Analyze these news items related to the prediction market event: "{event_context}"

News items:
{news_text}

For each news item, provide:
1. A relevance score (0-1) for how relevant it is to the event
2. Sentiment (positive, negative, neutral) regarding the event outcome being YES

Respond in JSON format:
{{
    "signals": [
        {{"index": 0, "relevance": 0.8, "sentiment": "positive"}},
        ...
    ]
}}

Only include the JSON, no other text."""

        try:
            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            import json
            import re

            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                signals_data = data.get("signals", [])

                # Update news items with analysis
                for signal_info in signals_data:
                    idx = signal_info.get("index", -1)
                    if 0 <= idx < len(news_items):
                        news_items[idx].relevance = signal_info.get("relevance", 0.5)
                        news_items[idx].sentiment = signal_info.get("sentiment", "neutral")

        except Exception as e:
            print(f"Error summarizing signals: {e}")

        return news_items


# Sync wrappers for CLI usage
def search_news_sync(query: str, days_back: int = 7) -> list[NewsSignal]:
    """Synchronous wrapper for search_news."""
    import asyncio
    collector = NewsCollector()
    try:
        return asyncio.run(collector.search_news(query, days_back))
    finally:
        asyncio.run(collector.close())


def summarize_signals_sync(
    news_items: list[NewsSignal], event_context: str
) -> list[NewsSignal]:
    """Synchronous wrapper for summarize_signals."""
    import asyncio
    collector = NewsCollector()
    try:
        return asyncio.run(collector.summarize_signals(news_items, event_context))
    finally:
        asyncio.run(collector.close())
