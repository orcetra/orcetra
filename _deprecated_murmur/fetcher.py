"""Polymarket data fetcher using CLOB and Gamma APIs."""

import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import httpx

from .models import Event, OrderBook, OrderBookLevel, OutcomeToken, PricePoint

# API endpoints
CLOB_API_BASE = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Default timeout and retry settings
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class PolymarketFetcher:
    """Fetches data from Polymarket APIs."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request_with_retry(
        self, method: str, url: str, **kwargs
    ) -> httpx.Response:
        """Make request with retry logic."""
        client = await self._get_client()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    import asyncio
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    last_error = e
                    continue
                raise
            except httpx.RequestError as e:
                import asyncio
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                last_error = e
                continue

        raise last_error or Exception("Max retries exceeded")

    def _extract_slug_from_url(self, url_or_slug: str) -> str:
        """Extract event slug from URL or return as-is if already a slug."""
        if url_or_slug.startswith("http"):
            parsed = urlparse(url_or_slug)
            path = parsed.path.strip("/")
            # Handle /event/slug format
            if path.startswith("event/"):
                return path.replace("event/", "")
            return path
        return url_or_slug

    async def fetch_event(self, url_or_slug: str) -> Optional[Event]:
        """Fetch event info, current prices, volume, outcome tokens."""
        slug = self._extract_slug_from_url(url_or_slug)

        # Try fetching from Gamma API by slug
        try:
            response = await self._request_with_retry(
                "GET",
                f"{GAMMA_API_BASE}/events",
                params={"slug": slug}
            )
            events = response.json()

            if not events:
                # Try searching by slug as part of ID
                response = await self._request_with_retry(
                    "GET",
                    f"{GAMMA_API_BASE}/events",
                    params={"slug_contains": slug, "limit": 1}
                )
                events = response.json()

            if events and len(events) > 0:
                event_data = events[0]
                return self._parse_event(event_data)

        except Exception as e:
            print(f"Error fetching event by slug: {e}")

        # Fallback: try markets endpoint
        try:
            response = await self._request_with_retry(
                "GET",
                f"{GAMMA_API_BASE}/markets",
                params={"slug": slug}
            )
            markets = response.json()

            if markets and len(markets) > 0:
                market_data = markets[0]
                return self._parse_market_as_event(market_data)

        except Exception as e:
            print(f"Error fetching market: {e}")

        return None

    @staticmethod
    def _parse_json_field(value, default=None):
        """Parse a field that might be a JSON string or already a list."""
        import json as _json
        if isinstance(value, str):
            try:
                return _json.loads(value)
            except (ValueError, TypeError):
                return default or []
        return value if value else (default or [])

    def _parse_event(self, data: dict) -> Event:
        """Parse event data from Gamma API response."""
        tokens = []
        markets = data.get("markets", [])

        for market in markets:
            # Each market can have YES/NO tokens — fields may be JSON strings
            outcomes = self._parse_json_field(market.get("outcomes"), ["Yes", "No"])
            prices = self._parse_json_field(market.get("outcomePrices"), ["0.5", "0.5"])
            token_ids = self._parse_json_field(market.get("clobTokenIds"), ["", ""])

            for i, outcome in enumerate(outcomes):
                token_id = token_ids[i] if i < len(token_ids) else ""
                price = float(prices[i]) if i < len(prices) else 0.5

                tokens.append(OutcomeToken(
                    token_id=token_id or market.get("conditionId", ""),
                    outcome=outcome,
                    price=price
                ))

        # Determine if resolved
        closed = data.get("closed", False)
        resolved = closed and data.get("resolutionSource") is not None

        # Parse end date
        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # For resolved events, compute outcome_price from first market's YES token
        outcome = data.get("outcome")
        outcome_price = None
        if closed and markets:
            first_market = markets[0]
            fm_outcomes = self._parse_json_field(first_market.get("outcomes"), ["Yes", "No"])
            fm_prices = self._parse_json_field(first_market.get("outcomePrices"), ["0.5", "0.5"])
            yes_price = None
            for i, o in enumerate(fm_outcomes):
                if o.lower() in ("yes", "true", "1"):
                    yes_price = float(fm_prices[i]) if i < len(fm_prices) else None
                    break
            if yes_price is None and len(fm_prices) >= 1:
                yes_price = float(fm_prices[0])
            if yes_price is not None:
                outcome_price = 1.0 if yes_price > 0.5 else 0.0
            if outcome is None and outcome_price is not None:
                outcome = "Yes" if outcome_price > 0.5 else "No"

        return Event(
            id=data.get("id", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            end_date=end_date,
            active=data.get("active", True),
            closed=closed,
            resolved=resolved,
            outcome=outcome,
            outcome_price=outcome_price,
            tokens=tokens,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
        )

    def _parse_market_as_event(self, data: dict) -> Event:
        """Parse market data as an event (single-market event)."""
        tokens = []
        outcomes = self._parse_json_field(data.get("outcomes"), ["Yes", "No"])
        prices = self._parse_json_field(data.get("outcomePrices"), ["0.5", "0.5"])
        token_ids = self._parse_json_field(data.get("clobTokenIds"), ["", ""])

        for i, outcome in enumerate(outcomes):
            token_id = token_ids[i] if i < len(token_ids) else ""
            price = float(prices[i]) if i < len(prices) else 0.5
            tokens.append(OutcomeToken(
                token_id=token_id,
                outcome=outcome,
                price=price
            ))

        closed = data.get("closed", False)
        resolved = closed

        # For resolved markets, determine the outcome
        # outcome_price = YES token final price (1.0 if Yes won, ~0.0 if No won)
        outcome = None
        outcome_price = None
        if resolved and prices:
            # Find YES token price
            yes_price = None
            for i, o in enumerate(outcomes):
                if o.lower() in ("yes", "true", "1"):
                    yes_price = float(prices[i])
                    break
            if yes_price is None and len(prices) >= 1:
                yes_price = float(prices[0])  # fallback: first token = YES

            outcome_price = 1.0 if yes_price > 0.5 else 0.0
            outcome = "Yes" if outcome_price > 0.5 else "No"

        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Event(
            id=data.get("conditionId", data.get("id", "")),
            slug=data.get("slug", ""),
            title=data.get("question", data.get("title", "")),
            description=data.get("description", ""),
            end_date=end_date,
            active=data.get("active", not closed),
            closed=closed,
            resolved=resolved,
            outcome=outcome,
            outcome_price=outcome_price,
            tokens=tokens,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
        )

    async def fetch_price_history(
        self, token_id: str, interval: str = "1d", days_back: int = 30
    ) -> list[PricePoint]:
        """Fetch historical price series for a token."""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)

            # Map interval to fidelity
            fidelity_map = {"1h": 60, "1d": 1440, "1w": 10080}
            fidelity = fidelity_map.get(interval, 1440)

            response = await self._request_with_retry(
                "GET",
                f"{CLOB_API_BASE}/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": fidelity,
                    "startTs": int(start_time.timestamp()),
                    "endTs": int(end_time.timestamp()),
                }
            )

            data = response.json()
            history = data.get("history", [])

            price_points = []
            for point in history:
                try:
                    ts = datetime.fromtimestamp(point.get("t", 0))
                    price = float(point.get("p", 0.5))
                    price_points.append(PricePoint(timestamp=ts, price=price))
                except (ValueError, TypeError):
                    continue

            return sorted(price_points, key=lambda x: x.timestamp)

        except Exception as e:
            print(f"Error fetching price history: {e}")
            return []

    async def fetch_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """Fetch current order book depth for a token."""
        try:
            response = await self._request_with_retry(
                "GET",
                f"{CLOB_API_BASE}/book",
                params={"token_id": token_id}
            )

            data = response.json()

            bids = [
                OrderBookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
                for a in data.get("asks", [])
            ]

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return OrderBook(
                token_id=token_id,
                bids=bids,
                asks=asks,
            )

        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return None

    async def list_active_events(self, limit: int = 50) -> list[Event]:
        """List trending/active events."""
        try:
            response = await self._request_with_retry(
                "GET",
                f"{GAMMA_API_BASE}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volume",
                    "ascending": "false",
                }
            )

            events_data = response.json()
            events = [self._parse_event(e) for e in events_data]
            return events

        except Exception as e:
            print(f"Error listing active events: {e}")
            return []

    async def list_resolved_events(self, limit: int = 100) -> list[Event]:
        """List resolved events for backtesting."""
        try:
            # Fetch closed/resolved markets
            response = await self._request_with_retry(
                "GET",
                f"{GAMMA_API_BASE}/markets",
                params={
                    "closed": "true",
                    "limit": limit,
                    "order": "endDate",
                    "ascending": "false",
                }
            )

            markets_data = response.json()
            events = []

            for market in markets_data:
                event = self._parse_market_as_event(market)
                # Only include if we can determine the outcome
                if event.resolved and event.outcome_price is not None:
                    events.append(event)

            return events

        except Exception as e:
            print(f"Error listing resolved events: {e}")
            return []


# Sync wrappers for CLI usage
def fetch_event_sync(url_or_slug: str) -> Optional[Event]:
    """Synchronous wrapper for fetch_event."""
    import asyncio
    fetcher = PolymarketFetcher()
    try:
        return asyncio.run(fetcher.fetch_event(url_or_slug))
    finally:
        asyncio.run(fetcher.close())


def fetch_price_history_sync(token_id: str, interval: str = "1d") -> list[PricePoint]:
    """Synchronous wrapper for fetch_price_history."""
    import asyncio
    fetcher = PolymarketFetcher()
    try:
        return asyncio.run(fetcher.fetch_price_history(token_id, interval))
    finally:
        asyncio.run(fetcher.close())


def _run_async(coro_fn):
    """Run an async fetcher operation synchronously, handling cleanup in same loop."""
    import asyncio

    async def _wrapper():
        fetcher = PolymarketFetcher()
        try:
            return await coro_fn(fetcher)
        finally:
            if fetcher._client and not fetcher._client.is_closed:
                await fetcher._client.aclose()

    return asyncio.run(_wrapper())


def fetch_event_sync(url_or_slug: str) -> Optional[Event]:
    return _run_async(lambda f: f.fetch_event(url_or_slug))


def fetch_price_history_sync(token_id: str, interval: str = "1d", days_back: int = 30) -> list[PricePoint]:
    return _run_async(lambda f: f.fetch_price_history(token_id, interval, days_back))


def fetch_orderbook_sync(token_id: str) -> Optional[OrderBook]:
    return _run_async(lambda f: f.fetch_orderbook(token_id))


def list_active_events_sync(limit: int = 50) -> list[Event]:
    return _run_async(lambda f: f.list_active_events(limit))


def list_resolved_events_sync(limit: int = 100) -> list[Event]:
    return _run_async(lambda f: f.list_resolved_events(limit))
