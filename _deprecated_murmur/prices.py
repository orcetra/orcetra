"""Real-time price fetcher for market context."""

import httpx
from typing import Optional


def get_crypto_prices() -> dict:
    """Get current crypto prices from CoinGecko (free, no API key)."""
    try:
        r = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin,ethereum,solana",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true",
            },
            timeout=10,
        )
        data = r.json()
        return {
            "bitcoin": {
                "price": data.get("bitcoin", {}).get("usd"),
                "change_24h": data.get("bitcoin", {}).get("usd_24h_change"),
                "volume_24h": data.get("bitcoin", {}).get("usd_24h_vol"),
                "market_cap": data.get("bitcoin", {}).get("usd_market_cap"),
            },
            "ethereum": {
                "price": data.get("ethereum", {}).get("usd"),
                "change_24h": data.get("ethereum", {}).get("usd_24h_change"),
            },
            "solana": {
                "price": data.get("solana", {}).get("usd"),
                "change_24h": data.get("solana", {}).get("usd_24h_change"),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def get_commodity_prices() -> dict:
    """Get commodity prices from Yahoo Finance (free, no API key)."""
    symbols = {
        "crude_oil": "CL=F",
        "gold": "GC=F",
        "natural_gas": "NG=F",
    }
    results = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for name, sym in symbols.items():
        try:
            r = httpx.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"interval": "1d", "range": "5d"},
                headers=headers,
                timeout=10,
            )
            meta = r.json()["chart"]["result"][0]["meta"]
            results[name] = {
                "price": meta.get("regularMarketPrice"),
                "prev_close": meta.get("chartPreviousClose"),
                "change_pct": round(
                    (meta.get("regularMarketPrice", 0) - meta.get("chartPreviousClose", 0))
                    / max(meta.get("chartPreviousClose", 1), 0.01)
                    * 100,
                    2,
                ),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def get_market_indices() -> dict:
    """Get major market indices."""
    symbols = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "vix": "^VIX",
    }
    results = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for name, sym in symbols.items():
        try:
            r = httpx.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"interval": "1d", "range": "5d"},
                headers=headers,
                timeout=10,
            )
            meta = r.json()["chart"]["result"][0]["meta"]
            results[name] = {
                "price": meta.get("regularMarketPrice"),
                "prev_close": meta.get("chartPreviousClose"),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def get_price_context(event_title: str) -> Optional[str]:
    """Build price context string relevant to the event."""
    title_lower = event_title.lower()
    context_parts = []

    # Detect what prices are relevant
    if any(kw in title_lower for kw in ["bitcoin", "btc", "crypto"]):
        crypto = get_crypto_prices()
        btc = crypto.get("bitcoin", {})
        if btc.get("price"):
            context_parts.append(
                f"CURRENT BITCOIN PRICE: ${btc['price']:,.0f} "
                f"(24h change: {btc.get('change_24h', 0):+.1f}%)"
            )

    if any(kw in title_lower for kw in ["ethereum", "eth"]):
        crypto = get_crypto_prices()
        eth = crypto.get("ethereum", {})
        if eth.get("price"):
            context_parts.append(f"CURRENT ETHEREUM PRICE: ${eth['price']:,.0f}")

    if any(kw in title_lower for kw in ["solana", "sol"]):
        crypto = get_crypto_prices()
        sol = crypto.get("solana", {})
        if sol.get("price"):
            context_parts.append(f"CURRENT SOLANA PRICE: ${sol['price']:,.0f}")

    if any(kw in title_lower for kw in ["crude oil", "oil", " cl "]):
        comm = get_commodity_prices()
        oil = comm.get("crude_oil", {})
        if oil.get("price"):
            context_parts.append(
                f"CURRENT CRUDE OIL PRICE: ${oil['price']:.2f}/barrel "
                f"(prev close: ${oil.get('prev_close', 0):.2f}, "
                f"change: {oil.get('change_pct', 0):+.1f}%)"
            )

    if any(kw in title_lower for kw in ["gold"]):
        comm = get_commodity_prices()
        gold = comm.get("gold", {})
        if gold.get("price"):
            context_parts.append(f"CURRENT GOLD PRICE: ${gold['price']:,.0f}/oz")

    if any(kw in title_lower for kw in ["s&p", "sp500", "stock market", "fed", "interest rate"]):
        idx = get_market_indices()
        sp = idx.get("sp500", {})
        vix = idx.get("vix", {})
        if sp.get("price"):
            context_parts.append(f"S&P 500: {sp['price']:,.0f}")
        if vix.get("price"):
            context_parts.append(f"VIX (volatility): {vix['price']:.1f}")

    return "\n".join(context_parts) if context_parts else None
