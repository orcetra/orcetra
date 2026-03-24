"""Context Planner — Auto-detect knowledge gaps and inject domain context.

The core alpha engine. Oscar experiment showed +8% edge with context injection.
Generalizes the ad-hoc Oscar approach to ALL event types.

Pipeline: parse_event → classify_domain → identify_gaps → generate_queries → 
          fetch_context → structure_signals → inject_into_prediction
"""

import re
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Domain(Enum):
    AWARDS = "awards"         # Oscar, Grammy, Emmy — precursor correlation
    ECONOMY = "economy"       # Fed, CPI, GDP, unemployment — leading indicators  
    POLICY = "policy"         # Government decisions, regulations — language patterns
    COMMODITIES = "commodities"  # Oil, gold, silver — current price + trend
    CRYPTO = "crypto"         # BTC, ETH, SOL — price + sentiment + on-chain
    SPORTS = "sports"         # NBA, NFL, soccer — injury/form/schedule
    GOLF = "golf"             # PGA — recent form, course history
    TECH = "tech"             # Product launches, acquisitions — supply chain signals
    GEOPOLITICS = "geopolitics"  # Wars, treaties, sanctions — escalation patterns
    ENTERTAINMENT = "entertainment"  # Box office, streaming — social buzz
    UNKNOWN = "unknown"


@dataclass
class ContextPlan:
    """What we need to know to predict this event."""
    domain: Domain
    event_title: str
    knowledge_gaps: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    time_sensitivity: str = "medium"  # low/medium/high
    predictability: str = "medium"    # low/medium/high (from Phase 1)
    tetlock_decomposition: list[str] = field(default_factory=list)


@dataclass
class ContextSignal:
    """A piece of context retrieved from search."""
    source: str
    text: str
    relevance: float  # 0-1
    novelty: float    # 0-1: how much new info vs common knowledge
    direction: Optional[str] = None  # "supports_yes", "supports_no", "neutral"


@dataclass 
class EnrichedContext:
    """Full context package ready for LLM injection."""
    plan: ContextPlan
    signals: list[ContextSignal] = field(default_factory=list)
    price_context: Optional[str] = None
    base_rate: Optional[float] = None  # Tetlock: historical base rate
    summary: str = ""


# === DOMAIN CLASSIFICATION ===

DOMAIN_KEYWORDS = {
    Domain.AWARDS: ["oscar", "grammy", "emmy", "tony", "golden globe", "bafta", 
                    "best picture", "best actor", "best director", "nomination",
                    "academy award", "sag", "critics choice"],
    Domain.ECONOMY: ["fed", "federal reserve", "unemployment", "cpi", "gdp", 
                     "inflation", "rate cut", "rate hike", "jobs report", "payroll",
                     "tsa passenger", "consumer confidence", "pmi", "retail sales"],
    Domain.POLICY: ["bill", "executive order", "tariff", "sanction", "regulation",
                    "congress", "senate vote", "legislation", "government shutdown"],
    Domain.COMMODITIES: ["oil", "crude", "gold", "silver", "natural gas", "copper",
                         "wheat", "corn", "commodity"],
    Domain.CRYPTO: ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
                    "token", "defi", "nft", "fdv", "market cap"],
    Domain.SPORTS: ["nba", "nfl", "mlb", "nhl", "premier league", "la liga", 
                    "champions league", "ufc", "boxing", "handicap", "o/u",
                    "spread", "moneyline", "celtics", "lakers", "warriors",
                    "heat", "nets", "knicks", "bucks", "nuggets", "76ers",
                    "mavericks", "suns", "packers", "chiefs", "eagles",
                    "cowboys", "win on", "beat the", "vs."],
    Domain.GOLF: ["golf", "pga", "players championship", "masters", "us open golf",
                  "top 5", "top 10", "top 20", "make the cut"],
    Domain.TECH: ["apple", "google", "meta", "microsoft", "tesla", "nvidia",
                  "iphone", "launch", "release date", "acquisition"],
    Domain.GEOPOLITICS: ["war", "ceasefire", "nato", "invasion", "missile",
                         "nuclear", "treaty", "diplomatic"],
    Domain.ENTERTAINMENT: ["box office", "streaming", "netflix", "disney",
                           "album", "concert", "ticket"],
}

DOMAIN_PREDICTABILITY = {
    Domain.AWARDS: "high",       # Precursor awards highly correlated
    Domain.ECONOMY: "high",      # Leading indicators + inertia
    Domain.POLICY: "medium",     # Language analysis useful but unpredictable
    Domain.COMMODITIES: "medium",  # Price context helps but volatile
    Domain.CRYPTO: "medium",     # Price helps, but very volatile
    Domain.SPORTS: "low",        # Bookmakers already price most info
    Domain.GOLF: "low",          # High variance, field too large
    Domain.TECH: "medium",       # Supply chain leaks useful
    Domain.GEOPOLITICS: "low",   # True uncertainty
    Domain.ENTERTAINMENT: "medium",
    Domain.UNKNOWN: "low",
}


def classify_domain(title: str, description: str = "") -> Domain:
    """Classify event into domain."""
    text = f"{title} {description}".lower()
    
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score
    
    if not scores:
        return Domain.UNKNOWN
    
    return max(scores, key=scores.get)


# === KNOWLEDGE GAP IDENTIFICATION ===

def identify_gaps(title: str, domain: Domain) -> list[str]:
    """What do we NOT know that would help predict this?"""
    gaps = []
    
    if domain == Domain.AWARDS:
        gaps = [
            "precursor award winners (PGA, DGA, SAG, Globes, BAFTA, Critics Choice)",
            "nomination counts and categories",
            "critical consensus and review aggregates",
            "historical correlation between precursors and final winner",
            "betting odds from other platforms",
        ]
    elif domain == Domain.ECONOMY:
        gaps = [
            "most recent data point for this indicator",
            "trend direction over last 3-6 months",
            "leading indicators that predict this metric",
            "analyst consensus forecast",
            "seasonal patterns",
        ]
    elif domain == Domain.COMMODITIES:
        gaps = [
            "current price vs threshold in question",
            "price trend over last 7-30 days",
            "supply/demand news (OPEC, inventory reports)",
            "related macro factors (USD strength, Fed policy)",
            "geopolitical events affecting supply",
        ]
    elif domain == Domain.CRYPTO:
        gaps = [
            "current price vs threshold",
            "7-day and 30-day trend",
            "major protocol updates or events",
            "regulatory news",
            "whale wallet movements / exchange flows",
        ]
    elif domain == Domain.SPORTS:
        gaps = [
            "injury reports for key players",
            "recent form (last 5-10 games)",
            "head-to-head record",
            "home/away record",
            "schedule density (back-to-back, travel)",
        ]
    elif domain == Domain.GOLF:
        gaps = [
            "player's recent tournament results (last 4-6 events)",
            "course history at this venue",
            "current world ranking",
            "strokes gained statistics",
            "weather forecast for tournament days",
        ]
    elif domain == Domain.POLICY:
        gaps = [
            "current bill status / legislative progress",
            "key sponsors and opposition",
            "committee votes or markups",
            "presidential position / veto threat",
            "similar historical legislation outcomes",
        ]
    elif domain == Domain.GEOPOLITICS:
        gaps = [
            "latest diplomatic communications",
            "military positioning changes",
            "economic pressure / sanctions status",
            "third-party mediator involvement",
            "historical precedent for similar situations",
        ]
    else:
        gaps = [
            "recent news about this topic",
            "expert opinions or forecasts",
            "historical precedent",
        ]
    
    return gaps


# === SEARCH QUERY GENERATION ===

def generate_queries(title: str, domain: Domain, gaps: list[str]) -> list[str]:
    """Generate targeted search queries to fill knowledge gaps."""
    queries = []
    
    # Extract key entities from title
    # Remove common prediction market phrasing
    clean = re.sub(r"^(will|does|is|are|can|should)\s+", "", title, flags=re.I)
    clean = re.sub(r"\?$", "", clean).strip()
    
    if domain == Domain.AWARDS:
        # Extract award name and category
        award_match = re.search(r"(oscar|grammy|emmy|tony|golden globe|bafta)", title, re.I)
        award = award_match.group(1) if award_match else "award"
        queries = [
            f"{award} 2026 predictions winners",
            f"{award} precursor awards results 2026",
            f"{clean} odds betting",
        ]
    elif domain == Domain.ECONOMY:
        queries = [
            f"{clean} latest data 2026",
            f"{clean} forecast consensus",
            f"{clean} trend analysis",
        ]
    elif domain == Domain.COMMODITIES:
        # Extract commodity name
        commodity = "oil"
        for c in ["oil", "gold", "silver", "natural gas", "copper", "wheat"]:
            if c in title.lower():
                commodity = c
                break
        queries = [
            f"{commodity} price today",
            f"{commodity} price forecast this week",
            f"{commodity} supply demand news 2026",
        ]
    elif domain == Domain.CRYPTO:
        token = "bitcoin"
        for t in ["bitcoin", "ethereum", "solana", "dogecoin"]:
            if t in title.lower():
                token = t
                break
        queries = [
            f"{token} price today",
            f"{token} price prediction this week",
            f"{token} news latest 2026",
        ]
    elif domain == Domain.SPORTS:
        queries = [
            f"{clean} injury report",
            f"{clean} preview prediction",
            f"{clean} odds",
        ]
    elif domain == Domain.GOLF:
        # Extract player name
        player_match = re.search(r"Will (.+?) finish", title)
        player = player_match.group(1) if player_match else clean
        tournament_match = re.search(r"at the (.+?)$", title)
        tournament = tournament_match.group(1) if tournament_match else "tournament"
        queries = [
            f"{player} recent results 2026",
            f"{player} {tournament} history",
            f"{tournament} odds favorites 2026",
        ]
    else:
        queries = [
            f"{clean} latest news",
            f"{clean} prediction forecast",
        ]
    
    return queries[:3]  # Max 3 queries to stay fast


# === CONTEXT FETCHING ===

def fetch_context(queries: list[str], brave_api_key: str = None) -> list[ContextSignal]:
    """Fetch context from Brave Search API."""
    if not brave_api_key:
        brave_api_key = os.environ.get("BRAVE_API_KEY", "")
    if not brave_api_key:
        return []
    
    signals = []
    
    for query in queries:
        try:
            r = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 3, "freshness": "pw"},
                headers={"X-Subscription-Token": brave_api_key},
                timeout=10,
            )
            results = r.json().get("web", {}).get("results", [])
            
            for result in results[:2]:  # Top 2 per query
                title = result.get("title", "")
                snippet = result.get("description", "")
                url = result.get("url", "")
                
                if not snippet:
                    continue
                
                signals.append(ContextSignal(
                    source=url,
                    text=f"{title}: {snippet}",
                    relevance=0.7,  # TODO: score relevance
                    novelty=0.5,    # TODO: compare to LLM training data
                ))
        except Exception as e:
            continue
        
        time.sleep(0.3)  # Rate limit
    
    return signals


# === TETLOCK DECOMPOSITION ===

def tetlock_decompose(title: str, domain: Domain, market_price: float) -> dict:
    """Apply Tetlock superforecaster methods.
    
    Key principles:
    1. Start from base rate (outside view)
    2. Decompose into sub-questions
    3. Update sequentially with each signal
    4. Extremize if confident
    """
    result = {
        "base_rate": None,
        "sub_questions": [],
        "adjustment_factors": [],
    }
    
    # Base rates by domain (from Phase 1 calibration data)
    base_rates = {
        Domain.AWARDS: 0.15,       # Most nominees don't win
        Domain.ECONOMY: 0.50,      # Binary econ indicators ~coin flip
        Domain.COMMODITIES: 0.50,  # Price direction ~random walk
        Domain.CRYPTO: 0.45,       # Slightly bearish base rate
        Domain.SPORTS: 0.50,       # By definition ~fair
        Domain.GOLF: 0.10,         # Low base rate for top-N finishes
        Domain.POLICY: 0.30,       # Most bills don't pass
        Domain.GEOPOLITICS: 0.30,  # Status quo bias
    }
    
    result["base_rate"] = base_rates.get(domain, 0.50)
    
    # Decomposition templates
    if domain == Domain.AWARDS:
        result["sub_questions"] = [
            "Has this nominee won relevant precursor awards?",
            "Is there a strong critical/audience consensus?",
            "Does the nominee fit the Academy's recent preference pattern?",
            "Are there any spoiler candidates splitting the vote?",
        ]
    elif domain == Domain.COMMODITIES:
        result["sub_questions"] = [
            "Is the current price above or below the threshold?",
            "What's the recent trend direction?",
            "Are there upcoming catalysts (OPEC, reports, Fed)?",
            "How much time remains until settlement?",
        ]
    elif domain == Domain.ECONOMY:
        result["sub_questions"] = [
            "What was the last data point?",
            "What's the trend direction (improving/worsening)?",
            "Are leading indicators aligned?",
            "Is there a seasonal pattern?",
        ]
    
    return result


# === MAIN PIPELINE ===

import os
import time

def build_context(event_title: str, description: str = "", 
                  market_price: float = 0.5) -> EnrichedContext:
    """Full context pipeline: classify → gaps → search → structure."""
    
    # Step 1: Classify domain
    domain = classify_domain(event_title, description)
    
    # Step 2: Identify knowledge gaps
    gaps = identify_gaps(event_title, domain)
    
    # Step 3: Generate search queries
    queries = generate_queries(event_title, domain, gaps)
    
    # Step 4: Build plan
    plan = ContextPlan(
        domain=domain,
        event_title=event_title,
        knowledge_gaps=gaps,
        search_queries=queries,
        predictability=DOMAIN_PREDICTABILITY.get(domain, "low"),
    )
    
    # Step 5: Tetlock decomposition
    tetlock = tetlock_decompose(event_title, domain, market_price)
    plan.tetlock_decomposition = tetlock.get("sub_questions", [])
    
    # Step 6: Fetch context from web
    signals = fetch_context(queries)
    
    # Step 7: Get price context (for commodities/crypto)
    price_ctx = None
    try:
        from orcetra.prices import get_price_context
        price_ctx = get_price_context(event_title)
    except ImportError:
        pass
    
    # Step 8: Build enriched context
    enriched = EnrichedContext(
        plan=plan,
        signals=signals,
        price_context=price_ctx,
        base_rate=tetlock.get("base_rate"),
    )
    
    # Step 9: Generate summary for LLM injection
    parts = []
    
    if enriched.base_rate is not None:
        parts.append(f"BASE RATE ({domain.value}): Historical base rate for this type of event is {enriched.base_rate:.0%}")
    
    if price_ctx:
        parts.append(f"REAL-TIME DATA: {price_ctx}")
    
    if signals:
        parts.append("RECENT CONTEXT:")
        for i, sig in enumerate(signals[:5], 1):
            parts.append(f"  [{i}] {sig.text[:200]}")
    
    if plan.tetlock_decomposition:
        parts.append("KEY SUB-QUESTIONS TO CONSIDER:")
        for sq in plan.tetlock_decomposition:
            parts.append(f"  • {sq}")
    
    enriched.summary = "\n".join(parts)
    
    return enriched


def format_for_llm(context: EnrichedContext) -> str:
    """Format enriched context as LLM prompt injection."""
    if not context.summary:
        return ""
    
    return f"""
--- CONTEXT PLANNER ANALYSIS ---
Domain: {context.plan.domain.value} (predictability: {context.plan.predictability})
{context.summary}
--- END CONTEXT ---
"""


# === CLI ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m orcetra.context_planner 'Event title here'")
        sys.exit(1)
    
    title = " ".join(sys.argv[1:])
    print(f"🔍 Analyzing: {title}\n")
    
    ctx = build_context(title, market_price=0.5)
    print(f"Domain: {ctx.plan.domain.value}")
    print(f"Predictability: {ctx.plan.predictability}")
    print(f"Knowledge gaps: {len(ctx.plan.knowledge_gaps)}")
    print(f"Search queries: {ctx.plan.search_queries}")
    print(f"Signals found: {len(ctx.signals)}")
    print(f"Base rate: {ctx.base_rate}")
    print(f"\n{format_for_llm(ctx)}")
