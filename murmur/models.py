"""Pydantic data models for Murmur prediction market system."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    """A single price point in time series."""
    timestamp: datetime
    price: float = Field(ge=0, le=1)


class OrderBookLevel(BaseModel):
    """A single level in the order book."""
    price: float
    size: float


class OrderBook(BaseModel):
    """Order book depth for a market."""
    token_id: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class OutcomeToken(BaseModel):
    """An outcome token for a market."""
    token_id: str
    outcome: str
    price: float = Field(ge=0, le=1)


class Event(BaseModel):
    """A Polymarket event with its markets."""
    id: str
    slug: str
    title: str
    description: str = ""
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    resolved: bool = False
    outcome: Optional[str] = None  # For resolved events
    outcome_price: Optional[float] = None  # Final price (0 or 1 for resolved)
    tokens: list[OutcomeToken] = Field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0


class NewsSignal(BaseModel):
    """A news signal extracted from search results."""
    title: str
    snippet: str
    source: str
    url: str
    date: Optional[datetime] = None
    sentiment: Optional[str] = None  # positive, negative, neutral
    relevance: float = Field(default=0.5, ge=0, le=1)


class Prediction(BaseModel):
    """A prediction output from a strategy."""
    probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    trajectory_7d: list[float] = Field(default_factory=list)  # 7 daily predictions
    market_price: Optional[float] = None  # Current market price for reference


class EventResult(BaseModel):
    """Result of evaluating a single event prediction."""
    event_id: str
    event_title: str
    predicted_prob: float
    market_prob: float  # What Polymarket predicted
    actual_outcome: float  # 0 or 1
    brier_score: float
    market_brier_score: float
    beat_market: bool
    reasoning: str = ""


class BacktestResult(BaseModel):
    """Aggregate backtest results for a strategy."""
    total_events: int
    brier_score: float
    market_brier_score: float  # Baseline from market prices
    log_loss: float
    accuracy_50: float  # Accuracy when confidence > 0.5
    accuracy_70: float  # Accuracy when confidence > 0.7
    calibration_buckets: dict[str, dict] = Field(default_factory=dict)
    beat_market_rate: float  # % of events where strategy beat market
    per_event_results: list[EventResult] = Field(default_factory=list)


class EvolutionRound(BaseModel):
    """A single round in the evolution loop."""
    round_number: int
    strategy_code: str
    backtest_result: Optional[BacktestResult] = None
    changes_made: str = ""
    improvement: float = 0.0  # Change in Brier score (negative is better)
    kept: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
