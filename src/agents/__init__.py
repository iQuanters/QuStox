"""External factors agents — public API."""
from .news_fetcher import fetch_market_context, MarketContext
from .sentiment_analyzer import analyze_sentiment, SentimentResult, REGIME_MULTIPLIERS
from .external_factors import get_external_factors, ExternalFactors

__all__ = [
    "fetch_market_context",
    "MarketContext",
    "analyze_sentiment",
    "SentimentResult",
    "REGIME_MULTIPLIERS",
    "get_external_factors",
    "ExternalFactors",
]
