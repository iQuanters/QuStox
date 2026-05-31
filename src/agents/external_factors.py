"""
External factors — public interface for the HQRC pipeline.

Single entry point: get_external_factors()

Wires together:
  1. news_fetcher.fetch_market_context()  →  headlines + VIX
  2. sentiment_analyzer.analyze_sentiment()  →  regime + multiplier + score

The returned ExternalFactors object is consumed by the HQRC predictor:

    factors = get_external_factors(ticker="SPY")
    result  = predictor.predict(data, tf_cfg,
                                regime_multiplier=factors.window_multiplier)

Typical call patterns
---------------------
    # Live (right now)
    factors = get_external_factors("SPY")

    # Historical / backtesting
    from datetime import datetime, timezone
    factors = get_external_factors("SPY", as_of=datetime(2024, 11, 5, tzinfo=timezone.utc))
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .news_fetcher import MarketContext, fetch_market_context
from .sentiment_analyzer import SentimentResult, analyze_sentiment


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExternalFactors:
    """
    All external-factor signals needed by the HQRC predictor.

    Attributes
    ----------
    regime            : market regime label
                        e.g. "FED_EVENT", "ELEVATED", "NORMAL"
    window_multiplier : factor to scale HQRC context windows
                        < 1.0 → compress windows (uncertain market)
                        > 1.0 → expand windows (stable market)
    sentiment_score   : −1.0 (very bearish) → +1.0 (very bullish)
    reasoning         : one-sentence explanation from OpenAI
    vix_current       : latest VIX close
    vix_trend         : "RISING" | "FALLING" | "STABLE"
    headlines_used    : number of headlines analysed
    as_of             : UTC timestamp this analysis refers to
    """
    regime:            str
    window_multiplier: float
    sentiment_score:   float
    reasoning:         str
    vix_current:       float
    vix_trend:         str
    headlines_used:    int
    as_of:             datetime


# ── Public ─────────────────────────────────────────────────────────────────────

def get_external_factors(
    ticker:         str                = "SPY",
    as_of:          Optional[datetime] = None,
    max_headlines:  int                = 20,
    lookback_hours: int                = 24,
    verbose:        bool               = True,
) -> ExternalFactors:
    """
    Fetch market context and classify regime + sentiment for a given ticker.

    Parameters
    ----------
    ticker          : stock/ETF symbol ("SPY", "AAPL", etc.)
    as_of           : point-in-time reference; None → now (live mode)
    max_headlines   : maximum headlines to analyse
    lookback_hours  : headline lookback window in hours
    verbose         : print progress to stdout

    Returns
    -------
    ExternalFactors ready to be passed as `regime_multiplier` to HQRCPredictor.

    Raises
    ------
    EnvironmentError  if ANTHROPIC_API_KEY is not set
    """
    if verbose:
        ts_label = "now" if as_of is None else as_of.strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n── External Factors  [{ticker}  @  {ts_label}] ──")

    # Step 1 — fetch headlines + VIX
    if verbose:
        print("   Fetching market context …")
    ctx: MarketContext = fetch_market_context(
        ticker=ticker,
        as_of=as_of,
        max_headlines=max_headlines,
        lookback_hours=lookback_hours,
    )

    if verbose:
        print(f"   VIX  : {ctx.vix_current:.1f}  "
              f"(p{ctx.vix_percentile:.0f}, {ctx.vix_trend})")
        print(f"   News : {len(ctx.headlines)} headlines")

    # Step 2 — classify with OpenAI
    if verbose:
        print("   Classifying regime via OpenAI …")
    sentiment: SentimentResult = analyze_sentiment(ctx)

    if verbose:
        print(f"   Regime     : {sentiment.regime}")
        print(f"   Multiplier : {sentiment.window_multiplier}×")
        print(f"   Sentiment  : {sentiment.sentiment_score:+.2f}")
        print(f"   Reasoning  : {sentiment.reasoning}")

    return ExternalFactors(
        regime=sentiment.regime,
        window_multiplier=sentiment.window_multiplier,
        sentiment_score=sentiment.sentiment_score,
        reasoning=sentiment.reasoning,
        vix_current=ctx.vix_current,
        vix_trend=ctx.vix_trend,
        headlines_used=sentiment.headlines_used,
        as_of=ctx.as_of,
    )
