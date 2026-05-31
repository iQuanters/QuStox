"""
Market context fetcher — headlines + VIX for a given point in time.

Data sources (no API keys needed):
  Headlines : yfinance.Ticker(ticker).news
  VIX       : yfinance.Ticker("^VIX").history()

as_of behaviour
---------------
  as_of=None      → current time  (live mode)
  as_of=datetime  → filters headlines from [as_of − lookback_hours, as_of]
                    and picks the VIX close on or before that date
                    (historical / backtesting mode)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import yfinance as yf


@dataclass(frozen=True)
class MarketContext:
    """All raw market context needed for sentiment analysis."""
    ticker:         str
    headlines:      list[str]   # most-recent first; may be empty if no news
    vix_current:    float       # latest VIX close value
    vix_percentile: float       # where VIX sits vs its 1-yr history  (0–100)
    vix_trend:      str         # "RISING" | "FALLING" | "STABLE"
    as_of:          datetime    # always timezone-aware (UTC)


# ── Public ─────────────────────────────────────────────────────────────────────

def fetch_market_context(
    ticker:         str           = "SPY",
    as_of:          Optional[datetime] = None,
    max_headlines:  int           = 20,
    lookback_hours: int           = 24,
) -> MarketContext:
    """
    Fetch headlines and VIX for a given point in time.

    Parameters
    ----------
    ticker          : stock/ETF symbol (headlines are fetched for this ticker)
    as_of           : point-in-time reference; None → now
    max_headlines   : cap on number of headlines returned
    lookback_hours  : how far back (hours) from as_of to look for headlines
    """
    as_of = _normalise_timestamp(as_of)

    headlines                          = _fetch_headlines(ticker, as_of, max_headlines, lookback_hours)
    vix_current, vix_pct, vix_trend   = _fetch_vix(as_of)

    return MarketContext(
        ticker=ticker,
        headlines=headlines,
        vix_current=vix_current,
        vix_percentile=vix_pct,
        vix_trend=vix_trend,
        as_of=as_of,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normalise_timestamp(ts: Optional[datetime]) -> datetime:
    """Return a UTC-aware datetime; default to now."""
    if ts is None:
        return datetime.now(tz=timezone.utc)
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _fetch_headlines(
    ticker:         str,
    as_of:          datetime,
    max_headlines:  int,
    lookback_hours: int,
) -> list[str]:
    """
    Pull headlines from yfinance and filter to the window
    [as_of − lookback_hours, as_of].

    Falls back to the N most-recent headlines (regardless of age) if the
    time-windowed result is empty — so the sentiment agent always has
    something to work with.
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    cutoff_ts      = as_of.timestamp()
    window_start   = cutoff_ts - lookback_hours * 3600

    in_window = [
        item for item in news_items
        if window_start <= item.get("providerPublishTime", 0) <= cutoff_ts
        and item.get("title")
    ]
    in_window.sort(key=lambda x: x["providerPublishTime"], reverse=True)

    headlines = [item["title"] for item in in_window[:max_headlines]]

    # Fallback: window is empty (e.g. weekend / historical date with no news)
    if not headlines:
        headlines = [
            item["title"]
            for item in news_items[:max_headlines]
            if item.get("title")
        ]

    return headlines


def _fetch_vix(as_of: datetime) -> tuple[float, float, str]:
    """
    Return (vix_current, vix_percentile_vs_1yr, vix_trend).

    vix_trend is computed by comparing the 5-day average ending at as_of
    against the prior 5-day average:
        > 5% higher  → RISING
        > 5% lower   → FALLING
        otherwise    → STABLE
    """
    _FALLBACK = (20.0, 50.0, "STABLE")

    try:
        hist  = yf.Ticker("^VIX").history(period="1y", interval="1d")
        if hist.empty:
            return _FALLBACK

        close = hist["Close"].dropna()

        # Strip timezone for comparison if index is tz-aware
        idx = close.index.tz_localize(None) if close.index.tzinfo else close.index
        as_of_naive = as_of.replace(tzinfo=None)

        # Most-recent close on or before as_of
        mask = idx <= as_of_naive
        vix_current = float(close[mask].iloc[-1]) if mask.any() else float(close.iloc[-1])

        # Percentile vs 1-yr distribution
        vix_percentile = float((close.values <= vix_current).mean() * 100)

        # Trend
        if len(close) >= 10:
            recent = close.iloc[-5:].mean()
            prior  = close.iloc[-10:-5].mean()
            if   recent > prior * 1.05:  vix_trend = "RISING"
            elif recent < prior * 0.95:  vix_trend = "FALLING"
            else:                        vix_trend = "STABLE"
        else:
            vix_trend = "STABLE"

        return vix_current, vix_percentile, vix_trend

    except Exception:
        return _FALLBACK
