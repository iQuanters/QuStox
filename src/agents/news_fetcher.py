"""
Market context fetcher — headlines + VIX for a given point in time.

Data sources:
  Headlines (live)       : yfinance.Ticker(ticker).news  — last ~10 articles
  Headlines (historical) : OpenAI GPT recall fallback    — training data covers
                           major market events back to ~2020
  VIX                    : yfinance.Ticker("^VIX").history()  — full history

as_of behaviour
---------------
  as_of=None      → current time, yfinance live headlines
  as_of=datetime  → yfinance headlines filtered to window;
                    if empty (historical date), OpenAI recalls headlines
                    from its training knowledge for that date
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import yfinance as yf


@dataclass(frozen=True)
class MarketContext:
    """All raw market context needed for sentiment analysis."""
    ticker:           str
    headlines:        list[str]   # most-recent first
    vix_current:      float       # VIX close on or before as_of
    vix_percentile:   float       # VIX vs 2yr history (0–100)
    vix_trend:        str         # "RISING" | "FALLING" | "STABLE"
    as_of:            datetime    # UTC-aware
    headlines_source: str = "yfinance"  # "yfinance" | "llm_recall" | "none"


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

    headlines, source                = _fetch_headlines(ticker, as_of, max_headlines, lookback_hours)
    vix_current, vix_pct, vix_trend  = _fetch_vix(as_of)

    return MarketContext(
        ticker=ticker,
        headlines=headlines,
        vix_current=vix_current,
        vix_percentile=vix_pct,
        vix_trend=vix_trend,
        as_of=as_of,
        headlines_source=source,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normalise_timestamp(ts: Optional[datetime]) -> datetime:
    """Return a UTC-aware datetime; default to now."""
    if ts is None:
        return datetime.now(tz=timezone.utc)
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _parse_news_item(item: dict) -> tuple[str, float]:
    """
    Parse a yfinance news item into (title, unix_timestamp).

    yfinance changed its news API schema in 2025:
      Old format: {"title": "...", "providerPublishTime": 1234567890}
      New format: {"content": {"title": "...", "pubDate": "2026-05-31T12:00:00Z"}}

    We handle both so the fetcher works regardless of yfinance version.
    """
    # New format — nested under "content"
    content = item.get("content", {})
    if content:
        title   = content.get("title", "")
        pub_str = content.get("pubDate", "") or content.get("displayTime", "")
        if pub_str:
            try:
                dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                return title, dt.timestamp()
            except ValueError:
                pass

    # Old format — flat keys
    title = item.get("title", "")
    ts    = float(item.get("providerPublishTime", 0))
    return title, ts


def _fetch_headlines(
    ticker:         str,
    as_of:          datetime,
    max_headlines:  int,
    lookback_hours: int,
) -> tuple[list[str], str]:
    """
    Pull headlines and return (headlines, source).

    source is one of:
      "yfinance"   — live headlines from yfinance within the time window
      "llm_recall" — OpenAI recalled historical headlines from training data
      "none"       — no headlines available from any source
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        news_items = []

    parsed = [_parse_news_item(item) for item in news_items]

    cutoff_ts    = as_of.timestamp()
    window_start = cutoff_ts - lookback_hours * 3600

    in_window = [
        (title, ts) for title, ts in parsed
        if window_start <= ts <= cutoff_ts and title
    ]
    in_window.sort(key=lambda x: x[1], reverse=True)
    headlines = [title for title, _ in in_window[:max_headlines]]

    if headlines:
        return headlines, "yfinance"

    # No headlines in window — try OpenAI recall for historical dates
    recalled = _recall_headlines_from_llm(ticker, as_of, max_headlines)
    if recalled:
        return recalled, "llm_recall"

    # Last resort: return whatever yfinance has regardless of date
    fallback = [title for title, _ in parsed if title][:max_headlines]
    return fallback, "yfinance" if fallback else "none"


def _recall_headlines_from_llm(
    ticker:        str,
    as_of:         datetime,
    max_headlines: int,
) -> list[str]:
    """
    Ask OpenAI to recall financial headlines for a historical date.

    GPT's training data covers major market events (elections, Fed decisions,
    geopolitical crises, pandemics, etc.) going back many years.
    This is used as a fallback when yfinance has no news for the requested date.

    Returns an empty list if OPENAI_API_KEY is not set or the call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return []

    # Only attempt recall for clearly historical dates (> 7 days ago)
    days_ago = (datetime.now(tz=timezone.utc) - as_of).days
    if days_ago < 7:
        return []

    try:
        from openai import OpenAI
        client    = OpenAI(api_key=api_key)
        # Use gpt-4o for recall — it has a later training cutoff than gpt-4o-mini
        # Override via OPENAI_RECALL_MODEL env var if needed
        model     = os.environ.get("OPENAI_RECALL_MODEL", "gpt-4o")
        date_str  = as_of.strftime("%B %d, %Y")

        response = client.chat.completions.create(
            model=model,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial historian with deep knowledge of market events. "
                        "Return only factual headlines from your training data. "
                        "Respond with a JSON object: {\"headlines\": [\"headline1\", ...]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Top {max_headlines} financial news headlines affecting "
                        f"{ticker} and the US stock market around {date_str}?"
                    ),
                },
            ],
        )

        data      = json.loads(response.choices[0].message.content)
        headlines = data.get("headlines", [])
        return [str(h) for h in headlines[:max_headlines]]

    except Exception:
        return []


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
        hist  = yf.Ticker("^VIX").history(period="2y", interval="1d")
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
