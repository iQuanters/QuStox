"""
Sentiment analyzer — OpenAI API classifies market regime + sentiment score.

Input  : MarketContext (headlines + VIX)
Output : SentimentResult (regime label, multiplier, sentiment score, reasoning)

GPT is given:
  - current VIX level + trend
  - up to 20 recent headlines
  - the full regime taxonomy + multiplier table

JSON mode (response_format=json_object) ensures structured, machine-readable output
consumed by the HQRC predictor and the output dashboard.

Model is read from OPENAI_MODEL env var (default: gpt-4o-mini).
API key is read from OPENAI_API_KEY env var.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .news_fetcher import MarketContext

# Load .env from project root (two levels up from src/agents/)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# ── Regime taxonomy (mirrors state.py REGIME_WINDOW_MULTIPLIERS) ──────────────

REGIME_MULTIPLIERS: dict[str, float] = {
    "CALM":          1.5,    # VIX < 15, no events
    "NORMAL":        1.0,    # VIX 15–25, baseline
    "ELEVATED":      0.7,    # VIX 25–35, volatility rising
    "CRISIS":        0.4,    # VIX > 35, crash/panic
    "ELECTION":      0.6,    # political binary event
    "FED_EVENT":     0.5,    # FOMC meeting / rate decision
    "GEOPOLITICAL":  0.55,   # war / sanctions
    "PANDEMIC":      0.3,    # systemic health shock
    "EARNINGS":      0.8,    # SPY earnings season
}

_DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You are a quantitative finance analyst classifying market regime and sentiment
from VIX data and recent news headlines.

Your job:
1. Choose ONE regime from this list (pick the most dominant):
   CALM | NORMAL | ELEVATED | CRISIS | ELECTION | FED_EVENT | GEOPOLITICAL | PANDEMIC | EARNINGS

2. Assign a sentiment_score between -1.0 (very bearish) and +1.0 (very bullish).
   Use 0.0 for neutral or mixed signals.

3. Write a one-sentence reasoning (max 20 words) explaining your choice.

Respond ONLY with a valid JSON object with exactly these keys:
  regime, sentiment_score, reasoning"""


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SentimentResult:
    regime:            str     # one of REGIME_MULTIPLIERS keys
    window_multiplier: float   # from REGIME_MULTIPLIERS[regime]
    sentiment_score:   float   # −1.0 → +1.0
    reasoning:         str     # one-sentence explanation
    headlines_used:    int     # how many headlines were analysed
    vix_current:       float


# ── Public ─────────────────────────────────────────────────────────────────────

def analyze_sentiment(ctx: MarketContext) -> SentimentResult:
    """
    Send MarketContext to OpenAI and return a structured SentimentResult.

    Uses the OPENAI_API_KEY environment variable for authentication.
    Uses OPENAI_MODEL env var to override the model (optional).

    JSON mode (response_format=json_object) guarantees parseable output
    without needing to strip markdown code fences.

    Raises
    ------
    EnvironmentError  if OPENAI_API_KEY is not set
    ValueError        if the model returns malformed JSON or an unknown regime
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Get your key at https://platform.openai.com and set:\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    model  = os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(ctx)

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        response_format={"type": "json_object"},   # guaranteed valid JSON back
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    return _parse_response(raw_text, ctx)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_prompt(ctx: MarketContext) -> str:
    """Format VIX + headlines into the user message sent to Claude."""
    # VIX context block
    vix_block = (
        f"VIX: {ctx.vix_current:.1f}  "
        f"(percentile={ctx.vix_percentile:.0f}%, trend={ctx.vix_trend})"
    )

    # Regime hints from VIX level alone (help Claude anchor quickly)
    if   ctx.vix_current < 15:  vix_hint = "VIX below 15 → lean toward CALM."
    elif ctx.vix_current < 25:  vix_hint = "VIX 15–25 → lean toward NORMAL or event-specific regime."
    elif ctx.vix_current < 35:  vix_hint = "VIX 25–35 → lean toward ELEVATED or event-specific regime."
    else:                       vix_hint = "VIX above 35 → lean toward CRISIS or PANDEMIC."

    # Date-aware event hint — override VIX-only bias for known calendar events
    date_hint = _date_event_hint(ctx.as_of)

    # Headlines block (numbered list, truncated to avoid token waste)
    if ctx.headlines:
        lines    = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(ctx.headlines))
        hl_block = f"Recent headlines ({len(ctx.headlines)}):\n{lines}"
    else:
        hl_block = "No recent headlines available."

    return (
        f"{vix_block}\n"
        f"{vix_hint}\n"
        f"{date_hint}"
        f"\n{hl_block}\n\n"
        f"Classify the regime and sentiment for {ctx.ticker} as of "
        f"{ctx.as_of.strftime('%Y-%m-%d %H:%M UTC')}."
    )


def _date_event_hint(dt) -> str:
    """
    Return a plain-English hint about known calendar events on/near a date.
    This nudges the LLM to pick the correct event-specific regime even when
    the recalled headlines lean bullish (e.g. election day with rising markets).
    """
    from datetime import date as _date

    d = dt.date() if hasattr(dt, "date") else dt

    # ── US Presidential Elections (first Tuesday after first Monday in Nov) ──
    _ELECTION_DAYS = {
        _date(2024, 11, 5),
        _date(2020, 11, 3),
        _date(2016, 11, 8),
        _date(2012, 11, 6),
    }
    if d in _ELECTION_DAYS:
        return "DATE CONTEXT: This is US Presidential Election Day. Regime should be ELECTION regardless of headline tone.\n"

    # ── FOMC meeting dates (approximate — covers decision day ± 1) ──
    # Check if within 1 day of a known FOMC decision date
    _FOMC_MONTHS_DAYS = {(1,29),(1,30),(3,19),(3,20),(5,7),(5,8),
                         (6,11),(6,12),(6,14),(6,15),          # Jun (covers 2022 75bps)
                         (7,30),(7,31),(9,17),(9,18),(9,19),(9,20),
                         (10,29),(10,30),(11,6),(11,7),         # Nov sometimes has FOMC
                         (12,10),(12,11),(12,17),(12,18)}
    if (d.month, d.day) in _FOMC_MONTHS_DAYS:
        return "DATE CONTEXT: This is an FOMC meeting / Fed rate decision day. Regime should be FED_EVENT.\n"

    return ""  # no special hint


def _parse_response(raw: str, ctx: MarketContext) -> SentimentResult:
    """
    Parse Claude's JSON response into a SentimentResult.

    Strips code fences if present, validates regime label, clamps score.
    """
    # Strip markdown code fences if Claude wraps JSON in them
    text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OpenAI returned invalid JSON.\nRaw response:\n{raw}"
        ) from exc

    regime = str(data.get("regime", "NORMAL")).upper()
    if regime not in REGIME_MULTIPLIERS:
        regime = "NORMAL"   # safe fallback

    sentiment_score = float(data.get("sentiment_score", 0.0))
    sentiment_score = max(-1.0, min(1.0, sentiment_score))   # clamp to [-1, 1]

    reasoning = str(data.get("reasoning", "")).strip()

    return SentimentResult(
        regime=regime,
        window_multiplier=REGIME_MULTIPLIERS[regime],
        sentiment_score=sentiment_score,
        reasoning=reasoning,
        headlines_used=len(ctx.headlines),
        vix_current=ctx.vix_current,
    )
