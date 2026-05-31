"""
QuStox — External Factors Agent
Run:  python run.py [TICKER] [--as-of "YYYY-MM-DD HH:MM"] [--scenario SCENARIO]

Examples
--------
  python run.py                              # SPY, right now (live)
  python run.py AAPL                         # AAPL, right now
  python run.py SPY --as-of "2024-11-05"    # US election day — GPT recalls headlines
  python run.py SPY --as-of "2024-09-18"    # Fed rate cut day — GPT recalls headlines
  python run.py SPY --as-of "2020-03-16"    # COVID crash — GPT recalls headlines
  python run.py SPY --scenario geopolitical # US-Israel-Iran war (past GPT cutoff)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env before importing anything that reads env vars
load_dotenv()

from src.agents import get_external_factors, ExternalFactors
from src.agents.news_fetcher import MarketContext
from src.agents.sentiment_analyzer import analyze_sentiment


# ── Scenario presets (pre-loaded headlines for demo / backtest) ───────────────
# yfinance does not provide historical news archives.
# These presets simulate real-world events with realistic headlines.

# Only events PAST the GPT training cutoff (~Apr 2025) need hardcoded headlines.
# Everything before that: use --as-of and GPT recalls headlines automatically.
#
#   --as-of "2024-11-05"  →  US Election Day      (GPT knows this)
#   --as-of "2024-09-18"  →  Fed 50bps cut        (GPT knows this)
#   --as-of "2020-03-16"  →  COVID crash           (GPT knows this)
#   --scenario geopolitical  →  US-Israel-Iran war (past GPT cutoff, hardcoded)

SCENARIOS: dict[str, dict] = {
    "geopolitical": {
        "as_of":    datetime(2025, 6, 13, 10, 0, tzinfo=timezone.utc),
        "vix":      31.4,
        "vix_pct":  88.0,
        "vix_trend": "RISING",
        "headlines": [
            "US launches airstrikes on Iranian nuclear facilities alongside Israel",
            "Iran threatens to close Strait of Hormuz, oil surges 14%",
            "S&P 500 futures plunge 3.2% as Middle East conflict escalates",
            "Pentagon confirms US forces engaged in joint operation with Israel",
            "Brent crude hits $112/barrel on Hormuz closure fears",
            "White House declares heightened threat level, markets reel",
            "Iran fires ballistic missiles at Israeli military bases",
            "Gold surges to record $3,200 as safe-haven demand spikes",
            "Airlines cancel flights over Iran airspace amid escalating conflict",
            "Congressional leaders briefed on emergency war powers invocation",
        ],
    },
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuStox external factors agent")
    p.add_argument(
        "ticker",
        nargs="?",
        default="SPY",
        help="Ticker symbol (default: SPY)",
    )
    p.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        metavar="DATETIME",
        help='Point-in-time reference, e.g. "2024-11-05" or "2024-11-05 14:30"',
    )
    p.add_argument(
        "--scenario",
        dest="scenario",
        default=None,
        choices=list(SCENARIOS.keys()),
        help="Demo scenario with pre-loaded headlines: geopolitical (US-Israel-Iran war)",
    )
    return p.parse_args()


def parse_as_of(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    print(f"[error] Cannot parse --as-of value: {raw!r}")
    print("        Use format: YYYY-MM-DD  or  YYYY-MM-DD HH:MM")
    sys.exit(1)


# ── Output dashboard ───────────────────────────────────────────────────────────

def print_dashboard(ticker: str, factors: ExternalFactors) -> None:
    ts = factors.as_of.strftime("%Y-%m-%d %H:%M UTC")

    # Sentiment bar:  ████░░░░░░  mapped from −1 → +1
    score  = factors.sentiment_score
    filled = round((score + 1) / 2 * 10)   # 0–10 blocks
    bar    = "█" * filled + "░" * (10 - filled)

    # Signal colour hint
    if   score >  0.2: signal_hint = "↑ BULLISH"
    elif score < -0.2: signal_hint = "↓ BEARISH"
    else:              signal_hint = "→ NEUTRAL"

    width = 54
    line  = "─" * width

    print(f"\n╔{'═' * width}╗")
    print(f"║  QuStox  │  {ticker:<6}  │  {ts:<27}  ║")
    print(f"╠{line}╣")
    print(f"║  VIX        {factors.vix_current:>6.1f}   "
          f"trend={factors.vix_trend:<8}              ║")
    print(f"║  Headlines  {factors.headlines_used:>6}   analysed by AI"
          f"                    ║")
    print(f"╠{line}╣")
    print(f"║  REGIME       {factors.regime:<20}             ║")
    print(f"║  MULTIPLIER   {factors.window_multiplier:<6.2f}×"
          f"  (HQRC window scale)           ║")
    print(f"╠{line}╣")
    print(f"║  SENTIMENT  [{bar}]  {score:+.2f}  {signal_hint:<12}  ║")
    print(f"╠{line}╣")
    # Wrap reasoning to fit dashboard width
    reason = factors.reasoning
    if len(reason) > width - 4:
        reason = reason[:width - 7] + "…"
    print(f"║  {reason:<{width - 2}}  ║")
    print(f"╚{'═' * width}╝")

    # Integration hint for teammate
    print(f"\n  → Pass to HQRC predictor:")
    print(f"      regime_multiplier = {factors.window_multiplier}")


# ── Guards ─────────────────────────────────────────────────────────────────────

def check_env() -> None:
    """Fail fast with a helpful message if required env vars are missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY is not set.")
        print("        Add it to .env or export it:")
        print("          export OPENAI_API_KEY=sk-...")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_scenario(ticker: str, scenario_key: str) -> ExternalFactors:
    """Run a pre-loaded demo scenario with realistic headlines."""
    s = SCENARIOS[scenario_key]
    print(f"\n── Scenario: {scenario_key.upper()}  [{ticker}  @  {s['as_of'].strftime('%Y-%m-%d %H:%M UTC')}] ──")
    print(f"   VIX      : {s['vix']:.1f}  (p{s['vix_pct']:.0f}, {s['vix_trend']})")
    print(f"   Headlines: {len(s['headlines'])} (pre-loaded scenario)")
    print("   Classifying regime via OpenAI …")

    ctx = MarketContext(
        ticker=ticker,
        headlines=s["headlines"],
        vix_current=s["vix"],
        vix_percentile=s["vix_pct"],
        vix_trend=s["vix_trend"],
        as_of=s["as_of"],
    )
    sentiment = analyze_sentiment(ctx)

    print(f"   Regime     : {sentiment.regime}")
    print(f"   Multiplier : {sentiment.window_multiplier}×")
    print(f"   Sentiment  : {sentiment.sentiment_score:+.2f}")
    print(f"   Reasoning  : {sentiment.reasoning}")

    from src.agents.sentiment_analyzer import REGIME_MULTIPLIERS
    return ExternalFactors(
        regime=sentiment.regime,
        window_multiplier=sentiment.window_multiplier,
        sentiment_score=sentiment.sentiment_score,
        reasoning=sentiment.reasoning,
        vix_current=ctx.vix_current,
        vix_trend=ctx.vix_trend,
        headlines_used=len(ctx.headlines),
        as_of=ctx.as_of,
    )


def main() -> None:
    args  = parse_args()
    as_of = parse_as_of(args.as_of)

    check_env()

    try:
        if args.scenario:
            factors = run_scenario(args.ticker.upper(), args.scenario)
        else:
            factors = get_external_factors(
                ticker=args.ticker.upper(),
                as_of=as_of,
                verbose=True,
            )
    except Exception as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)

    print_dashboard(args.ticker.upper(), factors)


if __name__ == "__main__":
    main()
