"""
QuStox — External Factors Agent
Run:  python run.py [TICKER] [--as-of "YYYY-MM-DD HH:MM"]

Examples
--------
  python run.py                          # SPY, right now
  python run.py AAPL                     # AAPL, right now
  python run.py SPY --as-of "2024-11-05" # SPY on US election day (backtest)
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
    print(f"║  Headlines  {factors.headlines_used:>6}   analysed by Claude"
          f"                ║")
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

def main() -> None:
    args  = parse_args()
    as_of = parse_as_of(args.as_of)

    check_env()

    try:
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
