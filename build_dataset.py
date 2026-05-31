#!/usr/bin/env python3
"""
build_dataset.py — fetch + schema + smart-target + normalize, in one pass.

This fuses data_fetch/main.py (download & lag-schema) with transform_aapl.py
(per-row normalization), and swaps the naive next-bar target for a trend-
comparison signal.

Pipeline
--------
1. FETCH      : intraday (10m/30m) → download 5m (60d, incl. pre/post) and
                resample up; daily (1d) → download max history directly.
                (exactly like data_fetch/main.py)
2. SCHEMA     : OHLC + Close_prev_1..N lag columns (N=12 intraday, 10 daily),
                drop the first N rows that lack full history.
3. TARGET     : *** the new step ***
                buy  if  mean(next FUTURE_WINDOW closes) > mean(prev PREV_WINDOW closes)
                sell otherwise.
                i.e. label is buy when the immediate future price level trends
                above the immediate past price level. Computed on RAW closes,
                before normalization, so there is no leakage (the future window
                is dropped from the features — only its mean feeds the label).
4. NORMALIZE  : identical to transform_aapl.py —
                  * Open/High/Low      ÷ Close      (per-row ratio)
                  * Close_prev_1..N    − Close      (per-row signed offset)
                  * reverse lag order  (Close_prev_1 = oldest, N = most recent)
                  * delete Close
5. WRITE      : {TICKER}_{INTERVAL}_v2.csv  (columns prepare_data.py expects).

Run:
    python build_dataset.py                          # AAPL 10m, fresh download
    python build_dataset.py GOOG --interval 1d       # GOOG daily
    python build_dataset.py AAPL --use-cache         # reuse cached raw if present
    python build_dataset.py SPY --outdir datasets_v2 # choose output folder
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent

# ============================================================ CONFIG (edit here)
PREV_WINDOW   = 5       # how many past closes the target averages over
FUTURE_WINDOW = 5       # how many future closes the target averages over
# lag-column count per interval (mirrors main.py: 10 for daily, else 12)
N_LAGS = {"10m": 12, "30m": 12, "1d": 10}


def get_timedelta(interval: str) -> pd.Timedelta:
    if interval == "10m":
        return pd.to_timedelta("10min")
    if interval == "30m":
        return pd.to_timedelta("30min")
    if interval == "1d":
        return pd.to_timedelta("1D")
    return pd.to_timedelta(interval)


def fetch(ticker: str, interval: str, raw_dir: Path, use_cache: bool) -> pd.DataFrame:
    """Download OHLCV for the interval (mirrors main.py's fetch paths)."""
    if interval == "1d":
        raw = raw_dir / f"{ticker}_1d.csv"
        if use_cache and raw.exists():
            print(f"[fetch] loading cached 1d raw ← {raw}")
            df = pd.read_csv(raw, index_col=0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        else:
            print(f"[fetch] downloading 1d data for {ticker} (period=max)…")
            df = yf.download(ticker, period="max", interval="1d",
                             auto_adjust=True, prepost=False, multi_level_index=False)
            if df.empty:
                raise SystemExit(f"No 1d data for {ticker}. Check the symbol.")
            raw_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        print(f"[fetch] daily bars: {len(df)} ({df.index[0].date()} … {df.index[-1].date()})")
        return df

    # intraday (10m / 30m): download 5m and resample up
    raw_5m = raw_dir / f"{ticker}_5m.csv"
    if use_cache and raw_5m.exists():
        print(f"[fetch] loading cached 5m raw ← {raw_5m}")
        df_5m = pd.read_csv(raw_5m, index_col=0)
        df_5m.index = pd.to_datetime(df_5m.index, utc=True).tz_convert("America/New_York")
    else:
        print(f"[fetch] downloading 5m data for {ticker} (period=60d, prepost)…")
        df_5m = yf.download(ticker, period="60d", interval="5m",
                            auto_adjust=True, prepost=True, multi_level_index=False)
        if df_5m.empty:
            raise SystemExit(f"No 5m data for {ticker}. Check the symbol.")
        raw_dir.mkdir(parents=True, exist_ok=True)
        df_5m.to_csv(raw_5m)

    df = (df_5m.resample(get_timedelta(interval))
                .agg({"Open": "first", "High": "max", "Low": "min",
                      "Close": "last", "Volume": "sum"})
                .dropna())
    print(f"[fetch] resampled to {interval}: {len(df)} bars "
          f"({df.index[0]} … {df.index[-1]})")
    return df


def build_schema(df: pd.DataFrame, interval: str, n_lags: int) -> pd.DataFrame:
    """OHLC + Close_prev_1..N lag columns; drop the first N rows (mirrors main.py)."""
    out = df[["Open", "High", "Low", "Close"]].copy()
    for j in range(1, n_lags + 1):
        out[f"Close_prev_{j}"] = out["Close"].shift(j)
    out = out.dropna()

    delta = get_timedelta(interval)
    out.insert(0, "start_ts", out.index)
    out.insert(1, "end_ts", out.index + delta)
    out.insert(2, "interval", interval)
    return out


def add_trend_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    *** the new step ***  buy / sell from a past-vs-future trend comparison.

    On RAW closes (pre-normalization):
        prev_mean   = mean(Close_prev_1 … Close_prev_PREV_WINDOW)   # last PREV_WINDOW closes
        future_mean = mean(Close[+1] … Close[+FUTURE_WINDOW])       # next FUTURE_WINDOW closes
        target      = 'buy' if future_mean > prev_mean else 'sell'

    The first PREV_WINDOW closes are already guaranteed by the lag schema; the
    final FUTURE_WINDOW rows lack a full future window and are dropped, so the
    label never leaks into the features.
    """
    prev_cols = [f"Close_prev_{i}" for i in range(1, PREV_WINDOW + 1)]
    prev_mean = df[prev_cols].mean(axis=1)

    future = pd.concat(
        [df["Close"].shift(-k) for k in range(1, FUTURE_WINDOW + 1)], axis=1
    )
    future_mean = future.mean(axis=1)

    df = df.copy()
    df["target"] = np.where(future_mean > prev_mean, "buy", "sell")
    df = df.iloc[:-FUTURE_WINDOW].copy()   # drop rows lacking a full future window
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row normalization identical to transform_aapl.py (target preserved)."""
    prev_cols_orig = sorted(
        [c for c in df.columns if c.lower().startswith("close_prev_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    n = len(prev_cols_orig)

    C = df["Close"]
    df = df.copy()
    for col in ["Open", "High", "Low"]:        # ÷ Close
        df[col] = df[col] / C
    for col in prev_cols_orig:                 # − Close
        df[col] = df[col] - C

    # reverse lag order: old lag-1 (newest) → new lag-n; old lag-n (oldest) → new lag-1
    rename_map = {prev_cols_orig[i]: f"Close_prev_{n - i}" for i in range(n)}
    df.rename(columns=rename_map, inplace=True)
    new_prev_cols = [f"Close_prev_{i}" for i in range(1, n + 1)]  # 1=oldest … n=newest

    df.drop(columns=["Close"], inplace=True)

    meta_cols  = [c for c in ["start_ts", "end_ts", "interval"] if c in df.columns]
    price_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]
    return df[meta_cols + price_cols + new_prev_cols + ["target"]]


def build(ticker: str, interval: str, use_cache: bool, outdir: Path) -> Path:
    n_lags = N_LAGS.get(interval, 12)
    raw_dir = HERE / "data_fetch" / "dataset" / ticker / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"{ticker}_{interval}_v2.csv"

    df = fetch(ticker, interval, raw_dir, use_cache)
    df = build_schema(df, interval, n_lags)
    df = add_trend_target(df)
    df = normalize(df)
    df.to_csv(dst, index=False)

    n_buy = int((df["target"] == "buy").sum())
    print(f"[target] buy if mean(next {FUTURE_WINDOW}) > mean(prev {PREV_WINDOW})  →  "
          f"buy={n_buy} ({n_buy/len(df):.1%})  sell={len(df)-n_buy} ({1-n_buy/len(df):.1%})")
    print(f"[done]   {ticker} {interval}: rows={len(df)}  cols={len(df.columns)}  →  {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="AAPL", help="ticker (default: AAPL)")
    parser.add_argument("--interval", default="10m", choices=["10m", "30m", "1d"],
                        help="bar interval (default: 10m)")
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cached raw CSV if present instead of downloading")
    parser.add_argument("--outdir", default=".", help="output folder (default: repo root)")
    args = parser.parse_args()

    build(args.ticker.upper(), args.interval, args.use_cache, Path(args.outdir))


if __name__ == "__main__":
    main()
