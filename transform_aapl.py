#!/usr/bin/env python3
"""
transform_aapl.py — turn an OHLC(V) CSV into the QRC-ready v2 dataset.

Works on EITHER kind of input:
  * a schema file that already has Close_prev_1..n  (e.g. AAPL_10m.csv), or
  * a raw OHLCV file with no lag columns             (e.g. AAPL_5m.csv) —
    in that case the Close_prev_1..n lags are built from Close automatically
    (Close_prev_i = Close.shift(i)), dropping the first n rows that lack history.

Per row, using each row's own Close value C:

  1. target          : *** smartly generated trend signal ***
                       buy  if  mean(next FUTURE_WINDOW closes) > mean(prev PREV_WINDOW closes)
                       sell otherwise.
                       Computed on RAW closes before normalization; the final
                       FUTURE_WINDOW rows (no full future window) are dropped, so
                       the label never leaks into the features.
  2. Open/High/Low   : divided by C
  3. Close_prev_1..n : subtracted by C  (C is the per-row reference point)
  4. Reverse order   : new Close_prev_1 = oldest lag … new Close_prev_n = most recent
  5. Close column    : deleted entirely

n is auto-detected (or built) so this works for any number of lags.

Run:
    python transform_aapl.py                       # default SRC below
    python transform_aapl.py data_fetch/dataset/AAPL/raw/AAPL_5m.csv
Output: <stem>_v2.csv  (e.g. AAPL_5m_v2.csv)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# ============================================================ CONFIG (edit here)
DEFAULT_SRC   = HERE / "AAPL_10m.csv"
N_LAGS        = 12   # lags to BUILD when the input has no Close_prev_* columns
PREV_WINDOW   = 5    # how many past closes the target averages over
FUTURE_WINDOW = 5    # how many future closes the target averages over

# timestamp column candidates (first match wins); falls back to the index
TS_CANDIDATES = ["start_ts", "Datetime", "Date", "Timestamp"]


def label_interval(delta: pd.Timedelta) -> str:
    """Human label for a bar spacing: 5min→'5m', 1h→'1h', 1 day→'1d'."""
    secs = int(delta.total_seconds())
    if secs and secs % 86400 == 0:
        return f"{secs // 86400}d"
    if secs and secs % 3600 == 0:
        return f"{secs // 3600}h"
    return f"{max(secs, 60) // 60}m"


def load(src: Path) -> tuple[pd.DataFrame, pd.Series | None, pd.Timedelta | None]:
    """Read the CSV; return (df, timestamp series or None, bar delta or None)."""
    df = pd.read_csv(src)
    ts = None
    for cand in TS_CANDIDATES:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], utc=True)
            break
    if ts is None and df.columns[0].lower() in ("unnamed: 0", "index", "price"):
        ts = pd.to_datetime(df[df.columns[0]], errors="coerce", utc=True)
        if ts.isna().all():
            ts = None
    delta = None
    if ts is not None:
        diffs = ts.diff().dropna()
        if len(diffs):
            delta = diffs.median()           # robust to overnight gaps
    return df, ts, delta


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    stem = src.stem if not src.stem.endswith("_v2") else src.stem[:-3]
    dst = HERE / f"{stem}_v2.csv"

    df, ts, delta = load(src)

    # ── detect or BUILD the Close_prev_* lag columns ──────────────────────────
    prev_cols_orig = sorted(
        [c for c in df.columns if c.lower().startswith("close_prev_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not prev_cols_orig:
        if "Close" not in df.columns:
            raise SystemExit(f"{src.name}: no Close_prev_* columns and no Close to build them from")
        print(f"[lags] no Close_prev_* found — building {N_LAGS} lags from Close")
        for j in range(1, N_LAGS + 1):
            df[f"Close_prev_{j}"] = df["Close"].shift(j)
        keep_from = N_LAGS                     # first N_LAGS rows lack full history
        df = df.iloc[keep_from:].reset_index(drop=True)
        if ts is not None:
            ts = ts.iloc[keep_from:].reset_index(drop=True)
        prev_cols_orig = [f"Close_prev_{j}" for j in range(1, N_LAGS + 1)]
    n = len(prev_cols_orig)
    print(f"[lags] {n} lag columns: {prev_cols_orig[0]} … {prev_cols_orig[-1]}")

    # keep only OHLC + lags (drop Volume etc.); ensure column order is stable
    df = df[["Open", "High", "Low", "Close"] + prev_cols_orig].copy()

    # ── build meta columns if absent (start_ts / end_ts / interval) ───────────
    if "start_ts" not in df.columns and ts is not None:
        interval = label_interval(delta) if delta is not None else ""
        df.insert(0, "start_ts", ts.values)
        df.insert(1, "end_ts", (ts + delta).values if delta is not None else ts.values)
        df.insert(2, "interval", interval)

    before = df.iloc[0].copy()
    C0 = before["Close"]

    # ── 1. SMART target: mean(next FUTURE_WINDOW) > mean(prev PREV_WINDOW) ─────
    prev_mean = df[[f"Close_prev_{i}" for i in range(1, PREV_WINDOW + 1)]].mean(axis=1)
    future = pd.concat([df["Close"].shift(-k) for k in range(1, FUTURE_WINDOW + 1)], axis=1)
    future_mean = future.mean(axis=1)
    df["target"] = np.where(future_mean > prev_mean, "buy", "sell")

    # ── 2. Open / High / Low  ÷  Close (per-row) ──────────────────────────────
    C = df["Close"]
    for col in ["Open", "High", "Low"]:
        df[col] = df[col] / C

    # ── 3. Close_prev_1..n  −  Close (per-row) ────────────────────────────────
    for col in prev_cols_orig:
        df[col] = df[col] - C

    # ── 4. Reverse order: prev_1 = oldest lag, prev_n = most recent lag ───────
    rename_map = {prev_cols_orig[i]: f"Close_prev_{n - i}" for i in range(n)}
    df.rename(columns=rename_map, inplace=True)
    new_prev_cols = [f"Close_prev_{i}" for i in range(1, n + 1)]

    # ── 5. Delete Close ───────────────────────────────────────────────────────
    df.drop(columns=["Close"], inplace=True)

    # ── reorder + drop the final FUTURE_WINDOW rows (no full future window) ────
    meta_cols  = [c for c in ["start_ts", "end_ts", "interval"] if c in df.columns]
    price_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]
    df = df[meta_cols + price_cols + new_prev_cols + ["target"]]
    df = df.iloc[:-FUTURE_WINDOW].copy()

    df.to_csv(dst, index=False)

    # ── show the work ─────────────────────────────────────────────────────────
    after = df.iloc[0]
    n_buy = int((df["target"] == "buy").sum())
    print(f"\nper-row reference value C (row 0 Close) = {C0:.4f}\n")
    print(f"{'column':<20}{'before':>14}{'after':>14}   transform")
    print("-" * 65)
    for col in ["Open", "High", "Low"]:
        print(f"{col:<20}{before[col]:>14.4f}{after[col]:>14.6f}   / C")
    print(f"{'Close':<20}{C0:>14.4f}{'(deleted)':>14}")
    for i, orig in enumerate(prev_cols_orig):
        new_name = rename_map[orig]
        print(f"{orig:<20}{before[orig]:>14.4f}{after[new_name]:>14.4f}   - C  |  "
              f"lag {int(orig.split('_')[-1]):>2} → lag {int(new_name.split('_')[-1]):>2}")
    print(f"{'target':<20}{'—':>14}{after['target']:>14}   "
          f"mean(next {FUTURE_WINDOW}) {'>' if after['target']=='buy' else '≤'} mean(prev {PREV_WINDOW})")
    print(f"\ntarget balance: buy={n_buy} ({n_buy/len(df):.1%})  "
          f"sell={len(df)-n_buy} ({1-n_buy/len(df):.1%})")
    print(f"rows: {len(df)}   columns: {len(df.columns)}   →  saved {dst.name}")


if __name__ == "__main__":
    main()
