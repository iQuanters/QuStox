#!/usr/bin/env python3
"""
transform_aapl.py — create a transformed version of AAPL_10m.csv.

Per-row, using each row's own Close value C:

  1. target        : 'buy' if the NEXT bar's Close > this Close, else 'sell'
                     (a genuine next-bar forecast — the label is NOT contained in
                      the features, so no leakage; the final row is dropped)
  2. Open/High/Low : divided by C
  3. Close_prev_1..n : subtracted by C  (C is the reference point)
  4. Reverse order : new Close_prev_1 = old Close_prev_n (oldest lag)
                     new Close_prev_n = old Close_prev_1 (most recent lag)
                     — so prev_1 is now the oldest, prev_n the most recent
  5. Close column  : deleted entirely

n is auto-detected from the column names so this works for any number of lags.
Output: AAPL_10m_v2.csv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
SRC  = HERE / "AAPL_10m.csv"
DST  = HERE / "AAPL_10m_v2.csv"

df = pd.read_csv(SRC)

# ── auto-detect Close_prev columns (sorted by lag number) ─────────────────────
prev_cols_orig = sorted(
    [c for c in df.columns if c.lower().startswith("close_prev_")],
    key=lambda c: int(c.split("_")[-1]),
)   # [Close_prev_1, Close_prev_2, ..., Close_prev_n]  ordered by lag ascending
n = len(prev_cols_orig)
print(f"detected {n} lag columns: {prev_cols_orig[0]} … {prev_cols_orig[-1]}")

# ── snapshot row 0 before transform ──────────────────────────────────────────
before = df.iloc[0].copy()
C0 = before["Close"]

# ── 1. target: direction of the NEXT bar (forecast — no leakage) ──────────────
next_close = df["Close"].shift(-1)
df["target"] = (next_close > df["Close"]).map({True: "buy", False: "sell"})

# ── 2. Open / High / Low  ÷  Close (per-row) ──────────────────────────────────
C = df["Close"]
for col in ["Open", "High", "Low"]:
    df[col] = df[col] / C

# ── 3. Close_prev_1..n  −  Close (per-row) ────────────────────────────────────
for col in prev_cols_orig:
    df[col] = df[col] - C

# ── 4. Reverse order: rename so prev_1=oldest lag, prev_n=most recent lag ─────
#    old Close_prev_1 (lag=1, most recent)  → new Close_prev_n
#    old Close_prev_n (lag=n, oldest)       → new Close_prev_1
rename_map = {
    prev_cols_orig[i]: f"Close_prev_{n - i}"
    for i in range(n)
}
df.rename(columns=rename_map, inplace=True)
new_prev_cols = [f"Close_prev_{i}" for i in range(1, n + 1)]  # now 1=oldest … n=most recent

# ── 5. Delete Close ───────────────────────────────────────────────────────────
df.drop(columns=["Close"], inplace=True)

# ── reorder columns: timestamps + interval, Open/High/Low, prev cols, target ──
meta_cols = [c for c in ["start_ts", "end_ts", "interval"] if c in df.columns]
price_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]
col_order = meta_cols + price_cols + new_prev_cols + ["target"]
df = df[col_order]

# drop the final row — it has no next bar, so its forecast target is undefined
df = df.iloc[:-1].copy()

df.to_csv(DST, index=False)

# ── show the work ─────────────────────────────────────────────────────────────
after = df.iloc[0]
print(f"\nper-row reference value C (row 0 Close) = {C0:.4f}\n")
print(f"{'column':<20}{'before':>14}{'after':>14}   transform")
print("-" * 65)
for col in ["Open", "High", "Low"]:
    print(f"{col:<20}{before[col]:>14.4f}{after[col]:>14.6f}   / C")
print(f"{'Close':<20}{C0:>14.4f}{'(deleted)':>14}")
for i, orig in enumerate(prev_cols_orig):
    new_name = rename_map[orig]
    orig_lag = int(orig.split("_")[-1])
    new_lag  = int(new_name.split("_")[-1])
    arrow = f"lag {orig_lag:>2} → lag {new_lag:>2}"
    print(f"{orig:<20}{before[orig]:>14.4f}{after[new_name]:>14.4f}   - C  |  {arrow}")
print(f"{'target':<20}{'—':>14}{after['target']:>14}   next Close "
      f"{'>' if after['target']=='buy' else '<'} this Close")
print(f"\nrows: {len(df)}   columns: {len(df.columns)}   →  saved {DST.name}")
