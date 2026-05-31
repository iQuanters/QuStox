#!/usr/bin/env python3
"""
prepare_data.py — convert AAPL_10m.csv to aapl_x.npy / aapl_y.npy for hqrc_nn.

  aapl_y : (N,)     — Close (target next-bar price)
  aapl_x : (N, 12)  — [Close_prev_12, ..., Close_prev_1]  (oldest → newest)

Output saved next to this file as aapl_x.npy and aapl_y.npy
(the exact names hqrc_nn.py loads directly by name).
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CSV  = HERE / "AAPL_10m.csv"

df = pd.read_csv(CSV)

prev_cols = [f"Close_prev_{i}" for i in range(12, 0, -1)]  # prev12 first → prev1 last
missing = [c for c in prev_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

X = df[prev_cols].to_numpy(dtype=np.float64)   # (N, 12)
y = df["Close"].to_numpy(dtype=np.float64)      # (N,)

np.save(HERE / "aapl_x.npy", X)
np.save(HERE / "aapl_y.npy", y)

print(f"aapl_x : {X.shape}  →  aapl_x.npy")
print(f"aapl_y : {y.shape}  →  aapl_y.npy")
print(f"X[0] = {X[0]}  →  y[0] = {y[0]:.4f}")
print(f"X[-1] = {X[-1]}  →  y[-1] = {y[-1]:.4f}")
