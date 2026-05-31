#!/usr/bin/env python3
"""
prepare_data.py — build the QRC classification arrays from AAPL_10m_v2.csv.

AAPL_10m_v2.csv (produced by transform_aapl.py) has, per row:
  Open, High, Low (÷ Close),  Close_prev_1..n (− Close, reversed: 1=oldest),  target

This script saves three arrays (the exact names hqrc_nn.py loads by name):
  aapl_x.npy   : (N, n) close_prev windows  [oldest … newest]  → reservoir input
  aapl_ohl.npy : (N, 3) Open, High, Low                        → concatenated to NN
  aapl_y.npy   : (N,)   labels  1 = buy, 0 = sell               → target

n is auto-detected, so any number of Close_prev_* columns works.
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC  = HERE / "AAPL_10m_v2.csv"

df = pd.read_csv(SRC)

# auto-detect close_prev columns, ordered by lag number (1 … n)
prev_cols = sorted(
    [c for c in df.columns if c.lower().startswith("close_prev_")],
    key=lambda c: int(c.split("_")[-1]),
)
n = len(prev_cols)

X   = df[prev_cols].to_numpy(dtype=np.float64)          # (N, n)
OHL = df[["Open", "High", "Low"]].to_numpy(dtype=np.float64)   # (N, 3)
y   = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()  # 1=buy,0=sell

np.save(HERE / "aapl_x.npy", X)
np.save(HERE / "aapl_ohl.npy", OHL)
np.save(HERE / "aapl_y.npy", y)

print(f"detected {n} close_prev columns: {prev_cols[0]} … {prev_cols[-1]}")
print(f"aapl_x   : {X.shape}    →  aapl_x.npy")
print(f"aapl_ohl : {OHL.shape}     →  aapl_ohl.npy")
print(f"aapl_y   : {y.shape}       →  aapl_y.npy")
print(f"labels   : buy={int(y.sum())} ({y.mean():.1%})  sell={int((1-y).sum())} ({1-y.mean():.1%})")
