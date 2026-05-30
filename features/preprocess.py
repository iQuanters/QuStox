"""
Univariate series preparation — everything is derived from the closing price.

  close   : raw closing price (kept for signal reconstruction & confidence)
  ret     : simple return  r_t = close_t / close_{t-1} − 1   (target source)
  enc     : tanh-bounded return fed into the quantum reservoir's angle encoding
  target  : next-bar return  r_{t+1}   (what the SVR regresses on)
"""

import numpy as np
import pandas as pd


def build_series(df: pd.DataFrame, return_norm: float = 0.02) -> pd.DataFrame:
    """Build the univariate frame (close, ret, enc, target) from OHLCV data."""
    close = df["Close"].squeeze()
    ret = close.pct_change()
    enc = np.tanh(ret / return_norm)        # bounded → stable angle encoding
    target = close.pct_change().shift(-1)   # next-bar return

    out = pd.DataFrame(
        {"close": close, "ret": ret, "enc": enc, "target": target},
        index=df.index,
    )
    return out.dropna()                      # drops first (no ret) & last (no target)


def train_test_split_temporal(series: pd.DataFrame, test_frac: float = 0.25):
    """Strict time-based split — no shuffling, no leakage."""
    n = len(series)
    split = int(n * (1 - test_frac))
    return series.iloc[:split].copy(), series.iloc[split:].copy()


def make_lag_windows(enc: np.ndarray, window: int) -> np.ndarray:
    """
    Classical baseline features: the raw window of the last `window` encoded
    returns at each step (left-padded with zeros). This is exactly the
    information the quantum reservoir is given, so the baseline is honest.
    """
    enc = np.asarray(enc, dtype=np.float64)
    n = len(enc)
    W = np.zeros((n, window), dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = enc[lo : t + 1]
        W[t, window - len(seg):] = seg
    return W
