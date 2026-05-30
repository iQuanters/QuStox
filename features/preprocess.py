import numpy as np
import pandas as pd

from .indicators import rsi, macd, bollinger_bands, atr

# 4 features fed into the quantum reservoir (one per qubit)
QUANTUM_FEATURE_COLS = ["rsi_norm", "macd_norm", "bb_pos", "ret_1_norm"]

# All features available to every model
ALL_FEATURE_COLS = [
    "ret_1_norm", "ret_5_norm", "ret_20_norm",
    "rsi_norm", "macd_norm", "bb_pos",
    "atr_norm", "vol_ratio",
    "hour_sin", "hour_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    f = pd.DataFrame(index=df.index)

    # ── returns (tanh-scaled to keep outliers bounded) ──────────────────────
    f["ret_1_norm"]  = np.tanh(close.pct_change(1)  / 0.02)
    f["ret_5_norm"]  = np.tanh(close.pct_change(5)  / 0.05)
    f["ret_20_norm"] = np.tanh(close.pct_change(20) / 0.10)

    # ── RSI → [0, 1] ─────────────────────────────────────────────────────────
    f["rsi_norm"] = rsi(close, 14) / 100.0

    # ── MACD histogram → tanh-normalized ─────────────────────────────────────
    _, _, hist = macd(close)
    rolling_std = hist.rolling(50, min_periods=10).std().replace(0, np.nan).ffill()
    f["macd_norm"] = np.tanh(hist / (rolling_std + 1e-12))

    # ── Bollinger Band position → [-1, 1] ────────────────────────────────────
    _, _, bp = bollinger_bands(close, 20)
    f["bb_pos"] = (bp * 2 - 1).clip(-1, 1)

    # ── ATR as fraction of price → tanh-scaled ──────────────────────────────
    raw_atr = atr(high, low, close, 14) / (close + 1e-12)
    f["atr_norm"] = np.tanh(raw_atr / raw_atr.rolling(50, min_periods=10).std().replace(0, np.nan).ffill())

    # ── volume ratio vs 20-bar average → tanh-scaled ────────────────────────
    vr = vol / (vol.rolling(20).mean() + 1e-12)
    f["vol_ratio"] = np.tanh((vr - 1.0) / 0.5)

    # ── intraday time encoding ───────────────────────────────────────────────
    idx = df.index
    if hasattr(idx, "hour"):
        hour_float = idx.hour + idx.minute / 60
    else:
        hour_float = np.full(len(df), 12.0)  # daily bars: use midday as neutral
    f["hour_sin"] = np.sin(2 * np.pi * np.asarray(hour_float) / 24)
    f["hour_cos"] = np.cos(2 * np.pi * np.asarray(hour_float) / 24)

    # ── next-bar direction label ─────────────────────────────────────────────
    f["label"] = (close.shift(-1) > close).astype(int)

    return f.dropna()


def train_test_split_temporal(feat: pd.DataFrame, test_frac: float = 0.25):
    """Strict time-based split — no data leakage."""
    n = len(feat)
    split = int(n * (1 - test_frac))
    return feat.iloc[:split].copy(), feat.iloc[split:].copy()


def get_Xy(feat: pd.DataFrame, cols: list):
    X = feat[cols].values.astype(np.float64)
    y = feat["label"].values.astype(int)
    return X, y
