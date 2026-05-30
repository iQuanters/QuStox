import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_spy(interval: str = "1h", period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"spy_{interval}_{period}.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"[data] loaded {len(df)} bars from cache ({interval}, {period})")
        return df

    print(f"[data] downloading SPY {interval} bars, period={period} …")
    raw = yf.download("SPY", period=period, interval=interval, progress=False, auto_adjust=True)

    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(cache_path)
    print(f"[data] downloaded {len(df)} bars → cached")
    return df
