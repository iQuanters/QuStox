# QuStox

A minimalist command-line tool to download and resample stock market data from Yahoo Finance into custom schema formats.

## Quick Start

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed.

### 2. Run Data Download & Resampling
Run the script with a stock ticker symbol (defaults to `SPY`):
```bash
# Fetch latest data and generate datasets
uv run python main.py AAPL

# Process using locally cached raw files (offline/saved api calls)
uv run python main.py AAPL --use-cache
```

## Datasets Output

Outputs are stored under the `dataset/{TICKER}/` folder:
* **Raw Files** (`dataset/{TICKER}/raw/`): Original Yahoo Finance data in standard OHLCV format (`5m`, `10m`, `30m`, `1h`, `1d`).
* **Processed Schema** (`dataset/{TICKER}/`): Lag-enriched datasets (`10m`, `30m`, `1d`) formatted with columns:
  `start_ts`, `end_ts`, `interval`, `Open`, `High`, `Low`, `Close`, and lagged closing prices (`Close_prev_1` to `Close_prev_N`).

## Pre-loaded Datasets

This repository includes pre-downloaded historical data for `SPY`, `AAPL`, `GOOG`, `CVX`, and `OXY`. To learn how to manage local changes, add new stock tickers, or remove existing ones, check the [Stock Datasets Guide](DATASETS.md).
