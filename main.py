import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf

def main():
    parser = argparse.ArgumentParser(description="Download and resample stock market data for different intervals.")
    parser.add_argument("ticker", nargs="?", default="SPY", help="Stock ticker symbol (default: SPY)")
    parser.add_argument("--use-cache", action="store_true", help="Use locally saved raw files if available instead of downloading")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    output_dir = os.path.join("dataset", ticker)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    use_cache = args.use_cache
    
    def get_data(period, interval, filename, prepost=False):
        filepath = os.path.join(raw_dir, filename)
        if use_cache and os.path.exists(filepath):
            print(f"Loading cached raw data from {filepath}...")
            df = pd.read_csv(filepath, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            if interval in ["5m", "1h"]:
                df.index = df.index.tz_convert("America/New_York")
            else:
                df.index = df.index.tz_localize(None)
            return df
        else:
            print(f"Downloading {interval} data for {ticker} (period={period})...")
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, prepost=prepost, multi_level_index=False)
            if not df.empty:
                df.to_csv(filepath)
            return df

    # ==========================================
    # LEVEL 1 & 2: Intraday (Max allowed is 60 days)
    # ==========================================
    print(f"\nProcessing intraday data for {ticker} (10-minute and 30-minute)...")
    
    # Fetch 5m data (used for both 10m and 30m resampling)
    df_5m = get_data(period="60d", interval="5m", filename=f"{ticker}_raw_5m.csv", prepost=True)
    if df_5m.empty:
        print(f"Error: No 5m data found for ticker {ticker}. Please check the symbol and try again.")
        return
        
    # 10-Minute Data (resample from 5m)
    df_10m = df_5m.resample("10min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_10m.to_csv(os.path.join(output_dir, f"{ticker}_10min_local.csv"))
    
    # 30-Minute Data (resample from 5m)
    df_30m = df_5m.resample("30min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_30m.to_csv(os.path.join(output_dir, f"{ticker}_30min_local.csv"))
    
    # Download raw 1h data (no aggregation required, just keep the raw file)
    df_1h = get_data(period="60d", interval="1h", filename=f"{ticker}_raw_1h.csv", prepost=True)
    if df_1h.empty:
        print(f"Error: No 1h data found for ticker {ticker}.")
        return
    
    # ==========================================
    # LEVEL 3: 1-Day Dataset (Max history)
    # ==========================================
    print(f"\nProcessing 1-day daily data for {ticker}...")
    
    # Download max historical daily data
    df_1d = get_data(period="max", interval="1d", filename=f"{ticker}_raw_1d.csv", prepost=False)
    if df_1d.empty:
        print(f"Error: No daily data found for ticker {ticker}.")
        return
    
    # Save the 1d data directly as the local output
    df_1d.to_csv(os.path.join(output_dir, f"{ticker}_1d_local.csv"))
    
    print(f"\nAll datasets for {ticker} processed and saved in {output_dir}!")

if __name__ == "__main__":
    main()