import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf

def main():
    parser = argparse.ArgumentParser(description="Download and resample stock market data for different intervals.")
    parser.add_argument("ticker", nargs="?", default="SPY", help="Stock ticker symbol (default: SPY)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    output_dir = os.path.join("dataset", ticker)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # ==========================================
    # LEVEL 1 & 2: Intraday (Max allowed is 60 days)
    # ==========================================
    print(f"Downloading 20-minute and 2-hour data for {ticker} (Max 60 days allowed)...")
    
    # 20-Minute Data (Fetch 5m data, resample to 20m)
    df_5m = yf.download(ticker, period="60d", interval="5m", auto_adjust=True, prepost=True, multi_level_index=False)
    if df_5m.empty:
        print(f"Error: No data found for ticker {ticker}. Please check the symbol and try again.")
        return
    df_5m.to_csv(os.path.join(raw_dir, f"{ticker}_raw_5m.csv"))
        
    df_20m = df_5m.resample("20min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_20m.to_csv(os.path.join(output_dir, f"{ticker}_20min_local.csv"))
    
    # 2-Hour Data (Fetch 1h data, resample to 2h)
    df_1h = yf.download(ticker, period="60d", interval="1h", auto_adjust=True, prepost=True, multi_level_index=False)
    if df_1h.empty:
        print(f"Error: No hourly data found for ticker {ticker}.")
        return
    df_1h.to_csv(os.path.join(raw_dir, f"{ticker}_raw_1h.csv"))
    
    df_2h = df_1h.resample("2h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_2h.to_csv(os.path.join(output_dir, f"{ticker}_2hours_local.csv"))
    
    # ==========================================
    # LEVEL 3: 2-Day Dataset (Full 5 Years allowed!)
    # ==========================================
    print(f"\nDownloading 5 years of daily data for {ticker} to build the 2-Day dataset...")
    
    # Step 1: Download 5 years of standard 1-day data
    df_1d = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, multi_level_index=False)
    if df_1d.empty:
        print(f"Error: No daily data found for ticker {ticker}.")
        return
    df_1d.to_csv(os.path.join(raw_dir, f"{ticker}_raw_1d.csv"))
        
    # Step 2: Group by every 2 trading days using integer division on row indices.
    # This ensures that the Open of Day 1 and Close of Day 2 match perfectly for each pair of consecutive trading days,
    # avoiding calendar-based artifacts (like single-day bars over weekends/holidays).
    df_2d = df_1d.groupby(np.arange(len(df_1d)) // 2).agg({
        "Open": "first", 
        "High": "max", 
        "Low": "min", 
        "Close": "last", 
        "Volume": "sum"
    })
    df_2d.index = df_1d.index[::2]
    
    # Save it locally
    df_2d.to_csv(os.path.join(output_dir, f"{ticker}_2days_local.csv"))
    
    print(f"\nAll datasets for {ticker} processed and saved in {output_dir}!")

if __name__ == "__main__":
    main()