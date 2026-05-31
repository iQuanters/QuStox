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
    
    def get_timedelta(interval):
        if interval == "10m":
            return pd.to_timedelta("10min")
        elif interval == "30m":
            return pd.to_timedelta("30min")
        elif interval == "1d":
            return pd.to_timedelta("1D")
        else:
            return pd.to_timedelta(interval)

    def apply_schema_and_save(df, interval, filepath):
        df_new = df.copy()
        
        # Drop Volume if present
        if 'Volume' in df_new.columns:
            df_new = df_new.drop(columns=['Volume'])
            
        # Reorder columns to OHLC
        df_new = df_new[['Open', 'High', 'Low', 'Close']]
        
        # Add previous closes
        num_prev = 10 if interval == "1d" else 12
        for j in range(1, num_prev + 1):
            df_new[f"Close_prev_{j}"] = df_new["Close"].shift(j)
            
        # Drop rows with NaN (drops the first num_prev rows where we don't have enough history)
        df_new = df_new.dropna()
        
        # Insert start_ts, end_ts, and interval columns
        delta = get_timedelta(interval)
        df_new.insert(0, 'start_ts', df_new.index)
        df_new.insert(1, 'end_ts', df_new.index + delta)
        df_new.insert(2, 'interval', interval)
        
        # Save to CSV without index
        df_new.to_csv(filepath, index=False)
    
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
            
        if os.path.exists(filepath):
            print(f"Existing raw file found at {filepath}. Performing incremental update...")
            # Load old data
            df_old = pd.read_csv(filepath, index_col=0)
            df_old.index = pd.to_datetime(df_old.index, utc=True)
            if interval in ["5m", "1h"]:
                df_old.index = df_old.index.tz_convert("America/New_York")
            else:
                df_old.index = df_old.index.tz_localize(None)
                
            last_ts = df_old.index[-1]
            print(f"Last available datapoint for {ticker} {interval}: {last_ts}")
            
            # Fetch only the new data starting from last_ts
            print(f"Downloading new {interval} data for {ticker} starting from {last_ts}...")
            df_new = yf.download(ticker, start=last_ts, interval=interval, auto_adjust=True, prepost=prepost, multi_level_index=False)
            
            if not df_new.empty:
                # Merge old and new
                df_combined = pd.concat([df_old, df_new])
                # Deduplicate, keeping the latest entry for any overlapping timestamps
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                # Sort chronologically
                df_combined.sort_index(inplace=True)
                # Save back to CSV
                df_combined.to_csv(filepath)
                print(f"Dataset updated. Total rows: {len(df_combined)} (added {len(df_combined) - len(df_old)} new rows).")
                return df_combined
            else:
                print("No new data found. Using existing cached data.")
                return df_old
        else:
            # File doesn't exist, perform full download
            print(f"No existing raw file found. Performing full download of {interval} data (period={period})...")
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, prepost=prepost, multi_level_index=False)
            if not df.empty:
                df.to_csv(filepath)
            return df
 
    # ==========================================
    # LEVEL 1 & 2: Intraday (Max allowed is 60 days)
    # ==========================================
    print(f"\nProcessing intraday data for {ticker} (10-minute and 30-minute)...")
    
    # Fetch 5m data (used for both 10m and 30m resampling)
    df_5m = get_data(period="60d", interval="5m", filename=f"{ticker}_5m.csv", prepost=True)
    if df_5m.empty:
        print(f"Error: No 5m data found for ticker {ticker}. Please check the symbol and try again.")
        return
        
    # 10-Minute Data (resample from 5m)
    df_10m = df_5m.resample("10min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_10m.to_csv(os.path.join(raw_dir, f"{ticker}_10m.csv"))
    apply_schema_and_save(df_10m, "10m", os.path.join(output_dir, f"{ticker}_10m.csv"))
    
    # 30-Minute Data (resample from 5m)
    df_30m = df_5m.resample("30min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df_30m.to_csv(os.path.join(raw_dir, f"{ticker}_30m.csv"))
    apply_schema_and_save(df_30m, "30m", os.path.join(output_dir, f"{ticker}_30m.csv"))
    
    # Download raw 1h data (no aggregation required, just keep the raw file)
    df_1h = get_data(period="60d", interval="1h", filename=f"{ticker}_1h.csv", prepost=True)
    if df_1h.empty:
        print(f"Error: No 1h data found for ticker {ticker}.")
        return
    
    # ==========================================
    # LEVEL 3: 1-Day Dataset (Max history)
    # ==========================================
    print(f"\nProcessing 1-day daily data for {ticker}...")
    
    # Download max historical daily data (saves directly as {ticker}_1d.csv)
    df_1d = get_data(period="max", interval="1d", filename=f"{ticker}_1d.csv", prepost=False)
    if df_1d.empty:
        print(f"Error: No daily data found for ticker {ticker}.")
        return
    apply_schema_and_save(df_1d, "1d", os.path.join(output_dir, f"{ticker}_1d.csv"))
    
    print(f"\nAll datasets for {ticker} processed and saved in {output_dir} (new schema) and {raw_dir} (raw OHLCV)!")

if __name__ == "__main__":
    main()