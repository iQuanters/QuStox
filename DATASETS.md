# Stock Datasets Guide

This repository contains pre-downloaded historical datasets for 5 major tickers (`SPY`, `AAPL`, `GOOG`, `CVX`, `OXY`) to allow you to get started immediately without hitting API rate limits.

Because of how Git works, these 5 stock folders are committed (tracked) in the repository so other users can download them upon cloning. However, the entire `dataset/` directory is listed in `.gitignore` to prevent tracking any local changes or new stock downloads.

Here is how to manage this setup:

## 1. How to Ignore Local Modifications to the 5 Tracked Stocks
If you run the downloader and modify the existing files for the 5 pre-loaded stocks, Git will mark them as modified. To tell Git to **ignore local modifications** to these files (so you don't accidentally commit or push them), run the following command in your terminal:

```bash
for d in SPY AAPL GOOG CVX OXY; do
  git update-index --skip-worktree dataset/$d/*.csv dataset/$d/raw/*.csv 2>/dev/null
done
```

If you ever want Git to track changes to them again, run:
```bash
for d in SPY AAPL GOOG CVX OXY; do
  git update-index --no-skip-worktree dataset/$d/*.csv dataset/$d/raw/*.csv 2>/dev/null
done
```

## 2. How to Add New Tickers
To add new stock tickers, simply run the downloader script:
```bash
uv run python main.py <TICKER>
```
The script will download and save the data under `dataset/<TICKER>/`. Because `dataset/` is in `.gitignore`, new folders (like `dataset/TSLA/`) are automatically ignored and will not show up in `git status` or be pushed.

## 3. How to Remove Tickers
* **To delete local files:** Simply delete the folder, e.g., `rm -rf dataset/TSLA`.
* **To stop tracking a pre-loaded stock entirely (remove it from GitHub):**
  If you want to untrack one of the 5 pre-loaded stocks from Git history, run:
  ```bash
  git rm -r --cached dataset/<TICKER>/
  git commit -m "remove <TICKER> from tracked datasets"
  git push origin master
  ```
