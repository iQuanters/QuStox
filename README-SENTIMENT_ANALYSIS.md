# Sentiment Analysis — External Factors Agent

Branch: `pabarot/feat-qustox-trade`  
Author: Parth Barot

This module reads live market context (VIX + news headlines) and classifies
a **regime** and **window multiplier** that the HQRC reservoir uses to
dynamically adjust its context window size.

---

## How it fits into the pipeline

```
yfinance (^VIX + SPY news)
        ↓
  External Factors Agent          ← this module
  (news_fetcher + OpenAI)
        ↓
  ExternalFactors.window_multiplier
        ↓
  Your HQRC predictor.predict()   ← your code
        ↓
  BUY / SELL / HOLD signal
```

---

## Setup

### 1. Merge / pull this branch

```bash
git fetch origin
git checkout pabarot/feat-qustox-trade
```

Or from your own branch, merge it in:

```bash
git merge origin/pabarot/feat-qustox-trade
```

### 2. Install dependencies

```bash
pip install openai yfinance python-dotenv
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root (never commit this):

```bash
cp .env.example .env
# open .env and fill in your key
```

`.env` contents:
```
OPENAI_API_KEY=sk-...
```

---

## Integration — copy this into your HQRC code

```python
from dotenv import load_dotenv
load_dotenv()  # load .env before anything else

from src.agents import get_external_factors

# Step 1 — get regime + multiplier (call once before prediction loop)
factors = get_external_factors(
    ticker="SPY",   # match your HQRC ticker
    verbose=True,   # prints VIX, headlines, regime to stdout
)

# Step 2 — pass multiplier into your HQRC predictor
result = predictor.predict(
    data=data,
    tf_cfg=tf_cfg,
    regime_multiplier=factors.window_multiplier,  # ← the one line that connects both
)
```

That's it. One import, one call, one value passed in.

---

## What `get_external_factors()` returns

```python
@dataclass
class ExternalFactors:
    regime:            str    # e.g. "NORMAL", "FED_EVENT", "CRISIS"
    window_multiplier: float  # e.g. 1.0, 0.5, 0.4  — pass to predictor
    sentiment_score:   float  # -1.0 (bearish) → +1.0 (bullish)
    reasoning:         str    # one-sentence explanation from OpenAI
    vix_current:       float  # latest VIX close
    vix_trend:         str    # "RISING" | "FALLING" | "STABLE"
    headlines_used:    int    # number of headlines analysed
    as_of:             datetime
```

### Regime → multiplier reference

| Regime | Multiplier | Meaning |
|---|---|---|
| `CALM` | `1.5×` | VIX < 15 — expand window, market is stable |
| `NORMAL` | `1.0×` | VIX 15–25 — baseline, no adjustment |
| `ELEVATED` | `0.7×` | VIX 25–35 — compress slightly, rising vol |
| `CRISIS` | `0.4×` | VIX > 35 — compress hard, trust only recent bars |
| `FED_EVENT` | `0.5×` | FOMC decision day |
| `ELECTION` | `0.6×` | Election night / result day |
| `GEOPOLITICAL` | `0.55×` | War / sanctions / major news event |
| `PANDEMIC` | `0.3×` | Systemic health shock |
| `EARNINGS` | `0.8×` | SPY earnings season |

---

## Full end-to-end example

```python
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from src.agents import get_external_factors
from src.hqrc import HQRCPredictor, HQRCConfig, load_csv, compute_features, TIMEFRAME_SETTINGS
from pathlib import Path

DATA_DIR = Path("dataset")
TICKER   = "SPY"

# ── 1. Get external factors (live, right now) ─────────────────────────────────
factors = get_external_factors(TICKER, verbose=True)

print(f"\nRegime     : {factors.regime}")
print(f"Multiplier : {factors.window_multiplier}x")
print(f"Sentiment  : {factors.sentiment_score:+.2f}")
print(f"Reasoning  : {factors.reasoning}")

# ── 2. Run HQRC on each timeframe with adjusted window ────────────────────────
cfg       = HQRCConfig(mode="pennylane")   # or "xpyq_qpu" for IBM hardware
predictor = HQRCPredictor(cfg)

for timeframe in ["10m", "30m", "1d"]:
    tf_cfg = TIMEFRAME_SETTINGS[timeframe]

    df   = load_csv(TICKER, timeframe, DATA_DIR)
    data = compute_features(df, TICKER, timeframe)

    result = predictor.predict(
        data=data,
        tf_cfg=tf_cfg,
        regime_multiplier=factors.window_multiplier,  # ← from sentiment agent
    )

    print(f"\n[{timeframe}]  {result.signal}  "
          f"conf={result.confidence:.0%}  "
          f"cv_acc={result.cv_accuracy:.1%}")

# ── 3. Final output ───────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"REGIME    : {factors.regime}  ({factors.window_multiplier}x window)")
print(f"SENTIMENT : {factors.sentiment_score:+.2f}  ({factors.vix_current:.1f} VIX)")
print(f"REASONING : {factors.reasoning}")
```

### Expected output

```
── External Factors  [SPY  @  now] ──
   Fetching market context …
   VIX  : 15.3  (p14, FALLING)
   News : 5 headlines
   Classifying regime via Claude …
   Regime     : NORMAL
   Multiplier : 1.0×
   Sentiment  : +0.20
   Reasoning  : Low VIX with stable macro backdrop suggests normal conditions.

[10m]  BUY   conf=63%  cv_acc=57.2%
[30m]  BUY   conf=68%  cv_acc=58.1%
[1d]   HOLD  conf=54%  cv_acc=55.9%

────────────────────────────────────────
REGIME    : NORMAL  (1.0x window)
SENTIMENT : +0.20  (15.3 VIX)
REASONING : Low VIX with stable macro backdrop suggests normal conditions.
```

---

## Historical / backtesting mode

Pass `as_of` to get the sentiment for any past date — useful for validating
that the regime classifier would have caught known events:

```python
from datetime import datetime, timezone

# US Election Day 2024
factors = get_external_factors(
    "SPY",
    as_of=datetime(2024, 11, 5, tzinfo=timezone.utc),
)
# → regime="ELECTION", multiplier=0.6, sentiment_score=-0.60

# COVID crash
factors = get_external_factors(
    "SPY",
    as_of=datetime(2020, 3, 16, tzinfo=timezone.utc),
)
# → regime="CRISIS", multiplier=0.4, sentiment_score=-0.95
```

---

## Standalone demo (no HQRC needed)

```bash
python3 run.py SPY                          # live signal right now
python3 run.py AAPL                         # different ticker
python3 run.py SPY --as-of "2024-11-05"     # election day backtest
python3 run.py SPY --as-of "2020-03-16"     # COVID crash
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key — get at platform.openai.com |
| `OPENAI_MODEL` | No | Override model (default: `gpt-4o-mini`) |

---

## Module structure

```
src/agents/
├── __init__.py               ← exports get_external_factors()
├── news_fetcher.py           ← VIX + headlines from yfinance
├── sentiment_analyzer.py     ← OpenAI JSON classification
└── external_factors.py       ← public get_external_factors()
```
