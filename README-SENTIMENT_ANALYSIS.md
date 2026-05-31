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
  news_fetcher.py
  ├── Live news: yfinance (last ~10 articles)
  └── Historical news: OpenAI gpt-4o recall (events back to ~2020)
        ↓
  sentiment_analyzer.py
  └── OpenAI gpt-4o-mini → regime label + sentiment score (JSON mode)
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
    regime:            str    # e.g. "NORMAL", "FED_EVENT", "GEOPOLITICAL"
    window_multiplier: float  # e.g. 1.0, 0.5, 0.55 — pass to predictor
    sentiment_score:   float  # -1.0 (bearish) → +1.0 (bullish)
    reasoning:         str    # one-sentence explanation from OpenAI
    vix_current:       float  # VIX close on or before as_of
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

## News headline sources

The module uses two sources depending on the date:

| Date range | Source | How |
|---|---|---|
| Last few days | yfinance live news | `Ticker.news` — real articles |
| Up to ~Apr 2025 | OpenAI `gpt-4o` recall | GPT's training data covers major market events |
| After ~Apr 2025 | `--scenario` flag | Pre-loaded headlines for events past GPT cutoff |

The source is tracked and printed:
```
News : 10 headlines  [recalled by OpenAI (historical)]
News : 3 headlines   [live via yfinance]
```

---

## Standalone demo (no HQRC needed)

### Live signal
```bash
python3 run.py SPY          # SPY right now
python3 run.py AAPL         # any ticker
```

### Historical backtesting — GPT recalls headlines automatically
```bash
python3 run.py SPY --as-of "2024-11-05"   # US Election Day 2024
python3 run.py SPY --as-of "2024-09-18"   # Fed 50bps rate cut
python3 run.py SPY --as-of "2020-03-16"   # COVID crash
python3 run.py SPY --as-of "2022-02-24"   # Russia invades Ukraine
```

### Events past GPT training cutoff — pre-loaded scenario
```bash
python3 run.py SPY --scenario geopolitical   # US-Israel-Iran war (Jun 2025)
```

**Rule:** if the date is before ~April 2025 → use `--as-of`, GPT recalls.  
After April 2025 → use `--scenario` (pre-loaded headlines required).

---

## Historical backtesting example (Python)

```python
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from src.agents import get_external_factors

# US Election Day 2024 — GPT recalls real election headlines
factors = get_external_factors("SPY", as_of=datetime(2024, 11, 5, tzinfo=timezone.utc))
# → regime="ELECTION", multiplier=0.6, sentiment_score ~-0.4 to +0.7

# Fed 50bps surprise cut Sep 2024
factors = get_external_factors("SPY", as_of=datetime(2024, 9, 18, tzinfo=timezone.utc))
# → regime="FED_EVENT", multiplier=0.5

# COVID crash Mar 2020
factors = get_external_factors("SPY", as_of=datetime(2020, 3, 16, tzinfo=timezone.utc))
# → regime="CRISIS", multiplier=0.4, sentiment_score ~-0.9
```

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

# ── 1. Get external factors ───────────────────────────────────────────────────
factors = get_external_factors(TICKER, verbose=True)

# ── 2. Run HQRC on each timeframe with adjusted window ────────────────────────
cfg       = HQRCConfig(mode="pennylane")   # or "xpyq_qpu" for IBM hardware
predictor = HQRCPredictor(cfg)

for timeframe in ["10m", "30m", "1d"]:
    tf_cfg = TIMEFRAME_SETTINGS[timeframe]
    df     = load_csv(TICKER, timeframe, DATA_DIR)
    data   = compute_features(df, TICKER, timeframe)

    result = predictor.predict(
        data=data,
        tf_cfg=tf_cfg,
        regime_multiplier=factors.window_multiplier,  # ← from sentiment agent
    )
    print(f"[{timeframe}]  {result.signal}  conf={result.confidence:.0%}")

# ── 3. Summary ────────────────────────────────────────────────────────────────
print(f"REGIME    : {factors.regime}  ({factors.window_multiplier}x window)")
print(f"SENTIMENT : {factors.sentiment_score:+.2f}  (VIX {factors.vix_current:.1f})")
print(f"REASONING : {factors.reasoning}")
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | — | Get at platform.openai.com |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model for sentiment classification |
| `OPENAI_RECALL_MODEL` | No | `gpt-4o` | Model for historical headline recall (needs later training cutoff) |

---

## Module structure

```
src/agents/
├── __init__.py               ← exports get_external_factors()
├── news_fetcher.py           ← VIX from yfinance + headlines (live or GPT recall)
├── sentiment_analyzer.py     ← OpenAI gpt-4o-mini JSON classification
└── external_factors.py       ← public get_external_factors() entry point
```
