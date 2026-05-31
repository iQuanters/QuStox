# QuStox — Hybrid Quantum Reservoir Trading System
**QAI Hackathon · Track 2 (Mid-Frequency Trading) · Qiskit + OpenAI**

A hybrid quantum-classical trading pipeline combining a **16-qubit Ising quantum reservoir** with an **AI sentiment agent** to generate buy/sell signals on equity data.

---

## How It Works

```
Market Data (AAPL 10-min bars)
        │
        ▼
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│   QUANTUM RESERVOIR (fixed)     │     │   SENTIMENT AGENT (live)     │
│                                 │     │                              │
│  16 qubits · 4 groups of 4      │     │  VIX level + trend           │
│  Random Ising Hamiltonian        │     │  News headlines (yfinance)   │
│  Sliding window · 1000 shots     │     │  OpenAI regime classifier    │
│  → 40 quantum features           │     │  → regime + confidence gate  │
│    (16 ⟨Z⟩ + 24 ⟨ZZ⟩)           │     │                              │
└──────────────┬──────────────────┘     └──────────────┬───────────────┘
               │                                       │
               ▼                                       ▼
        Pre-trained MLP              window_multiplier / conf_gate
        [43 → 6 → 1]                 (scales risk by market regime)
               │                                       │
               └───────────────┬───────────────────────┘
                               ▼
                    Final Signal: BUY / SELL / HOLD
                    + P(buy) · conf% · pos_size
```

**Sentiment regimes and their effect on the signal:**

| Regime | Trigger | Conf. Gate | Position Size |
|---|---|---|---|
| CALM | VIX < 15, quiet news | ≥ 40% | 1.00× |
| NORMAL | VIX 15–25, baseline | ≥ 50% | 1.00× |
| ELEVATED | VIX 25–35, rising vol | ≥ 60% | 0.70× |
| CRISIS | VIX > 35, crash/panic | ≥ 75% | 0.40× |
| ELECTION | US election day | ≥ 65% | 0.60× |
| FED_EVENT | FOMC meeting day | ≥ 65% | 0.50× |
| GEOPOLITICAL | War / sanctions | ≥ 70% | 0.55× |
| PANDEMIC | Systemic health shock | ≥ 80% | 0.30× |
| EARNINGS | Earnings season | ≥ 55% | 0.80× |

---

## Environment Setup

### 1. Clone and enter the repo
```bash
git clone https://github.com/iQuanters/QuStox.git
cd QuStox
git checkout qustox/final
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

Or manually:
```bash
pip3 install qiskit qiskit-aer scikit-learn openai python-dotenv yfinance pandas numpy joblib
```

### 3. Add your OpenAI API key
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
OPENAI_API_KEY=sk-...
```
Get a key at [platform.openai.com](https://platform.openai.com).
> ⚠️ `.env` is gitignored — it will **never** be committed to GitHub.

### 4. Verify setup
```bash
python3 -c "from src.agents.external_factors import get_external_factors; print('sentiment OK')"
python3 -c "import qiskit, sklearn, joblib; print('quantum OK')"
```

---

## Entry Points

| Script | Purpose | Speed |
|---|---|---|
| `infer_pipeline.py` | **Single-bar inference** · local Aer simulator + sentiment | ~30s |
| `infer_xpyq.py` | **Single-bar inference** · xpyq cloud / real IBM QPU + sentiment | ~30s–5min |
| `run.py` | **Sentiment dashboard** · no quantum · historical testing | ~5s |
| `main.py` | **Full backtest** · trains from scratch + tests over 2 years | ~10–20 min |

---

## `infer_pipeline.py` — Single-Bar Inference

The main production script. Loads pre-trained MLP weights, runs the quantum reservoir on **one bar**, applies sentiment confidence gate, outputs a signal. **No retraining.**

### Commands

```bash
# Most recent bar in the dataset (live mode)
python3 infer_pipeline.py

# Market close of a specific day
python3 infer_pipeline.py 2026-04-30

# Specific time — nearest 10-min bar
python3 infer_pipeline.py "2026-04-30 09:30"

# Skip sentiment (no OpenAI API call, uses NORMAL defaults)
python3 infer_pipeline.py 2026-04-30 --no-sentiment

# Different ticker for sentiment (data still AAPL unless --dataset is set)
python3 infer_pipeline.py 2026-04-30 --ticker AAPL

# Override shots (default from readout_weights.json)
python3 infer_pipeline.py 2026-04-30 --shots 500
```

### Output explained
```
timestamp             raw   P(buy)  conf%  gated   size  actual  hit
regime: FED_EVENT   gate:    65%          pos:  0.50×
2026-04-30 14:00:00   BUY   0.714   42.7    BUY   0.50    buy     ✓

raw    → quantum MLP raw signal at default 50% threshold
P(buy) → probability of BUY from the MLP
conf%  → certainty: 0 = coin-flip, 100 = certain
gated  → final signal after sentiment confidence gate applied
          HOLD = signal exists but confidence below regime's gate
size   → position size multiplier from current market regime
actual → ground truth label from dataset (for backtesting)
hit    → ✓ correct prediction / ✗ incorrect
```

### Dataset date range
The included `AAPL_10m_v2.csv` covers **March 5 – May 29, 2026**. Use dates within this range:

```bash
# FOMC meeting day — Fed decision at 14:00
python3 infer_pipeline.py "2026-04-30 14:00"

# Market open — first bar of the day
python3 infer_pipeline.py "2026-04-30 09:30"

# Market close
python3 infer_pipeline.py 2026-04-30

# Latest bar available
python3 infer_pipeline.py
```

---

## `infer_xpyq.py` — Cloud Inference (xpyq + IBM QPU)

Same as `infer_pipeline.py` but the quantum circuits run on the **xpyq cloud platform** — free simulator or real IBM QPU via your BYOK token. Requires `XPYQ_API_KEY` in `.env`.

```bash
# xpyq cloud simulator (free, validates full cloud round-trip)
python3 infer_xpyq.py --index -1

# Specific date, cloud sim, with sentiment
python3 infer_xpyq.py 2026-04-30 --ticker AAPL

# Real IBM QPU — prints cost estimate only (safe to run)
python3 infer_xpyq.py --index -1 --mode qpu --backend ibm_marrakesh

# Real IBM QPU — actually submits (uses your IBM quota)
python3 infer_xpyq.py --index -1 --mode qpu --backend ibm_marrakesh --confirm

# Skip sentiment
python3 infer_xpyq.py --index -1 --no-sentiment
```

> Set `XPYQ_API_KEY` and optionally `IBM_QUANTUM_TOKEN` in your `.env` file.

---

## `run.py` — Sentiment Dashboard

Fast sentiment analysis for any ticker and any date. Tests the AI agent in isolation — no quantum simulation.

### Commands

```bash
# Live market sentiment right now
python3 run.py AAPL
python3 run.py SPY
python3 run.py TSLA

# Historical date
python3 run.py SPY --as-of "2024-11-05"

# Hardcoded scenario (events past GPT knowledge cutoff ~Apr 2025)
python3 run.py SPY --scenario geopolitical
```

### 🔴 Negative / Bearish Events

```bash
# COVID market crash — VIX hit ~82, circuit breakers triggered
# Expected: PANDEMIC · 0.30× · -0.80 bearish
python3 run.py SPY --as-of "2020-03-16"

# Russia invades Ukraine — instant geopolitical shock
# Expected: GEOPOLITICAL · 0.55× · -0.80 bearish
python3 run.py SPY --as-of "2022-02-24"

# Fed 75bps rate hike — largest since 1994, inflation panic
# Expected: FED_EVENT · 0.50× · -0.70 bearish
python3 run.py SPY --as-of "2022-06-15"

# SVB bank collapse — US banking sector fear
# Expected: ELEVATED · 0.70× · -0.70 bearish
python3 run.py SPY --as-of "2023-03-13"

# Yen carry trade unwind — VIX spiked to ~65
# Expected: CRISIS · 0.40× · -0.50 bearish
python3 run.py SPY --as-of "2024-08-05"

# US-Israel-Iran war escalation (hardcoded — past GPT cutoff)
# Expected: GEOPOLITICAL · 0.55× · -0.80 bearish
python3 run.py SPY --scenario geopolitical
```

### 🟡 Neutral / Mixed Events

```bash
# US Presidential Election 2024 — binary outcome, market cautious
# Expected: ELECTION · 0.60× · +0.50 slight bullish
python3 run.py SPY --as-of "2024-11-05"

# US Presidential Election 2020 — high uncertainty
# Expected: ELECTION · 0.60× · 0.00 neutral
python3 run.py SPY --as-of "2020-11-03"

# Fed first rate cut (50bps) — positive but uncertain size
# Expected: FED_EVENT · 0.50× · +0.30 cautiously bullish
python3 run.py SPY --as-of "2024-09-19"
```

### 🟢 Positive / Bullish Events

```bash
# Pfizer vaccine announcement — market surged ~3%, pandemic fear drops
# Expected: NORMAL · 1.00× · +0.60 bullish
python3 run.py SPY --as-of "2020-11-09"

# CPI cooled sharply — soft landing confirmed, big rally
# Expected: NORMAL · 1.00× · +0.50 bullish
python3 run.py SPY --as-of "2023-11-14"

# Day after Trump election win — market surged on policy optimism
# Expected: FED_EVENT · 0.50× · +0.50 bullish
python3 run.py SPY --as-of "2024-11-06"
```

### Example output
```
╔══════════════════════════════════════════════════════╗
║  QuStox  │  SPY     │  2022-02-24 00:00 UTC         ║
╠──────────────────────────────────────────────────────╣
║  VIX          31.4   trend=RISING                   ║
║  Headlines      10   analysed by AI                  ║
╠──────────────────────────────────────────────────────╣
║  REGIME       GEOPOLITICAL                          ║
║  MULTIPLIER   0.55  ×  (HQRC window scale)          ║
╠──────────────────────────────────────────────────────╣
║  SENTIMENT  [█░░░░░░░░░]  -0.80  ↓ BEARISH          ║
╚══════════════════════════════════════════════════════╝
  → regime_multiplier = 0.55
```

---

## `main.py` — Full Training + Backtest Pipeline

Trains the SVR readout from scratch over 2 years of SPY data, then backtests. Sentiment agent scales the reservoir window before training.

```bash
# Full run — downloads + caches SPY data (~10–20 min)
python3 main.py

# Tune reservoir window and shots
python3 main.py --window 30 --shots 1000

# Fast dev run — last 200 bars only (~2 min)
python3 main.py --limit 200

# Skip sentiment (no OpenAI API key needed)
python3 main.py --no-sentiment

# Enable quantum noise model (slow — use with --limit)
python3 main.py --noise --limit 400

# 5-minute bars instead of hourly
python3 main.py --interval 5m

# Force re-download data and re-compute features (ignore cache)
python3 main.py --no-cache
```

**Example output:**
```
========================== Backtest Results ==========================
Model                  Dir.Acc    RMSE   Sharpe  MaxDD  TotRet
HQRC (quantum SVR)      53.2%  0.00312   1.243  -8.1%   14.2%
Classical SVR           51.1%  0.00318   0.821  -9.4%   10.1%
Naive persistence       49.8%     ...
Buy & Hold              ...
Track 2 baseline (Markowitz Sharpe): 0.033 | Target: Sharpe > 1.0
```

---

## Stock Selection

| Script | Ticker support |
|---|---|
| `run.py` | ✅ Any ticker — `python3 run.py TSLA --as-of "2024-08-05"` |
| `infer_pipeline.py` | ⚠️ Sentiment: any ticker via `--ticker`. **Data + weights: AAPL only** (default). |
| `main.py` | ⚠️ SPY only (hardcoded in data fetcher). |

**To run inference on a different stock**, retrain on that stock's dataset first:
```bash
# 1. Retrain on SPY 10-min data
python3 hqrc_nn.py --dataset datasets_v2/SPY_10m_v2.csv

# 2. Infer on SPY
python3 infer_pipeline.py 2026-04-30 --dataset datasets_v2/SPY_10m_v2.csv --ticker SPY
```

Available datasets in `datasets_v2/`: `AAPL`, `CVX`, `GOOG`, `OXY`, `SPY` (10m and 1d intervals).

---

## Project Structure

```
QuStox/
├── src/agents/
│   ├── external_factors.py    # public entry: get_external_factors()
│   ├── sentiment_analyzer.py  # OpenAI gpt-4o-mini regime classifier
│   └── news_fetcher.py        # VIX + yfinance/LLM headline fetcher
│
├── quantum/
│   ├── reservoir.py           # 16-qubit Ising reservoir
│   └── noise.py               # Aer noise model
│
├── models/
│   ├── readout.py             # SVR + NN readout, signals_and_confidence()
│   └── baselines.py           # classical SVR + naive persistence
│
├── data/fetch.py              # SPY yfinance fetcher + cache
├── features/preprocess.py     # returns, tanh encoding, lag windows
├── backtest/engine.py         # Sharpe, drawdown, directional accuracy
│
├── datasets_v2/               # AAPL/CVX/GOOG/OXY/SPY 10m + 1d
├── readout_weights.joblib     # Pre-trained MLP (AAPL 10-min)
├── readout_weights.json       # Weights config (window, encode_scale, seed)
│
├── infer_pipeline.py          # ⭐ Inference: local Aer simulator + sentiment
├── infer_xpyq.py              # ⭐ Inference: xpyq cloud / real IBM QPU + sentiment
├── infer.py                   # Shared utilities (QuantumReservoir, helpers)
├── run.py                     # Sentiment-only dashboard CLI
├── main.py                    # Full training + backtest pipeline
├── reservoir_diagram.png      # Architecture diagram (16-qubit topology)
│
├── .env                       # API keys (gitignored — never committed)
├── .env.example               # Template
└── requirements.txt
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes (sentiment) | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model for regime classification |
| `OPENAI_RECALL_MODEL` | No | `gpt-4o` | Model for historical headline recall |
| `XPYQ_API_KEY` | Yes (`infer_xpyq.py`) | — | xpyq cloud platform API key |
| `IBM_QUANTUM_TOKEN` | Yes (`--mode qpu`) | — | IBM Quantum token for real QPU |

---

## Sentiment Integration Contract

```python
from src.agents.external_factors import get_external_factors

factors = get_external_factors("SPY")   # or any ticker, optional as_of=datetime

print(factors.regime)            # "GEOPOLITICAL"
print(factors.window_multiplier) # 0.55  — scale reservoir window by this
print(factors.sentiment_score)   # -0.80 — (-1 very bearish → +1 very bullish)
print(factors.vix_current)       # 31.4
print(factors.vix_trend)         # "RISING"
print(factors.reasoning)         # one-sentence explanation
print(factors.headlines_used)    # number of headlines analysed
```
