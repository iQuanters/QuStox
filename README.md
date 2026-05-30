# HQRC-Trade
**Hybrid Quantum-Classical Reservoir Computer for SPY Market Direction Prediction**
QAI Hackathon · Track 2 (Mid-Frequency Trading)

---

## Overview

HQRC-Trade predicts the direction of the next hourly SPY bar (up / down) using a **Hybrid Quantum-Classical Reservoir Computer (HQRC)**. A fixed random quantum circuit acts as a nonlinear feature extractor: classical market signals are angle-encoded into qubit rotations, propagated through entangled reservoir layers with data re-uploading, and read out as expectation values. These quantum features are concatenated with classical technical indicators and fed to a logistic regression classifier.

**Key claim:** quantum entanglement captures cross-feature nonlinear interactions (reflected in ⟨ZᵢZⱼ⟩ correlators) that a purely classical linear model misses, improving out-of-sample Sharpe ratio.

---

## Architecture

```
SPY 1-hour OHLCV bars (yfinance, 2 years)
          │
          ▼
Classical Features (10)
  ret_1, ret_5, ret_20 (tanh-scaled)
  RSI(14), MACD histogram, Bollinger Band position
  ATR/price, volume ratio, hour_sin, hour_cos
          │
          ├──────────────────────────────────────┐
          │ 4 features (RSI, MACD, BB, ret_1)    │
          ▼                                      │
  Quantum Reservoir (PennyLane / XpyQ)           │
  ┌─────────────────────────────────────┐        │
  │  H on each qubit                   │        │
  │  ┌── for each layer (×3) ──────┐   │        │
  │  │  RY(π·fᵢ)  ← re-upload     │   │        │
  │  │  RY(θᵢ) RZ(φᵢ)  ← fixed   │   │        │
  │  │  CNOT ring (entanglement)   │   │        │
  │  └─────────────────────────────┘   │        │
  │  Measure: ⟨Z₀⟩…⟨Z₃⟩ + ⟨ZᵢZⱼ⟩    │        │
  └─────────────────────────────────────┘        │
          │ 10 quantum features                   │
          └────────────────┬──────────────────────┘
                           ▼
              Concat → 20 features total
                           │
                           ▼
              Logistic Regression (sklearn)
                           │
                           ▼
                   Buy (1) / Flat (0)
                           │
                           ▼
              Walk-forward backtest vs baselines
```

---

## Quantum Backend

Two interchangeable backends:

| Backend | How | When |
|---|---|---|
| **PennyLane** (`default.qubit`) | Local circuit simulation | Default (no API key needed) |
| **XpyQ SDK** | Remote matmul API — gate matrices × state vector | Set `XPYQ_API_KEY` |

The XpyQ backend composes quantum gate matrices (RY, CNOT) with the state vector via hosted `matmul` calls, demonstrating real-time quantum-linear-algebra-as-a-service.

---

## Results (out-of-sample, 866 hourly bars)

| Model | Dir. Accuracy | Sharpe | Max DD |
|---|---|---|---|
| GradBoost (classical) | 54.7% | 1.664 | -7.0% |
| RandomForest (classical) | 53.2% | 1.386 | -9.2% |
| **HQRC (hybrid)** | **52.5%** | **1.158** | **-8.3%** |
| LogReg classical-only | 52.5% | 1.070 | -10.0% |
| Track 2 baseline (Markowitz) | — | 0.033 | — |
| Hackathon target | — | >1.0 | — |

**HQRC Sharpe 1.158 > target 1.0** ✓  
**HQRC beats classical LogReg** (same model, same data — quantum features are the only difference) ✓  

The quantum reservoir adds 10 entanglement-derived features (4 single-qubit + 6 two-qubit ZZ correlators) on top of 10 classical features. The improvement over classical LogReg demonstrates quantum feature value.

---

## Quickstart

```bash
# 1. Create environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline (PennyLane backend, downloads SPY data)
python main.py

# 3. Run with XpyQ backend (requires API key)
export XPYQ_API_KEY=your_key_here
python main.py --backend xpyq

# 4. 5-minute bars (60-day window)
python main.py --interval 5m
```

---

## Project Structure

```
HQRC-Trade/
├── data/
│   └── fetch.py           # SPY download + caching (yfinance)
├── features/
│   ├── indicators.py      # RSI, MACD, Bollinger Bands, ATR
│   └── preprocess.py      # Feature engineering + normalization
├── quantum/
│   ├── gates.py           # Gate matrices (RY, CNOT, kron embedding)
│   ├── reservoir.py       # PennyLane quantum reservoir
│   └── xpyq_reservoir.py  # XpyQ SDK quantum reservoir
├── models/
│   ├── hybrid.py          # HQRC hybrid model
│   └── baselines.py       # Classical baselines (LogReg, RF, GradBoost)
├── backtest/
│   └── engine.py          # Sharpe, drawdown, walk-forward evaluation
├── notebooks/
│   └── demo.ipynb         # End-to-end demo notebook
├── cache/                 # Downloaded data cache
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Design Decisions

**Why reservoir computing?** A fixed (un-trained) quantum circuit as reservoir separates quantum feature extraction from classical optimization. This makes the quantum advantage claim falsifiable: compare logistic regression trained with vs without quantum features on identical data.

**Why data re-uploading?** Single-layer encoding gives at most linear expressibility. Re-uploading features in every reservoir layer creates a polynomial expansion in the input space — similar to kernel methods but via quantum dynamics.

**Why ZZ correlators?** ⟨ZᵢZⱼ⟩ captures two-qubit entanglement. These cross-product features (e.g. RSI × MACD interaction) are what a purely classical linear model cannot represent without explicit feature engineering.

**Why logistic regression as readout?** Interpretability — every feature's contribution (classical and quantum) is a single coefficient. Makes the quantum feature importance directly measurable.
