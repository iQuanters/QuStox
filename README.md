# HQRC-Trade
**Quantum Reservoir Computer for SPY Market Prediction**
QAI Hackathon · Track 2 (Mid-Frequency Trading) · fully Qiskit

---

## Overview

HQRC-Trade predicts the next SPY bar from the **closing price alone**, using a
16-qubit **quantum reservoir** with genuine *temporal memory*. The reservoir is
a fixed (never-trained) random-Ising quantum system; only a small classical SVR
read-out is fitted on top. The SVR regresses the next-bar return, from which we
derive a **buy/sell** signal and a **0–100 confidence**.

**The core idea — fading quantum memory.** For each target time `t` we feed the
last `n` returns into the reservoir and let a *single* quantum state evolve
coherently through the whole window, with no reset and no intermediate
measurement. Each new input perturbs the already-evolved, entangled state, so
past inputs influence the final read-out through the quantum dynamics. We
measure only at the end. This is what reservoir computing is supposed to be: the
temporal correlations of the signal are stored in, and processed by, the quantum
state itself.

---

## Architecture

```
SPY closing price (yfinance)
        │  r_t = close_t/close_{t-1} − 1,   enc = tanh(r_t / 0.02)
        ▼
For each time t:  window = [enc_{t-n+1}, …, enc_t]      (n ≈ 10–50, default 30)
        │
        ▼
16-qubit Ising Quantum Reservoir  ──  4 groups × 4 qubits
  ┌──────────────────────────────────────────────────────────┐
  │ |0…0⟩                                                     │
  │  repeat for each step r in the window:                    │
  │    encode  : RX(π · 2^k · r) on qubit k of every group    │  ← freq. multiplex
  │    evolve  : fixed random transverse-field Ising           │  ← coherent memory
  │              Π RX(hx)RY(hy)RZ(hz) · Π RZZ(Jz)RXX(Jx)       │
  │  (no reset, no mid-window measurement → memory persists)   │
  │ measure all 16 qubits · 1000 shots                         │
  └──────────────────────────────────────────────────────────┘
        │  features = ⟨Z_i⟩ (16) + intra-group ⟨Z_iZ_j⟩ (24) = 40
        ▼
   SVR read-out (RBF, quantum features only)  →  predicted next return
        ▼
   signal = sign(pred − current) → BUY / SELL
   confidence = clamp(|pred − current| / |current − prev| × 100, 0, 100)
        ▼
   Long/short backtest vs Classical SVR · Naive persistence · Buy & Hold
```

### Topology — 4 groups of 4
* **Intra-group:** all-to-all Ising coupling (6 couplings per group → 24 total).
* **Inter-group:** a *ring* of bridge couplings between consecutive groups
  (qubits 3–4, 7–8, 11–12, 15–0).

### Encoding — exponential frequency multiplexing
Within every group of 4, qubit `k` encodes the same scalar return at multiplier
`2^k` (×1, ×2, ×4, ×8). The four groups share the input; the fixed random Ising
fields/couplings and the ring break the symmetry so the groups diverge.

### Realistic noise (opt-in)
A gentle superconducting-style noise model (1q ≈ 1e-3, 2q ≈ 1e-2 depolarizing,
≈1e-2 readout, T1/T2 relaxation) validates that the dynamics survive on near-term
hardware. Enable with `--noise`. It is **off by default and slow**: 16 qubits
rule out density-matrix simulation, so Aer runs shot-by-shot statevector
trajectories (1000× the work of the ideal path). Use it with `--limit` for a
robustness check on a subsample, not for a full-history run.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full run (ideal simulator; downloads + caches SPY data and reservoir features)
python main.py

# Tune the temporal memory window / shots
python main.py --window 30 --shots 1000

# Fast dev run on the last 400 bars
python main.py --limit 400

# Noise-robustness check on a subsample (slow — per-shot trajectories)
python main.py --noise --limit 400
```

Reservoir features are cached to `cache/` keyed by config + data, so re-runs of
the read-out are instant. The ideal path samples all 1000 shots from a single
statevector (fast). The `--noise` path is orders of magnitude slower — keep it
to `--limit`ed subsamples.

---

## Project Structure

```
HQRC-Trade/
├── data/
│   └── fetch.py           # SPY download + caching (yfinance)
├── features/
│   └── preprocess.py      # returns, tanh encoding signal, next-return target, lag windows
├── quantum/
│   ├── reservoir.py       # 16-qubit Ising reservoir, sliding-window temporal memory
│   └── noise.py           # realistic Aer noise model
├── models/
│   ├── readout.py         # SVR read-out + buy/sell signal + confidence
│   └── baselines.py       # classical SVR (same input) + naive persistence
├── backtest/
│   └── engine.py          # long/short Sharpe, drawdown, directional accuracy, RMSE
├── cache/                 # data + reservoir-feature cache
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Design Decisions

**Why a fixed reservoir + trained read-out?** It makes the quantum claim
falsifiable: the `Classical SVR` baseline gets the *exact same input window* but
without the quantum dynamics, so any difference is attributable to the reservoir.

**Why a random Ising Hamiltonian?** It is the physically-grounded standard for
quantum reservoirs — rich, tunable fading-memory dynamics that map onto real
spin/superconducting hardware, and pair naturally with the weighted group
topology.

**Why encode returns, not the raw price?** Angle encoding needs a stationary,
bounded signal. Raw SPY price drifts across the sample, which would push test-set
encodings outside the training range and break the read-out. Returns are derived
purely from the close and stay in-distribution.

**Why SVR?** A nonlinear (RBF) read-out can exploit the 40-dimensional quantum
feature geometry while remaining a small, fast, well-regularized model. Targets
are standardized so the tiny bar-to-bar returns are learnable.
