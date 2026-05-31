#!/usr/bin/env python3
"""
HQRC-NN — single-file, self-sufficient train/test pipeline.
================================================================================
Quantum Reservoir Computer (16-qubit Ising, Qiskit) + 2-layer NN read-out [40,6,1]
for SPY next-close prediction from a sliding window of closing prices.

Everything needed to TRAIN and TEST lives in this one file:
  * data loading (SPY closes from CSV, or yfinance download if absent)
  * sliding-window dataset:  x = W consecutive closes,  y = next close
  * 16-qubit random-Ising quantum reservoir with genuine temporal memory
  * NN read-out [40, 6, 1]  (sklearn MLP) + classical no-quantum baseline
  * long/short backtest metrics + buy/sell signal + 0-100 confidence
  * live progress messages (percent, elapsed, ETA) for the slow reservoir pass

Dependencies: numpy, pandas, scikit-learn, qiskit, qiskit-aer  (yfinance optional).

Usage
-----
  python hqrc_nn.py                      # full dataset, ideal simulator
  python hqrc_nn.py --limit 600          # quick run on the last 600 bars
  python hqrc_nn.py --classical          # add a no-quantum NN baseline
  python hqrc_nn.py --noise --limit 400  # realistic-noise robustness check (slow)
  python hqrc_nn.py --target level        # regress raw close instead of return
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# ----------------------------------------------------------------------- config
HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CSV_PATH = CACHE_DIR / "spy_1h_2y.csv"

W_DEFAULT = 15            # sliding-window length
RETURN_NORM = 0.02       # tanh scale for the reservoir's return encoding
BARS_PER_YEAR = 252 * 6  # hourly-bar Sharpe annualization

# native gates the reservoir is simulated with (noise attaches to these)
BASIS_GATES = ["rx", "ry", "rz", "rxx", "rzz"]


# ============================================================ realistic noise
def build_noise_model(
    p1: float = 1e-3, p2: float = 1e-2, p_readout: float = 1e-2,
    t1_us: float = 120.0, t2_us: float = 90.0,
    t_1q_ns: float = 50.0, t_2q_ns: float = 300.0,
) -> NoiseModel:
    """Gentle, realistic superconducting-style noise on the native gates."""
    nm = NoiseModel(basis_gates=BASIS_GATES)
    t1, t2 = t1_us * 1e3, t2_us * 1e3  # ns

    relax_1q = thermal_relaxation_error(t1, t2, t_1q_ns)
    err_1q = depolarizing_error(p1, 1).compose(relax_1q)
    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry"])

    relax_2q = thermal_relaxation_error(t1, t2, t_2q_ns).tensor(
        thermal_relaxation_error(t1, t2, t_2q_ns)
    )
    err_2q = depolarizing_error(p2, 2).compose(relax_2q)
    nm.add_all_qubit_quantum_error(err_2q, ["rxx", "rzz"])

    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    )
    return nm


# ============================================================ quantum reservoir
class QuantumReservoir:
    """
    16 qubits = 4 groups of 4.
      intra-group : all-to-all Ising coupling (6 per group → 24)
      inter-group : ring bridges 3-4, 7-8, 11-12, 15-0 (4)        = 28 edges
    Sliding window: encode each step's return (RX, 2^k freq. multiplex per group),
    evolve a single state through the whole window via a fixed random Ising
    Hamiltonian (no reset, no mid-window measurement → fading quantum memory),
    then measure. Features per window: ⟨Z_i⟩ (16) + intra-group ⟨Z_iZ_j⟩ (24) = 40.
    """

    def __init__(self, n_groups=4, group_size=4, window=W_DEFAULT, shots=1000,
                 trotter_steps=1, evo_time=1.0, encode_scale=1.0,
                 noise=False, seed=42):
        self.n_groups, self.group_size = n_groups, group_size
        self.n_qubits = n_groups * group_size
        self.window, self.shots = window, shots
        self.trotter_steps, self.evo_time = trotter_steps, evo_time
        self.encode_scale, self.noise, self.seed = encode_scale, noise, seed

        self.groups = [list(range(g * group_size, (g + 1) * group_size))
                       for g in range(n_groups)]
        self.intra_pairs = []
        for grp in self.groups:
            self.intra_pairs.extend(combinations(grp, 2))
        self.inter_pairs = [(self.groups[g][-1], self.groups[(g + 1) % n_groups][0])
                            for g in range(n_groups)]
        self.edges = self.intra_pairs + self.inter_pairs

        rng = np.random.default_rng(seed)
        self.hx = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hy = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hz = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.Jx = rng.uniform(-1.0, 1.0, len(self.edges))
        self.Jz = rng.uniform(-1.0, 1.0, len(self.edges))

        self.sim = AerSimulator(noise_model=build_noise_model() if noise else None)

    @property
    def output_dim(self) -> int:
        return self.n_qubits + len(self.intra_pairs)

    def _append_ising(self, qc: QuantumCircuit) -> None:
        t = self.evo_time
        for _ in range(self.trotter_steps):
            for i in range(self.n_qubits):
                qc.rx(t * self.hx[i], i)
                qc.ry(t * self.hy[i], i)
                qc.rz(t * self.hz[i], i)
            for (i, j), jx, jz in zip(self.edges, self.Jx, self.Jz):
                qc.rzz(t * jz, i, j)
                qc.rxx(t * jx, i, j)

    def _encode(self, qc: QuantumCircuit, r: float) -> None:
        a = self.encode_scale * np.pi * float(r)
        for grp in self.groups:
            for k, q in enumerate(grp):
                qc.rx((2 ** k) * a, q)

    def build_circuit(self, window_returns: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for r in window_returns:
            self._encode(qc, r)
            self._append_ising(qc)
        qc.measure_all()
        return qc

    def _counts_to_features(self, counts: dict[str, int]) -> np.ndarray:
        keys = list(counts.keys())
        weights = np.array([counts[k] for k in keys], dtype=np.float64)
        probs = weights / weights.sum()
        # Qiskit strings are little-endian: reverse so index i ↔ qubit i.
        bits = np.array(
            [[1 - 2 * int(c) for c in k.replace(" ", "")[::-1]] for k in keys],
            dtype=np.float64,
        )
        z = probs @ bits
        zz = np.array([probs @ (bits[:, i] * bits[:, j]) for i, j in self.intra_pairs])
        return np.concatenate([z, zz])

    def _build_windows(self, returns: np.ndarray) -> np.ndarray:
        n = len(returns)
        W = np.zeros((n, self.window), dtype=np.float64)
        for t in range(n):
            lo = max(0, t - self.window + 1)
            seg = returns[lo:t + 1]
            W[t, self.window - len(seg):] = seg
        return W

    def _cache_key(self, returns: np.ndarray) -> Path:
        cfg = (f"{self.n_qubits}_{self.window}_{self.shots}_{self.trotter_steps}_"
               f"{self.evo_time}_{self.encode_scale}_{int(self.noise)}_{self.seed}")
        h = hashlib.sha1(cfg.encode() + returns.tobytes()).hexdigest()[:16]
        return CACHE_DIR / f"qrc_features_{h}.npy"

    def transform(self, returns: np.ndarray, use_cache=True,
                  batch_size=200, verbose=True) -> np.ndarray:
        """Return-series → (len, output_dim) feature matrix, with live progress."""
        returns = np.asarray(returns, dtype=np.float64)
        cache_path = self._cache_key(returns)
        if use_cache and cache_path.exists():
            if verbose:
                print(f"[qrc] loaded cached features ← {cache_path.name}")
            return np.load(cache_path)

        windows = self._build_windows(returns)
        n = len(windows)
        feats = np.zeros((n, self.output_dim), dtype=np.float64)
        if verbose:
            print(f"[qrc] simulating {n} windows  "
                  f"({self.n_qubits} qubits, window={self.window}, "
                  f"shots={self.shots}, noise={'ON' if self.noise else 'OFF'})")
        t0 = time.time()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            circuits = [self.build_circuit(windows[t]) for t in range(start, end)]
            tqc = transpile(circuits, self.sim, optimization_level=1)
            result = self.sim.run(tqc, shots=self.shots).result()
            for i in range(end - start):
                feats[start + i] = self._counts_to_features(result.get_counts(i))

            if verbose:
                elapsed = time.time() - t0
                rate = end / elapsed
                eta = (n - end) / rate if rate > 0 else 0.0
                bar = _progress_bar(end / n)
                print(f"\r[qrc] {bar} {end:>5}/{n} ({end/n:5.1%}) | "
                      f"{elapsed:6.1f}s elapsed | ETA {eta:6.1f}s | "
                      f"{rate:5.1f} win/s", end="", flush=True)
        if verbose:
            print()  # newline after the progress line

        if use_cache:
            np.save(cache_path, feats)
            if verbose:
                print(f"[qrc] cached features → {cache_path.name}")
        return feats


def _progress_bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ============================================================ NN read-out
class NNReadout:
    """2-layer MLP read-out [n_features, 6, 1] with standardized features+target."""

    def __init__(self, hidden=6, alpha=1e-3, max_iter=3000, seed=42):
        self.model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu",
                             solver="adam", alpha=alpha, max_iter=max_iter,
                             early_stopping=True, n_iter_no_change=40,
                             random_state=seed),
            ),
            transformer=StandardScaler(),
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def signals_and_confidence(pred_ret: np.ndarray, close: np.ndarray):
    """signal = sign(pred-current); confidence = clamp(|pred-cur|/|cur-prev|*100,0,100)."""
    pred_ret = np.asarray(pred_ret, dtype=np.float64)
    current = np.asarray(close, dtype=np.float64)
    pred_price = current * (1.0 + pred_ret)
    prev = np.empty_like(current)
    prev[0], prev[1:] = current[0], current[:-1]
    conf = np.clip(np.abs(pred_price - current) / (np.abs(current - prev) + 1e-12)
                   * 100.0, 0.0, 100.0)
    signal = np.where(pred_price > current, 1.0, -1.0)
    return signal, conf, pred_price


# ============================================================ backtest metrics
def strategy_returns(signals, actual_returns, transaction_cost=0.0001):
    signals = np.asarray(signals, dtype=np.float64)
    costs = np.abs(np.diff(signals, prepend=signals[0])) * transaction_cost
    return signals * actual_returns - costs


def sharpe_ratio(returns, bars_per_year=BARS_PER_YEAR):
    return 0.0 if returns.std() < 1e-12 else \
        float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cumulative)
    return float(((cumulative - peak) / (peak + 1e-12)).min())


def evaluate(pred_ret, signals, actual_ret, pred_close, next_close, name):
    ret = strategy_returns(signals, actual_ret)
    return {
        "model": name,
        "directional_acc": float((np.sign(pred_ret) == np.sign(actual_ret)).mean()),
        "rmse": float(np.sqrt(np.mean((pred_ret - actual_ret) ** 2))),
        "rmse_close": float(np.sqrt(np.mean((pred_close - next_close) ** 2))),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_trades": int(np.abs(np.diff(signals, prepend=signals[0])).sum() / 2),
    }


def buy_and_hold(actual_ret):
    return {
        "model": "Buy & Hold",
        "directional_acc": float((actual_ret > 0).mean()),
        "rmse": float("nan"), "rmse_close": float("nan"),
        "sharpe": sharpe_ratio(actual_ret),
        "max_drawdown": max_drawdown(actual_ret),
        "total_return": float((1 + actual_ret).prod() - 1),
        "n_trades": 0,
    }


# ============================================================ data + dataset
def load_closes(csv_path: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if not csv_path.exists():
        print(f"[data] {csv_path.name} not found — downloading SPY 1h/2y via yfinance …")
        import yfinance as yf
        raw = yf.download("SPY", period="2y", interval="1h",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        raw.to_csv(csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df["Close"].to_numpy(dtype=np.float64), df.index


def build_close_windows(close: np.ndarray, window: int):
    """x[i] = close[i:i+window] (window closes); y[i] = close[i+window] (next)."""
    m = len(close) - window
    X = np.stack([close[i:i + window] for i in range(m)])
    y = np.array([close[i + window] for i in range(m)])
    return X, y


def encoding_series(close: np.ndarray, return_norm: float) -> np.ndarray:
    """tanh-normalized returns aligned to the close index."""
    enc = np.zeros_like(close, dtype=np.float64)
    enc[1:] = np.tanh((close[1:] / close[:-1] - 1.0) / return_norm)
    return enc


def make_lag_windows(enc: np.ndarray, window: int) -> np.ndarray:
    n = len(enc)
    Wm = np.zeros((n, window), dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = enc[lo:t + 1]
        Wm[t, window - len(seg):] = seg
    return Wm


# ============================================================ main
def main():
    p = argparse.ArgumentParser(description="HQRC + NN read-out [40,6,1] (single file)")
    p.add_argument("--csv", default=str(CSV_PATH))
    p.add_argument("--window", type=int, default=W_DEFAULT)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--target", choices=["return", "level"], default="return")
    p.add_argument("--classical", action="store_true",
                   help="also train an NN on the raw return windows (no quantum)")
    p.add_argument("--noise", action="store_true", help="realistic noise (slow)")
    p.add_argument("--limit", type=int, default=0, help="use only the last N bars")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"\n{'='*68}")
    print(f"  HQRC + NN read-out [40, 6, 1]  |  16 qubits (4×4) Ising reservoir")
    print(f"  window={args.window}  shots={args.shots}  target={args.target}  "
          f"noise={'ON' if args.noise else 'OFF'}")
    print(f"{'='*68}\n")

    # 1. data
    close, index = load_closes(Path(args.csv))
    if args.limit:
        close, index = close[-args.limit:], index[-args.limit:]
    print(f"[data] {len(close)} closes  ({index[0]} → {index[-1]})")

    # 2. sliding-window close dataset (literal x / y)
    X_close, y_close = build_close_windows(close, args.window)
    np.save(CACHE_DIR / "nn_X_close_windows.npy", X_close)
    np.save(CACHE_DIR / "nn_y_close.npy", y_close)
    print(f"[dataset] x={X_close.shape}  y={y_close.shape}  → saved to cache/")
    print(f"[dataset] example: x[0] ends at {X_close[0][-1]:.2f}  →  y[0]={y_close[0]:.2f}")

    # 3. reservoir features aligned to each window
    enc = encoding_series(close, RETURN_NORM)
    res = QuantumReservoir(window=args.window, shots=args.shots,
                           noise=args.noise, seed=args.seed)
    Q_all = res.transform(enc, use_cache=not args.no_cache, batch_size=args.batch_size)

    t_idx = np.arange(args.window - 1, len(close) - 1)   # "current" index per row
    Q = Q_all[t_idx]
    current = close[t_idx]
    next_close = close[t_idx + 1]
    next_ret = next_close / current - 1.0

    # 4. temporal split
    m = len(Q)
    split = int(m * (1 - args.test_frac))
    cur_te, ynext_te, ret_te = current[split:], next_close[split:], next_ret[split:]
    print(f"[split] train={split}  test={m - split}")

    if args.target == "return":
        y_tr = next_ret[:split]
    else:
        y_tr = next_close[:split]

    def to_close_and_ret(pred):
        if args.target == "return":
            return cur_te * (1 + pred), pred
        return pred, pred / cur_te - 1.0

    results = []

    # 5. NN read-out on quantum features
    print("[train] fitting NN read-out [40,6,1] on quantum features …")
    nn = NNReadout(hidden=6, seed=args.seed).fit(Q[:split], y_tr)
    print("[test ] predicting on held-out set …")
    pred = nn.predict(Q[split:])
    pred_close, pred_ret = to_close_and_ret(pred)
    sig, conf, _ = signals_and_confidence(pred_ret, cur_te)
    results.append(evaluate(pred_ret, sig, ret_te, pred_close, ynext_te,
                            "NN readout (quantum)"))

    # 6. optional classical NN baseline (raw return windows, no quantum)
    if args.classical:
        print("[train] fitting NN baseline on raw return windows (no quantum) …")
        Wlag = make_lag_windows(enc, args.window)[t_idx]
        nn_c = NNReadout(hidden=6, seed=args.seed).fit(Wlag[:split], y_tr)
        pred_c = nn_c.predict(Wlag[split:])
        pc_close, pc_ret = to_close_and_ret(pred_c)
        sig_c, _, _ = signals_and_confidence(pc_ret, cur_te)
        results.append(evaluate(pc_ret, sig_c, ret_te, pc_close, ynext_te,
                                "NN readout (classical)"))

    results.append(buy_and_hold(ret_te))

    # 7. report
    print("\n" + "=" * 92)
    print(f"{'Model':<26}{'Dir.Acc':>9}{'RMSE(ret)':>11}{'RMSE($)':>10}"
          f"{'Sharpe':>9}{'MaxDD':>9}{'TotRet':>9}{'Trades':>8}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['model']:<26}{r['directional_acc']:>8.1%} "
              f"{r['rmse']:>10.5f} {r['rmse_close']:>9.3f} {r['sharpe']:>8.3f} "
              f"{r['max_drawdown']:>8.1%} {r['total_return']:>8.1%} {r['n_trades']:>7}")
    print("=" * 92)

    print("\n  Sample NN (quantum) predictions (last 8 test bars):")
    print(f"  {'current':>10}{'pred_next':>11}{'actual':>10}{'signal':>8}{'conf%':>8}")
    for i in range(max(0, len(cur_te) - 8), len(cur_te)):
        s = "BUY " if sig[i] > 0 else "SELL"
        print(f"  {cur_te[i]:>10.2f}{pred_close[i]:>11.2f}{ynext_te[i]:>10.2f}"
              f"{s:>8}{conf[i]:>7.1f}")
    print()

    return results


if __name__ == "__main__":
    main()
