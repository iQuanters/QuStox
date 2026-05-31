"""
Sliding-window close-price → quantum reservoir → NN read-out [40, 6, 1].

Fully runnable training + testing script.

Dataset
-------
From the SPY closing-price series we build a sliding window of length W (=15):

    x[i] = [close_{i}, close_{i+1}, …, close_{i+W-1}]   (W closes)
    y[i] =  close_{i+W}                                 (the next close)

i.e. each row is W consecutive closes and the label is the very next close
(sliding by 1). These arrays are saved to cache/ for inspection.

Pipeline
--------
Each window is processed by the 16-qubit Ising reservoir (which encodes the
window's tanh-normalized returns and evolves a single state through all W
steps) → 40 features → NN read-out [40, 6, 1] → next-close prediction.

Target note
-----------
By default the NN regresses the next *return* and we reconstruct the next close
( pred_close = current · (1 + pred_ret) ). Returns are stationary, so the model
generalizes to the held-out test period even though absolute prices there are
higher than anything seen in training. Use --target level to instead regress the
raw close directly (literal y, but extrapolates poorly out-of-sample).

Usage
-----
  python train_nn.py                       # full dataset, ideal sim
  python train_nn.py --limit 600           # quick run on the last 600 bars
  python train_nn.py --classical           # also train an NN on raw windows (no quantum)
  python train_nn.py --target level        # regress raw close instead of return
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantum.reservoir import QuantumReservoir
from features.preprocess import make_lag_windows
from models.readout import NNReadout, signals_and_confidence
from backtest.engine import evaluate, buy_and_hold

CSV_PATH = Path(__file__).parent / "cache" / "spy_1h_2y.csv"
CACHE_DIR = Path(__file__).parent / "cache"
W = 15                       # sliding-window length
RETURN_NORM = 0.02           # tanh scale for the reservoir's return encoding
BARS_PER_YEAR = 252 * 6      # hourly bars


# --------------------------------------------------------------------- dataset
def build_close_windows(close: np.ndarray, window: int):
    """x[i] = close[i:i+window] (window closes), y[i] = close[i+window] (next)."""
    n = len(close)
    m = n - window
    X = np.empty((m, window), dtype=np.float64)
    y = np.empty(m, dtype=np.float64)
    for i in range(m):
        X[i] = close[i : i + window]
        y[i] = close[i + window]
    return X, y


def encoding_series(close: np.ndarray, return_norm: float) -> np.ndarray:
    """tanh-normalized returns aligned to the close index (enc[k] is the return into close[k])."""
    enc = np.zeros_like(close, dtype=np.float64)
    enc[1:] = np.tanh((close[1:] / close[:-1] - 1.0) / return_norm)
    return enc


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description="QRC + NN read-out [40,6,1] train/test")
    p.add_argument("--csv", default=str(CSV_PATH))
    p.add_argument("--window", type=int, default=W)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--target", choices=["return", "level"], default="return")
    p.add_argument("--classical", action="store_true",
                   help="also train an NN on the raw return windows (no quantum)")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="use only the last N bars")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"\n{'='*64}")
    print(f"  QRC + NN read-out [40, 6, 1]  |  window={args.window}  shots={args.shots}")
    print(f"  target={args.target}  noise={'ON' if args.noise else 'OFF'}")
    print(f"{'='*64}\n")

    # --- load close series ---------------------------------------------------
    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    close = df["Close"].to_numpy(dtype=np.float64)
    if args.limit:
        close = close[-args.limit:]
    print(f"[data] {len(close)} closes  ({df.index[0]} → {df.index[-1]})")

    # --- 1. sliding-window close dataset (literal x / y) ---------------------
    X_close, y_close = build_close_windows(close, args.window)
    np.save(CACHE_DIR / "nn_X_close_windows.npy", X_close)
    np.save(CACHE_DIR / "nn_y_close.npy", y_close)
    print(f"[dataset] X={X_close.shape}  y={y_close.shape}  "
          f"(saved → cache/nn_X_close_windows.npy, nn_y_close.npy)")
    print(f"[dataset] example  x[0][:3]={X_close[0][:3].round(2)} … "
          f"x[0][-1]={X_close[0][-1]:.2f}  →  y[0]={y_close[0]:.2f}")

    # --- 2. reservoir features, aligned to each window ----------------------
    # Reservoir state Q[t] summarizes returns up to close[t]; we predict close[t+1].
    # Window i (closes i..i+W-1) has "current" = close[i+W-1] at series index t=i+W-1.
    enc = encoding_series(close, RETURN_NORM)
    reservoir = QuantumReservoir(
        window=args.window, shots=args.shots, noise=args.noise, seed=args.seed,
    )
    print(f"[qrc] output_dim={reservoir.output_dim}  "
          f"({reservoir.n_qubits} ⟨Z⟩ + {len(reservoir.intra_pairs)} ⟨ZZ⟩)")
    Q_all = reservoir.transform(enc, use_cache=not args.no_cache)

    t_idx = np.arange(args.window - 1, len(close) - 1)   # current index per dataset row
    Q = Q_all[t_idx]                                     # (M, 40) features as-of current
    current = close[t_idx]                               # current close
    next_close = close[t_idx + 1]                        # y (next close)
    next_ret = next_close / current - 1.0                # realized next return

    # --- 3. temporal train/test split ---------------------------------------
    m = len(Q)
    split = int(m * (1 - args.test_frac))
    cur_te, ynext_te, ret_te = current[split:], next_close[split:], next_ret[split:]
    print(f"[split] train={split}  test={m - split}")

    # target: next return (reconstruct close) or raw level
    if args.target == "return":
        y_tr, y_te = next_ret[:split], next_ret[split:]
    else:
        y_tr, y_te = next_close[:split], next_close[split:]

    def to_close_and_ret(pred):
        if args.target == "return":
            return cur_te * (1 + pred), pred
        return pred, pred / cur_te - 1.0

    results = []

    # --- 4. NN read-out on quantum features ---------------------------------
    nn = NNReadout(hidden=6, seed=args.seed).fit(Q[:split], y_tr)
    pred = nn.predict(Q[split:])
    pred_close, pred_ret = to_close_and_ret(pred)
    sig, conf, _ = signals_and_confidence(pred_ret, cur_te)
    r = evaluate(pred_ret, sig, ret_te, "NN readout (quantum)", BARS_PER_YEAR, conf)
    r["rmse_close"] = float(np.sqrt(np.mean((pred_close - ynext_te) ** 2)))
    results.append(r)

    # --- 5. optional classical NN baseline (raw return windows) -------------
    if args.classical:
        Wlag = make_lag_windows(enc, args.window)[t_idx]
        nn_c = NNReadout(hidden=6, seed=args.seed).fit(Wlag[:split], y_tr)
        pred_c = nn_c.predict(Wlag[split:])
        pc_close, pc_ret = to_close_and_ret(pred_c)
        sig_c, conf_c, _ = signals_and_confidence(pc_ret, cur_te)
        rc = evaluate(pc_ret, sig_c, ret_te, "NN readout (classical)", BARS_PER_YEAR, conf_c)
        rc["rmse_close"] = float(np.sqrt(np.mean((pc_close - ynext_te) ** 2)))
        results.append(rc)

    # buy & hold reference
    bh = buy_and_hold(ret_te, BARS_PER_YEAR)
    bh["rmse_close"] = float("nan")
    results.append(bh)

    # --- 6. report -----------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'Model':<26}{'Dir.Acc':>9}{'RMSE(ret)':>11}{'RMSE($)':>10}"
          f"{'Sharpe':>9}{'MaxDD':>9}{'TotRet':>9}{'Trades':>8}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['model']:<26}"
              f"{r['directional_acc']:>8.1%} "
              f"{r['rmse']:>10.5f} "
              f"{r.get('rmse_close', float('nan')):>9.3f} "
              f"{r['sharpe']:>8.3f} "
              f"{r['max_drawdown']:>8.1%} "
              f"{r['total_return']:>8.1%} "
              f"{r['n_trades']:>7}")
    print("=" * 92)

    print("\n  Sample NN (quantum) predictions (last 8 test bars):")
    print(f"  {'current':>10}{'pred_next':>11}{'actual':>10}{'signal':>8}{'conf%':>8}")
    for i in range(max(0, len(cur_te) - 8), len(cur_te)):
        s = "BUY " if sig[i] > 0 else "SELL"
        print(f"  {cur_te[i]:>10.2f}{pred_close[i]:>11.2f}{ynext_te[i]:>10.2f}{s:>8}{conf[i]:>7.1f}")
    print()

    return results


if __name__ == "__main__":
    main()
