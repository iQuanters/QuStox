"""
HQRC-Trade — Hybrid Quantum-Classical Reservoir Computer for SPY direction.
QAI Hackathon · Track 2 (MFT) · fully Qiskit.

Pipeline
--------
1. Fetch SPY bars (closing price only).
2. Build univariate series: returns → tanh-bounded encoding signal, next-return target.
3. External factors agent: VIX + news headlines → regime classification → window multiplier.
4. Quantum reservoir: 16 qubits (4 groups × 4), random-Ising dynamics, sliding
   window of `n` steps evolved coherently → ⟨Z⟩ + intra-group ⟨ZZ⟩ features.
   The reservoir window is scaled by the regime multiplier (e.g. CRISIS → shorter
   window to prioritise recent data; CALM → longer window for more context).
5. SVR readout (quantum-only) → predicted next return → buy/sell + confidence.
6. Backtest vs classical SVR (same input), naive persistence, and buy & hold.

Usage
-----
  python main.py                       # full run, ideal simulator (fast)
  python main.py --window 30 --shots 1000
  python main.py --noise --limit 400   # noise-robustness check on a subsample (slow)
  python main.py --no-sentiment        # skip external-factors agent (offline / no API key)
"""

import argparse
import numpy as np

from data.fetch import fetch_spy
from features.preprocess import (
    build_series,
    train_test_split_temporal,
    make_lag_windows,
)
from quantum.reservoir import QuantumReservoir
from models.readout import SVRReadout, signals_and_confidence
from models.baselines import ClassicalSVR, naive_persistence
from backtest.engine import evaluate, buy_and_hold

BARS_PER_YEAR = {"1h": 252 * 6, "5m": 252 * 78}

# ---------------------------------------------------------------------------
# Optional external-factors agent (requires OPENAI_API_KEY in .env).
# If the import fails or the flag --no-sentiment is passed, the pipeline runs
# without regime adjustment — equivalent to window_multiplier = 1.0.
# ---------------------------------------------------------------------------
try:
    from src.agents.external_factors import get_external_factors as _get_ext
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False


def print_results(results: list[dict]):
    print("\n" + "=" * 88)
    print(f"{'Model':<24}{'Dir.Acc':>9}{'RMSE':>10}{'Sharpe':>9}"
          f"{'Shp(cw)':>9}{'MaxDD':>9}{'TotRet':>9}{'Trades':>8}")
    print("-" * 88)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(
            f"{r['model']:<24}"
            f"{r['directional_acc']:>8.1%} "
            f"{r['rmse']:>9.5f} "
            f"{r['sharpe']:>8.3f} "
            f"{r.get('sharpe_confw', float('nan')):>8.3f} "
            f"{r['max_drawdown']:>8.1%} "
            f"{r['total_return']:>8.1%} "
            f"{r['n_trades']:>7}"
        )
    print("=" * 88)
    print("  Track 2 baseline (Markowitz Sharpe): 0.033   |   Hackathon target: Sharpe > 1.0\n")


def main():
    p = argparse.ArgumentParser(description="HQRC-Trade (Qiskit)")
    p.add_argument("--interval", default="1h", choices=["1h", "5m"])
    p.add_argument("--period", default="2y")
    p.add_argument("--window", type=int, default=30, help="reservoir lookback (10–50)")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--trotter-steps", type=int, default=1)
    p.add_argument("--evo-time", type=float, default=1.0)
    p.add_argument("--noise", action="store_true",
                   help="enable realistic noise (slow: per-shot trajectories; "
                        "use with --limit for a robustness check)")
    p.add_argument("--limit", type=int, default=0, help="use only the last N bars (dev)")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sentiment", action="store_true",
                   help="skip external-factors agent (no OPENAI_API_KEY needed)")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  HQRC-Trade | Qiskit | 16 qubits (4×4) | Ising reservoir")
    print(f"  window={args.window}  shots={args.shots}  "
          f"noise={'ON' if args.noise else 'OFF'}  "
          f"sentiment={'OFF' if args.no_sentiment else 'ON'}")
    print(f"{'='*60}\n")

    # 1–2. Data → univariate series
    df = fetch_spy(args.interval, args.period, use_cache=not args.no_cache)
    series = build_series(df)
    if args.limit:
        series = series.iloc[-args.limit:]
    print(f"[data] {series.index[0]} → {series.index[-1]}  ({len(series)} bars)")

    train, test = train_test_split_temporal(series, test_frac=0.25)
    n_train = len(train)
    y_train, y_test = train["target"].values, test["target"].values
    close_test = test["close"].values
    bpy = BARS_PER_YEAR[args.interval]
    print(f"[split] train={n_train}  test={len(test)}")

    # 3. External factors — VIX + news sentiment → regime → window multiplier
    #    Skip if --no-sentiment flag is set or agent import failed (no API key).
    regime_label      = "NORMAL"
    window_multiplier = 1.0
    sentiment_score   = 0.0
    if _SENTIMENT_AVAILABLE and not args.no_sentiment:
        try:
            factors = _get_ext("SPY", as_of=None, verbose=True)
            regime_label      = factors.regime
            window_multiplier = factors.window_multiplier
            sentiment_score   = factors.sentiment_score
            print(
                f"[sentiment] regime={regime_label}  "
                f"multiplier={window_multiplier:.2f}  "
                f"VIX={factors.vix_current:.1f}  "
                f"sentiment={sentiment_score:+.2f}\n"
                f"  reasoning: {factors.reasoning}"
            )
        except Exception as exc:
            print(f"[sentiment] WARNING: external factors failed ({exc}); "
                  "using default window_multiplier=1.0")
    else:
        print("[sentiment] skipped (--no-sentiment or agent unavailable)")

    # Clamp the adjusted window to [5, 60] so the reservoir is always valid.
    adjusted_window = int(np.clip(round(args.window * window_multiplier), 5, 60))
    if adjusted_window != args.window:
        print(f"[qrc] window adjusted: {args.window} × {window_multiplier:.2f} "
              f"→ {adjusted_window}  (regime: {regime_label})")

    # 4. Quantum reservoir features (full series, then split)
    enc_all = series["enc"].values
    reservoir = QuantumReservoir(
        window=adjusted_window, shots=args.shots, trotter_steps=args.trotter_steps,
        evo_time=args.evo_time, noise=args.noise, seed=args.seed,
    )
    print(f"[qrc] reservoir output_dim={reservoir.output_dim} "
          f"({reservoir.n_qubits} ⟨Z⟩ + {len(reservoir.intra_pairs)} ⟨ZZ⟩)")
    Q = reservoir.transform(enc_all, use_cache=not args.no_cache,
                            batch_size=args.batch_size)
    Q_train, Q_test = Q[:n_train], Q[n_train:]

    results = []

    # 4. HQRC — SVR on quantum features only
    qrc = SVRReadout().fit(Q_train, y_train)
    pred_q = qrc.predict(Q_test)
    sig_q, conf_q, price_q = signals_and_confidence(pred_q, close_test)
    results.append(evaluate(pred_q, sig_q, y_test, "HQRC (quantum SVR)", bpy, conf_q))

    # 5. Classical SVR baseline — same input (raw lag window), no quantum
    Wlag = make_lag_windows(enc_all, adjusted_window)
    csvr = ClassicalSVR().fit(Wlag[:n_train], y_train)
    pred_c = csvr.predict(Wlag[n_train:])
    sig_c, conf_c, _ = signals_and_confidence(pred_c, close_test)
    results.append(evaluate(pred_c, sig_c, y_test, "Classical SVR", bpy, conf_c))

    # naive persistence + buy & hold references
    pred_n = naive_persistence(Wlag[n_train:])
    sig_n, _, _ = signals_and_confidence(pred_n, close_test)
    results.append(evaluate(pred_n, sig_n, y_test, "Naive persistence", bpy))
    results.append(buy_and_hold(y_test, bpy))

    print_results(results)
    print(f"  [sentiment] regime={regime_label}  "
          f"window_multiplier={window_multiplier:.2f}  "
          f"adjusted_window={adjusted_window}  "
          f"sentiment_score={sentiment_score:+.2f}\n")

    # sample of HQRC predictions with buy/sell + confidence
    print("  Sample HQRC signals (last 8 test bars):")
    print(f"  {'current':>10}{'pred_next':>11}{'signal':>8}{'conf%':>8}")
    for i in range(max(0, len(close_test) - 8), len(close_test)):
        sig = "BUY " if sig_q[i] > 0 else "SELL"
        print(f"  {close_test[i]:>10.2f}{price_q[i]:>11.2f}{sig:>8}{conf_q[i]:>7.1f}")
    print()

    return results


if __name__ == "__main__":
    main()
