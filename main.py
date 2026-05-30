"""
HQRC-Trade: Hybrid Quantum-Classical Reservoir Computer for SPY Direction Prediction
QAI Hackathon — Track 2 (MFT)

Pipeline
--------
1. Fetch SPY hourly bars (2 years)
2. Build classical technical features
3. Quantum reservoir transform (PennyLane or XpyQ)
4. Train hybrid model (quantum + classical → logistic regression)
5. Walk-forward backtest vs classical baselines
6. Print results table

Usage
-----
  python main.py                        # PennyLane backend (default)
  XPYQ_API_KEY=xxx python main.py       # XpyQ backend
  python main.py --interval 1h          # hourly bars (default)
  python main.py --interval 5m          # 5-min bars (max 60 days)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from data.fetch import fetch_spy
from features.preprocess import (
    build_features,
    train_test_split_temporal,
    get_Xy,
    ALL_FEATURE_COLS,
    QUANTUM_FEATURE_COLS,
)
from models.baselines import get_all_baselines
from models.hybrid import HQRCModel
from backtest.engine import evaluate, walk_forward_evaluate


def get_reservoir(backend: str, n_qubits: int = 4, n_layers: int = 3):
    if backend == "xpyq":
        from quantum.xpyq_reservoir import XpyQReservoir
        return XpyQReservoir(n_qubits=n_qubits, n_layers=n_layers)
    else:
        from quantum.reservoir import QuantumReservoir
        return QuantumReservoir(n_qubits=n_qubits, n_layers=n_layers)


def print_results(results: list[dict]):
    print("\n" + "=" * 72)
    print(f"{'Model':<28} {'Dir.Acc':>8} {'Sharpe':>8} {'MaxDD':>8} {'TotRet':>8}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(
            f"{r['model']:<28} "
            f"{r['directional_acc']:>7.1%} "
            f"{r['sharpe']:>8.3f} "
            f"{r['max_drawdown']:>7.1%} "
            f"{r['total_return']:>7.1%}"
        )
    print("=" * 72)
    print(f"  Track 2 baseline (Markowitz Sharpe): 0.033")
    print(f"  Hackathon target: Sharpe > 1.0")
    print()


def main():
    parser = argparse.ArgumentParser(description="HQRC-Trade")
    parser.add_argument("--interval", default="1h", choices=["1h", "5m"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--backend", default=None, choices=["pennylane", "xpyq"])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Auto-detect backend
    if args.backend is None:
        args.backend = "xpyq" if os.environ.get("XPYQ_API_KEY") else "pennylane"

    print(f"\n{'='*60}")
    print(f"  HQRC-Trade | Track 2 MFT | backend={args.backend.upper()}")
    print(f"  {args.n_qubits} qubits | {args.n_layers} reservoir layers")
    print(f"{'='*60}\n")

    # 1. Data
    df = fetch_spy(interval=args.interval, period=args.period, use_cache=not args.no_cache)
    print(f"[data] {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)")

    # 2. Features
    feat = build_features(df)
    print(f"[features] {len(feat)} rows after indicator warm-up")

    # 3. Actual next-bar returns (computed on full series to avoid gaps)
    actual_returns = (
        df["Close"].squeeze()
        .pct_change(1)
        .shift(-1)
        .reindex(feat.index)
        .fillna(0)
        .values
    )

    # 4. Train / test split (time-based, no shuffling)
    feat_train, feat_test = train_test_split_temporal(feat, test_frac=0.25)
    print(f"[split] train={len(feat_train)} bars | test={len(feat_test)} bars")

    X_train, y_train = get_Xy(feat_train, ALL_FEATURE_COLS)
    X_test, y_test = get_Xy(feat_test, ALL_FEATURE_COLS)

    actual_ret_train = actual_returns[: len(feat_train)]
    actual_ret_test = actual_returns[len(feat_train): len(feat_train) + len(feat_test)]

    # Indices of quantum input features within ALL_FEATURE_COLS
    q_idx = [ALL_FEATURE_COLS.index(c) for c in QUANTUM_FEATURE_COLS]

    # bars per year for Sharpe annualisation
    bars_per_year = {"1h": 252 * 6, "5m": 252 * 78}[args.interval]

    results = []

    # 5a. Classical baselines
    print("\n[baselines] training classical models …")
    for name, model in get_all_baselines().items():
        model.fit(X_train, y_train)
        r = evaluate(model, X_test, y_test, actual_ret_test, name, bars_per_year)
        results.append(r)
        print(f"  {name:<28}  acc={r['directional_acc']:.1%}  sharpe={r['sharpe']:.3f}")

    # 5b. HQRC hybrid model
    print(f"\n[HQRC] initialising {args.backend.upper()} reservoir …")
    reservoir = get_reservoir(args.backend, args.n_qubits, args.n_layers)
    hqrc = HQRCModel(reservoir, q_idx)
    hqrc.fit(X_train, y_train)
    r = evaluate(hqrc, X_test, y_test, actual_ret_test, "HQRC (hybrid)", bars_per_year)
    results.append(r)
    print(f"  {'HQRC (hybrid)':<28}  acc={r['directional_acc']:.1%}  sharpe={r['sharpe']:.3f}")

    # 6. Results table
    print_results(results)

    return results


if __name__ == "__main__":
    main()
