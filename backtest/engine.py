"""
Walk-forward backtesting engine.

Computes strategy returns, Sharpe ratio, directional accuracy, and
maximum drawdown for any model with a .predict(X) interface.
"""

import numpy as np
import pandas as pd


def strategy_returns(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost: float = 0.0001,   # 1 bps per trade
) -> np.ndarray:
    """
    Convert direction predictions → bar-level strategy returns.

    Signal: 1 = long SPY, 0 = flat (no position, no short).
    Cost charged whenever signal flips (one round-trip = 2× cost).
    """
    signals = predictions.astype(float)
    costs = np.abs(np.diff(signals, prepend=signals[0])) * transaction_cost
    return signals * actual_returns - costs


def sharpe_ratio(returns: np.ndarray, bars_per_year: int = 252 * 6) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free rate)."""
    if returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def directional_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def max_drawdown(returns: np.ndarray) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / (rolling_max + 1e-12)
    return float(drawdown.min())


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    actual_returns: np.ndarray,
    model_name: str = "model",
    bars_per_year: int = 252 * 6,
) -> dict:
    preds = model.predict(X_test)
    ret = strategy_returns(preds, actual_returns)
    return {
        "model": model_name,
        "directional_acc": directional_accuracy(preds, y_test),
        "sharpe": sharpe_ratio(ret, bars_per_year),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_trades": int(np.abs(np.diff(preds.astype(float), prepend=0)).sum()),
    }


def walk_forward_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    actual_returns: np.ndarray,
    train_frac: float = 0.7,
    n_folds: int = 3,
    model_name: str = "model",
    bars_per_year: int = 252 * 6,
    refit: bool = True,
) -> dict:
    """
    Rolling walk-forward evaluation.
    Trains on an expanding window and tests on the next fold.
    """
    n = len(X)
    fold_size = int(n * (1 - train_frac) / n_folds)
    train_end = int(n * train_frac)

    all_preds, all_labels, all_rets = [], [], []

    for fold in range(n_folds):
        test_start = train_end + fold * fold_size
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break

        X_tr, y_tr = X[:test_start], y[:test_start]
        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]
        rets_te = actual_returns[test_start:test_end]

        if refit:
            model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        all_preds.append(preds)
        all_labels.append(y_te)
        all_rets.append(rets_te)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_rets = np.concatenate(all_rets)

    ret = strategy_returns(all_preds, all_rets)
    return {
        "model": model_name,
        "directional_acc": directional_accuracy(all_preds, all_labels),
        "sharpe": sharpe_ratio(ret, bars_per_year),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_trades": int(np.abs(np.diff(all_preds.astype(float), prepend=0)).sum()),
        "n_test_bars": len(all_preds),
    }
