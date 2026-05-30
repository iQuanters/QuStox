"""
Backtesting for a long/short, return-predicting strategy.

Signal ∈ {+1 long, −1 short}. Strategy return at bar t is
    signal_t · realized_return_{t+1} − cost·|Δsignal|
A confidence-weighted variant scales position size by confidence/100.
"""

import numpy as np


def strategy_returns(
    signals: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost: float = 0.0001,   # 1 bp per unit position change
) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float64)
    costs = np.abs(np.diff(signals, prepend=signals[0])) * transaction_cost
    return signals * actual_returns - costs


def sharpe_ratio(returns: np.ndarray, bars_per_year: int) -> float:
    if returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / (rolling_max + 1e-12)
    return float(drawdown.min())


def directional_accuracy(pred_ret: np.ndarray, actual_ret: np.ndarray) -> float:
    return float((np.sign(pred_ret) == np.sign(actual_ret)).mean())


def evaluate(
    pred_ret: np.ndarray,
    signals: np.ndarray,
    actual_ret: np.ndarray,
    model_name: str,
    bars_per_year: int,
    confidence: np.ndarray | None = None,
) -> dict:
    """Compute regression + trading metrics for one model."""
    ret = strategy_returns(signals, actual_ret)
    out = {
        "model": model_name,
        "directional_acc": directional_accuracy(pred_ret, actual_ret),
        "rmse": float(np.sqrt(np.mean((pred_ret - actual_ret) ** 2))),
        "sharpe": sharpe_ratio(ret, bars_per_year),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_trades": int(np.abs(np.diff(signals, prepend=signals[0])).sum() / 2),
    }
    if confidence is not None:
        w = signals * (confidence / 100.0)
        wret = strategy_returns(w, actual_ret)
        out["sharpe_confw"] = sharpe_ratio(wret, bars_per_year)
    return out


def buy_and_hold(actual_ret: np.ndarray, bars_per_year: int) -> dict:
    """Always-long reference."""
    ret = actual_ret
    return {
        "model": "Buy & Hold",
        "directional_acc": float((actual_ret > 0).mean()),
        "rmse": float("nan"),
        "sharpe": sharpe_ratio(ret, bars_per_year),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_trades": 0,
    }
