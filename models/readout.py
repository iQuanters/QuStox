"""
SVR readout + trading-signal logic.

The reservoir is fixed; only this readout is trained. It is a regression task:
the SVR predicts the next-bar return, from which we reconstruct the implied
next price and derive a buy/sell signal and a 0–100 confidence.

Target scaling
--------------
Bar-to-bar returns are tiny (~1e-3). A raw SVR with epsilon=0.1 would predict
~0 everywhere, so we standardize BOTH the features and the target
(TransformedTargetRegressor) — epsilon then acts in units of target std.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


class SVRReadout:
    """RBF Support-Vector Regression on (standardized) reservoir features."""

    def __init__(self, C: float = 10.0, gamma="scale", epsilon: float = 0.1):
        self.model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon),
            ),
            transformer=StandardScaler(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRReadout":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted next-bar return."""
        return self.model.predict(X)


def signals_and_confidence(pred_ret: np.ndarray, close: np.ndarray):
    """
    Convert predicted next-bar returns into trading signals.

    Returns
    -------
    signal     : +1 (buy/long) if predicted price > current, −1 (sell/short) otherwise
    confidence : clamp(|pred − current| / |current − prev| × 100, 0, 100)
                 i.e. predicted move size relative to the last realized move
    pred_price : reconstructed implied next price = current · (1 + pred_ret)
    """
    pred_ret = np.asarray(pred_ret, dtype=np.float64)
    current = np.asarray(close, dtype=np.float64)
    pred_price = current * (1.0 + pred_ret)

    prev = np.empty_like(current)
    prev[0] = current[0]
    prev[1:] = current[:-1]

    move_pred = np.abs(pred_price - current)
    move_prev = np.abs(current - prev)
    confidence = np.clip(move_pred / (move_prev + 1e-12) * 100.0, 0.0, 100.0)

    signal = np.where(pred_price > current, 1.0, -1.0)
    return signal, confidence, pred_price
