"""
Read-out models + trading-signal logic.

The reservoir is fixed; only the read-out is trained. Two read-outs are
available, both regressing the next-bar return:

  * SVRReadout : RBF Support-Vector Regression.
  * NNReadout  : a 2-layer neural network [n_features, 6, 1] (sklearn MLP).

Target scaling
--------------
Bar-to-bar returns are tiny (~1e-3). A raw model with default tolerances would
predict ~0 everywhere, so we standardize BOTH the features and the target
(TransformedTargetRegressor) for every read-out.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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


class NNReadout:
    """
    2-layer neural-network read-out: [n_features, 6, 1].

    Layer 1: n_features → 6  (ReLU)
    Layer 2: 6 → 1           (linear)

    With the reservoir's 40 features this is exactly the requested [40, 6, 1].
    Features and target are standardized (returns are ~1e-3).
    """

    def __init__(
        self,
        hidden: int = 6,
        alpha: float = 1e-3,
        max_iter: int = 3000,
        seed: int = 42,
    ):
        self.model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(hidden,),
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=40,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NNReadout":
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
