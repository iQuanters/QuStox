"""
HQRC hybrid model: quantum reservoir features + classical features → logistic regression.

The key experiment:
  classical_only  — LogReg on classical features
  quantum_only    — LogReg on quantum reservoir features
  hybrid (HQRC)   — LogReg on [classical | quantum] concatenated
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HQRCModel:
    """
    Hybrid Quantum-Classical Reservoir Computer.

    Parameters
    ----------
    reservoir     : QuantumReservoir or XpyQReservoir instance.
    quantum_cols  : Indices of features to feed into the quantum reservoir.
    C             : Logistic regression regularization (inverse strength).
    """

    def __init__(self, reservoir, quantum_feature_indices: list[int], C: float = 1.0):
        self.reservoir = reservoir
        self.q_idx = quantum_feature_indices
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        self._fitted = False

    def _quantum_features(self, X: np.ndarray) -> np.ndarray:
        X_q = X[:, self.q_idx]
        return self.reservoir.transform(X_q)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HQRCModel":
        print(f"[HQRC] extracting quantum features for {len(X)} training samples …")
        Q = self._quantum_features(X)
        combined = np.hstack([X, Q])
        combined_scaled = self.scaler.fit_transform(combined)
        self.clf.fit(combined_scaled, y)
        self._fitted = True
        print(f"[HQRC] trained on {combined.shape[1]} features "
              f"({X.shape[1]} classical + {Q.shape[1]} quantum)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Q = self._quantum_features(X)
        combined = np.hstack([X, Q])
        return self.clf.predict(self.scaler.transform(combined))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Q = self._quantum_features(X)
        combined = np.hstack([X, Q])
        return self.clf.predict_proba(self.scaler.transform(combined))
