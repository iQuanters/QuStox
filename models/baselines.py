"""
Classical baseline models for honest benchmarking against HQRC.
Each exposes the same .fit(X, y) / .predict(X) interface.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def logistic_regression(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])


def random_forest(n_estimators: int = 200, max_depth: int = 5) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )


def gradient_boosting(n_estimators: int = 100, max_depth: int = 3) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )),
    ])


def get_all_baselines() -> dict:
    return {
        "LogReg (classical)": logistic_regression(),
        "RandomForest": random_forest(),
        "GradBoost": gradient_boosting(),
    }
