"""Margin model: ridge regression on pre-game rating differences.

Predicts home margin (home score - away score); win probability is
P(margin > 0) under a normal error with sigma estimated from training residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


class MarginModel:
    def __init__(self, alpha: float = 10.0, features=FEATURES):
        self.features = list(features)
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.sigma = 13.5

    def fit(self, df: pd.DataFrame) -> "MarginModel":
        X, y = df[self.features].to_numpy(float), df.result.to_numpy(float)
        self.pipe.fit(X, y)
        self.sigma = float(np.std(y - self.pipe.predict(X)))
        return self

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(df[self.features].to_numpy(float))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.predict_margin(df) / self.sigma)

    def coefficients(self) -> pd.Series:
        sc, rg = self.pipe.named_steps.values()
        return pd.Series(rg.coef_ / sc.scale_, index=self.features)


def metrics(df: pd.DataFrame, p: np.ndarray, margin: np.ndarray | None = None) -> dict:
    y = (df.result > 0).astype(float).to_numpy()
    keep = df.result.to_numpy() != 0          # ties excluded from accuracy
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {
        "games": int(len(df)),
        "accuracy": float(((p > 0.5) == (y == 1))[keep].mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }
    if margin is not None:
        out["margin_mae"] = float(np.mean(np.abs(margin - df.result.to_numpy())))
    return out


def vegas_proba(df: pd.DataFrame, sigma: float = 13.5) -> np.ndarray:
    """Win probability implied by the closing spread (nflverse: positive = home favoured)."""
    return norm.cdf(df.spread_line.to_numpy(float) / sigma)
