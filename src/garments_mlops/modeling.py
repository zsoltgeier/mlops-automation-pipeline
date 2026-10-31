"""Deterministic regression pipelines and comparable validation metrics."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from garments_mlops.preprocessing import build_preprocessor

ModelKind = Literal["baseline", "random_forest"]


@dataclass(frozen=True)
class ModelSpec:
    """Small, explicit model search space with deterministic fitting."""

    kind: ModelKind = "random_forest"
    random_seed: int = 42
    n_estimators: int = 100
    max_depth: int = 8
    min_samples_leaf: int = 3

    def __post_init__(self) -> None:
        if self.kind not in ("baseline", "random_forest"):
            raise ValueError("Model kind must be baseline or random_forest")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer between 0 and 2**32 - 1")
        for value in (self.n_estimators, self.max_depth, self.min_samples_leaf):
            if type(value) is not int or value <= 0:
                raise ValueError("Forest parameters must be positive integers")


def build_model(spec: ModelSpec) -> Pipeline:
    """Bundle raw-input preprocessing and the regressor into one artifact."""
    estimator = DummyRegressor(strategy="mean") if spec.kind == "baseline" else RandomForestRegressor(
        n_estimators=spec.n_estimators, max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf,
        random_state=spec.random_seed, n_jobs=1,
    )
    return Pipeline([("preprocessing", build_preprocessor()), ("regressor", estimator)])


def regression_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Reject invalid predictions and compute finite MAE, RMSE and R-squared."""
    truth = np.asarray(actual, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if truth.ndim != 1 or prediction.shape != truth.shape or len(truth) < 2:
        raise ValueError("Evaluation requires matching one-dimensional arrays with at least two rows")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Evaluation values must be finite")
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(root_mean_squared_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
    }


@dataclass(frozen=True)
class PromotionDecision:
    """Explain a quality gate without coupling it to Registry operations."""

    promoted: bool
    reason: str
    candidate_mae: float
    champion_mae: float | None
    minimum_improvement: float


def decide_promotion(
    candidate_mae: float, champion_mae: float | None, minimum_improvement: float = 0,
) -> PromotionDecision:
    """Promote the first valid model or a strictly better candidate meeting the margin."""
    if not isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not isfinite(candidate_mae) or candidate_mae < 0:
        raise ValueError("Candidate MAE must be finite and nonnegative")
    if champion_mae is None:
        return PromotionDecision(True, "initial_champion", candidate_mae, None, minimum_improvement)
    if not isfinite(champion_mae) or champion_mae < 0:
        raise ValueError("Champion MAE must be finite and nonnegative")
    better = candidate_mae < champion_mae and champion_mae - candidate_mae >= minimum_improvement
    reason = "validation_mae_improved" if better else "insufficient_validation_improvement"
    return PromotionDecision(better, reason, candidate_mae, champion_mae, minimum_improvement)
