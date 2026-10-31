"""Check regression artifacts and the quality gate independently of MLflow."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from garments_mlops.data import RAW_FEATURES, TARGET, clean_dataset
from garments_mlops.modeling import ModelSpec, build_model, decide_promotion, regression_metrics
from garments_mlops.splitting import split_by_date


@pytest.mark.parametrize("kind", ["baseline", "random_forest"])
def test_complete_model_is_deterministic_and_uses_raw_features(records: pd.DataFrame, kind: str) -> None:
    split = split_by_date(clean_dataset(records))
    spec = ModelSpec(kind=kind, n_estimators=5)
    first = build_model(spec).fit(split.train[RAW_FEATURES], split.train[TARGET])
    second = build_model(spec).fit(split.train[RAW_FEATURES], split.train[TARGET])
    prediction = first.predict(split.validation[RAW_FEATURES])
    assert_allclose(prediction, second.predict(split.validation[RAW_FEATURES]))
    metrics = regression_metrics(split.validation[TARGET], prediction)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["rmse"] >= metrics["mae"] >= 0
    if kind == "baseline":
        assert_allclose(prediction, split.train[TARGET].mean())


@pytest.mark.parametrize("options", [{"kind": "unknown"}, {"random_seed": True}, {"random_seed": -1},
                                     {"n_estimators": 0}, {"max_depth": 1.5}, {"min_samples_leaf": -1}])
def test_invalid_model_configuration_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelSpec(**options)


def test_regression_metrics_are_correct_and_handle_constant_targets() -> None:
    assert regression_metrics([0, 1], [0, 1]) == {"mae": 0, "rmse": 0, "r2": 1}
    assert regression_metrics([1, 1], [0, 0])["r2"] == 0


@pytest.mark.parametrize("actual,predicted", [
    ([], []), ([1], [1]), ([1, 2], [1]), ([[1], [2]], [[1], [2]]),
    ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf]),
])
def test_invalid_predictions_are_rejected(actual: object, predicted: object) -> None:
    with pytest.raises(ValueError):
        regression_metrics(actual, predicted)


@pytest.mark.parametrize("candidate,champion,margin,promoted,reason", [
    (0.2, None, 0, True, "initial_champion"),
    (0.1, 0.2, 0, True, "validation_mae_improved"),
    (0.2, 0.1, 0, False, "insufficient_validation_improvement"),
    (0.2, 0.2, 0, False, "insufficient_validation_improvement"),
    (0.1, 0.2, 0.15, False, "insufficient_validation_improvement"),
    (0.125, 0.25, 0.125, True, "validation_mae_improved"),
])
def test_promotion_protects_existing_champion(
    candidate: float, champion: float | None, margin: float, promoted: bool, reason: str,
) -> None:
    decision = decide_promotion(candidate, champion, margin)
    assert decision.promoted is promoted
    assert decision.reason == reason
    assert decision.candidate_mae == candidate
    assert decision.champion_mae == champion
    assert decision.minimum_improvement == margin


@pytest.mark.parametrize("candidate,champion,margin", [
    (np.nan, None, 0), (-1, None, 0), (0, np.inf, 0), (0, -1, 0), (0, 1, -1), (0, 1, np.inf),
])
def test_nonfinite_scores_and_invalid_margins_cannot_promote(
    candidate: float, champion: float | None, margin: float,
) -> None:
    with pytest.raises(ValueError):
        decide_promotion(candidate, champion, margin)
