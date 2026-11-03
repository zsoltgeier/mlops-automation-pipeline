"""Check inference artifacts, finite output validation and audit failures."""

import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import numpy as np
import pandas as pd
import pytest

from garments_mlops.api_models import PredictRequest
from garments_mlops import prediction
from garments_mlops.tracking import Champion, tracking_client
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def prediction_request(records: pd.DataFrame) -> PredictRequest:
    frame = records.iloc[:1].drop(columns="actual_productivity").copy()
    frame["date"] = "2015-01-01"
    return PredictRequest.model_validate({"records": frame.to_dict(orient="records") * 2})


@pytest.fixture
def prediction_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}", artifact_dir=tmp_path / "artifacts")


def test_prediction_audit_is_usable_for_monitoring(
    prediction_request: PredictRequest, prediction_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    model.predict.return_value = np.array([0.7, 0.8])
    monkeypatch.setattr(prediction, "load_champion", Mock(return_value=Champion("2", model)))
    result = prediction.predict_records(prediction_config, prediction_request, "b" * 32)
    assert result.predictions == [0.7, 0.8]
    assert result.model_version == "2"
    client = tracking_client(prediction_config)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["model_version"] == "2"
    assert run.data.metrics["record_count"] == 2
    body = json.loads(Path(client.download_artifacts(result.run_id, "api/request.json")).read_text())
    assert body["records"][0]["date"] == "2015-01-01"
    response = json.loads(Path(client.download_artifacts(result.run_id, "api/response.json")).read_text())
    assert response == result.model_dump(mode="json")
    current = pd.read_csv(client.download_artifacts(result.run_id, "monitoring/current.csv"))
    assert current.prediction.tolist() == result.predictions
    assert len(current) == 2
    assert "actual_productivity" not in current


@pytest.mark.parametrize("values", [[0.7], [[0.7], [0.8]], [np.nan, 0.8], [0.7, np.inf]])
def test_invalid_model_predictions_are_rejected(
    prediction_request: PredictRequest, prediction_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
    values: object,
) -> None:
    model = Mock(predict=Mock(return_value=values))
    monkeypatch.setattr(prediction, "load_champion", Mock(return_value=Champion("2", model)))
    with pytest.raises(prediction.PredictionOutputError):
        prediction.predict_records(prediction_config, prediction_request, "b" * 32)


def test_failed_audit_does_not_return_an_unlogged_success(
    prediction_request: PredictRequest, prediction_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock(predict=Mock(return_value=[.7, .8]))
    monkeypatch.setattr(prediction, "load_champion", Mock(return_value=Champion("2", model)))
    monkeypatch.setattr(mlflow, "log_text", Mock(side_effect=OSError("artifact unavailable")))
    with pytest.raises(OSError, match="artifact unavailable"):
        prediction.predict_records(prediction_config, prediction_request, "b" * 32)
    client = tracking_client(prediction_config)
    experiment = client.get_experiment_by_name(f"{prediction_config.experiment_name}-requests")
    assert client.search_runs([experiment.experiment_id])[0].info.status == "FAILED"
