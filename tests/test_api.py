"""Exercise public HTTP contracts, redacted errors and real model lifecycle calls."""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from filelock import Timeout
import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import pytest

from garments_mlops import api
from garments_mlops.api_models import PredictResponse
from garments_mlops.modeling import decide_promotion
from garments_mlops.monitoring import MonitoringSummary
from garments_mlops.prediction import PredictionOutputError
from garments_mlops.prepare import prepare_dataset
from garments_mlops.tracking import MissingChampionError
from garments_mlops.training import TrainingResult
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def http_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}", artifact_dir=tmp_path / "artifacts",
                          data_dir=tmp_path / "prepared", lock_path=tmp_path / "training.lock", n_estimators=5)


@pytest.fixture
def body(records: pd.DataFrame) -> dict[str, object]:
    frame = records.iloc[:1].drop(columns="actual_productivity").copy()
    frame["date"] = "2015-01-01"
    return {"records": frame.to_dict(orient="records") * 2}


def test_prediction_and_training_responses_are_correlated(
    http_config: TrainingConfig, body: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prediction(config: TrainingConfig, request: object, request_id: str) -> PredictResponse:
        assert config == http_config
        return PredictResponse(request_id=request_id, model_version="2", run_id="prediction-run", predictions=[.7, .8])

    monkeypatch.setattr(api, "predict_records", prediction)
    result = TrainingResult("training-run", "3", "models:/model/3", "2", {"mae": .2}, decide_promotion(.2, .1))
    training = Mock(return_value=result)
    monkeypatch.setattr(api, "train_candidate", training)
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        prediction_response = client.post("/predict", json=body, headers={"X-Request-ID": "untrusted-header"})
        assert prediction_response.status_code == 200
        assert prediction_response.json()["request_id"] == prediction_response.headers["X-Request-ID"]
        assert prediction_response.json()["request_id"] != "untrusted-header"
        response = client.post("/train", json={"model": "baseline"})
        assert response.status_code == 200
        expected = {"status": "completed", "request_id": response.headers["X-Request-ID"], **asdict(result)}
        assert response.json() == expected
        assert training.call_args.kwargs["request_id"] == response.json()["request_id"]
        assert training.call_args.args[0].lock_timeout == 0


@pytest.mark.parametrize("error,status", [
    (MissingChampionError("missing"), 503), (Timeout("secret-lock-path"), 409),
    (MlflowException("private-server-url"), 503), (OSError("secret-path"), 503),
    (ValueError("private-data"), 503), (PredictionOutputError("secret-array"), 500),
    (RuntimeError("private-token"), 500),
])
def test_errors_do_not_expose_backend_details(
    http_config: TrainingConfig, body: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    error: Exception, status: int, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(api, "predict_records", Mock(side_effect=error))
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == status
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert str(error) not in response.text
    assert str(error) not in caplog.text


def test_validation_logs_and_responses_do_not_echo_unknown_secrets(
    http_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    prediction = Mock()
    monkeypatch.setattr(api, "predict_records", prediction)
    caplog.set_level("INFO", logger="garments_mlops.api")
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        response = client.post("/predict", json={"records": [], "password": "do-not-leak"},
                               headers={"Authorization": "Bearer do-not-leak"})
        assert response.status_code == 422
        assert "do-not-leak" not in response.text
        assert "do-not-leak" not in caplog.text
        assert client.get("/does-not-exist?secret=do-not-leak").status_code == 404
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert {"/train", "/predict", "/health"} <= set(schema["paths"])
    prediction.assert_not_called()
    assert len(caplog.records) >= 4
    for entry in caplog.records:
        assert json.loads(entry.message)["event"] == "request_completed"


def test_health_reflects_registry_readiness(http_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        assert client.get("/health").status_code == 503
        monkeypatch.setattr(api, "champion_version", Mock(return_value=Mock(version="2")))
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "model_version": "2"}
        monkeypatch.setattr(api, "tracking_client", Mock(side_effect=OSError("private-backend")))
        assert client.get("/health").json() == {"status": "unavailable", "model_version": None}


def test_factory_reads_explicit_environment_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("{}")
    monkeypatch.setenv("MLOPS_TRAINING_CONFIG", str(path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert api.create_app().state.training_config.tracking_uri == "http://localhost:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert api.create_app().state.training_config.tracking_uri == TrainingConfig().tracking_uri


def test_real_api_training_prediction_reload_and_audit(
    http_config: TrainingConfig, body: dict[str, object], records: pd.DataFrame, parameters: Path, tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    prepare_dataset(source, http_config.data_dir, parameters)
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        assert client.post("/predict", json=body).status_code == 503
        baseline = client.post("/train", json={"model": "baseline"})
        assert baseline.status_code == 200
        assert baseline.json()["decision"]["promoted"]
        assert client.get("/health").json()["model_version"] == "1"
        assert client.post("/predict", json=body).json()["model_version"] == "1"
        forest = client.post("/train", json={"model": "random_forest"})
        assert forest.status_code == 200
        assert forest.json()["decision"]["promoted"]
        response = client.post("/predict", json=body)
        assert response.status_code == 200
        assert response.json()["model_version"] == "2"
        assert len(response.json()["predictions"]) == 2
    tracking = mlflow.MlflowClient(tracking_uri=http_config.tracking_uri)
    artifact = tracking.download_artifacts(forest.json()["run_id"], "api/response.json")
    assert json.loads(Path(artifact).read_text()) == forest.json()


def test_lifespan_initialization_and_environment_flags(
    http_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = Mock(return_value="2")
    monkeypatch.setattr(api, "initialize_system", initializer)
    monkeypatch.setenv("MLOPS_BOOTSTRAP", "true")
    with TestClient(api.create_app(http_config)) as client:
        assert client.app.state.bootstrap_version == "2"
    initializer.assert_called_once_with(http_config)
    monkeypatch.setenv("MLOPS_BOOTSTRAP", "false")
    with TestClient(api.create_app(http_config)):
        assert initializer.call_count == 1
    monkeypatch.setenv("MLOPS_BOOTSTRAP", "typo")
    with pytest.raises(ValueError, match="MLOPS_BOOTSTRAP"):
        api.create_app(http_config)


def test_monitor_endpoint_reuses_the_existing_drift_service_and_validates_source(
    http_config: TrainingConfig, monitoring_summary: MonitoringSummary, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate = Mock(return_value=monitoring_summary)
    monkeypatch.setattr(api, "generate_monitoring", generate)
    with TestClient(api.create_app(http_config, bootstrap=False)) as client:
        response = client.post("/monitor", json={"source": "demo"})
        assert response.status_code == 200
        assert response.json() == monitoring_summary.model_dump(mode="json")
        assert response.headers["X-Request-ID"]
        generate.assert_called_once_with(http_config, "demo")
        assert client.post("/monitor", json={"source": "unknown"}).status_code == 422
        assert client.post("/monitor", json={"path": "private-path"}).status_code == 422
