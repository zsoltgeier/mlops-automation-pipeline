"""Run real prediction-to-report persistence and prove monitoring never promotes a model."""

import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pandas as pd
import pytest

from garments_mlops.api_models import PredictRequest
from garments_mlops import monitoring
from garments_mlops.prediction import predict_records
from garments_mlops.prepare import prepare_dataset
from garments_mlops.tracking import tracking_client
from garments_mlops.training import train_candidate
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def monitored_system(records: pd.DataFrame, parameters: Path, tmp_path: Path) -> TrainingConfig:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    config = TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}", artifact_dir=tmp_path / "artifacts",
                            data_dir=tmp_path / "prepared", lock_path=tmp_path / "training.lock", n_estimators=5)
    prepare_dataset(source, config.data_dir, parameters)
    train_candidate(config, "baseline")
    frame = pd.read_csv(config.data_dir / "reference.csv")
    request = PredictRequest.model_validate({"records": json.loads(frame.to_json(orient="records"))})
    predict_records(config, request, "b" * 32)
    return config


def test_report_reuses_identical_snapshot_and_preserves_model_lifecycle(monitored_system: TrainingConfig) -> None:
    client = tracking_client(monitored_system)
    before = client.get_model_version_by_alias(monitored_system.model_name, "champion").version
    result = monitoring.generate_monitoring(monitored_system)
    assert result.source == "live"
    assert result.current_rows == result.reference_rows == 12
    assert result.prediction_model_versions == ["1"]
    assert result.champion_at_report == "1"
    assert result.drift_share == 0
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["method"] == "jensenshannon"
    assert run.data.metrics["drift_share"] == 0
    html = monitoring.read_artifact(client, result.run_id, "monitoring/report.html")
    assert "<html" in html.lower()
    document = json.loads(monitoring.read_artifact(client, result.run_id, "monitoring/report.json"))
    assert len(document["metrics"]) == 16
    saved = monitoring.MonitoringSummary.model_validate_json(
        monitoring.read_artifact(client, result.run_id, "monitoring/summary.json"))
    assert saved == result
    assert monitoring.generate_monitoring(monitored_system) == result
    assert len(client.search_runs([run.info.experiment_id])) == 1
    assert len(client.search_model_versions(f"name = '{monitored_system.model_name}'")) == 1
    assert client.get_model_version_by_alias(monitored_system.model_name, "champion").version == before
    overview = monitoring.training_overview(client, monitored_system)
    assert overview[0]["model"] == "baseline"
    assert overview[0]["validation_mae"] is not None


def test_new_request_window_gets_its_own_report(monitored_system: TrainingConfig) -> None:
    first = monitoring.generate_monitoring(monitored_system)
    frame = pd.read_csv(monitored_system.data_dir / "reference.csv").assign(smv=10000)
    request = PredictRequest.model_validate({"records": json.loads(frame.to_json(orient="records")), "source": "demo"})
    predict_records(monitored_system, request, "c" * 32)
    demo = monitoring.generate_monitoring(monitored_system, "demo")
    assert demo.run_id != first.run_id
    assert demo.source == "demo"
    assert demo.current_rows == 12
    assert demo.columns[1].drift is True
    assert monitoring.generate_monitoring(monitored_system).run_id == first.run_id


def test_failed_artifact_write_is_not_a_successful_report(
    monitored_system: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlflow, "log_text", Mock(side_effect=OSError("storage unavailable")))
    with pytest.raises(OSError):
        monitoring.generate_monitoring(monitored_system)
    client = tracking_client(monitored_system)
    experiment = client.get_experiment_by_name(f"{monitored_system.experiment_name}-monitoring")
    assert client.search_runs([experiment.experiment_id])[0].info.status == "FAILED"
    assert str(client.get_model_version_by_alias(monitored_system.model_name, "champion").version) == "1"


def test_absent_champion_and_training_experiment_are_supported(
    monitored_system: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitoring, "champion_version", Mock(return_value=None))
    assert monitoring.generate_monitoring(monitored_system).champion_at_report is None
    client = Mock()
    client.get_experiment_by_name.return_value = None
    assert monitoring.training_overview(client, monitored_system) == []
