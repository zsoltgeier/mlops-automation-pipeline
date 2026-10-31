"""Integrate real model artifacts, tracking, aliases and quality protection."""

from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import Mock

from filelock import FileLock, Timeout
import mlflow.sklearn as mlflow_sklearn
from mlflow.models import get_model_info
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from garments_mlops.data import RAW_FEATURES
from garments_mlops.prepare import prepare_dataset
from garments_mlops.tracking import load_champion, tracking_client
from garments_mlops.training import train_candidate
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def training_config(records: pd.DataFrame, parameters: Path, tmp_path: Path) -> TrainingConfig:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    data = tmp_path / "prepared"
    prepare_dataset(source, data, parameters)
    return TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}", artifact_dir=tmp_path / "artifacts",
                          data_dir=data, lock_path=tmp_path / "train.lock", n_estimators=5)


def test_initial_model_registers_full_pipeline_and_tracking_artifacts(training_config: TrainingConfig) -> None:
    result = train_candidate(training_config, "baseline")
    assert result.decision.promoted
    assert result.decision.reason == "initial_champion"
    assert result.model_version == result.champion_version == "1"
    client = tracking_client(training_config)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["validation_mae"] == result.metrics["mae"]
    assert run.data.params["kind"] == "baseline"
    assert run.data.tags["promotion_approved"] == "true"
    assert {item.dataset.name for item in run.inputs.dataset_inputs} == {"train", "validation"}
    champion = load_champion(training_config)
    features = pd.read_csv(training_config.data_dir / "validation.csv")[RAW_FEATURES]
    assert np.isfinite(champion.model.predict(features)).all()
    assert champion.version == result.model_version
    artifact = client.download_artifacts(result.run_id, "promotion/decision.json")
    decision = json.loads(Path(artifact).read_text())
    assert decision["candidate_version"] == "1"
    assert decision["champion_before"] is None
    assert client.list_artifacts(result.run_id, "data")[0].path == "data/provenance.json"
    assert client.list_artifacts(result.run_id, "evaluation")[0].path == "evaluation/validation_predictions.csv"
    info = get_model_info(result.model_uri)
    assert info.flavors["sklearn"]["serialization_format"] == "skops"
    assert set(info.signature.inputs.input_names()) == set(RAW_FEATURES)


def test_identical_candidate_is_rejected_and_model_loads_without_restart(training_config: TrainingConfig) -> None:
    first = train_candidate(training_config, "baseline")
    before = load_champion(training_config)
    second = train_candidate(training_config, "baseline")
    after = load_champion(training_config)
    assert not second.decision.promoted
    assert second.model_version == "2"
    assert first.champion_version == second.champion_version == after.version == before.version
    assert second.decision.candidate_mae == second.decision.champion_mae
    features = pd.read_csv(training_config.data_dir / "validation.csv")[RAW_FEATURES]
    assert_allclose(before.model.predict(features), after.model.predict(features))


def test_training_is_serialized_by_shared_file_lock(training_config: TrainingConfig) -> None:
    with FileLock(training_config.lock_path):
        with pytest.raises(Timeout):
            train_candidate(replace(training_config, lock_timeout=0), "baseline")


def test_improved_forest_promotes_and_worse_baseline_cannot_replace_it(training_config: TrainingConfig) -> None:
    baseline = train_candidate(training_config, "baseline")
    forest = train_candidate(training_config, "random_forest")
    assert forest.metrics["mae"] < baseline.metrics["mae"]
    assert forest.decision.promoted
    assert forest.champion_version == forest.model_version == "2"
    before = load_champion(training_config)
    rejected = train_candidate(training_config, "baseline")
    after = load_champion(training_config)
    assert rejected.decision.candidate_mae > rejected.decision.champion_mae
    assert not rejected.decision.promoted
    assert rejected.model_version == "3"
    assert rejected.champion_version == after.version == before.version == forest.model_version
    features = pd.read_csv(training_config.data_dir / "validation.csv")[RAW_FEATURES]
    assert_allclose(before.model.predict(features), after.model.predict(features))
    client = tracking_client(training_config)
    previous = client.get_model_version(training_config.model_name, baseline.model_version)
    assert "champion" not in previous.aliases


def test_configurable_margin_can_reject_an_improved_candidate(training_config: TrainingConfig) -> None:
    baseline = train_candidate(training_config, "baseline")
    candidate = train_candidate(replace(training_config, minimum_improvement=1), "random_forest")
    assert candidate.metrics["mae"] < baseline.metrics["mae"]
    assert not candidate.decision.promoted
    assert candidate.champion_version == baseline.model_version


def test_failed_artifact_registration_does_not_change_champion(
    training_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = train_candidate(training_config, "baseline")
    monkeypatch.setattr(mlflow_sklearn, "log_model", Mock(return_value=Mock(registered_model_version=None)))
    with pytest.raises(RuntimeError, match="registered model version"):
        train_candidate(training_config, "baseline")
    client = tracking_client(training_config)
    active = client.get_model_version_by_alias(training_config.model_name, "champion")
    assert str(active.version) == initial.model_version
    runs = client.search_runs([client.get_experiment_by_name(training_config.experiment_name).experiment_id])
    assert any(run.info.status == "FAILED" for run in runs)


@pytest.mark.parametrize("failure", ["artifact_storage", "alias_update"])
def test_backend_failure_cannot_activate_a_candidate(
    training_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    initial = train_candidate(training_config, "baseline")
    client = tracking_client(training_config)
    if failure == "artifact_storage":
        monkeypatch.setattr(mlflow_sklearn, "log_model", Mock(side_effect=OSError("artifact storage unavailable")))
        error = OSError
    else:
        monkeypatch.setattr(type(client), "set_registered_model_alias",
                            Mock(side_effect=MlflowException("alias update denied", error_code="PERMISSION_DENIED")))
        error = MlflowException
    with pytest.raises(error):
        train_candidate(training_config, "random_forest")
    active = client.get_model_version_by_alias(training_config.model_name, "champion")
    assert str(active.version) == initial.model_version
    runs = client.search_runs([client.get_experiment_by_name(training_config.experiment_name).experiment_id])
    assert any(run.info.status == "FAILED" for run in runs)


def test_model_loads_from_its_packaged_code_in_a_fresh_process(training_config: TrainingConfig, tmp_path: Path) -> None:
    result = train_candidate(training_config, "random_forest")
    program = """
import json
import sys
import mlflow
import mlflow.sklearn as store
import pandas as pd
mlflow.set_tracking_uri(sys.argv[1])
mlflow.set_registry_uri(sys.argv[1])
model = store.load_model(sys.argv[2])
import garments_mlops.preprocessing as preprocessing
features = pd.read_csv(sys.argv[3])[json.loads(sys.argv[4])]
print(json.dumps({'prediction': model.predict(features).tolist(), 'code': preprocessing.__file__}))
"""
    process = subprocess.run([
        sys.executable, "-I", "-c", program, training_config.tracking_uri,
        f"models:/{training_config.model_name}/{result.model_version}",
        str(training_config.data_dir / "validation.csv"), json.dumps(RAW_FEATURES),
    ], cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=60, check=True)
    restored = json.loads(process.stdout)
    assert "code/garments_mlops/preprocessing.py" in Path(restored["code"]).as_posix()
    features = pd.read_csv(training_config.data_dir / "validation.csv")[RAW_FEATURES]
    assert_allclose(restored["prediction"], load_champion(training_config).model.predict(features))
