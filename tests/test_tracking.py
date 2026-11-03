"""Exercise SQLite experiment and Registry aliases with narrow error handling."""

from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_ALREADY_EXISTS
import pytest
from sklearn.pipeline import Pipeline

from garments_mlops.tracking import (
    MissingChampionError, champion_version, experiment_id, load_champion, tracking_client,
)
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def tracking_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'db/tracking.db'}", artifact_dir=tmp_path / "artifacts")


def test_experiment_is_persistent_and_deleted_experiments_are_not_replaced(tracking_config: TrainingConfig) -> None:
    client = tracking_client(tracking_config)
    first = experiment_id(client, tracking_config)
    assert experiment_id(client, tracking_config) == first
    assert tracking_config.artifact_dir.is_dir()
    client.delete_experiment(first)
    with pytest.raises(ValueError, match="experiment is deleted"):
        experiment_id(client, tracking_config)


def test_remote_experiments_use_server_artifact_storage(tmp_path: Path) -> None:
    config = TrainingConfig(tracking_uri="http://localhost:5000", artifact_dir=tmp_path / "unused")
    client = tracking_client(config)
    assert client.tracking_uri == config.tracking_uri
    fake = Mock()
    fake.get_experiment_by_name.return_value = None
    fake.create_experiment.return_value = "experiment-1"
    assert experiment_id(fake, config) == "experiment-1"
    fake.create_experiment.assert_called_once_with(config.experiment_name, artifact_location=None)
    assert not config.artifact_dir.exists()


def test_missing_champion_is_distinct_from_backend_failure(tracking_config: TrainingConfig) -> None:
    client = tracking_client(tracking_config)
    assert champion_version(client, tracking_config.model_name) is None
    with pytest.raises(MissingChampionError):
        load_champion(tracking_config)
    fake = Mock()
    fake.get_model_version_by_alias.side_effect = MlflowException("denied", error_code=PERMISSION_DENIED)
    with pytest.raises(MlflowException, match="denied"):
        champion_version(fake, tracking_config.model_name)


def test_alias_resolution_pins_the_model_uri(tracking_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    client = tracking_client(tracking_config)
    client.create_registered_model(tracking_config.model_name)
    version = client.create_model_version(tracking_config.model_name, source=tracking_config.artifact_dir.as_uri())
    client.set_registered_model_alias(tracking_config.model_name, "champion", version.version)
    pipeline = Pipeline([])
    loader = Mock(return_value=pipeline)
    monkeypatch.setattr(mlflow.sklearn, "load_model", loader)
    champion = load_champion(tracking_config)
    assert champion.version == str(version.version)
    assert champion.model is pipeline
    loader.assert_called_once_with(f"models:/{tracking_config.model_name}/{version.version}")
    loader.return_value = object()
    with pytest.raises(TypeError, match="complete sklearn Pipeline"):
        load_champion(tracking_config)


def test_concurrent_experiment_creation_reuses_only_an_active_experiment() -> None:
    config = TrainingConfig(tracking_uri="http://localhost:5000")
    fake = Mock()
    fake.get_experiment_by_name.side_effect = [None, SimpleNamespace(lifecycle_stage="active", experiment_id="new")]
    fake.create_experiment.side_effect = MlflowException("created concurrently", error_code=RESOURCE_ALREADY_EXISTS)
    assert experiment_id(fake, config) == "new"
    for replacement in [None, SimpleNamespace(lifecycle_stage="deleted")]:
        fake.get_experiment_by_name.side_effect = [None, replacement]
        with pytest.raises(MlflowException, match="created concurrently"):
            experiment_id(fake, config)
    fake.get_experiment_by_name.side_effect = [None]
    fake.create_experiment.side_effect = MlflowException("denied", error_code=PERMISSION_DENIED)
    with pytest.raises(MlflowException, match="denied"):
        experiment_id(fake, config)
