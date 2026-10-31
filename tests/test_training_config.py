"""Validate Tracking Server, artifact and quality-gate settings."""

from pathlib import Path

import numpy as np
import pytest

from garments_mlops.training_config import TrainingConfig, load_training_config


def test_defaults_are_persistent_and_model_parameters_are_reusable(tmp_path: Path) -> None:
    config = TrainingConfig()
    assert config.tracking_uri == "sqlite:///runtime/mlflow.db"
    assert config.model_spec("baseline").kind == "baseline"
    assert config.model_spec("random_forest").random_seed == 42
    path = tmp_path / "training.yaml"
    path.write_text("{}")
    assert load_training_config(path) == config


def test_yaml_paths_and_remote_uri_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("tracking_uri: http://localhost:5000\nartifact_dir: artifacts\n"
                    "data_dir: prepared\nlock_path: train.lock\n")
    config = load_training_config(path)
    assert config.artifact_dir == Path("artifacts")
    assert config.data_dir == Path("prepared")
    assert config.lock_path == Path("train.lock")
    assert config.tracking_uri == "http://localhost:5000"
    assert TrainingConfig(tracking_uri="https://tracking.example.com").tracking_uri.startswith("https://")


@pytest.mark.parametrize("options", [
    {"tracking_uri": None}, {"tracking_uri": "file:///tmp/tracking"}, {"tracking_uri": "sqlite:///"},
    {"tracking_uri": "sqlite:///:memory:"}, {"experiment_name": " "}, {"experiment_name": None},
    {"model_name": "bad/model@name"}, {"model_name": None}, {"minimum_improvement": True},
    {"minimum_improvement": -1}, {"minimum_improvement": np.inf}, {"lock_timeout": "30"},
    {"lock_timeout": np.nan}, {"random_seed": -1},
])
def test_invalid_settings_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**options)


@pytest.mark.parametrize("content", ["", "[]", "typo: 42", "data_dir: null", "artifact_dir: ''", "lock_path: 42"])
def test_invalid_yaml_configuration_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_training_config(path)
