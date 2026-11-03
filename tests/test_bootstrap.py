"""Ensure cold start and restart preserve Registry versions and data integrity."""

from pathlib import Path

import pandas as pd

from garments_mlops.bootstrap import initialize_system
from garments_mlops.prepare import sha256
from garments_mlops.tracking import tracking_client
from garments_mlops.training_config import TrainingConfig


def test_initialization_is_idempotent_and_repairs_generated_data(
    records: pd.DataFrame, parameters: Path, tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    config = TrainingConfig(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}", artifact_dir=tmp_path / "artifacts",
                            data_dir=tmp_path / "prepared", lock_path=tmp_path / "training.lock", n_estimators=5)
    first = initialize_system(config, source, parameters)
    assert first == "2"
    expected = {path.name: sha256(path) for path in config.data_dir.iterdir()}
    (config.data_dir / "train.csv").write_text("corrupted")
    second = initialize_system(config, source, parameters)
    assert second == first
    assert expected == {path.name: sha256(path) for path in config.data_dir.iterdir()}
    client = tracking_client(config)
    versions = client.search_model_versions(f"name = '{config.model_name}'")
    assert len(versions) == 2
    experiment = client.get_experiment_by_name(config.experiment_name)
    assert len(client.search_runs([experiment.experiment_id])) == 2
