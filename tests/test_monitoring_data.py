"""Validate DVC reference hashes, bounded request windows and legacy audit compatibility."""

from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from garments_mlops import monitoring_data
from garments_mlops.prepare import prepare_dataset
from garments_mlops.training_config import TrainingConfig


def test_reference_uses_only_hashed_training_inputs(
    records: pd.DataFrame, parameters: Path, tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    config = TrainingConfig(data_dir=tmp_path / "prepared")
    prepare_dataset(source, config.data_dir, parameters)
    frame, digest = monitoring_data.load_reference(config)
    assert len(frame) == 12
    assert "actual_productivity" not in frame
    assert digest == sha256((config.data_dir / "reference.csv").read_bytes()).hexdigest()
    assert frame.date.max() < pd.read_csv(config.data_dir / "validation.csv").date.min()
    (config.data_dir / "reference.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="hash"):
        monitoring_data.load_reference(config)


@pytest.mark.parametrize("failure", [None, "request", "model", "prediction", "source", "empty"])
def test_window_bounds_origin_and_artifact_integrity(
    records: pd.DataFrame, tmp_path: Path, failure: str | None,
) -> None:
    frame = records.drop(columns="actual_productivity").assign(
        request_id="a" * 32, model_version="2", prediction=.7, data_source="live")
    if failure == "request":
        frame["request_id"] = "mismatch"
    if failure == "model":
        frame["model_version"] = "3"
    if failure == "prediction":
        frame["prediction"] = float("inf")
    if failure == "source":
        frame["data_source"] = "demo"
    if failure == "empty":
        frame = frame.iloc[:0]
    path = tmp_path / "current.csv"
    frame.to_csv(path, index=False)
    client = Mock()
    tags = {"request_id": "a" * 32, "model_version": "2"}
    client.search_runs.return_value = [Mock(info=Mock(run_id="skip"), data=Mock(tags={**tags, "data_source": "demo"})),
                                       Mock(info=Mock(run_id="one"), data=Mock(tags=tags)),
                                       Mock(info=Mock(run_id="two"), data=Mock(tags=tags))]
    client.download_artifacts.return_value = str(path)
    if failure is not None:
        with pytest.raises(ValueError):
            monitoring_data.collect_current(client, TrainingConfig(), "live")
        return
    window = monitoring_data.collect_current(client, TrainingConfig(), "live", max_rows=23)
    assert len(window.frame) == 23
    assert window.frame.data_source.unique().tolist() == ["live"]
    assert [entry["rows"] for entry in window.manifest] == [20, 3]
    assert window.manifest[0]["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert "FINISHED" in client.search_runs.call_args.kwargs["filter_string"]
    assert len(monitoring_data.collect_current(client, TrainingConfig(), "live", max_rows=2).frame) == 2
    frame.assign(data_source="demo").to_csv(path, index=False)
    assert len(monitoring_data.collect_current(client, TrainingConfig(), "demo").frame) == 20


def test_missing_experiment_and_empty_source_are_explicit() -> None:
    client = Mock()
    client.get_experiment_by_name.return_value = None
    with pytest.raises(monitoring_data.NoCurrentData):
        monitoring_data.collect_current(client, TrainingConfig(), "live")
    client.get_experiment_by_name.return_value = Mock()
    client.search_runs.return_value = []
    with pytest.raises(monitoring_data.NoCurrentData):
        monitoring_data.collect_current(client, TrainingConfig(), "demo")
    with pytest.raises(ValueError, match="bounds"):
        monitoring_data.collect_current(client, TrainingConfig(), "live", max_runs=0)
