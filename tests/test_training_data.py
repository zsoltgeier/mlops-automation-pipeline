"""Guard against mutated partitions, time leakage and test-set consumption."""

import json
from pathlib import Path

import pandas as pd
import pytest

from garments_mlops.prepare import prepare_dataset, sha256
from garments_mlops.training_data import load_training_data


@pytest.fixture
def prepared(records: pd.DataFrame, parameters: Path, tmp_path: Path) -> Path:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    output = tmp_path / "prepared"
    prepare_dataset(source, output, parameters)
    return output


def rewrite_partition(directory: Path, name: str, frame: pd.DataFrame) -> None:
    """Update a fixture and its manifest to test semantic checks separately."""
    path = directory / name
    frame.to_csv(path, index=False)
    manifest = directory / "metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata["files_sha256"][name] = sha256(path)
    manifest.write_text(json.dumps(metadata))


def test_only_train_and_validation_are_consumed(prepared: Path) -> None:
    (prepared / "test.csv").write_text("This must never be parsed during training")
    data = load_training_data(prepared)
    assert len(data.train) == 12
    assert len(data.validation) == 4
    assert data.train.date.max() < data.validation.date.min()
    assert data.provenance["train_sha256"] == sha256(prepared / "train.csv")
    assert data.provenance["validation_sha256"] == sha256(prepared / "validation.csv")
    assert data.provenance["validation_rows"] == 4


def test_changed_csv_is_rejected(prepared: Path) -> None:
    path = prepared / "train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="integrity check failed"):
        load_training_data(prepared)


@pytest.mark.parametrize("metadata", [[], {}, {"files_sha256": {}},
                                      {"files_sha256": {}, "source_sha256": 123},
                                      {"files_sha256": {}, "source_sha256": "not-a-hash"}])
def test_incomplete_metadata_is_rejected(prepared: Path, metadata: object) -> None:
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata must contain"):
        load_training_data(prepared)


@pytest.mark.parametrize("mode", ["duplicate", "too_small", "overlap"])
def test_invalid_partition_semantics_are_rejected(prepared: Path, mode: str) -> None:
    validation = pd.read_csv(prepared / "validation.csv")
    if mode == "duplicate":
        validation = pd.concat([validation, validation.iloc[:1]], ignore_index=True)
    elif mode == "too_small":
        validation = validation.iloc[:1]
    else:
        validation["date"] = "2015-01-01"
    rewrite_partition(prepared, "validation.csv", validation)
    with pytest.raises(ValueError):
        load_training_data(prepared)
