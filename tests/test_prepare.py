"""Exercise the complete data workflow, CLI and provenance contract."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from garments_mlops.data import RAW_FEATURES, TARGET
from garments_mlops.prepare import load_config, main, prepare_dataset, sha256


def test_complete_workflow_is_reproducible(records: pd.DataFrame, parameters: Path, tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    records.to_csv(source, index=False)
    output = tmp_path / "processed"
    metadata = prepare_dataset(source, output, parameters)
    assert metadata["raw_rows"] == metadata["clean_rows"] == 20
    assert metadata["source_sha256"] == sha256(source)
    assert metadata == json.loads((output / "metadata.json").read_text())
    first_hashes = {path.name: sha256(path) for path in output.iterdir()}
    assert len(first_hashes) == 11
    for name, checksum in metadata["files_sha256"].items():
        assert first_hashes[name] == checksum
    main(["--source", str(source), "--output", str(output), "--parameters", str(parameters)])
    assert first_hashes == {path.name: sha256(path) for path in output.iterdir()}
    reference = pd.read_csv(output / "reference.csv")
    train = pd.read_csv(output / "train.csv")
    assert len(reference) == len(train) == 12
    assert TARGET not in reference
    assert set(reference.columns) == set(RAW_FEATURES)
    numeric = pd.read_csv(output / "preprocessed.csv")
    assert len(numeric) == 20
    assert set(numeric.split) == {"train", "validation", "test"}
    assert np.isfinite(numeric.drop(columns="split").to_numpy()).all()
    assert len(pd.read_csv(output / "cleaned.csv")) == 20
    preprocessor = joblib.load(output / "preprocessor.joblib")
    for name in ["train", "validation", "test"]:
        raw = pd.read_csv(output / f"{name}.csv")
        transformed = pd.read_csv(output / f"{name}_transformed.csv")
        assert_allclose(transformed[TARGET], raw[TARGET])
        assert_allclose(transformed.drop(columns=TARGET), preprocessor.transform(raw[RAW_FEATURES]), atol=1e-14)


@pytest.mark.parametrize("content", ["", "[]", "data: 1", "other: {}", "data: {}",
                                     "data: {train_fraction: 0.6, validation_fraction: 0.2}",
                                     "data: {train_fraction: 0.6, validation_fraction: 0.2, random_seed: 42, typo: 1}"])
def test_invalid_configuration_is_rejected(content: str, parameters: Path) -> None:
    parameters.write_text(content)
    with pytest.raises(ValueError):
        load_config(parameters)


def test_invalid_source_does_not_create_outputs(records: pd.DataFrame, parameters: Path, tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    records.drop(columns=TARGET).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing target"):
        prepare_dataset(source, tmp_path / "not-created", parameters)
    assert not (tmp_path / "not-created").exists()


def test_cli_defaults_and_help(records: pd.DataFrame, parameters: Path, tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records.to_csv(data / "garments_worker_productivity.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["garments-prepare"])
    main()
    assert (data / "processed/preprocessed.csv").is_file()
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_real_dataset_smoke(parameters: Path, tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data/garments_worker_productivity.csv"
    output = tmp_path / "real-data"
    metadata = prepare_dataset(source, output, parameters)
    assert metadata["raw_rows"] == metadata["clean_rows"] == 1197
    assert metadata["duplicates_removed"] == 0
    assert metadata["source_sha256"] == "763ffcdd39b7f7f360243c2af8f89ed733e8b5f9e511efeb0c886b83e1b76866"
    cleaned = pd.read_csv(output / "cleaned.csv")
    assert cleaned.split.value_counts().to_dict() == {"train": 711, "test": 275, "validation": 211}
    assert cleaned[TARGET].gt(1).sum() == 37
