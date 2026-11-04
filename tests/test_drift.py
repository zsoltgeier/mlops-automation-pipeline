"""Exercise real Evidently reports on unchanged, shifted and missing populations."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from garments_mlops import drift


def test_unchanged_and_shifted_populations_produce_native_reports(records: pd.DataFrame) -> None:
    reference = records.drop(columns="actual_productivity")
    same = drift.analyze_drift(reference, reference)
    assert same.drift_share == 0
    assert same.dataset_drift is False
    assert same.evaluated_columns == 15
    assert "<html" in same.html.lower()
    assert len(json.loads(same.report_json)["metrics"]) == 16
    shifted = reference.copy()
    for name in drift.NUMERIC_FEATURES:
        shifted[name] = shifted[name].fillna(0) + 10000
    changed = drift.analyze_drift(reference, shifted)
    assert changed.dataset_drift is True
    assert changed.drift_share >= 0.5
    assert next(item for item in changed.columns if item.column == "smv").drift is True
    assert all(item.column not in {"date", "prediction", "request_id", "model_version"} for item in changed.columns)


def test_all_missing_column_has_missingness_but_no_invented_drift_score(records: pd.DataFrame) -> None:
    reference = records.drop(columns="actual_productivity")
    current = reference.assign(wip=float("nan"))
    result = drift.analyze_drift(reference, current)
    item = next(column for column in result.columns if column.column == "wip")
    assert item.score is None and item.drift is None
    assert item.reference_missing == 0.5
    assert item.current_missing == 1
    assert result.evaluated_columns == 14


def test_nonfinite_library_score_is_rejected(records: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = Mock()
    snapshot.json.return_value = '{"metrics":[{"config":{"column":"smv"},"value":NaN}]}'
    monkeypatch.setattr(drift, "Report", Mock(return_value=Mock(run=Mock(return_value=snapshot))))
    with pytest.raises(ValueError, match="nonfinite"):
        drift.analyze_drift(records.drop(columns="actual_productivity"), records.drop(columns="actual_productivity"))
