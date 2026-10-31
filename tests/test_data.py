"""Check normalization, schema validation and preservation of valid records."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from garments_mlops.data import NUMERIC_FEATURES, RAW_FEATURES, TARGET, clean_dataset, clean_features


def test_cleaning_is_idempotent_and_does_not_mutate_inputs(records: pd.DataFrame) -> None:
    original = records.copy(deep=True)
    cleaned = clean_dataset(records)
    assert_frame_equal(records, original)
    assert_frame_equal(clean_dataset(cleaned), cleaned)
    assert set(cleaned.department) == {"sewing", "finishing"}
    assert set(cleaned.team) == {"1", "2"}
    assert cleaned.date.min() == "2015-01-01"
    assert cleaned.month.eq(1).all()
    assert cleaned.day_of_month.max() == 10
    assert cleaned[TARGET].max() == 1.12
    assert cleaned.no_of_workers.eq(30.5).all()
    assert cleaned.wip.isna().sum() == 10


def test_positional_label_alignment_and_duplicate_removal(records: pd.DataFrame) -> None:
    changed_index = records.set_axis(np.arange(100, 120))
    assert_frame_equal(clean_dataset(records), clean_dataset(changed_index))
    duplicated = pd.concat([records, records.iloc[:1]], ignore_index=True)
    assert_frame_equal(clean_dataset(duplicated), clean_dataset(records))


def test_blank_categories_and_missing_teams_are_normalized(records: pd.DataFrame) -> None:
    records.loc[0, "department"] = "  "
    records.loc[1, "team"] = np.nan
    cleaned = clean_features(records[RAW_FEATURES])
    assert pd.isna(cleaned.loc[0, "department"])
    assert pd.isna(cleaned.loc[1, "team"])


@pytest.mark.parametrize("name", RAW_FEATURES)
def test_missing_required_features_are_rejected(records: pd.DataFrame, name: str) -> None:
    with pytest.raises(ValueError, match="Missing required"):
        clean_features(records[RAW_FEATURES].drop(columns=name))


def test_duplicate_names_are_rejected(records: pd.DataFrame) -> None:
    duplicate_columns = pd.concat([records, records[["date"]]], axis=1)
    for cleaner in [clean_dataset, clean_features]:
        with pytest.raises(ValueError, match="Duplicate column"):
            cleaner(duplicate_columns)


def test_target_is_required_for_training_and_forbidden_in_features(records: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing target"):
        clean_dataset(records[RAW_FEATURES])
    with pytest.raises(ValueError, match="Target must not"):
        clean_features(records)


def test_empty_dataset_is_rejected(records: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="At least one"):
        clean_dataset(records.iloc[:0])


@pytest.mark.parametrize("value", [None, "not-a-date", "2015-02-30", 12345, "2015-01-01T12:00:00"])
def test_invalid_dates_are_rejected(records: pd.DataFrame, value: object) -> None:
    records.loc[0, "date"] = value
    with pytest.raises(ValueError):
        clean_dataset(records)


@pytest.mark.parametrize("name", NUMERIC_FEATURES)
@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, "invalid"])
def test_invalid_numbers_are_rejected(records: pd.DataFrame, name: str, value: object) -> None:
    records[name] = records[name].astype(object)
    records.loc[0, name] = value
    with pytest.raises(ValueError):
        clean_dataset(records)


@pytest.mark.parametrize("value", [0, -1, 1.5, np.inf, "invalid"])
def test_invalid_teams_are_rejected(records: pd.DataFrame, value: object) -> None:
    records["team"] = records.team.astype(object)
    records.loc[0, "team"] = value
    with pytest.raises(ValueError):
        clean_dataset(records)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.1, "invalid"])
def test_invalid_targets_are_rejected(records: pd.DataFrame, value: object) -> None:
    records[TARGET] = records[TARGET].astype(object)
    records.loc[0, TARGET] = value
    with pytest.raises(ValueError):
        clean_dataset(records)
