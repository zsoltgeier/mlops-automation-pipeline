"""Verify temporal separation and validation of reproducibility parameters."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from garments_mlops.data import clean_dataset
from garments_mlops.splitting import SplitConfig, split_by_date


def test_whole_dates_are_disjoint_and_every_record_is_preserved(records: pd.DataFrame) -> None:
    cleaned = clean_dataset(records)
    splits = split_by_date(cleaned.sample(frac=1, random_state=42))
    assert [len(frame) for _, frame in splits.items()] == [12, 4, 4]
    assert splits.train.date.max() < splits.validation.date.min()
    assert splits.validation.date.max() < splits.test.date.min()
    recombined = pd.concat([frame for _, frame in splits.items()], ignore_index=True)
    assert_frame_equal(recombined.sort_values(["date", "team"]).reset_index(drop=True),
                       cleaned.sort_values(["date", "team"]).reset_index(drop=True))
    for _, frame in splits.items():
        assert frame.date.value_counts().eq(2).all()


def test_partition_dates_do_not_depend_on_seed(records: pd.DataFrame) -> None:
    cleaned = clean_dataset(records)
    first = split_by_date(cleaned, SplitConfig(random_seed=0))
    second = split_by_date(cleaned, SplitConfig(random_seed=2**32 - 1))
    for (_, left), (_, right) in zip(first.items(), second.items(), strict=True):
        assert_frame_equal(left, right)


@pytest.mark.parametrize("train,validation", [
    (0, 0.2), (-0.1, 0.2), (0.6, 0), (0.8, 0.2), (np.nan, 0.2), (0.6, np.inf),
])
def test_invalid_fractions_are_rejected(train: float, validation: float) -> None:
    with pytest.raises(ValueError, match="finite, positive"):
        SplitConfig(train, validation)


@pytest.mark.parametrize("train,validation", [(True, 0.2), ("0.6", 0.2), (0.6, None)])
def test_nonnumeric_fractions_are_rejected(train: object, validation: object) -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        SplitConfig(train, validation)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "42", None])
def test_invalid_seeds_are_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        SplitConfig(random_seed=seed)


@pytest.mark.parametrize("count,config", [(0, SplitConfig()), (4, SplitConfig()), (20, SplitConfig(0.95, 0.01))])
def test_too_few_days_are_rejected(records: pd.DataFrame, count: int, config: SplitConfig) -> None:
    with pytest.raises(ValueError, match="Not enough distinct dates"):
        split_by_date(clean_dataset(records).iloc[:count], config)


def test_custom_ratios_allocate_remaining_days_to_test(records: pd.DataFrame) -> None:
    split = split_by_date(clean_dataset(records), SplitConfig(0.5, 0.3))
    assert [frame.date.nunique() for _, frame in split.items()] == [5, 3, 2]
