"""Keep entire observation dates in reproducible, disjoint partitions."""

from dataclasses import dataclass
from math import isfinite

import pandas as pd


def _validate_fraction(value: object) -> float:
    """Validate runtime YAML values before arithmetic uses them."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Split fractions must be numeric")
    if not isfinite(value) or value <= 0:
        raise ValueError("Split fractions must be finite, positive and sum to less than one")
    return float(value)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration shared by dataset preparation and future training."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    random_seed: int = 42

    def __post_init__(self) -> None:
        fractions = (_validate_fraction(self.train_fraction), _validate_fraction(self.validation_fraction))
        if sum(fractions) >= 1:
            raise ValueError("Split fractions must be finite, positive and sum to less than one")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer between 0 and 2**32 - 1")


@dataclass(frozen=True)
class DatasetSplits:
    """Chronological partitions containing validated observations and labels."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        """Return datasets in chronological order."""
        return [("train", self.train), ("validation", self.validation), ("test", self.test)]


def split_by_date(frame: pd.DataFrame, config: SplitConfig = SplitConfig()) -> DatasetSplits:
    """Split cleaned ISO-date records without randomizing future observations."""
    ordered = frame.sort_values("date", kind="stable")
    days = sorted(ordered["date"].unique())
    train_end = int(len(days) * config.train_fraction)
    validation_end = train_end + int(len(days) * config.validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end >= len(days):
        raise ValueError("Not enough distinct dates for nonempty train, validation and test sets")
    train = ordered[ordered["date"].isin(days[:train_end])].reset_index(drop=True)
    validation = ordered[ordered["date"].isin(days[train_end:validation_end])].reset_index(drop=True)
    test = ordered[ordered["date"].isin(days[validation_end:])].reset_index(drop=True)
    return DatasetSplits(train, validation, test)
