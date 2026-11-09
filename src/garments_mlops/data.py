"""Stateless cleaning of garment productivity observations."""

import numpy as np
import pandas as pd

from garments_mlops.feature_schema import (
    CALENDAR_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, TARGET,
)

__all__ = ["CALENDAR_FEATURES", "CATEGORICAL_FEATURES", "NUMERIC_FEATURES", "RAW_FEATURES", "TARGET",
           "clean_features", "clean_dataset"]


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize records without estimating any population statistics."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if TARGET in frame.columns:
        raise ValueError("Target must not be included in model inputs")
    if frame.empty:
        raise ValueError("At least one observation is required")
    result = frame.loc[:, RAW_FEATURES].copy().reset_index(drop=True)
    date_text = result["date"].astype("string")
    valid_dates = date_text.str.fullmatch(r"(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})").fillna(False)
    if not valid_dates.all():
        raise ValueError("Dates must be nonmissing ISO dates or month/day/year strings")
    dates = pd.to_datetime(date_text, format="mixed", errors="raise")
    result["date"] = dates.dt.strftime("%Y-%m-%d")
    result["month"] = dates.dt.month
    result["day_of_month"] = dates.dt.day
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(result[name], errors="raise").astype(float)
        if np.isinf(values).any() or values.dropna().lt(0).any():
            raise ValueError(f"{name} must contain finite nonnegative values or null")
        result[name] = values
    team = pd.to_numeric(result["team"], errors="raise").astype(float)
    if np.isinf(team).any() or team.dropna().le(0).any() or team.dropna().mod(1).ne(0).any():
        raise ValueError("team must contain positive integers or null")
    result["team"] = team.map(lambda value: str(int(value)) if pd.notna(value) else np.nan)
    for name in CATEGORICAL_FEATURES:
        text = result[name].astype("string").str.strip().replace("", pd.NA)
        if name == "department":
            text = text.replace({"sweing": "sewing"})
        result[name] = text.astype(object).where(text.notna(), np.nan)
    return result


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate labels, deduplicate normalized records and order observations."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    features = clean_features(frame.drop(columns=TARGET))
    target = pd.to_numeric(frame[TARGET], errors="raise").astype(float)
    if not np.isfinite(target).all() or target.lt(0).any():
        raise ValueError("Target must contain finite nonnegative values")
    features[TARGET] = target.to_numpy()
    return features.drop_duplicates().sort_values("date", kind="stable").reset_index(drop=True)
