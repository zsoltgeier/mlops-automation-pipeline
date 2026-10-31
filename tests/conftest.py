"""Small synthetic records with realistic garment feature types."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def records() -> pd.DataFrame:
    """Return two observations per day over ten complete days."""
    dates = pd.date_range("2015-01-01", periods=10)
    return pd.DataFrame({
        "date": np.repeat(dates.strftime("%m/%d/%Y"), 2),
        "quarter": ["Quarter1"] * 20,
        "department": ["sweing", "finishing "] * 10,
        "day": np.repeat(dates.day_name(), 2),
        "team": [1, 2] * 10,
        "targeted_productivity": [0.8] * 20,
        "smv": np.arange(20, dtype=float) + 1,
        "wip": [100.0, np.nan] * 10,
        "over_time": [60.0] * 20,
        "incentive": [5.0] * 20,
        "idle_time": [0.0] * 20,
        "idle_men": [0.0] * 20,
        "no_of_style_change": [0] * 20,
        "no_of_workers": [30.5] * 20,
        "actual_productivity": np.linspace(0.4, 1.12, 20),
    })
