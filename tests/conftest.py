"""Small synthetic records with realistic garment feature types."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


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


@pytest.fixture
def parameters(tmp_path: Path) -> Path:
    """Write the explicit data preparation configuration."""
    path = tmp_path / "params.yaml"
    path.write_text("data:\n  train_fraction: 0.6\n  validation_fraction: 0.2\n  random_seed: 42\n")
    return path


@pytest.fixture
def monitoring_summary() -> object:
    """A persisted report contract shared by CLI and Streamlit boundary tests."""
    from garments_mlops.monitoring import MonitoringSummary

    return MonitoringSummary(
        run_id="report-run", created_at_utc="2026-11-04T00:00:00+00:00", source="live", fingerprint="f" * 64,
        reference_sha256="a" * 64, current_sha256="b" * 64, reference_rows=711, current_rows=200,
        prediction_model_versions=["2"], champion_at_report="2", source_runs=[], drift_share=.6,
        dataset_drift=True, evaluated_columns=15, columns=[],
    )
