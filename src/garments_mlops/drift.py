"""Feature-distribution analysis only; this module never trains or promotes a model."""

import json
from math import isfinite

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
import numpy as np
import pandas as pd
from pydantic import BaseModel

from garments_mlops.data import CALENDAR_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_features


class ColumnDrift(BaseModel):
    """A drift score is unavailable when either population has no observed values."""

    column: str
    score: float | None
    drift: bool | None
    reference_missing: float
    current_missing: float


class DriftAnalysis(BaseModel):
    """Portable summary plus the native Evidently JSON and interactive HTML."""

    drift_share: float
    dataset_drift: bool
    evaluated_columns: int
    columns: list[ColumnDrift]
    html: str
    report_json: str


def analyze_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftAnalysis:
    """Compare observed feature values with fixed Jensen-Shannon settings, retaining missingness separately."""
    names = [*NUMERIC_FEATURES, *CALENDAR_FEATURES, *CATEGORICAL_FEATURES]
    ref = clean_features(reference).loc[:, names]
    cur = clean_features(current).loc[:, names]
    available = [name for name in names if ref[name].notna().any() and cur[name].notna().any()]
    definition = DataDefinition(
        numerical_columns=[name for name in available if name not in CATEGORICAL_FEATURES],
        categorical_columns=[name for name in available if name in CATEGORICAL_FEATURES],
    )
    # Evidently also computes correlations, undefined for valid constant columns (e.g. idle_men).
    # Suppress only NumPy divide/invalid signals here; nonfinite drift scores are rejected below.
    with np.errstate(invalid="ignore", divide="ignore"):
        snapshot = Report([DataDriftPreset(columns=available, method="jensenshannon", threshold=0.1, drift_share=0.5)],
                          include_tests=True).run(
            Dataset.from_pandas(cur.loc[:, available], data_definition=definition),
            Dataset.from_pandas(ref.loc[:, available], data_definition=definition),
        )
    document = json.loads(snapshot.json())
    scores = {metric["config"]["column"]: float(metric["value"]) for metric in document["metrics"]
              if "column" in metric["config"]}
    if not all(isfinite(score) for score in scores.values()):
        raise ValueError("Evidently returned a nonfinite drift score")
    columns = [ColumnDrift(column=name, score=scores.get(name),
                           drift=scores[name] >= 0.1 if name in scores else None,
                           reference_missing=float(ref[name].isna().mean()),
                           current_missing=float(cur[name].isna().mean())) for name in names]
    share = sum(item.drift is True for item in columns) / len(available)
    return DriftAnalysis(drift_share=share, dataset_drift=share >= 0.5, evaluated_columns=len(available),
                         columns=columns, html=snapshot.get_html_str(as_iframe=False), report_json=snapshot.json())
