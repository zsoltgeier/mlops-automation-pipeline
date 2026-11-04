"""Verified training reference and bounded, provenance-preserving prediction windows."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from mlflow import MlflowClient
import numpy as np
import pandas as pd

from garments_mlops.data import RAW_FEATURES, clean_features
from garments_mlops.training_config import TrainingConfig

DataSource = Literal["live", "demo"]


class NoCurrentData(ValueError):
    """No completed prediction requests are available for the selected source."""


@dataclass(frozen=True)
class CurrentWindow:
    """Newest completed request batches, with exact source artifact hashes."""

    frame: pd.DataFrame
    manifest: list[dict[str, str | int]]
    source: DataSource


def load_reference(config: TrainingConfig) -> tuple[pd.DataFrame, str]:
    """Check the DVC-produced reference against its preprocessing provenance."""
    content = (config.data_dir / "reference.csv").read_bytes()
    digest = sha256(content).hexdigest()
    metadata = json.loads((config.data_dir / "metadata.json").read_text(encoding="utf-8"))
    if digest != metadata["files_sha256"]["reference.csv"]:
        raise ValueError("Reference hash differs from preprocessing metadata")
    return clean_features(pd.read_csv(BytesIO(content))), digest


def collect_current(
    client: MlflowClient, config: TrainingConfig, source: DataSource, *, max_runs: int = 50, max_rows: int = 5000,
) -> CurrentWindow:
    """Select one source from the latest completed requests, never silently skipping corrupt artifacts."""
    if source not in {"live", "demo"} or not 1 <= max_runs <= 100 or not 1 <= max_rows <= 10000:
        raise ValueError("Invalid monitoring source or window bounds")
    experiment = client.get_experiment_by_name(f"{config.experiment_name}-requests")
    if experiment is None:
        raise NoCurrentData("No prediction experiment exists")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' AND tags.endpoint = 'predict' "
                      f"AND tags.model_name = '{config.model_name}'",
        order_by=["attributes.start_time DESC", "attributes.run_id DESC"], max_results=max_runs,
    )
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, str | int]] = []
    remaining = max_rows
    for run in runs:
        if run.data.tags.get("data_source", "live") != source:
            continue
        with TemporaryDirectory() as directory:
            path = client.download_artifacts(run.info.run_id, "monitoring/current.csv", directory)
            content = Path(path).read_bytes()
        frame = pd.read_csv(BytesIO(content), dtype={"request_id": str, "model_version": str})
        clean_features(frame)
        if (not frame.request_id.eq(run.data.tags["request_id"]).all()
                or not frame.model_version.eq(run.data.tags["model_version"]).all()
                or not np.isfinite(frame.prediction).all()
                or ("data_source" in frame and not frame.data_source.eq(source).all())):
            raise ValueError("Prediction artifact does not match its audit run")
        selected = frame.iloc[:remaining].copy()
        selected["data_source"] = source
        frames.append(selected.loc[:, [*RAW_FEATURES, "prediction", "request_id", "model_version", "data_source"]])
        manifest.append({"run_id": run.info.run_id, "sha256": sha256(content).hexdigest(), "rows": len(selected)})
        remaining -= len(selected)
        if remaining == 0:
            break
    if not frames:
        raise NoCurrentData("No completed prediction requests exist for this source")
    return CurrentWindow(pd.concat(frames, ignore_index=True), manifest, source)
