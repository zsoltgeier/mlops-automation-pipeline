"""Persist reproducible feature-drift reports without modifying model lifecycle state."""

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel

from garments_mlops.drift import ColumnDrift, analyze_drift
from garments_mlops.monitoring_data import DataSource, collect_current, load_reference
from garments_mlops.tracking import champion_version, experiment_id, tracking_client
from garments_mlops.training_config import TrainingConfig


class MonitoringSummary(BaseModel):
    """Persisted report identity, source populations and interpretable drift results."""

    run_id: str
    created_at_utc: str
    source: DataSource
    fingerprint: str
    reference_sha256: str
    current_sha256: str
    reference_rows: int
    current_rows: int
    prediction_model_versions: list[str]
    champion_at_report: str | None
    source_runs: list[dict[str, str | int]]
    drift_share: float
    dataset_drift: bool
    evaluated_columns: int
    columns: list[ColumnDrift]


def read_artifact(client: MlflowClient, run_id: str, artifact: str) -> str:
    """Download text without relying on shared host/container artifact paths."""
    with TemporaryDirectory() as directory:
        path = client.download_artifacts(run_id, artifact, directory)
        return Path(path).read_text(encoding="utf-8")


def generate_monitoring(config: TrainingConfig, source: DataSource = "live") -> MonitoringSummary:
    """Reuse an identical completed report or log native reports, populations and provenance."""
    client = tracking_client(config)
    reference, reference_hash = load_reference(config)
    window = collect_current(client, config, source)
    csv = window.frame.to_csv(index=False)
    current_hash = sha256(csv.encode()).hexdigest()
    identity = {"reference": reference_hash, "current": current_hash, "runs": window.manifest,
                "source": source, "analysis": "evidently-0.7.23-js0.1-share0.5-v1"}
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    experiment = experiment_id(client, replace(config, experiment_name=f"{config.experiment_name}-monitoring"))
    query = f"attributes.status = 'FINISHED' AND tags.fingerprint = '{fingerprint}'"
    previous = client.search_runs([experiment], filter_string=query, max_results=1)
    if previous:
        return MonitoringSummary.model_validate_json(
            read_artifact(client, previous[0].info.run_id, "monitoring/summary.json"))
    analysis = analyze_drift(reference, window.frame)
    active = champion_version(client, config.model_name)
    with mlflow.start_run(experiment_id=experiment, run_name=f"drift-{source}") as run:
        summary = MonitoringSummary(
            run_id=run.info.run_id, created_at_utc=datetime.now(timezone.utc).isoformat(), source=source,
            fingerprint=fingerprint, reference_sha256=reference_hash, current_sha256=current_hash,
            reference_rows=len(reference), current_rows=len(window.frame),
            prediction_model_versions=sorted(window.frame.model_version.unique().tolist()),
            champion_at_report=str(active.version) if active is not None else None, source_runs=window.manifest,
            drift_share=analysis.drift_share, dataset_drift=analysis.dataset_drift,
            evaluated_columns=analysis.evaluated_columns, columns=analysis.columns,
        )
        mlflow.set_tags({"operation": "data_drift", "data_source": source, "fingerprint": fingerprint,
                         "model_name": config.model_name})
        mlflow.log_params({"method": "jensenshannon", "threshold": .1, "dataset_drift_share": .5,
                           "reference_sha256": reference_hash, "current_sha256": current_hash})
        mlflow.log_metrics({"drift_share": analysis.drift_share, "dataset_drift": int(analysis.dataset_drift),
                            "reference_rows": len(reference), "current_rows": len(window.frame)})
        mlflow.log_text(analysis.html, "monitoring/report.html")
        mlflow.log_text(analysis.report_json, "monitoring/report.json")
        mlflow.log_dict(summary.model_dump(mode="json"), "monitoring/summary.json")
        mlflow.log_text(reference.to_csv(index=False), "monitoring/reference.csv")
        mlflow.log_text(csv, "monitoring/current.csv")
    return summary


def training_overview(client: MlflowClient, config: TrainingConfig) -> list[dict[str, str | float | None]]:
    """Expose recent completed training results, without interpreting drift as model accuracy."""
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=20)
    return [{"run_id": run.info.run_id, "model": run.data.params.get("kind"),
             "validation_mae": run.data.metrics.get("validation_mae"),
             "promotion": run.data.tags.get("promotion_reason")} for run in runs]
