"""Validate the live monitoring service and persisted artifacts alongside the API smoke test."""

from io import StringIO
import json

import httpx2
import pandas as pd
from pydantic import BaseModel

from garments_mlops.monitoring import MonitoringSummary, generate_monitoring, read_artifact
from garments_mlops.tracking import tracking_client
from garments_mlops.training_config import TrainingConfig


class MonitoringEvidence(BaseModel):
    """Live HTTP status and the exact report runs checked through MLflow."""

    dashboard_status: int
    dashboard_health_status: int
    reports: list[MonitoringSummary]


def verify_monitoring(config: TrainingConfig, dashboard_url: str) -> MonitoringEvidence:
    """Verify both origins, native reports and row/version consistency over the Tracking Server boundary."""
    with httpx2.Client(base_url=dashboard_url, timeout=30, trust_env=False) as http:
        health = http.get("/_stcore/health")
        health.raise_for_status()
        page = http.get("/")
        page.raise_for_status()
    client = tracking_client(config)
    reports: list[MonitoringSummary] = []
    for source in ("live", "demo"):
        report = generate_monitoring(config, source)
        if client.get_run(report.run_id).info.status != "FINISHED":
            raise RuntimeError("Monitoring run is not FINISHED")
        html = read_artifact(client, report.run_id, "monitoring/report.html")
        native = json.loads(read_artifact(client, report.run_id, "monitoring/report.json"))
        current = pd.read_csv(StringIO(read_artifact(client, report.run_id, "monitoring/current.csv")),
                              dtype={"model_version": str})
        if "<html" not in html.lower() or not native.get("metrics"):
            raise RuntimeError("Native Evidently report artifacts are missing or invalid")
        if (len(current) != report.current_rows or current.empty or not current.data_source.eq(source).all()
                or sorted(current.model_version.unique().tolist()) != report.prediction_model_versions):
            raise RuntimeError("Monitoring population differs from its summary")
        saved = MonitoringSummary.model_validate_json(read_artifact(client, report.run_id, "monitoring/summary.json"))
        if saved != report:
            raise RuntimeError("Persisted monitoring summary differs from the generated report")
        reports.append(report)
    return MonitoringEvidence(dashboard_status=page.status_code, dashboard_health_status=health.status_code,
                              reports=reports)
