"""Thin Airflow SDK DAG; the tested HTTP workflow client lives under src/."""

from datetime import datetime, timezone
import os

from airflow.sdk import dag, task

from garments_mlops.orchestration import monitor_demo, predict_reference, train_challenger


@dag(dag_id="garment_productivity", schedule=os.getenv("MLOPS_AIRFLOW_SCHEDULE") or None,
     start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), catchup=False, max_active_runs=1,
     is_paused_upon_creation=False,
     default_args={"retries": 0}, tags=["mlops", "garments"])
def productivity_workflow() -> None:
    """Train through the quality gate, predict labeled samples, then persist an independent drift report."""
    train = task(train_challenger)()
    predict = task(predict_reference)()
    monitor = task(monitor_demo)()
    train >> predict >> monitor


productivity_workflow()
