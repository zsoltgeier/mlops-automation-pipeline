"""Exercise a real scheduler-run Airflow DAG over its local REST API."""

from time import sleep
from uuid import uuid4

import httpx2
from pydantic import BaseModel

DAG_ID = "garment_productivity"
TASK_IDS = {"train_challenger", "predict_reference", "monitor_demo"}


class AirflowEvidence(BaseModel):
    """Exact scheduler run and successful task states used as final evidence."""

    dag_id: str
    run_id: str
    state: str
    tasks: dict[str, str]


def verify_airflow(url: str, *, attempts: int = 60) -> AirflowEvidence:
    """Trigger the configured workflow, wait for completion and reject missing or failed tasks."""
    run_id = f"smoke-{uuid4().hex}"
    prefix = f"/api/v2/dags/{DAG_ID}"
    with httpx2.Client(base_url=url, timeout=15, trust_env=False) as client:
        errors = client.get("/api/v2/importErrors")
        errors.raise_for_status()
        if errors.json()["total_entries"]:
            raise RuntimeError("Airflow reports DAG import errors")
        dag = client.get(prefix)
        dag.raise_for_status()
        if dag.json()["is_paused"]:
            unpause = client.patch(prefix, json={"is_paused": False})
            unpause.raise_for_status()
        trigger = client.post(f"{prefix}/dagRuns", json={"dag_run_id": run_id, "logical_date": None, "conf": {}})
        trigger.raise_for_status()
        for _ in range(attempts):
            response = client.get(f"{prefix}/dagRuns/{run_id}")
            response.raise_for_status()
            state = response.json()["state"]
            if state == "failed":
                raise RuntimeError("Airflow workflow failed")
            if state == "success":
                break
            sleep(2)
        else:
            raise TimeoutError("Airflow workflow did not complete within the polling limit")
        response = client.get(f"{prefix}/dagRuns/{run_id}/taskInstances")
        response.raise_for_status()
        tasks = {item["task_id"]: item["state"] for item in response.json()["task_instances"]}
        if set(tasks) != TASK_IDS or any(value != "success" for value in tasks.values()):
            raise RuntimeError("Airflow task results are incomplete or unsuccessful")
    return AirflowEvidence(dag_id=DAG_ID, run_id=run_id, state="success", tasks=tasks)
