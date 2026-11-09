"""Prove scheduler evidence rejects failed, stalled and incomplete DAG runs."""

import httpx2
import pytest

from garments_mlops import airflow_smoke


@pytest.mark.parametrize("failure", [None, "unpaused", "http", "imports", "failed", "timeout", "missing", "task"])
def test_scheduler_run_and_task_evidence(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    polls = 0
    unpauses = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal polls, unpauses
        if request.url.path.endswith("importErrors"):
            return httpx2.Response(503 if failure == "http" else 200,
                                   json={"total_entries": 1 if failure == "imports" else 0})
        if request.method == "PATCH":
            unpauses += 1
            return httpx2.Response(200, json={})
        if request.method == "POST":
            return httpx2.Response(200, json={})
        if request.url.path.endswith(airflow_smoke.DAG_ID):
            return httpx2.Response(200, json={"is_paused": failure != "unpaused"})
        if request.url.path.endswith("taskInstances"):
            tasks = [] if failure == "missing" else [
                {"task_id": task, "state": "failed" if failure == "task" else "success"}
                for task in airflow_smoke.TASK_IDS]
            return httpx2.Response(200, json={"task_instances": tasks})
        polls += 1
        state = "failed" if failure == "failed" else "success"
        return httpx2.Response(200, json={"state": "running" if polls == 1 or failure == "timeout" else state})

    original = httpx2.Client
    monkeypatch.setattr(airflow_smoke.httpx2, "Client", lambda **kwargs: original(
        **kwargs, transport=httpx2.MockTransport(handle)))
    monkeypatch.setattr(airflow_smoke, "sleep", lambda seconds: None)
    if failure not in {None, "unpaused"}:
        with pytest.raises((RuntimeError, TimeoutError, httpx2.HTTPStatusError)):
            airflow_smoke.verify_airflow("http://airflow", attempts=2)
        return
    result = airflow_smoke.verify_airflow("http://airflow")
    assert result.state == "success" and set(result.tasks) == airflow_smoke.TASK_IDS
    assert unpauses == int(failure != "unpaused")
