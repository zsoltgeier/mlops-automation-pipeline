"""Reject unreachable services, incomplete reports and inconsistent monitoring artifacts."""

import json
from unittest.mock import Mock

import httpx2
import pandas as pd
import pytest

from garments_mlops import system_smoke
from garments_mlops.monitoring import MonitoringSummary
from garments_mlops.training_config import TrainingConfig


@pytest.mark.parametrize("failure", [None, "http", "status", "html", "json", "rows", "source", "version", "summary"])
def test_live_monitoring_evidence_is_checked(
    monitoring_summary: MonitoringSummary, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    reports = {source: monitoring_summary.model_copy(update={"source": source, "current_rows": 2})
               for source in ["live", "demo"]}
    original_client = httpx2.Client
    monkeypatch.setattr(system_smoke.httpx2, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx2.MockTransport(lambda request: httpx2.Response(
            503 if failure == "http" else 200, text="Streamlit"))))
    client = Mock()
    client.get_run.return_value.info.status = "FAILED" if failure == "status" else "FINISHED"
    monkeypatch.setattr(system_smoke, "tracking_client", Mock(return_value=client))
    current_source = "live"

    def generate(config: TrainingConfig, source: str) -> MonitoringSummary:
        nonlocal current_source
        current_source = source
        return reports[source]

    def artifact(tracking: object, run: str, path: str) -> str:
        if path.endswith("report.html"):
            return "broken" if failure == "html" else "<html>Evidently report</html>"
        if path.endswith("report.json"):
            return json.dumps({"metrics": [] if failure == "json" else [{"value": .6}]})
        if path.endswith("summary.json"):
            return reports[current_source].model_copy(
                update={"drift_share": 0 if failure == "summary" else .6}).model_dump_json()
        return pd.DataFrame({"model_version": ["1" if failure == "version" else "2"] * 2,
                             "data_source": ["wrong" if failure == "source" else current_source] * 2,
                             }).iloc[:0 if failure == "rows" else 2].to_csv(index=False)

    monkeypatch.setattr(system_smoke, "generate_monitoring", generate)
    monkeypatch.setattr(system_smoke, "read_artifact", artifact)
    if failure is not None:
        with pytest.raises((RuntimeError, httpx2.HTTPStatusError)):
            system_smoke.verify_monitoring(TrainingConfig(), "http://dashboard")
        return
    evidence = system_smoke.verify_monitoring(TrainingConfig(), "http://dashboard")
    assert evidence.dashboard_status == evidence.dashboard_health_status == 200
    assert [report.source for report in evidence.reports] == ["live", "demo"]
