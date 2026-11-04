"""Check reproducible demo traffic, source selection and CLI configuration."""

import json
from pathlib import Path
from unittest.mock import Mock

import httpx2
import pandas as pd
import pytest

from garments_mlops import monitor_cli
from garments_mlops.api_models import PredictRequest
from garments_mlops.data import clean_features
from garments_mlops.monitoring import MonitoringSummary
from garments_mlops.monitoring_data import NoCurrentData
from garments_mlops.training_config import TrainingConfig


@pytest.mark.parametrize("failure", [None, "count", "id", "http"])
def test_demo_is_reproducible_labeled_and_uses_the_real_api_contract(
    records: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    reference = clean_features(records.drop(columns="actual_productivity"))
    monkeypatch.setattr(monitor_cli, "load_reference", Mock(return_value=(reference, "hash")))
    requests: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        validated = PredictRequest.model_validate(body)
        assert request.url.path == "/predict"
        assert validated.source == "demo"
        assert len(validated.records) == 200
        assert validated.records[0].incentive == 105
        return httpx2.Response(503 if failure == "http" else 200, headers={"X-Request-ID": "a" * 32},
                               json={"request_id": "wrong" if failure == "id" else "a" * 32,
                                     "model_version": "2", "run_id": "prediction-run",
                                     "predictions": [] if failure == "count" else [.7] * 200})

    original_client = httpx2.Client
    monkeypatch.setattr(monitor_cli.httpx2, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx2.MockTransport(handle)))
    if failure is not None:
        with pytest.raises((RuntimeError, httpx2.HTTPStatusError)):
            monitor_cli.submit_demo(TrainingConfig(), "http://api")
        return
    result = monitor_cli.submit_demo(TrainingConfig(), "http://api")
    assert result.model_version == "2"
    monitor_cli.submit_demo(TrainingConfig(), "http://api")
    assert requests[0] == requests[1]


@pytest.mark.parametrize("existing", ["live", "demo", "empty"])
def test_initialization_prefers_live_data_and_only_seeds_an_empty_installation(
    existing: str, monitoring_summary: MonitoringSummary, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate = Mock(side_effect={"live": [monitoring_summary],
                                 "demo": [NoCurrentData(), monitoring_summary],
                                 "empty": [NoCurrentData(), NoCurrentData(), monitoring_summary]}[existing])
    demo = Mock()
    monkeypatch.setattr(monitor_cli, "generate_monitoring", generate)
    monkeypatch.setattr(monitor_cli, "submit_demo", demo)
    assert monitor_cli.initialize_monitoring(TrainingConfig(), "http://api") == monitoring_summary
    assert demo.call_count == int(existing == "empty")
    assert generate.call_args_list[0].args[1] == "live"
    if existing != "live":
        assert generate.call_args_list[-1].args[1] == "demo"


def test_cli_loads_environment_and_exports_report(
    tmp_path: Path, monitoring_summary: MonitoringSummary, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("{}")
    monkeypatch.setenv("MLOPS_TRAINING_CONFIG", str(path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert monitor_cli.service_config().tracking_uri == TrainingConfig().tracking_uri
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    assert monitor_cli.service_config().tracking_uri == "http://tracking:5000"
    generate = Mock(return_value=monitoring_summary)
    initialize = Mock(return_value=monitoring_summary)
    monkeypatch.setattr(monitor_cli, "generate_monitoring", generate)
    monkeypatch.setattr(monitor_cli, "initialize_monitoring", initialize)
    monitor_cli.main(["--source", "demo"])
    assert json.loads(capsys.readouterr().out)["run_id"] == "report-run"
    assert generate.call_args.args[1] == "demo"
    monitor_cli.main(["--demo-if-empty", "--api-url", "http://api:8000"])
    assert initialize.call_args.args[1] == "http://api:8000"
    assert json.loads(capsys.readouterr().out)["run_id"] == "report-run"
    with pytest.raises(SystemExit) as result:
        monitor_cli.main(["--help"])
    assert result.value.code == 0
