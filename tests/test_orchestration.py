"""Check the dependency-free Airflow client against the existing API contract."""

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pandas as pd
import pytest

from garments_mlops import orchestration
from garments_mlops.api_models import PredictRequest
from garments_mlops.data import clean_features


@pytest.mark.parametrize("failure", [None, "http", "body"])
def test_http_client_uses_fixed_json_contract_and_propagates_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    response = Mock()
    response.__enter__ = Mock(return_value=BytesIO(b'[]' if failure == "body" else b'{"run_id":"run"}'))
    response.__exit__ = Mock(return_value=False)
    error = HTTPError("http://api/train", 503, "unavailable", {}, None) if failure == "http" else None
    client = Mock(return_value=response, side_effect=error)
    monkeypatch.setattr(orchestration, "urlopen", client)
    monkeypatch.setenv("MLOPS_API_URL", "http://api:8000/")
    if failure is not None:
        with pytest.raises((ValueError, HTTPError)):
            orchestration.call_api("train", {"model": "random_forest"})
        return
    assert orchestration.call_api("train", {"model": "random_forest"}) == {"run_id": "run"}
    request = client.call_args.args[0]
    assert request.full_url == "http://api:8000/train"
    assert json.loads(request.data) == {"model": "random_forest"}
    assert client.call_args.kwargs["timeout"] == 180


def test_reference_payload_is_valid_target_free_and_labeled(
    records: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.csv"
    clean_features(records.drop(columns="actual_productivity")).to_csv(reference, index=False)
    samples = orchestration.sample_reference(reference, 2)
    validated = PredictRequest.model_validate({"records": samples, "source": "demo"})
    assert validated.records[1].wip is None
    assert validated.records[0].team == 1
    assert validated.records[0].no_of_workers == 30.5
    assert "month" not in samples[0]
    assert "actual_productivity" not in samples[0]
    monkeypatch.setenv("MLOPS_REFERENCE", str(reference))
    call = Mock(return_value={"run_id": "run"})
    monkeypatch.setattr(orchestration, "call_api", call)
    assert orchestration.train_challenger() == {"run_id": "run"}
    call.assert_called_with("train", {"model": "random_forest"})
    assert orchestration.predict_reference() == {"run_id": "run"}
    assert call.call_args.args[0] == "predict"
    assert call.call_args.args[1]["source"] == "demo"
    assert len(call.call_args.args[1]["records"]) == 20
    assert orchestration.monitor_demo() == {"run_id": "run"}
    call.assert_called_with("monitor", {"source": "demo"})


@pytest.mark.parametrize("failure", ["schema", "empty", "fractional", "infinite", "size"])
def test_invalid_workflow_inputs_fail_before_sending_requests(
    records: pd.DataFrame, tmp_path: Path, failure: str,
) -> None:
    frame = clean_features(records.drop(columns="actual_productivity"))
    if failure == "schema":
        frame = frame.drop(columns="smv")
    if failure == "empty":
        frame = frame.iloc[:0]
    if failure == "fractional":
        frame["team"] = "1.5"
    if failure == "infinite":
        frame["team"] = "inf"
    path = tmp_path / "reference.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        orchestration.sample_reference(path, 0 if failure == "size" else 200)
