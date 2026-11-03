"""Test smoke evidence checks without depending on a running Docker daemon."""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import Mock

import httpx2
import pandas as pd
import pytest

from garments_mlops import smoke
from garments_mlops.api_models import HealthResponse, PredictResponse, TrainResponse
from garments_mlops.data import RAW_FEATURES
from garments_mlops.modeling import decide_promotion


@pytest.fixture
def evidence() -> smoke.SmokeResult:
    return smoke.SmokeResult(
        health=HealthResponse(status="ready", model_version="2"),
        predict=PredictResponse(request_id="b" * 32, model_version="2", run_id="prediction", predictions=[.7]),
        train=TrainResponse(request_id="a" * 32, run_id="training", model_version="3", model_uri="models:/model/3",
                            champion_version="2", metrics={"mae": .2}, decision=decide_promotion(.2, .1)),
        swagger_status=200, openapi_status=200, validation_status=422,
    )


@pytest.mark.parametrize("failure", [None, "count", "correlation", "promotion", "validation", "http_error"])
def test_smoke_checks_actual_http_response_semantics(evidence: smoke.SmokeResult, failure: str | None) -> None:
    prediction = evidence.predict.model_dump(mode="json")
    training = evidence.train.model_dump(mode="json")
    if failure == "count":
        prediction["predictions"] = []
    if failure == "correlation":
        prediction["request_id"] = "wrong"
    if failure == "promotion":
        training["decision"] = asdict(decide_promotion(.1, .2))

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/predict":
            if not json.loads(request.content)["records"]:
                return httpx2.Response(200 if failure == "validation" else 422, json={})
            return httpx2.Response(200, json=prediction, headers={"X-Request-ID": "b" * 32})
        responses = {
            "/health": httpx2.Response(503 if failure == "http_error" else 200, json=evidence.health.model_dump()),
            "/docs": httpx2.Response(200, text="Swagger"),
            "/openapi.json": httpx2.Response(200, json={"paths": {}}),
            "/train": httpx2.Response(200, json=training),
        }
        return responses[request.url.path]

    with httpx2.Client(base_url="http://service", transport=httpx2.MockTransport(handle)) as client:
        if failure is None:
            assert smoke.exercise_api(client, [{}]) == evidence
        else:
            with pytest.raises((RuntimeError, httpx2.HTTPStatusError)):
                smoke.exercise_api(client, [{}])


@pytest.mark.parametrize("failure", [None, "status", "request", "response", "rows", "columns"])
def test_audit_download_and_monitoring_checks(
    evidence: smoke.SmokeResult, failure: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[tuple[str, str], str] = {}
    for response in [evidence.predict, evidence.train]:
        files[(response.run_id, "api/request.json")] = json.dumps({"request_id": response.request_id})
        files[(response.run_id, "api/response.json")] = response.model_dump_json()
    current = pd.DataFrame([{key: 0 for key in RAW_FEATURES}])
    if failure == "request":
        files[("prediction", "api/request.json")] = json.dumps({"request_id": "wrong"})
    if failure == "response":
        files[("prediction", "api/response.json")] = "{}"
    if failure == "rows":
        current = current.iloc[:0]
    if failure == "columns":
        current = pd.DataFrame({"wrong": [0]})
    files[("prediction", "monitoring/current.csv")] = current.to_csv(index=False)

    def download(run_id: str, artifact: str, directory: str) -> str:
        path = Path(directory) / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[(run_id, artifact)])
        return str(path)

    client = Mock()
    client.get_run.return_value.info.status = "FAILED" if failure == "status" else "FINISHED"
    client.download_artifacts.side_effect = download
    monkeypatch.setattr(smoke, "MlflowClient", Mock(return_value=client))
    if failure is None:
        smoke.verify_audit("http://tracking", evidence)
    else:
        with pytest.raises(RuntimeError):
            smoke.verify_audit("http://tracking", evidence)


def test_cli_exports_valid_sample_records_and_evidence(
    evidence: smoke.SmokeResult, records: pd.DataFrame, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "results/smoke.json"
    records.to_csv(source, index=False)
    exercise = Mock(return_value=evidence)
    audit = Mock()
    monkeypatch.setattr(smoke, "exercise_api", exercise)
    monkeypatch.setattr(smoke, "verify_audit", audit)
    smoke.main(["--source", str(source), "--output", str(output)])
    submitted = exercise.call_args.args[1]
    assert len(submitted) == 3
    assert submitted[0]["date"] == "2015-01-01"
    assert submitted[1]["wip"] is None
    assert set(submitted[0]) == set(RAW_FEATURES)
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text())["checked_at_utc"] is not None
    audit.assert_called_once_with("http://127.0.0.1:5000", evidence)
    with pytest.raises(SystemExit) as result:
        smoke.main(["--help"])
    assert result.value.code == 0
