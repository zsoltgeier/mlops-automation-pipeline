"""Verify a running API, quality protection and downloadable MLflow audit data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import httpx2
from mlflow import MlflowClient
import pandas as pd
from pydantic import BaseModel

from garments_mlops.api_models import HealthResponse, PredictResponse, TrainResponse
from garments_mlops.data import RAW_FEATURES


class SmokeResult(BaseModel):
    """Evidence from serving, validation and audited training requests."""

    health: HealthResponse
    predict: PredictResponse
    train: TrainResponse
    swagger_status: int
    openapi_status: int
    validation_status: int
    checked_at_utc: str | None = None


def exercise_api(client: httpx2.Client, records: list[dict[str, object]]) -> SmokeResult:
    """Check serving, validation and rejection of a non-improving baseline."""
    health = client.get("/health")
    health.raise_for_status()
    docs = client.get("/docs")
    docs.raise_for_status()
    schema = client.get("/openapi.json")
    schema.raise_for_status()
    prediction = client.post("/predict", json={"records": records})
    prediction.raise_for_status()
    predicted = PredictResponse.model_validate(prediction.json())
    if len(predicted.predictions) != len(records) or predicted.request_id != prediction.headers["X-Request-ID"]:
        raise RuntimeError("Prediction response has inconsistent records or request correlation")
    training = client.post("/train", json={"model": "baseline"})
    training.raise_for_status()
    trained = TrainResponse.model_validate(training.json())
    if trained.decision.promoted or trained.champion_version != predicted.model_version:
        raise RuntimeError("A non-improving baseline changed the initialized champion")
    invalid = client.post("/predict", json={"records": []})
    if invalid.status_code != 422:
        raise RuntimeError("Empty prediction batches must return HTTP 422")
    return SmokeResult(health=HealthResponse.model_validate(health.json()), predict=predicted, train=trained,
                       swagger_status=docs.status_code, openapi_status=schema.status_code,
                       validation_status=invalid.status_code)


def verify_audit(tracking_uri: str, result: SmokeResult) -> None:
    """Download the actual request, response and monitoring files across the server boundary."""
    client = MlflowClient(tracking_uri=tracking_uri)
    with TemporaryDirectory(prefix="garments-smoke-") as directory:
        for index, response in enumerate((result.predict, result.train)):
            run_id = response.run_id
            if client.get_run(run_id).info.status != "FINISHED":
                raise RuntimeError("API audit run is not FINISHED")
            destination = Path(directory) / str(index)
            destination.mkdir()
            request_file = client.download_artifacts(run_id, "api/request.json", str(destination))
            response_file = client.download_artifacts(run_id, "api/response.json", str(destination))
            if json.loads(Path(request_file).read_text())["request_id"] != response.request_id:
                raise RuntimeError("Audit request correlation differs from the API response")
            if json.loads(Path(response_file).read_text()) != response.model_dump(mode="json"):
                raise RuntimeError("Audit response differs from the delivered API response")
            if isinstance(response, PredictResponse):
                current_file = client.download_artifacts(run_id, "monitoring/current.csv", str(destination))
                current = pd.read_csv(current_file)
                if len(current) != len(response.predictions) or not set(RAW_FEATURES).issubset(current.columns):
                    raise RuntimeError("Monitoring artifact does not contain the prediction inputs")


def main(argv: Sequence[str] | None = None) -> None:
    """Run a live smoke test and persist a timestamped, reproducible evidence format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000")
    parser.add_argument("--source", type=Path, default=Path("data/garments_worker_productivity.csv"))
    parser.add_argument("--output", type=Path, default=Path("runtime/smoke.json"))
    args = parser.parse_args(argv)
    features = pd.read_csv(args.source).head(3)[RAW_FEATURES].copy()
    features["date"] = pd.to_datetime(features["date"], format="mixed").dt.strftime("%Y-%m-%d")
    records = json.loads(features.to_json(orient="records"))
    with httpx2.Client(base_url=args.api_url, timeout=120, trust_env=False) as client:
        result = exercise_api(client, records)
    verify_audit(args.mlflow_url, result)
    result.checked_at_utc = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
