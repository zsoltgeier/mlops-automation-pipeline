"""Generate persisted reports and, when needed, labeled demo requests through the real API."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import httpx2

from garments_mlops.api_models import PredictRequest, PredictResponse
from garments_mlops.data import RAW_FEATURES
from garments_mlops.monitoring import MonitoringSummary, generate_monitoring
from garments_mlops.monitoring_data import NoCurrentData, load_reference
from garments_mlops.training_config import TrainingConfig, load_training_config


def service_config() -> TrainingConfig:
    """Use the same YAML and tracking override as the API process."""
    config = load_training_config(Path(os.getenv("MLOPS_TRAINING_CONFIG", "training.yaml")))
    return replace(config, tracking_uri=os.getenv("MLFLOW_TRACKING_URI", config.tracking_uri))


def submit_demo(config: TrainingConfig, api_url: str) -> PredictResponse:
    """Send 200 reproducible shifted training-input samples, explicitly labeled as synthetic demo traffic."""
    reference, _ = load_reference(config)
    frame = reference.loc[:, RAW_FEATURES].sample(n=200, replace=True, random_state=42).reset_index(drop=True)
    for name in ["targeted_productivity", "no_of_workers"]:
        frame[name] *= .6
    for name in ["smv", "over_time"]:
        frame[name] *= 1.8
    for name, increase in {"incentive": 100, "idle_time": 15, "idle_men": 5, "wip": 2000}.items():
        frame[name] = frame[name].fillna(0) + increase
    frame["team"] = frame["team"].astype(int)
    frame["no_of_style_change"] = frame["no_of_style_change"].astype(int) + 2
    request = PredictRequest.model_validate({"records": json.loads(frame.to_json(orient="records")), "source": "demo"})
    with httpx2.Client(base_url=api_url, timeout=120, trust_env=False) as client:
        response = client.post("/predict", json=request.model_dump(mode="json"))
        response.raise_for_status()
    result = PredictResponse.model_validate(response.json())
    if len(result.predictions) != len(frame) or result.request_id != response.headers.get("X-Request-ID"):
        raise RuntimeError("Demo API response does not match the submitted batch")
    return result


def initialize_monitoring(config: TrainingConfig, api_url: str) -> MonitoringSummary:
    """Prefer live data; create demo traffic only when neither source has any completed requests."""
    try:
        return generate_monitoring(config, "live")
    except NoCurrentData:
        try:
            return generate_monitoring(config, "demo")
        except NoCurrentData:
            submit_demo(config, api_url)
            return generate_monitoring(config, "demo")


def main(arguments: list[str] | None = None) -> None:
    """Create a report from live/demo traffic; optional cold-start demo initialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["live", "demo"], default="live")
    parser.add_argument("--demo-if-empty", action="store_true")
    parser.add_argument("--api-url", default=os.getenv("MLOPS_API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args(arguments)
    config = service_config()
    result = (initialize_monitoring(config, args.api_url) if args.demo_if_empty
              else generate_monitoring(config, args.source))
    print(result.model_dump_json(indent=2))
