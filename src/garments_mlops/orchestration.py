"""Small standard-library workflow client usable inside the separate Airflow environment."""

import csv
from itertools import islice
import json
from math import isfinite
import os
from pathlib import Path
from urllib.request import Request, urlopen

from garments_mlops.feature_schema import NUMERIC_FEATURES, RAW_FEATURES


def call_api(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    """Call a fixed workflow endpoint and fail the task on an HTTP or malformed response."""
    base = os.getenv("MLOPS_API_URL", "http://api:8000").rstrip("/")
    request = Request(f"{base}/{endpoint}", data=json.dumps(payload, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        document = json.load(response)
    if not isinstance(document, dict):
        raise ValueError("Workflow API response must be a JSON object")
    return document


def sample_reference(path: Path, count: int = 200) -> list[dict[str, object]]:
    """Read target-free reference inputs and preserve the API's numeric and nullable contracts."""
    if not 1 <= count <= 1000:
        raise ValueError("Workflow batch size must be between 1 and 1000")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(RAW_FEATURES).issubset(reader.fieldnames or []):
            raise ValueError("Workflow reference is missing model inputs")
        records: list[dict[str, object]] = []
        for row in islice(reader, count):
            record: dict[str, object] = {name: row[name] for name in RAW_FEATURES}
            for name in NUMERIC_FEATURES:
                record[name] = float(row[name]) if row[name] else None
            for name in ["team", "no_of_style_change"]:
                value = float(row[name])
                if not isfinite(value) or not value.is_integer():
                    raise ValueError("Workflow integer feature contains an invalid value")
                record[name] = int(value)
            records.append(record)
    if not records:
        raise ValueError("Workflow reference must not be empty")
    return records


def train_challenger() -> dict[str, object]:
    """Explicitly train a candidate; the existing validation gate decides its promotion."""
    return call_api("train", {"model": "random_forest"})


def predict_reference() -> dict[str, object]:
    """Send historical reference samples as labeled demo traffic, never as production observations."""
    path = Path(os.getenv("MLOPS_REFERENCE", "/data/reference.csv"))
    return call_api("predict", {"records": sample_reference(path), "source": "demo"})


def monitor_demo() -> dict[str, object]:
    """Generate a separate drift report after prediction, without triggering training or promotion."""
    return call_api("monitor", {"source": "demo"})
