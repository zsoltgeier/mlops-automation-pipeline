"""Load verified train/validation partitions without consuming the test set."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd

from garments_mlops.data import clean_dataset
from garments_mlops.prepare import sha256


@dataclass(frozen=True)
class TrainingData:
    """Verified observations and content references for MLflow provenance."""

    train: pd.DataFrame
    validation: pd.DataFrame
    provenance: dict[str, object]


def load_training_data(directory: Path) -> TrainingData:
    """Verify hashes and temporal separation before any model is fitted."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("files_sha256"), dict):
        raise ValueError("Prepared data metadata must contain file hashes")
    source_hash = metadata.get("source_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError("Prepared data metadata must contain a source SHA-256")
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, object] = {"source_sha256": source_hash}
    for name in ("train", "validation"):
        path = directory / f"{name}.csv"
        digest = sha256(path)
        if metadata["files_sha256"].get(path.name) != digest:
            raise ValueError(f"Prepared data integrity check failed for {path.name}")
        raw = pd.read_csv(path, float_precision="round_trip")
        clean = clean_dataset(raw)
        if len(clean) != len(raw) or len(clean) < 2:
            raise ValueError("Training partitions must contain at least two unique observations")
        frames[name] = clean
        provenance[f"{name}_sha256"] = digest
        provenance[f"{name}_rows"] = len(clean)
        provenance[f"{name}_first_date"] = clean.date.min()
        provenance[f"{name}_last_date"] = clean.date.max()
    if frames["train"].date.max() >= frames["validation"].date.min():
        raise ValueError("Validation dates must follow all training dates")
    return TrainingData(frames["train"], frames["validation"], provenance)
