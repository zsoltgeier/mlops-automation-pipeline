"""Explicit configuration for local tracking and serialized model training."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
import re

import yaml

from garments_mlops.modeling import ModelKind, ModelSpec


def _nonnegative(value: object, name: str) -> None:
    """Validate scalar values loaded from YAML before using them as policy."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _text(value: object, name: str) -> str:
    """Require nonblank strings at configuration boundaries."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    """Keep one durable local default while supporting a remote Tracking Server."""

    tracking_uri: str = "sqlite:///runtime/mlflow.db"
    artifact_dir: Path = Path("runtime/artifacts")
    data_dir: Path = Path("data/processed")
    experiment_name: str = "garment-productivity-training"
    model_name: str = "garment-productivity"
    minimum_improvement: float = 0.0
    lock_path: Path = Path("runtime/training.lock")
    lock_timeout: float = 30.0
    random_seed: int = 42
    n_estimators: int = 100
    max_depth: int = 8
    min_samples_leaf: int = 3

    def __post_init__(self) -> None:
        uri = _text(self.tracking_uri, "tracking_uri")
        if not uri.startswith(("sqlite:///", "http://", "https://")):
            raise ValueError("tracking_uri must use SQLite or HTTP(S)")
        if self.tracking_uri.startswith("sqlite:///") and self.tracking_uri[10:] in ("", ":memory:"):
            raise ValueError("SQLite tracking must use a persistent database file")
        _text(self.experiment_name, "experiment_name")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", _text(self.model_name, "model_name")) is None:
            raise ValueError("model_name must contain letters, numbers, underscores, dots or hyphens")
        _nonnegative(self.minimum_improvement, "minimum_improvement")
        _nonnegative(self.lock_timeout, "lock_timeout")
        self.model_spec("random_forest")

    def model_spec(self, kind: ModelKind) -> ModelSpec:
        """Return validated parameters for a single candidate model."""
        return ModelSpec(kind, self.random_seed, self.n_estimators, self.max_depth, self.min_samples_leaf)


def load_training_config(path: Path) -> TrainingConfig:
    """Read a flat YAML document and reject misspelled configuration keys."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) - {field.name for field in fields(TrainingConfig)}:
        raise ValueError("Training configuration must be a mapping of known settings")
    for name in ("artifact_dir", "data_dir", "lock_path"):
        if name in document:
            value = document[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
            document[name] = Path(value)
    return TrainingConfig(**document)
