"""MLflow experiment setup and version-pinned champion resolution."""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from garments_mlops.training_config import TrainingConfig


class MissingChampionError(LookupError):
    """Raised when no production model has been promoted yet."""


@dataclass(frozen=True)
class Champion:
    """A model and the exact Registry version used to load it."""

    version: str
    model: Pipeline


def tracking_client(config: TrainingConfig) -> MlflowClient:
    """Configure the SDK and ensure the local SQLite parent exists."""
    if config.tracking_uri.startswith("sqlite:///"):
        database = config.tracking_uri[10:].partition("?")[0]
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.tracking_uri)
    return MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)


def experiment_id(client: MlflowClient, config: TrainingConfig) -> str:
    """Reuse active experiments and let remote servers select artifact storage."""
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        if experiment.lifecycle_stage != "active":
            raise ValueError("Training experiment is deleted; restore it or choose a new name")
        return experiment.experiment_id
    location = None
    if config.tracking_uri.startswith("sqlite:///"):
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        location = config.artifact_dir.resolve().as_uri()
    try:
        return client.create_experiment(config.experiment_name, artifact_location=location)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        created = client.get_experiment_by_name(config.experiment_name)
        if created is None or created.lifecycle_stage != "active":
            raise
        return created.experiment_id


def champion_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    """Treat only a missing Registry resource as an absent champion."""
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def load_champion(config: TrainingConfig) -> Champion:
    """Resolve the alias on every call and pin the loaded model to that version."""
    client = tracking_client(config)
    version = champion_version(client, config.model_name)
    if version is None:
        raise MissingChampionError("No champion model is registered")
    model = mlflow_sklearn.load_model(f"models:/{config.model_name}/{version.version}")
    if not isinstance(model, Pipeline):
        raise TypeError("Champion artifact must be a complete sklearn Pipeline")
    return Champion(str(version.version), model)
