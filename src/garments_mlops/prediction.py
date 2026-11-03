"""Version-pinned batch inference with MLflow request and monitoring artifacts."""

from dataclasses import replace

import mlflow
import numpy as np
import pandas as pd

from garments_mlops.api_models import PredictRequest, PredictResponse
from garments_mlops.tracking import experiment_id, load_champion, tracking_client
from garments_mlops.training_config import TrainingConfig


class PredictionOutputError(RuntimeError):
    """A registered model returned malformed or nonfinite predictions."""


def predict_records(config: TrainingConfig, request: PredictRequest, request_id: str) -> PredictResponse:
    """Return predictions only after their validated inputs and results are logged."""
    records = [record.model_dump(mode="json") for record in request.records]
    features = pd.DataFrame(records)
    champion = load_champion(config)
    values = np.asarray(champion.model.predict(features), dtype=float)
    if values.shape != (len(records),) or not np.isfinite(values).all():
        raise PredictionOutputError("Champion returned invalid predictions")
    client = tracking_client(config)
    audit_config = replace(config, experiment_name=f"{config.experiment_name}-requests")
    experiment = experiment_id(client, audit_config)
    with mlflow.start_run(experiment_id=experiment, run_name="predict") as run:
        response = PredictResponse(request_id=request_id, model_version=champion.version,
                                   run_id=run.info.run_id, predictions=values.tolist())
        mlflow.set_tags({"endpoint": "predict", "request_id": request_id,
                         "model_name": config.model_name, "model_version": champion.version})
        mlflow.log_dict({"request_id": request_id, "records": records}, "api/request.json")
        mlflow.log_dict(response.model_dump(mode="json"), "api/response.json")
        current = features.assign(prediction=values, request_id=request_id, model_version=champion.version)
        mlflow.log_text(current.to_csv(index=False), "monitoring/current.csv")
        mlflow.log_metric("record_count", len(records))
    return response
