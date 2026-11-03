"""FastAPI application with correlated, sanitized service errors and access logs."""

from dataclasses import asdict, replace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from filelock import Timeout
from mlflow.exceptions import MlflowException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.concurrency import run_in_threadpool

from garments_mlops.api_models import HealthResponse, PredictRequest, PredictResponse, TrainRequest, TrainResponse
from garments_mlops.bootstrap import initialize_system
from garments_mlops.prediction import PredictionOutputError, predict_records
from garments_mlops.tracking import MissingChampionError, champion_version, tracking_client
from garments_mlops.training import train_candidate
from garments_mlops.training_config import TrainingConfig, load_training_config

LOGGER = logging.getLogger(__name__)


def _error_response(request: Request, error: Exception) -> JSONResponse:
    """Expose stable messages without serializing backend URLs or raw input values."""
    if isinstance(error, Timeout):
        status, message = 409, "Training is already in progress"
    elif isinstance(error, MissingChampionError):
        status, message = 503, "No champion model is available"
    elif isinstance(error, PredictionOutputError):
        status, message = 500, "Champion returned invalid predictions"
    elif isinstance(error, (MlflowException, OSError, ValueError)):
        status, message = 503, "Configured model data or tracking storage is unavailable"
    else:
        status, message = 500, "Internal service error"
    request_id = str(request.state.request_id)
    LOGGER.error(json.dumps({"event": "request_failed", "request_id": request_id,
                             "status_code": status, "error_type": type(error).__name__}))
    return JSONResponse(status_code=status, content={"detail": message, "request_id": request_id})


def create_app(config: TrainingConfig | None = None, *, bootstrap: bool | None = None) -> FastAPI:
    """Create an application for one configured Tracking Server and Registry."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    if config is None:
        config = load_training_config(Path(os.getenv("MLOPS_TRAINING_CONFIG", "training.yaml")))
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri is not None:
            config = replace(config, tracking_uri=tracking_uri)
    settings = config
    if bootstrap is None:
        flag = os.getenv("MLOPS_BOOTSTRAP", "true").lower()
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError("MLOPS_BOOTSTRAP must be true, false, 1 or 0")
        bootstrap = flag in {"true", "1"}

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if bootstrap:
            application.state.bootstrap_version = await run_in_threadpool(initialize_system, settings)
        yield

    app = FastAPI(title="Garment Productivity API", version="0.1.0", lifespan=lifespan)
    app.state.training_config = settings

    async def correlate(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = uuid4().hex
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            response = _error_response(request, error)
        response.headers["X-Request-ID"] = request.state.request_id
        path = request.url.path
        endpoint = path if path in {"/train", "/predict", "/health", "/docs", "/openapi.json", "/redoc"} else "other"
        LOGGER.info(json.dumps({"event": "request_completed", "request_id": request.state.request_id,
                                "method": request.method, "endpoint": endpoint, "status_code": response.status_code,
                                "elapsed_ms": round((perf_counter() - started) * 1000, 3)}))
        return response

    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        errors = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": errors, "request_id": request.state.request_id})

    def health(response: Response) -> HealthResponse:
        try:
            active = champion_version(tracking_client(settings), settings.model_name)
        except Exception as error:
            LOGGER.warning(json.dumps({"event": "readiness_failed", "error_type": type(error).__name__}))
            active = None
        if active is None:
            response.status_code = 503
            return HealthResponse(status="unavailable")
        return HealthResponse(status="ready", model_version=str(active.version))

    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        return predict_records(settings, payload, str(request.state.request_id))

    def train(payload: TrainRequest, request: Request) -> TrainResponse:
        request_id = str(request.state.request_id)
        result = train_candidate(replace(settings, lock_timeout=0), payload.model, request_id=request_id)
        return TrainResponse(request_id=request_id, **asdict(result))

    app.middleware("http")(correlate)
    app.exception_handler(RequestValidationError)(invalid_request)
    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/predict", predict, methods=["POST"], response_model=PredictResponse)
    app.add_api_route("/train", train, methods=["POST"], response_model=TrainResponse)
    return app
