FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/runtime /app/data/processed \
    && chown -R app:app /app

COPY --chown=app:app pyproject.toml requirements.in training.yaml params.yaml ./
COPY --chown=app:app src ./src
COPY --chown=app:app data/garments_worker_productivity.csv ./data/garments_worker_productivity.csv
RUN python -m pip install --no-build-isolation --no-deps .

# Runtime models carry their own source; Git metadata and interactive progress bars are optional.
ENV GIT_PYTHON_REFRESH=quiet \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false

USER app
EXPOSE 8000

CMD ["uvicorn", "garments_mlops.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
