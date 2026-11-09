#!/usr/bin/env bash
# Build and test disposable services; preserve any developer project and its volumes.
set -euo pipefail

cd "$(dirname "$0")/.."
project="garments-check-$(date +%s)-$$"
output="${1:-runtime/system-smoke}/$project"
mkdir -p "$output"
export API_PORT=0 MLFLOW_PORT=0 DASHBOARD_PORT=0 AIRFLOW_PORT=0

compose() {
    local limit=$1
    shift
    timeout --kill-after=5s "$limit" docker compose --file compose.yaml -p "$project" "$@"
}

if ! timeout --kill-after=5s 15s docker info > "$output/engine.log" 2>&1; then
    printf 'Docker Engine is unavailable; see %s/engine.log\n' "$output" >&2
    exit 1
fi

cleanup() {
    local result=$?
    trap - EXIT
    compose 10s ps --all > "$output/services.txt" 2>&1 || true
    compose 15s logs --no-color > "$output/services.log" 2>&1 || true
    if ! compose 45s down --volumes --remove-orphans \
        > "$output/cleanup.log" 2>&1; then
        result=1
        cat "$output/cleanup.log"
    fi
    exit "$result"
}
trap cleanup EXIT

printf 'Building and starting isolated project %s\n' "$project"
if ! compose 600s up -d --wait --wait-timeout 300 \
    > "$output/build-start.log" 2>&1; then
    tail -n 80 "$output/build-start.log"
    exit 1
fi
compose 180s exec -T api garments-smoke \
    --api-url http://127.0.0.1:8000 --mlflow-url http://mlflow:5000 \
    --dashboard-url http://dashboard:8501 --output runtime/system-smoke.json \
    --airflow-url http://airflow:8080 \
    > "$output/check.log" 2>&1
compose 30s cp api:/app/runtime/system-smoke.json "$output/smoke.json"
printf 'API, MLflow, monitoring and Airflow checks passed. Evidence: %s/smoke.json\n' "$output"
