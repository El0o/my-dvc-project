FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ARG MLFLOW_TRACKING_URI
ARG MLFLOW_EXPERIMENT_NAME
ARG MLFLOW_RUN_NAME

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME}
ENV MLFLOW_RUN_NAME=${MLFLOW_RUN_NAME}

RUN --mount=type=secret,id=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=MLFLOW_TRACKING_PASSWORD \
    MLFLOW_ALLOW_MODEL_CODE_EXECUTION=true python - <<'PY'

import os
import mlflow
import mlflow.sklearn
import joblib

from mlflow.tracking import MlflowClient

# Read temporary Docker build secrets
with open("/run/secrets/MLFLOW_TRACKING_USERNAME") as f:
    os.environ["MLFLOW_TRACKING_USERNAME"] = f.read().strip()

with open("/run/secrets/MLFLOW_TRACKING_PASSWORD") as f:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = f.read().strip()

# Configure MLflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

client = MlflowClient()

experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
run_name = os.environ["MLFLOW_RUN_NAME"]

# Find experiment
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise RuntimeError(f"Experiment not found: {experiment_name}")

# Find latest finished run with matching name
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        f"tags.mlflow.runName = '{run_name}' "
        "and attributes.status = 'FINISHED'"
    ),
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError(f"No finished run found: {run_name}")

run = runs[0]

print(f"Selected run: {run.info.run_id}")

# Find Logged Models associated with this run
logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id = '{run.info.run_id}'",
    output_format="list"
)

# Keep only successfully logged models
ready_models = [
    model for model in logged_models
    if model.status == "READY"
]

if not ready_models:
    raise RuntimeError(
        f"No ready Logged Model found for run {run.info.run_id}"
    )

if len(ready_models) > 1:
    raise RuntimeError(
        f"Expected one Logged Model, found {len(ready_models)}"
    )

logged_model = ready_models[0]

print(f"Selected Logged Model: {logged_model.model_id}")

# Retrieve model using its MLflow 3 URI
model_uri = f"models:/{logged_model.model_id}"

print(f"Downloading model from: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

# Save local copy for FastAPI
joblib.dump(model, "/app/model.joblib")

print("Model successfully downloaded!")

PY

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
