"""Tracked candidate training with serialized champion-challenger promotion."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
import mlflow.sklearn as mlflow_sklearn

from garments_mlops.data import RAW_FEATURES, TARGET
from garments_mlops.modeling import ModelKind, PromotionDecision, build_model, decide_promotion, regression_metrics
from garments_mlops.tracking import MissingChampionError, experiment_id, load_champion, tracking_client
from garments_mlops.training_config import TrainingConfig
from garments_mlops.training_data import load_training_data


@dataclass(frozen=True)
class TrainingResult:
    """Run, candidate and active version identifiers for callers and the future API."""

    run_id: str
    model_version: str
    model_uri: str
    champion_version: str
    metrics: dict[str, float]
    decision: PromotionDecision


def train_candidate(config: TrainingConfig, kind: ModelKind = "random_forest") -> TrainingResult:
    """Fit on train, compare on validation and change the alias only after successful logging."""
    spec = config.model_spec(kind)
    data = load_training_data(config.data_dir)
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(config.lock_path, timeout=config.lock_timeout):
        client = tracking_client(config)
        experiment = experiment_id(client, config)
        with mlflow.start_run(experiment_id=experiment, run_name=kind, nested=mlflow.active_run() is not None) as run:
            mlflow.log_params({**asdict(spec), "minimum_improvement": config.minimum_improvement})
            mlflow.set_tags({"primary_metric": "validation_mae",
                             "dataset_source_sha256": data.provenance["source_sha256"]})
            mlflow.log_dict(data.provenance, "data/provenance.json")
            for name, frame in (("train", data.train), ("validation", data.validation)):
                dataset = from_pandas(frame[RAW_FEATURES + [TARGET]], name=name, targets=TARGET,
                                      digest=str(data.provenance[f"{name}_sha256"])[:32],
                                      source=CodeDatasetSource({
                                          "file_uri": (config.data_dir / f"{name}.csv").resolve().as_uri(),
                                          "sha256": data.provenance[f"{name}_sha256"],
                                      }))
                mlflow.log_input(dataset, context=name)
            model = build_model(spec).fit(data.train[RAW_FEATURES], data.train[TARGET])
            train_prediction = model.predict(data.train[RAW_FEATURES])
            prediction = model.predict(data.validation[RAW_FEATURES])
            metrics = regression_metrics(data.validation[TARGET], prediction)
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            mlflow.log_metrics({f"train_{key}": value for key, value in
                                regression_metrics(data.train[TARGET], train_prediction).items()})
            try:
                incumbent = load_champion(config)
            except MissingChampionError:
                incumbent = None
            champion_mae = None
            evaluation = data.validation.assign(candidate_prediction=prediction)
            if incumbent is not None:
                champion_prediction = incumbent.model.predict(data.validation[RAW_FEATURES])
                champion_mae = regression_metrics(data.validation[TARGET], champion_prediction)["mae"]
                evaluation["champion_prediction"] = champion_prediction
                mlflow.log_metric("champion_validation_mae", champion_mae)
            decision = decide_promotion(metrics["mae"], champion_mae, config.minimum_improvement)
            info = mlflow_sklearn.log_model(
                model, name="model", registered_model_name=config.model_name,
                signature=infer_signature(data.train[RAW_FEATURES], train_prediction),
                input_example=data.train[RAW_FEATURES].head(3),
                code_paths=[str(Path(__file__).resolve().parent)],
                serialization_format="skops",
                # Allow only our cleaner and the dtype/tree structures produced by our fitted pipeline.
                skops_trusted_types=["garments_mlops.preprocessing.FeatureCleaner", "numpy.dtype",
                                     "sklearn.tree._tree.Tree"],
                pip_requirements=[f"{name}=={version(name)}" for name in (
                    "mlflow", "scikit-learn", "pandas", "numpy", "scipy", "joblib", "threadpoolctl", "skops",
                )],
            )
            if info.registered_model_version is None:
                raise RuntimeError("MLflow did not return a registered model version")
            candidate_version = str(info.registered_model_version)
            mlflow.log_text(evaluation.to_csv(index=False), "evaluation/validation_predictions.csv")
            mlflow.log_dict({**asdict(decision), "candidate_version": candidate_version,
                             "champion_before": incumbent.version if incumbent is not None else None},
                            "promotion/decision.json")
            mlflow.set_tags({"promotion_approved": str(decision.promoted).lower(), "promotion_reason": decision.reason})
            for key, value in {"validation_mae": str(metrics["mae"]), "promotion_reason": decision.reason,
                               "validation_sha256": str(data.provenance["validation_sha256"])}.items():
                client.set_model_version_tag(config.model_name, candidate_version, key, value)
            if decision.promoted:
                client.set_registered_model_alias(config.model_name, "champion", candidate_version)
            active = str(client.get_model_version_by_alias(config.model_name, "champion").version)
            return TrainingResult(run.info.run_id, candidate_version, info.model_uri, active, metrics, decision)
