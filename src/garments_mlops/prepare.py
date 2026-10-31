"""Export deterministic datasets and provenance using train-only preprocessing."""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
import yaml

from garments_mlops.data import RAW_FEATURES, TARGET, clean_dataset
from garments_mlops.preprocessing import build_preprocessor
from garments_mlops.splitting import SplitConfig, split_by_date

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> SplitConfig:
    """Require explicit data parameters and reject unknown settings."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"data"} or not isinstance(document["data"], dict):
        raise ValueError("Parameters must contain a data mapping")
    settings = document["data"]
    if set(settings) != {"train_fraction", "validation_fraction", "random_seed"}:
        raise ValueError("Expected train_fraction, validation_fraction and random_seed")
    return SplitConfig(**settings)


def sha256(path: Path) -> str:
    """Fingerprint inputs and generated artifacts without recording timestamps."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(source: Path, output: Path, parameters: Path) -> dict[str, object]:
    """Create splits, a reusable preprocessor and submission-ready numeric data."""
    config = load_config(parameters)
    original = pd.read_csv(source)
    cleaned = clean_dataset(original)
    splits = split_by_date(cleaned, config)
    preprocessor = build_preprocessor().fit(splits.train[RAW_FEATURES])
    names = preprocessor.named_steps["columns"].get_feature_names_out().tolist()
    output.mkdir(parents=True, exist_ok=True)
    clean_parts: list[pd.DataFrame] = []
    numeric_parts: list[pd.DataFrame] = []
    partition_details: dict[str, object] = {}
    generated: list[Path] = []
    for name, partition in splits.items():
        raw_path = output / f"{name}.csv"
        numeric_path = output / f"{name}_transformed.csv"
        transformed = pd.DataFrame(preprocessor.transform(partition[RAW_FEATURES]), columns=names)
        transformed[TARGET] = partition[TARGET].to_numpy()
        partition.to_csv(raw_path, index=False, float_format="%.17g")
        transformed.to_csv(numeric_path, index=False, float_format="%.17g")
        generated.extend([raw_path, numeric_path])
        clean_parts.append(partition.assign(split=name))
        numeric_parts.append(transformed.assign(split=name))
        partition_details[name] = {
            "rows": len(partition), "days": int(partition.date.nunique()),
            "first_date": partition.date.min(), "last_date": partition.date.max(),
        }
    for filename, frame in [
        ("cleaned.csv", pd.concat(clean_parts, ignore_index=True)),
        ("preprocessed.csv", pd.concat(numeric_parts, ignore_index=True)),
        ("reference.csv", splits.train[RAW_FEATURES]),
    ]:
        path = output / filename
        frame.to_csv(path, index=False, float_format="%.17g")
        generated.append(path)
    artifact = output / "preprocessor.joblib"
    joblib.dump(preprocessor, artifact)
    generated.append(artifact)
    metadata: dict[str, object] = {
        "source_sha256": sha256(source), "parameters_sha256": sha256(parameters),
        "configuration": asdict(config), "target": TARGET, "task": "regression",
        "split_strategy": "chronological_complete_dates", "raw_rows": len(original), "clean_rows": len(cleaned),
        "duplicates_removed": len(original) - len(cleaned), "feature_names": names,
        "missing_values": {name: int(value) for name, value in original.isna().sum().items()},
        "partitions": partition_details,
        "files_sha256": {path.name: sha256(path) for path in sorted(generated)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Prepared %s observations in %s", len(cleaned), output)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    """Expose data preparation as an installed console command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/garments_worker_productivity.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--parameters", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    prepare_dataset(args.source, args.output, args.parameters)
