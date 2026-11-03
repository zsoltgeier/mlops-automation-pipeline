"""Prepare deterministic data and initialize a persistent champion when absent."""

from pathlib import Path

from filelock import FileLock

from garments_mlops.prepare import prepare_dataset
from garments_mlops.tracking import champion_version, load_champion, tracking_client
from garments_mlops.training import train_candidate
from garments_mlops.training_config import TrainingConfig


def initialize_system(
    config: TrainingConfig, source: Path = Path("data/garments_worker_productivity.csv"),
    parameters: Path = Path("params.yaml"),
) -> str:
    """Reuse model history while rebuilding small deterministic data artifacts."""
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    startup_lock = config.lock_path.with_name(f"{config.lock_path.name}.bootstrap")
    with FileLock(startup_lock, timeout=config.lock_timeout):
        with FileLock(config.lock_path, timeout=config.lock_timeout):
            prepare_dataset(source, config.data_dir, parameters)
        client = tracking_client(config)
        if champion_version(client, config.model_name) is None:
            train_candidate(config, "baseline")
            train_candidate(config, "random_forest")
        return load_champion(config).version
