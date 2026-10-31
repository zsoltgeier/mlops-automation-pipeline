"""Command-line entry point for a tracked baseline or challenger experiment."""

import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from typing import Sequence

from garments_mlops.training import train_candidate
from garments_mlops.training_config import load_training_config


def main(argv: Sequence[str] | None = None) -> None:
    """Train one candidate and print run/version/promotion metadata as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training.yaml"))
    parser.add_argument("--model", choices=["baseline", "random_forest"], default="random_forest")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--minimum-improvement", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_training_config(args.config)
    if args.tracking_uri is not None:
        config = replace(config, tracking_uri=args.tracking_uri)
    if args.minimum_improvement is not None:
        config = replace(config, minimum_improvement=args.minimum_improvement)
    result = train_candidate(config, args.model)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
