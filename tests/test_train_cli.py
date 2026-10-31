"""Keep CLI arguments and JSON output independent of model fitting tests."""

from pathlib import Path
import json
import sys
from unittest.mock import Mock

import pytest

from garments_mlops.modeling import decide_promotion
from garments_mlops import train_cli
from garments_mlops.training import TrainingResult


def test_default_cli_and_overrides_are_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("{}")
    result = TrainingResult("run-1", "2", "models:/candidate/2", "2", {"mae": 0.1}, decide_promotion(0.1, 0.2))
    training = Mock(return_value=result)
    monkeypatch.setattr(train_cli, "train_candidate", training)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["garments-train"])
    train_cli.main()
    assert training.call_args.args[1] == "random_forest"
    assert json.loads(capsys.readouterr().out)["run_id"] == "run-1"
    train_cli.main(["--config", str(path), "--model", "baseline", "--tracking-uri", "http://localhost:5000",
                    "--minimum-improvement", "0.1"])
    config, kind = training.call_args.args
    assert config.tracking_uri == "http://localhost:5000"
    assert config.minimum_improvement == 0.1
    assert kind == "baseline"
    assert json.loads(capsys.readouterr().out)["decision"]["promoted"] is True


@pytest.mark.parametrize("arguments,code", [(["--help"], 0), (["--model", "invalid"], 2)])
def test_cli_usage_errors_and_help(arguments: list[str], code: int) -> None:
    with pytest.raises(SystemExit) as result:
        train_cli.main(arguments)
    assert result.value.code == code
