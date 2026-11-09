"""Prove public data restoration fails closed and keeps existing datasets untouched."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import runpy
from unittest.mock import Mock
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from garments_mlops import fetch_data


def archive(csv: bytes) -> bytes:
    """Pack one CSV exactly as the public UCI archive does."""
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr(fetch_data.DATASET_FILE, csv)
    return stream.getvalue()


@pytest.mark.parametrize("failure", [None, "hash", "archive_size", "csv_size", "network", "write"])
def test_verified_atomic_download_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    csv = b"verified fixture content\n"
    monkeypatch.setattr(fetch_data, "DATASET_SHA256", sha256(csv).hexdigest())
    content = archive(b"wrong" if failure == "hash" else csv)
    if failure == "archive_size":
        content = b"0" * (fetch_data.MAX_ARCHIVE_BYTES + 1)
    if failure == "csv_size":
        content = archive(b"0" * (fetch_data.MAX_CSV_BYTES + 1))
    response = Mock()
    response.__enter__ = Mock(return_value=BytesIO(content))
    response.__exit__ = Mock(return_value=False)
    download = Mock(return_value=response, side_effect=OSError("network") if failure == "network" else None)
    monkeypatch.setattr(fetch_data, "urlopen", download)
    if failure == "write":
        monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("write")))
    destination = tmp_path / "data/raw.csv"
    if failure is not None:
        with pytest.raises((ValueError, OSError)):
            fetch_data.fetch_dataset(destination)
        assert not destination.exists()
        assert not list(destination.parent.glob("tmp*"))
        return
    assert fetch_data.fetch_dataset(destination) == destination
    assert destination.read_bytes() == csv
    assert fetch_data.fetch_dataset(destination) == destination
    download.assert_called_once_with(fetch_data.DATASET_URL, timeout=30)
    destination.write_bytes(b"changed locally")
    with pytest.raises(ValueError, match="Existing"):
        fetch_data.fetch_dataset(destination)
    assert destination.read_bytes() == b"changed locally"


def test_cli_and_public_dataset_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "dataset.csv"
    restore = Mock(return_value=destination)
    monkeypatch.setattr(fetch_data, "fetch_dataset", restore)
    fetch_data.main(["--output", str(destination)])
    restore.assert_called_once_with(destination)
    assert str(destination) in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        fetch_data.main(["--help"])
    assert result.value.code == 0
    source = Path(__file__).resolve().parents[1] / "data" / fetch_data.DATASET_FILE
    assert sha256(source.read_bytes()).hexdigest() == fetch_data.DATASET_SHA256


def test_standalone_ci_entrypoint_verifies_existing_public_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = Path(__file__).resolve().parents[1] / "data" / fetch_data.DATASET_FILE
    monkeypatch.setattr("sys.argv", ["fetch_data.py", "--output", str(source)])
    runpy.run_path(fetch_data.__file__, run_name="__main__")
    assert str(source) in capsys.readouterr().out
