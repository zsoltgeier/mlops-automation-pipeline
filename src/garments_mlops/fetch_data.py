"""Restore the exact public UCI CSV used by the DVC pipeline, including on clean CI runners."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/597/productivity+prediction+of+garment+employees.zip"
DATASET_FILE = "garments_worker_productivity.csv"
DATASET_SHA256 = "763ffcdd39b7f7f360243c2af8f89ed733e8b5f9e511efeb0c886b83e1b76866"
MAX_ARCHIVE_BYTES = 2_000_000
MAX_CSV_BYTES = 1_000_000


def fetch_dataset(destination: Path = Path("data") / DATASET_FILE) -> Path:
    """Verify existing data or atomically restore the pinned CSV; never replace an unverified file."""
    if destination.exists():
        if sha256(destination.read_bytes()).hexdigest() != DATASET_SHA256:
            raise ValueError("Existing dataset differs from the pinned UCI version")
        return destination
    with urlopen(DATASET_URL, timeout=30) as response:
        content = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("UCI archive exceeds the expected size limit")
    with ZipFile(BytesIO(content)) as archive:
        if archive.getinfo(DATASET_FILE).file_size > MAX_CSV_BYTES:
            raise ValueError("UCI CSV exceeds the expected size limit")
        csv = archive.read(DATASET_FILE)
    if sha256(csv).hexdigest() != DATASET_SHA256:
        raise ValueError("Downloaded CSV differs from the pinned UCI version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            temporary.write(csv)
            temporary.close()
            pending.replace(destination)
        finally:
            pending.unlink(missing_ok=True)
    return destination


def main(arguments: list[str] | None = None) -> None:
    """Fetch the public versioned CSV without requiring private DVC credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data") / DATASET_FILE)
    args = parser.parse_args(arguments)
    print(fetch_dataset(args.output))


if __name__ == "__main__":
    main()
