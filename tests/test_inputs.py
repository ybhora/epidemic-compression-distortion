"""Check bundled reference integrity and download/hash behavior without network access."""

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_data

with (ROOT / "data/reference_manifest.csv").open() as handle:
    for row in csv.DictReader(handle):
        assert fetch_data.digest(ROOT / row["path"]) == row["sha256"], row["path"]

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "data").mkdir()
    source = root / "public.csv"
    source.write_bytes(b"x,y\n1,2\n")
    (root / "data/reference_manifest.csv").write_text("path,bytes,sha256\n")
    with (root / "data/sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "bytes", "sha256", "group", "url"]
        )
        writer.writeheader()
        writer.writerow(
            dict(
                path="data/raw/input.csv",
                bytes=source.stat().st_size,
                sha256=fetch_data.digest(source),
                group="analysis",
                url=source.as_uri(),
            )
        )
    fetch_data.ROOT = root
    fetch_data.check(download=True)
    fetch_data.check()
    target = root / "data/raw/input.csv"
    target.write_bytes(b"changed")
    try:
        fetch_data.check()
        raise AssertionError("Changed source accepted")
    except SystemExit as error:
        assert "Changed" in str(error)
    target.unlink()
    source.write_bytes(b"changed upstream")
    try:
        fetch_data.check(download=True)
        raise AssertionError("Changed upstream source accepted")
    except SystemExit as error:
        assert "source hash differs" in str(error)
    assert not target.exists() and not target.with_suffix(".csv.partial").exists()

fetch_data.ROOT = ROOT
print("Reference hashes, acquisition, reuse and changed-input rejection passed.")
