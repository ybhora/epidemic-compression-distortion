"""Download public inputs and verify the exact bytes used for the paper."""

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def entries(group="analysis"):
    with (ROOT / "data/sources.csv").open() as handle:
        return [
            r
            for r in csv.DictReader(handle)
            if r["group"] == "analysis" or group == "all"
        ]


def check(group="analysis", download=False):
    problems = []
    with (ROOT / "data/reference_manifest.csv").open() as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["path"]
            if not path.is_file() or digest(path) != row["sha256"]:
                problems.append(f"Changed or missing bundled reference: {row['path']}")
    rows = entries(group)
    for number, row in enumerate(rows, 1):
        path = ROOT / row["path"]
        if not path.exists() and download:
            print(f"[{number}/{len(rows)}] Downloading {path.name}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".partial")
            try:
                request = Request(
                    row["url"], headers={"User-Agent": "coarsening-paper-reproduction"}
                )
                with urlopen(request, timeout=180) as response, partial.open(
                    "wb"
                ) as output:
                    while block := response.read(2**20):
                        output.write(block)
                if digest(partial) != row["sha256"]:
                    raise ValueError("source hash differs from the paper snapshot")
                partial.replace(path)
            except Exception as error:
                partial.unlink(missing_ok=True)
                problems.append(f"{row['path']}: {error}")
                continue
        if not path.is_file():
            problems.append(f"Missing {row['path']}")
        elif path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
            problems.append(f"Changed {row['path']}")
    if problems:
        raise SystemExit(
            "\n".join(problems)
            + "\nSee docs/data.md. Inputs were not replaced or accepted silently."
        )
    print(f"Verified {len(rows)} source files.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["analysis", "all"],
        default="analysis",
        help="all also downloads pandemic mobility for rebuilding the archived predictor",
    )
    parser.add_argument(
        "--check", action="store_true", help="verify files without downloading"
    )
    args = parser.parse_args()
    check(args.group, download=not args.check)
