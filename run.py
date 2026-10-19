"""One entry point for the paper analyses. Run --help for the workflow."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "prepare": ["prepare_data", "prepare_nyc_panel", "validate_nys_alignment"],
    "empirical": [
        "recompute_empirical",
        "additional_checks",
        "summarize_results",
        "predictive_check",
        "supplement_checks",
    ],
    "simulations": ["plot_simulations", "figure1", "simulation_inference"],
    "paper": [
        "paper_analysis",
        "geography",
        "common_support_recluster",
        "seed_robustness",
        "paper_tables",
        "revision_handling_check",
        "nyc_date_sensitivity",
        "nyc_death_timing_check",
    ],
    "figures": ["plot_empirical", "plot_publication", "paper_figures"],
}


def execute(script, *args):
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    print(f"\nRunning {script.name}", flush=True)
    subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start: python run.py test\nQuick figure: python run.py demo\nFull workflow: python run.py download; python run.py all\nSee docs/reproduction.md for individual stages.",
    )
    parser.add_argument(
        "command", choices=["test", "demo", "download", "check-inputs", "all", *STAGES]
    )
    parser.add_argument(
        "--fresh-partitions",
        action="store_true",
        help="remove the generated membership cache before empirical/all",
    )
    args = parser.parse_args()
    for directory in [
        "results/analysis",
        "results/paper",
        "results/figures",
        ".cache/matplotlib",
    ]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    if args.fresh_partitions:
        if args.command not in ["empirical", "all"]:
            parser.error("--fresh-partitions applies to empirical or all")
        (ROOT / "results/analysis/partitions.pkl").unlink(missing_ok=True)
    if args.command in ["download", "check-inputs"]:
        execute(
            ROOT / "scripts/fetch_data.py",
            *(["--check"] if args.command == "check-inputs" else []),
        )
    elif args.command == "test":
        for name in ["test_metrics", "test_science", "test_inputs"]:
            execute(ROOT / f"tests/{name}.py")
    elif args.command == "demo":
        execute(ROOT / "scripts/figure1.py")
    elif args.command == "all":
        execute(ROOT / "scripts/fetch_data.py", "--check")
        for name in ["test_metrics", "test_science", "test_inputs"]:
            execute(ROOT / f"tests/{name}.py")
        for stage in STAGES.values():
            for script in stage:
                execute(ROOT / f"scripts/{script}.py")
    else:
        for script in STAGES[args.command]:
            execute(ROOT / f"scripts/{script}.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Stopped at {Path(error.cmd[1]).name}. See the error above and docs/reproduction.md."
        )
