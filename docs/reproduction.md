# Reproducing and resuming analyses

Follow the installation steps in the [README](../README.md). Use Python 3.14 and the package versions in `requirements.txt` to reproduce the results. Run `python run.py --help` to see the commands.

## Workflow

The complete workflow is `python run.py download` followed by `python run.py all`. To run stages separately, use this order:

| Command | What it does | Needs |
| --- | --- | --- |
| `python run.py test` | Independent metric, SIR-conservation, and interpolation checks | Installed packages |
| `python run.py demo` | Figure 1 | Installed packages |
| `python run.py download` | Downloads and verifies the public source files | Internet, about 4 GB |
| `python run.py check-inputs` | Verifies downloaded files without network access | Downloaded source files |
| `python run.py prepare` | Builds Census master, G1/G2, daily case panel, NYC augmentation, and source-date validation | Verified source files |
| `python run.py empirical` | Communities; static, rolling and death analyses; permutations; prediction; graph/null summaries | `prepare` |
| `python run.py simulations` | Canonical simulations and homogeneous-SIR fitting demonstrations, Figures 1 and S2a–e | Installed packages |
| `python run.py paper` | Three-phase frontiers, final gradients, geographic diagnostics, common-support reclustering, ten-seed sweep, and sensitivities | `empirical` |
| `python run.py figures` | Final empirical and supplementary figures | All previous analysis stages |

The `all` command runs these stages in the required order and stops on the first failure. Downloads are a separate, explicit operation. After download, `all` operates offline.

For a single analysis, run its script from the repository folder, for example `python scripts/seed_robustness.py`. These scripts use the same output files as the staged runner. `scripts/figure1.py`, `plot_simulations.py`, and `simulation_inference.py` are independent of the empirical data.

## After an interruption

Inspect the last `Running …` line and the error above it. Correct the issue, then rerun that stage and the later stages. Rerunning a stage replaces its generated outputs. Files left by an interrupted stage should not be treated as complete results.

The empirical analysis saves generated memberships in `results/analysis/partitions.pkl` to avoid reclustering during subsequent runs. This is a locally generated cache, never a downloaded input. Load only your own cache. To recompute memberships:

```bash
python run.py empirical --fresh-partitions
```

If you change the graph, county population/support, resolution grid, community-detection versions, or seed, use `--fresh-partitions` and rerun every downstream stage. Keep the pinned package versions and edge order for numerical reproduction. Seed alone does not define an identical community partition across implementations.

## Common problems

- **A package cannot be installed:** check `python --version` and use Python 3.14 in a new virtual environment. Avoid installing into a system Python or an unrelated research environment.
- **Missing input:** run `python run.py download`, or use the paths and sources in [data.md](data.md). A later stage also needs the outputs of earlier stages.
- **Hash mismatch:** the source may have changed. The downloader does not overwrite existing files or accept revised inputs silently. Recover the recorded snapshot or conduct a separately documented updated analysis. See `data/sources.csv` for the exact expected hash.
- **Download interrupted:** rerun `python run.py download`. Complete, matching files are reused; incomplete `.partial` files are discarded.
- **Very slow empirical run:** the permutation and rolling analyses are substantial calculations. Progress messages identify completed partitions/windows/controls. Do not replace the full analysis with a smaller grid when trying to match the paper.
- **Runtime warnings about empty means:** a constant epidemic series has undefined Pearson correlation. Such values are intentionally missing and are omitted from metric means. Membership population remains in the compression denominator.
