"""Reconstruct the ten-seed Leiden sensitivity in Supplementary Table S4."""

import pickle
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from validate_pipeline import DATA, OUT, ROOT, load_master, metrics
from summarize_results import interp


def main():
    master = load_master()
    parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
    ids = parts["core85", 33.471236].index
    edges = pd.read_parquet(DATA / "graphs/edge_tables/graph_2.parquet")
    edges = edges[edges.geoid_i.isin(ids) & edges.geoid_j.isin(ids)]
    graph = ig.Graph.TupleList(
        edges[["geoid_i", "geoid_j", "weight"]].itertuples(index=False, name=None),
        directed=False,
        weights=True,
    )
    graph.add_vertices(sorted(set(ids) - set(graph.vs["name"])))
    resolutions = sorted(r for f, r in parts if f == "core85")
    panel = pd.read_parquet(DATA / "epidemic_panel_daily_with_nyc.parquet")
    raw = (
        panel[panel.GEOID.isin(ids)]
        .pivot(index="GEOID", columns="date", values="smoothed_cases")
        .fillna(0)
    )
    phases = {
        "Lockdown": ("2020-03-15", "2020-06-15"),
        "Delta": ("2021-07-01", "2021-11-30"),
        "Omicron": ("2021-12-01", "2022-03-31"),
    }
    canonical = pd.read_csv(ROOT / "results/paper/static_frontiers.csv")
    rows, comparisons = [], []
    for seed in [42, *range(1, 10)]:
        current = []
        for resolution in resolutions:
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=seed,
                n_iterations=-1,
            )
            unit = pd.Series(partition.membership, index=graph.vs["name"]).reindex(ids)
            compression = master.loc[ids, "population"].sum() / unit.nunique()
            for phase, (lo, hi) in phases.items():
                _, units = metrics(
                    raw.loc[:, (raw.columns >= lo) & (raw.columns <= hi)], unit
                )
                current.append(
                    dict(
                        seed=seed,
                        phase=phase,
                        resolution=resolution,
                        compression=compression,
                        n_units=unit.nunique(),
                        inc=units.inc.mean(),
                        lag=units.lag.mean(),
                    )
                )
        frame = pd.DataFrame(current)
        rows.extend(current)
        for phase in phases:
            markers = canonical[
                (canonical.signal == "cases")
                & (canonical.phase == phase)
                & canonical.family.isin(["States", "CSA", "CBSA>=2"])
            ]
            for marker in markers.itertuples():
                for key in ["lag", "inc"]:
                    comparisons.append(
                        dict(
                            seed=seed,
                            phase=phase,
                            partition=marker.family,
                            metric=key,
                            core=interp(
                                frame[frame.phase == phase], key, marker.compression
                            ),
                            marker=getattr(marker, key),
                        )
                    )
        print(f"Seed {seed} complete", flush=True)
    destination = ROOT / "results/paper"
    pd.DataFrame(rows).to_csv(destination / "seed_frontiers.csv", index=False)
    matched = pd.DataFrame(comparisons)
    matched.to_csv(destination / "seed_matched.csv", index=False)
    matched.groupby(["phase", "partition", "metric"]).agg(
        minimum=("core", "min"), maximum=("core", "max"), marker=("marker", "first")
    ).to_csv(destination / "seed_ranges.csv")


if __name__ == "__main__":
    main()
