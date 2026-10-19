"""S6: distinguish restricted core memberships from freshly clustered common supports."""

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
    core = parts["core85", 33.471236].index
    cbsa = master.CBSA.dropna()
    canonical = {
        "States": master.state,
        "CSA": master.CSA.dropna(),
        "CBSA>=2": cbsa[cbsa.map(cbsa.value_counts()) >= 2],
    }
    edges = pd.read_parquet(DATA / "graphs/edge_tables/graph_2.parquet")
    panel = pd.read_parquet(DATA / "epidemic_panel_daily_with_nyc.parquet")
    raw = (
        panel[panel.GEOID.isin(core)]
        .pivot(index="GEOID", columns="date", values="smoothed_cases")
        .fillna(0)
    )
    phases = {
        "Lockdown": ("2020-03-15", "2020-06-15"),
        "Delta": ("2021-07-01", "2021-11-30"),
        "Omicron": ("2021-12-01", "2022-03-31"),
    }
    resolutions = sorted(r for family, r in parts if family == "core85")
    rows, matched = [], []
    for name, reference in canonical.items():
        ids = sorted(set(reference.index) & set(core))
        reference = reference.loc[ids]
        e = edges[edges.geoid_i.isin(ids) & edges.geoid_j.isin(ids)]
        graph = ig.Graph.TupleList(
            e[["geoid_i", "geoid_j", "weight"]].itertuples(index=False, name=None),
            directed=False,
            weights=True,
        )
        graph.add_vertices(sorted(set(ids) - set(graph.vs["name"])))
        memberships = {}
        for resolution in resolutions:
            if name == "States":
                memberships[resolution] = parts["core85", resolution].loc[ids]
            else:
                partition = la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=resolution,
                    seed=42,
                    n_iterations=-1,
                )
                memberships[resolution] = pd.Series(
                    partition.membership, index=graph.vs["name"]
                ).reindex(ids)
        for phase, (lo, hi) in phases.items():
            signal = raw.loc[ids, (raw.columns >= lo) & (raw.columns <= hi)]
            _, canonical_units = metrics(signal, reference)
            target = master.loc[ids, "population"].sum() / reference.nunique()
            curve = []
            for resolution, membership in memberships.items():
                _, units = metrics(signal, membership)
                curve.append(
                    dict(
                        reference=name,
                        phase=phase,
                        resolution=resolution,
                        n_counties=len(ids),
                        n_units=membership.nunique(),
                        compression=master.loc[ids, "population"].sum()
                        / membership.nunique(),
                        inc=units.inc.mean(),
                        lag=units.lag.mean(),
                    )
                )
            frame = pd.DataFrame(curve)
            rows.extend(curve)
            matched.append(
                dict(
                    reference=name,
                    phase=phase,
                    n_counties=len(ids),
                    compression=target,
                    canonical_lag=canonical_units.lag.mean(),
                    reclustered_lag=interp(frame, "lag", target),
                    canonical_inc=canonical_units.inc.mean(),
                    reclustered_inc=interp(frame, "inc", target),
                )
            )
        print(
            f"Common-support reclustering complete: {name} ({len(ids)} counties)",
            flush=True,
        )
    destination = ROOT / "results/paper"
    pd.DataFrame(rows).to_csv(
        destination / "common_support_reclustered_frontiers.csv", index=False
    )
    pd.DataFrame(matched).to_csv(
        destination / "common_support_reclustered_comparisons.csv", index=False
    )


if __name__ == "__main__":
    main()
