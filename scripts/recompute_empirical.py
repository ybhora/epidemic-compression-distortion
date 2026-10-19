"""Empirical partition, static, rolling and death analyses.
Metrics: unclipped 1-rho0 (range 0..2), |optimal lag|, unit-first averaging.
Bootstrap conditions on fixed unit aggregates and partitions, resampling county summaries
within units. It measures composition stability, not reporting or spatial uncertainty.
"""

from nyc_deaths import append_borough_deaths
from pathlib import Path
import sys, json, pickle, random, warnings
import numpy as np, pandas as pd, igraph as ig, leidenalg as la
from validate_pipeline import (
    load_master,
    metrics,
    ROOT,
    DATA,
    OUT,
    EX,
    select_coarse_partition,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 42


def bootstrap(cm, B=500, stat="mean"):
    rng = np.random.default_rng(SEED)
    units = []
    for _, g in cm.groupby("unit"):
        a = g[["inc", "lag"]].to_numpy()
        idx = rng.integers(len(a), size=(B, len(a)))
        units.append(np.nanmean(a[idx], axis=1))
    a = np.stack(units, axis=1)
    v = np.nanmean(a, axis=1) if stat == "mean" else np.nanmedian(a, axis=1)
    return np.nanquantile(v, [0.025, 0.975], axis=0)


def main():
    master = load_master()
    master = master[~master.index.str[:2].isin(EX)]
    panel = pd.read_parquet(
        DATA / "epidemic_panel_daily_with_nyc.parquet",
        columns=["GEOID", "date", "cases", "smoothed_cases"],
    )
    panel = panel[panel.GEOID.isin(master.index)]
    raw = (
        panel.pivot(index="GEOID", columns="date", values="smoothed_cases")
        .sort_index()
        .fillna(0)
    )
    master = master.loc[raw.index]
    units = {
        "States": master.state,
        "CSA": master.CSA.dropna(),
        "CBSA": master.CBSA.dropna(),
    }
    cb = units["CBSA"]
    units["CBSA>=2"] = cb[cb.map(cb.value_counts()) >= 2]
    units["CBSA>=3"] = cb[cb.map(cb.value_counts()) >= 3]
    cache = OUT / "partitions.pkl"
    if cache.exists():
        parts = pickle.loads(cache.read_bytes())
    else:
        parts = {}
        edges = pd.read_parquet(DATA / "graphs/edge_tables/graph_2.parquet")
        edges = edges[
            edges.geoid_i.isin(master.index) & edges.geoid_j.isin(master.index)
        ]
        strength = (
            pd.concat(
                [
                    edges.groupby("geoid_i").weight.sum(),
                    edges.groupby("geoid_j").weight.sum(),
                ],
                axis=1,
            )
            .sum(axis=1)
            .reindex(master.index)
            .fillna(0)
        )
        rank = (
            pd.DataFrame({"strength": strength, "population": master.population})
            .rename_axis("geoid")
            .reset_index()
            .sort_values(["strength", "geoid"], ascending=[False, True])
        )
        rank["cum_pop"] = rank.population.cumsum()
        supports = {"full": sorted(master.index)}
        sr = []
        for cut in [0.85, 0.90, 0.95]:
            k = int(np.searchsorted(rank.cum_pop, cut * rank.population.sum())) + 1
            ids = sorted(rank.iloc[:k].geoid)
            family = f"core{int(cut*100)}"
            supports[family] = ids
            inside = edges.geoid_i.isin(ids) & edges.geoid_j.isin(ids)
            sr.append(
                dict(
                    family=family,
                    n_counties=k,
                    population=master.loc[ids, "population"].sum(),
                    population_fraction=master.loc[ids, "population"].sum()
                    / master.population.sum(),
                    internal_flow_share=edges.loc[inside, "weight"].sum()
                    / edges.weight.sum(),
                )
            )
        pd.DataFrame(sr).to_csv(OUT / "retained_supports.csv", index=False)
        resolutions = sorted(
            set(
                pd.read_csv(
                    ROOT / "data/reference/window_resolution_registry.csv"
                ).resolution.round(6)
            )
        )
        for family, ids in supports.items():
            e = edges[edges.geoid_i.isin(ids) & edges.geoid_j.isin(ids)]
            g = ig.Graph.TupleList(
                e[["geoid_i", "geoid_j", "weight"]].itertuples(index=False, name=None),
                directed=False,
                weights=True,
            )
            g.add_vertices(sorted(set(ids) - set(g.vs["name"])))
            for r in resolutions:
                p = la.find_partition(
                    g,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=r,
                    seed=SEED,
                    n_iterations=-1,
                )
                u = pd.Series(p.membership, index=g.vs["name"]).reindex(ids)
                parts[(family, r)] = u
                print("partition", family, r, u.nunique(), flush=True)
            if family == "core85":
                for r in resolutions[::2]:
                    ig.set_random_number_generator(random.Random(SEED))
                    p = g.community_multilevel(
                        weights="weight", resolution=r if r > 0 else 1e-9
                    )
                    parts[("Louvain", r)] = pd.Series(
                        p.membership, index=g.vs["name"]
                    ).reindex(ids)
        cache.write_bytes(pickle.dumps(parts))
    # Native memberships retained; signal-specific constant county exclusions are reported.
    dates = raw.columns
    delta = raw.loc[:, (dates >= "2021-07-01") & (dates <= "2021-11-30")]
    allparts = {**{(n, np.nan): u for n, u in units.items()}, **parts}
    rows = []
    for (family, r), u in allparts.items():
        cm, um = metrics(delta, u)
        ci = bootstrap(cm)
        rows.append(
            dict(
                family=family,
                resolution=r,
                n_counties=len(cm),
                n_valid_counties=cm.inc.notna().sum(),
                n_units=len(um),
                n_valid_units_inc=um.inc.notna().sum(),
                n_valid_units_lag=um.lag.notna().sum(),
                compression=master.loc[u.index, "population"].sum() / u.nunique(),
                inc=um.inc.mean(),
                lag=um.lag.mean(),
                inc_lo=ci[0, 0],
                inc_hi=ci[1, 0],
                lag_lo=ci[0, 1],
                lag_hi=ci[1, 1],
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "corrected_delta_frontiers.csv", index=False)
    print("static done", flush=True)
    # Native-support gradient: each partition uses its own included counties.
    rlist = sorted(r for f, r in parts if f == "core85")
    near = lambda t: min(rlist, key=lambda x: abs(x - t))
    grad = {
        "States": units["States"],
        "CSA": units["CSA"],
        "CBSA>=2": units["CBSA>=2"],
        "Leiden coarse": select_coarse_partition(parts, master)[1],
        "Leiden intermediate": parts[("core85", near(33.47))],
        "Leiden fine": parts[("core85", near(111.19))],
    }
    gd = []
    gu = []
    for name, u in grad.items():
        ids = raw.index.intersection(u.dropna().index)
        x = raw.loc[ids, (dates >= "2020-03-15") & (dates <= "2020-06-15")]
        cm, um = metrics(x, u.reindex(ids))
        ci = bootstrap(cm, stat="median")
        gd.append(
            dict(
                partition=name,
                n_counties=len(ids),
                n_valid_counties=cm.inc.notna().sum(),
                n_units=len(um),
                n_valid_units_inc=um.inc.notna().sum(),
                n_valid_units_lag=um.lag.notna().sum(),
                compression=master.loc[ids, "population"].sum() / len(um),
                inc=um.inc.median(),
                lag=um.lag.median(),
                inc_lo=ci[0, 0],
                inc_hi=ci[1, 0],
                lag_lo=ci[0, 1],
                lag_hi=ci[1, 1],
            )
        )
        gu.append(um.reset_index().assign(partition=name))
    pd.DataFrame(gd).to_csv(OUT / "corrected_gradient.csv", index=False)
    pd.concat(gu).to_csv(OUT / "corrected_gradient_units.csv", index=False)
    # Dynamic: preserve the exact existing window dates and resolution grid.
    windows = pd.read_csv(ROOT / "data/reference/window_resolution_registry.csv")[
        ["window_center_date", "window_start", "window_end"]
    ].drop_duplicates()
    dyn = []
    for k, w in enumerate(windows.itertuples(index=False)):
        x = raw.loc[:, (dates >= w.window_start) & (dates <= w.window_end)]
        for (family, r), u in parts.items():
            if family != "core85":
                continue
            cm, um = metrics(x, u)
            dyn.append(
                dict(
                    center=w.window_center_date,
                    start=w.window_start,
                    end=w.window_end,
                    resolution=r,
                    compression=master.loc[u.index, "population"].sum() / u.nunique(),
                    inc=um.inc.mean(),
                    lag=um.lag.mean(),
                    n_valid_counties=cm.inc.notna().sum(),
                )
            )
        if k % 10 == 0:
            print("dynamic", k + 1, "/", len(windows), flush=True)
    pd.DataFrame(dyn).to_csv(OUT / "corrected_dynamic.csv", index=False)
    # NYT deaths use the same observed dates, differencing and centered smoothing conventions.
    ds = []
    for year in [2020, 2021, 2022]:
        d = pd.read_csv(
            ROOT / f"data/raw/us_covid19_nyt/us-counties-{year}.csv",
            dtype={"fips": str},
            usecols=["date", "fips", "deaths"],
        )
        ds.append(d[d.fips.isin(master.index)])
    d = (
        pd.concat(ds)
        .drop_duplicates(["fips", "date"], keep="last")
        .sort_values(["fips", "date"])
    )
    d["daily"] = d.groupby("fips").deaths.diff()
    first = ~d.fips.duplicated()
    d.loc[first, "daily"] = d.loc[first, "deaths"]
    d["smooth"] = d.groupby("fips").daily.transform(
        lambda s: s.rolling(7, center=True, min_periods=1).mean()
    )
    deaths = d.pivot(index="fips", columns="date", values="smooth").fillna(0)
    deaths = append_borough_deaths(deaths)
    deaths.to_parquet(OUT / "deaths_smoothed.parquet")
    dr = []
    for (family, r), u in allparts.items():
        u = u.loc[u.index.intersection(deaths.index)]
        if family == "Louvain":
            continue
        x = deaths.loc[
            :, (deaths.columns >= "2021-07-01") & (deaths.columns <= "2021-11-30")
        ]
        cm, um = metrics(x, u)
        ci = bootstrap(cm)
        dr.append(
            dict(
                family=family,
                resolution=r,
                n_valid_counties=cm.inc.notna().sum(),
                n_units=len(um),
                n_valid_units_inc=um.inc.notna().sum(),
                n_valid_units_lag=um.lag.notna().sum(),
                compression=master.loc[u.index, "population"].sum() / u.nunique(),
                inc=um.inc.mean(),
                lag=um.lag.mean(),
                inc_lo=ci[0, 0],
                inc_hi=ci[1, 0],
                lag_lo=ci[0, 1],
                lag_hi=ci[1, 1],
            )
        )
    pd.DataFrame(dr).to_csv(OUT / "deaths_delta_frontiers.csv", index=False)
    dg = []
    for name, u in grad.items():
        death_common = sorted(set(u.dropna().index) & set(deaths.index))
        x = deaths.loc[
            death_common,
            (deaths.columns >= "2020-03-15") & (deaths.columns <= "2020-06-15"),
        ]
        cm, um = metrics(x, u.reindex(death_common))
        ci = bootstrap(cm, stat="median")
        dg.append(
            dict(
                partition=name,
                n_counties=len(death_common),
                compression=master.loc[death_common, "population"].sum() / len(um),
                n_valid_counties=cm.inc.notna().sum(),
                n_units=len(um),
                n_valid_units_inc=um.inc.notna().sum(),
                n_valid_units_lag=um.lag.notna().sum(),
                inc=um.inc.median(),
                lag=um.lag.median(),
                inc_lo=ci[0, 0],
                inc_hi=ci[1, 0],
                lag_lo=ci[0, 1],
                lag_hi=ci[1, 1],
            )
        )
    pd.DataFrame(dg).to_csv(OUT / "deaths_gradient.csv", index=False)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
