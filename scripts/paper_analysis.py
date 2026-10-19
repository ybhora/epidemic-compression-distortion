"""Three-phase frontiers, lag-bound sensitivity and timing/shape decomposition."""

from pathlib import Path
import sys, os, pickle, json, warnings

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from validate_pipeline import metrics, load_master, EX, DATA
from recompute_empirical import bootstrap
from summarize_results import interp, phase

warnings.filterwarnings("ignore", category=RuntimeWarning)
O = ROOT / "results/paper"
OLD = ROOT / "results/analysis"
PH = {
    "Lockdown": ("2020-03-15", "2020-06-15"),
    "Delta": ("2021-07-01", "2021-11-30"),
    "Omicron": ("2021-12-01", "2022-03-31"),
}
m = load_master()
m = m[~m.index.str[:2].isin(EX)]
p = pd.read_parquet(DATA / "epidemic_panel_daily_with_nyc.parquet")
raw = (
    p.pivot(index="GEOID", columns="date", values="smoothed_cases")
    .reindex(m.index)
    .fillna(0)
)
deaths = pd.read_parquet(OLD / "deaths_smoothed.parquet")
deaths.columns = pd.to_datetime(deaths.columns)
parts = pickle.loads((OLD / "partitions.pkl").read_bytes())
can = {"States": m.state, "CSA": m.CSA.dropna(), "CBSA": m.CBSA.dropna()}
for n in [2, 3]:
    s = can["CBSA"]
    can[f"CBSA>={n}"] = s[s.map(s.value_counts()) >= n]
allparts = {**{(k, np.nan): v for k, v in can.items()}, **parts}
grad = {
    "States": can["States"],
    "CSA": can["CSA"],
    "CBSA>=2": can["CBSA>=2"],
    "Leiden coarse": parts["full", 4.650257],
    "Leiden intermediate": parts["core85", 33.471236],
    "Leiden fine": parts["core85", 111.186983],
}


def bootstrap_draws(cm, B=500, stat="mean"):
    rng = np.random.default_rng(SEED)
    units = []
    for _, g in cm.groupby("unit"):
        a = g[["inc", "lag"]].to_numpy()
        idx = rng.integers(len(a), size=(B, len(a)))
        units.append(np.nanmean(a[idx], axis=1))
    a = np.stack(units, axis=1)
    v = np.nanmean(a, axis=1) if stat == "mean" else np.nanmedian(a, axis=1)
    return v


def calc(x, u, fam, r, ph, signal, bound=28, boot=True):
    u = u.loc[u.index.intersection(x.index)]
    cm, um = metrics(x, u, max_lag=bound)
    cm["shape"] = cm.optimized_inc
    cm["timing"] = cm.inc - cm.optimized_inc
    decomp = cm.groupby("unit")[["shape", "timing"]].mean().mean()
    row = dict(
        phase=ph,
        signal=signal,
        family=fam,
        resolution=r,
        support=(
            fam
            if fam in ["full", "core85", "core90", "core95"]
            else ("core85" if fam == "Louvain" else "native")
        ),
        n_counties=len(cm),
        n_valid_counties=int(cm.inc.notna().sum()),
        n_valid_lag=int(cm.lag.notna().sum()),
        n_units=len(um),
        compression=m.loc[u.index, "population"].sum() / u.nunique(),
        inc=um.inc.mean(),
        lag=um.lag.mean(),
        shape=decomp["shape"],
        timing=decomp["timing"],
        bound_pct=100 * cm.lag.eq(bound).sum() / cm.lag.notna().sum(),
        max_lag=bound,
    )
    draws = bootstrap_draws(cm) if boot else None
    if boot:
        ci = np.nanquantile(draws, [0.025, 0.975], axis=0)
        for j, k in enumerate(["inc", "lag"]):
            row[k + "_lo"] = ci[0, j]
            row[k + "_hi"] = ci[1, j]
    return row, cm, um, draws


SEED = 42
rows = []
cs = []
us = []
draws = {}
for signal, a in [("cases", raw), ("deaths", deaths)]:
    for ph, (lo, hi) in PH.items():
        x = a.loc[:, a.columns.to_series().between(lo, hi).values]
        for (fam, r), u in allparts.items():
            if signal == "deaths" and fam == "Louvain":
                continue
            key = f"{signal}|{ph}|{fam}|{r}"
            row, cm, um, b = calc(x, u, fam, r, ph, signal)
            rows.append(row)
            draws[key] = b
            cs.append(cm.assign(phase=ph, signal=signal, family=fam, resolution=r))
            us.append(
                um.reset_index().assign(
                    phase=ph, signal=signal, family=fam, resolution=r
                )
            )
        pd.DataFrame(rows).to_csv(O / "static_frontiers.csv", index=False)
        print("static", signal, ph, flush=True)
pd.concat(cs).to_parquet(O / "county_metrics.parquet", index=False)
pd.concat(us).to_parquet(O / "unit_metrics.parquet", index=False)
with open(O / "bootstrap_draws.pkl", "wb") as f:
    pickle.dump(draws, f)
# Native-support gradient, retaining county tails and equal-unit resampling.
grows = []
gcs = []
gus = []
for name, u in grad.items():
    x = raw.loc[:, raw.columns.to_series().between(*PH["Lockdown"]).values]
    cm, um = metrics(x, u)
    ci = bootstrap(cm, stat="median")
    v = cm.dropna(subset=["lag"])
    pop = m.loc[v.geoid, "population"].to_numpy()
    n = u.value_counts()
    row = dict(
        phase="Lockdown",
        partition=name,
        support=(
            "full"
            if name == "Leiden coarse"
            else ("core85" if name.startswith("Leiden") else "native")
        ),
        n_counties=len(cm),
        n_valid_counties=len(v),
        n_units=len(um),
        singletons=int((n == 1).sum()),
        compression=m.loc[u.index, "population"].sum() / u.nunique(),
        unit_median_lag=um.lag.median(),
        lag_lo=ci[0, 1],
        lag_hi=ci[1, 1],
        median_inc=um.inc.median(),
        county_median=v.lag.median(),
        county_p90=v.lag.quantile(0.9),
        county_max=v.lag.max(),
    )
    for t in [7, 14]:
        row[f"county_gt{t}"] = 100 * v.lag.gt(t).mean()
        row[f"population_gt{t}"] = 100 * pop[v.lag.to_numpy() > t].sum() / pop.sum()
    grows.append(row)
    gcs.append(cm.assign(partition=name))
    gus.append(um.reset_index().assign(partition=name))
pd.DataFrame(grows).to_csv(O / "M3_gradient.csv", index=False)
pd.concat(gcs).to_csv(O / "M3_counties.csv", index=False)
pd.concat(gus).to_csv(O / "M3_units.csv", index=False)
# Pairwise common-support curves; recompute both aggregates and compression.
matched = []
for ph, (lo, hi) in PH.items():
    x = raw.loc[:, raw.columns.to_series().between(lo, hi).values]
    for name in ["States", "CSA", "CBSA>=2"]:
        ids = can[name].index.intersection(parts["core85", 33.471236].index)
        for fam, r, u in [(name, np.nan, can[name].loc[ids])] + [
            ("Leiden", r, u.loc[ids]) for (f, r), u in parts.items() if f == "core85"
        ]:
            row, _, _, _ = calc(x.loc[ids], u, fam, r, ph, "cases", boot=False)
            matched.append(dict(row, reference=name, support="pairwise common"))
    print("matched", ph, flush=True)
pd.DataFrame(matched).to_csv(O / "S-F_common_frontiers.csv", index=False)
# Full core curves under alternative search bounds; supports unchanged.
sens = []
for ph, (lo, hi) in PH.items():
    x = raw.loc[:, raw.columns.to_series().between(lo, hi).values]
    for bound in [14, 42]:
        for (fam, r), u in allparts.items():
            if fam not in ["States", "CSA", "CBSA>=2", "core85"] and not (
                bound == 42 and ph == "Lockdown" and fam in ["full", "CBSA", "CBSA>=3"]
            ):
                continue
            row, _, _, b = calc(
                x,
                u,
                fam,
                r,
                ph,
                "cases",
                bound,
                boot=(ph == "Lockdown" and bound == 42),
            )
            sens.append(row)
            if b is not None:
                draws[f"cases|{ph}|{fam}|{r}|42"] = b
    print("bounds", ph, flush=True)
pd.DataFrame(sens).to_csv(O / "S-I_lag_bound_frontiers.csv", index=False)
with open(O / "bootstrap_draws.pkl", "wb") as f:
    pickle.dump(draws, f)
# Dynamic: exact CSA compression from existing reference; also literal rounded 1.43M.
dy = pd.read_csv(OLD / "corrected_dynamic.csv")
dr = []
for center, g in dy.groupby("center"):
    c = m.loc[can["CSA"].index, "population"].sum() / can["CSA"].nunique()
    dr.append(
        dict(
            center=center,
            phase=phase(center),
            compression=c,
            lag_at_csa=interp(g, "lag", c),
            inc_at_csa=interp(g, "inc", c),
            lag_at_1430000=interp(g, "lag", 1430000),
            min_compression=g.compression.min(),
            max_compression=g.compression.max(),
        )
    )
pd.DataFrame(dr).to_csv(O / "M6_dynamic.csv", index=False)
# Reuse the same smoothing sensitivity values in the final S12 plotting stage.
pd.read_csv(OLD / "sensitivity.csv").to_csv(
    O / "S-I_existing_smoothing.csv", index=False
)
print("Paper analyses complete", flush=True)
