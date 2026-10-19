from pathlib import Path
import pickle, sys, warnings
import numpy as np, pandas as pd
from validate_pipeline import load_master, metrics, ROOT, DATA, OUT, EX
from recompute_empirical import bootstrap

warnings.filterwarnings("ignore", category=RuntimeWarning)
master = load_master()
master = master[~master.index.str[:2].isin(EX)]
panel = pd.read_parquet(
    DATA / "epidemic_panel_daily_with_nyc.parquet",
    columns=["GEOID", "date", "cases", "smoothed_cases"],
)
panel = panel[panel.GEOID.isin(master.index)]
raw = panel.pivot(index="GEOID", columns="date", values="smoothed_cases").fillna(0)
master = master.loc[raw.index]
parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
u0 = parts[("core85", 33.471236)]
units = {
    "States": master.state,
    "CSA": master.CSA.dropna(),
    "CBSA>=2": master.CBSA.dropna(),
}
cb = units["CBSA>=2"]
units["CBSA>=2"] = cb[cb.map(cb.value_counts()) >= 2]
delta = raw.loc[:, (raw.columns >= "2021-07-01") & (raw.columns <= "2021-11-30")]
# Permute membership labels, preserving unit sizes; recompute each aggregate.
rng = np.random.default_rng(42)
nr = []
for phase, lo, hi in [
    ("Lockdown", "2020-03-15", "2020-06-15"),
    ("Delta", "2021-07-01", "2021-11-30"),
]:
    x = raw.loc[:, (raw.columns >= lo) & (raw.columns <= hi)]
    for name, u in {**units, "Leiden": u0}.items():
        cm, um = metrics(x, u)
        nr.append(
            dict(
                phase=phase,
                partition=name,
                rep=-1,
                inc=um.inc.mean(),
                lag=um.lag.mean(),
            )
        )
        for b in range(200):
            v = pd.Series(rng.permutation(u.values), index=u.index)
            cm, um = metrics(x, v)
            nr.append(
                dict(
                    phase=phase,
                    partition=name,
                    rep=b,
                    inc=um.inc.mean(),
                    lag=um.lag.mean(),
                )
            )
        print("null", phase, name, flush=True)
pd.DataFrame(nr).to_csv(OUT / "corrected_nulls.csv", index=False)
# Smoothing and lag-window sensitivities; same support/partitions, change one convention.
from importlib import import_module
import validate_pipeline as vp

sr = []
base = (
    panel.pivot(index="GEOID", columns="date", values="cases")
    .fillna(0)
    .sort_index(axis=1)
)
for days in [1, 7, 14]:
    smooth = base.T.rolling(days, center=True, min_periods=1).mean().T
    x = smooth.loc[
        :, (smooth.columns >= "2021-07-01") & (smooth.columns <= "2021-11-30")
    ]
    for maxlag in [14, 28, 42]:
        for name, u in {**units, "Leiden": u0}.items():
            cm, um = metrics(x, u, max_lag=maxlag)
            sr.append(
                dict(
                    smoothing=days,
                    max_lag=maxlag,
                    partition=name,
                    inc=um.inc.mean(),
                    lag=um.lag.mean(),
                )
            )
pd.DataFrame(sr).to_csv(OUT / "sensitivity.csv", index=False)
print("sensitivities done", flush=True)
