"""NYC event-date sensitivity, not an inferred reconstruction of report dates."""

import pickle
import numpy as np, pandas as pd
from validate_pipeline import load_master, OUT, metrics, select_coarse_partition
from nyc_deaths import BOROUGHS, append_borough_deaths
from summarize_results import interp


def main():
    d = pd.read_parquet(OUT / "deaths_smoothed.parquet")
    base = d.drop(index=list(BOROUGHS.values()))
    m = load_master().loc[d.index]
    p = pickle.loads((OUT / "partitions.pkl").read_bytes())
    can = {"States": m.state, "CSA": m.CSA.dropna(), "CBSA": m.CBSA.dropna()}
    for k in [2, 3]:
        can[f"CBSA>={k}"] = can["CBSA"][
            can["CBSA"].map(can["CBSA"].value_counts()) >= k
        ]
    cand = {
        **{(n, np.nan): u for n, u in can.items()},
        **{k: u for k, u in p.items() if k[0] == "core85"},
    }
    grad = {n: can[n] for n in ["States", "CSA", "CBSA>=2"]}
    grad.update(
        {
            "Leiden coarse": select_coarse_partition(p, m)[1],
            "Leiden intermediate": p["core85", 33.471236],
            "Leiden fine": p["core85", 111.186983],
        }
    )
    rows = []
    gs = []
    comps = []
    for shift in [0, 7, 14, 21, 28]:
        a = append_borough_deaths(base, shift)
        x = a.loc[:, (a.columns >= "2021-07-01") & (a.columns <= "2021-11-30")]
        rr = []
        for (family, r), u in cand.items():
            _, v = metrics(x, u)
            rr.append(
                dict(
                    shift_days=shift,
                    family=family,
                    resolution=r,
                    compression=m.loc[u.index, "population"].sum() / u.nunique(),
                    inc=v.inc.mean(),
                    lag=v.lag.mean(),
                )
            )
        frame = pd.DataFrame(rr)
        rows += rr
        for c in frame[frame.resolution.isna()].itertuples():
            for key in ["inc", "lag"]:
                value = interp(frame[frame.family == "core85"], key, c.compression)
                comps.append(
                    dict(
                        shift_days=shift,
                        canonical=c.family,
                        metric=key,
                        canonical_value=getattr(c, key),
                        matched_core=value,
                        advantage=getattr(c, key) - value,
                    )
                )
        x = a.loc[:, (a.columns >= "2020-03-15") & (a.columns <= "2020-06-15")]
        for name, u in grad.items():
            _, v = metrics(x, u)
            gs.append(
                dict(
                    shift_days=shift,
                    partition=name,
                    inc=v.inc.median(),
                    lag=v.lag.median(),
                )
            )
        print("NYC death timing scenario", shift, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nyc_death_timing_frontiers.csv", index=False)
    pd.DataFrame(gs).to_csv(OUT / "nyc_death_timing_gradient.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / "nyc_death_timing_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
