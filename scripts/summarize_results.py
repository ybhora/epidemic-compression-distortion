import numpy as np, pandas as pd, json
from pathlib import Path

O = Path(__file__).resolve().parents[1] / "results/analysis"
PH = [
    ("Lockdown", "2020-03-15", "2020-06-15"),
    ("Reopening", "2020-06-15", "2020-11-01"),
    ("Fall/Winter", "2020-11-01", "2021-03-01"),
    ("Recovery/Delta", "2021-03-01", "2021-11-01"),
    ("Omicron", "2021-11-01", "2022-03-02"),
]


def phase(d):
    return next((n for n, a, b in PH if a <= str(d) < b), "Outside")


def interp(d, key, x):
    a = d[["compression", key]].dropna().groupby("compression")[key].mean().sort_index()
    return (
        np.interp(np.log(x), np.log(a.index), a.values)
        if a.index.min() <= x <= a.index.max()
        else np.nan
    )


def admissible(d, t):
    a = (
        d[["compression", "lag"]]
        .dropna()
        .groupby("compression")
        .lag.mean()
        .sort_index()
    )
    x = np.log(a.index.to_numpy())
    y = a.to_numpy()
    ok = list(x[y <= t])
    for i in range(len(x) - 1):
        if (y[i] - t) * (y[i + 1] - t) < 0:
            ok.append(x[i] + (t - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return np.exp(max(ok)) if ok else np.nan


if __name__ == "__main__":
    d = pd.read_csv(O / "corrected_dynamic.csv")
    static = pd.read_csv(O / "corrected_delta_frontiers.csv")
    csa = static.loc[static.family == "CSA", "compression"].iloc[0]
    rows = []
    for c, g in d.groupby("center"):
        for t in [2, 3, 4, 5]:
            rows.append(
                dict(
                    center=c,
                    phase=phase(c),
                    tolerance=t,
                    admissible=admissible(g, t),
                    inc_at_csa=interp(g, "inc", csa),
                    csa_reference=csa,
                )
            )
    s = pd.DataFrame(rows)
    s.to_csv(O / "dynamic_summary.csv", index=False)
    a = s.groupby(["tolerance", "phase"]).admissible.median().unstack()
    a.to_csv(O / "phase_medians.csv")
    print(a.to_string())
    print(
        "below CSA",
        s.assign(below=s.admissible < csa).groupby("tolerance").below.mean().to_dict(),
    )
    allrows = []
    for name in ["corrected_delta_frontiers", "deaths_delta_frontiers"]:
        d = pd.read_csv(O / (name + ".csv"))
        can = d[d.resolution.isna()]
        for _, m in can.iterrows():
            for f in ["core85", "full", "Louvain"]:
                g = d[d.family == f]
                if not len(g):
                    continue
                for metric in ["inc", "lag"]:
                    allrows.append(
                        dict(
                            signal=name,
                            canonical=m.family,
                            frontier=f,
                            metric=metric,
                            canonical_value=m[metric],
                            matched_value=interp(g, metric, m.compression),
                            gap=m[metric] - interp(g, metric, m.compression),
                        )
                    )
    pd.DataFrame(allrows).to_csv(O / "frontier_comparisons.csv", index=False)
    print(pd.DataFrame(allrows).to_string(index=False))
