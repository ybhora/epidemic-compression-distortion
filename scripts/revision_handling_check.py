"""Check the native-support frontier ordering under alternative negative-revision handling."""

from nyc_deaths import append_borough_deaths
import pickle
import numpy as np
import pandas as pd
from validate_pipeline import ROOT, DATA, OUT, load_master, metrics, EX
from summarize_results import interp


def main():
    master = load_master()
    master = master[~master.index.str[:2].isin(EX)]
    panel = pd.read_parquet(
        DATA / "epidemic_panel_daily_with_nyc.parquet",
        columns=["GEOID", "date", "cases", "smoothed_cases"],
    )
    ids = sorted(set(master.index) & set(panel.GEOID))
    master = master.loc[ids]
    can = {
        "States": master.state,
        "CSA": master.CSA.dropna(),
        "CBSA": master.CBSA.dropna(),
    }
    for n in [2, 3]:
        can[f"CBSA>={n}"] = can["CBSA"][
            can["CBSA"].map(can["CBSA"].value_counts()) >= n
        ]
    parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
    parts = {
        **{(k, np.nan): u for k, u in can.items()},
        **{k: u for k, u in parts.items() if k[0] == "core85"},
    }
    raw = pd.concat(
        pd.read_csv(
            ROOT / f"data/raw/us_covid19_nyt/us-counties-{year}.csv",
            dtype={"fips": str},
            usecols=["date", "fips", "cases", "deaths"],
        )
        for year in [2020, 2021, 2022]
    )
    raw = (
        raw[raw.fips.isin(ids)]
        .drop_duplicates(["fips", "date"], keep="last")
        .sort_values(["fips", "date"])
    )
    rows = []
    for signal, clip in [("cases", False), ("deaths", True)]:
        daily = raw.groupby("fips")[signal].diff()
        first = ~raw.fips.duplicated()
        daily.loc[first] = raw.loc[first, signal]
        if clip:
            daily = daily.clip(lower=0)
        sm = daily.groupby(raw.fips).transform(
            lambda x: x.rolling(7, center=True, min_periods=1).mean()
        )
        a = (
            raw[["fips", "date"]]
            .assign(value=sm)
            .pivot(index="fips", columns="date", values="value")
            .reindex(ids)
            .fillna(0)
        )
        if signal == "cases":
            nyc = panel[
                panel.GEOID.isin(["36005", "36047", "36061", "36081", "36085"])
            ].pivot(index="GEOID", columns="date", values="smoothed_cases")
            nyc.columns = nyc.columns.astype(str)
            a.loc[nyc.index] = nyc.reindex(columns=a.columns).fillna(0)
        else:
            a = append_borough_deaths(a.loc[a.index.intersection(raw.fips.unique())])
        a = a.loc[:, (a.columns >= "2021-07-01") & (a.columns <= "2021-11-30")]
        for (family, r), u in parts.items():
            u = u.loc[u.index.intersection(a.index)]
            c, v = metrics(a, u)
            rows.append(
                dict(
                    signal=signal,
                    negative_revisions="zeroed" if clip else "retained",
                    family=family,
                    resolution=r,
                    n_valid_counties=c.inc.notna().sum(),
                    compression=master.loc[u.index, "population"].sum() / u.nunique(),
                    inc=v.inc.mean(),
                    lag=v.lag.mean(),
                )
            )
        print(signal, "alternative handling complete", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "revision_handling_frontiers.csv", index=False)
    comparisons = []
    for signal, g in d.groupby("signal"):
        for c in g[g.resolution.isna()].itertuples():
            for metric in ["inc", "lag"]:
                value = interp(g[g.family == "core85"], metric, c.compression)
                comparisons.append(
                    dict(
                        signal=signal,
                        canonical=c.family,
                        metric=metric,
                        canonical_value=getattr(c, metric),
                        matched_core_value=value,
                        advantage=getattr(c, metric) - value,
                    )
                )
    out = pd.DataFrame(comparisons)
    out.to_csv(OUT / "revision_handling_comparisons.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
