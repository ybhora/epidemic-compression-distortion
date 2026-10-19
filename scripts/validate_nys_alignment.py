"""S1: validate NYS processing dates against 57 non-NYC counties, without fitting NYC."""

import numpy as np
import pandas as pd
from validate_pipeline import ROOT, OUT


def main():
    nys = pd.read_csv(ROOT / "data/raw/nys/nys_county_cases_2020_2022.csv")
    nys["date"] = pd.to_datetime(nys.test_date)
    census = pd.read_csv(
        ROOT / "data/raw/census/co-est2019-alldata.csv",
        encoding="latin1",
        dtype={"STATE": str, "COUNTY": str},
    )
    census = census[(census.STATE == "36") & (census.SUMLEV == 50)].copy()
    census["county"] = census.CTYNAME.str.replace(" County", "", regex=False)
    census["GEOID"] = census.STATE + census.COUNTY
    assert not nys.duplicated(["county", "date"]).any()
    nys = nys.merge(
        census[["county", "GEOID"]], on="county", validate="many_to_one"
    ).sort_values(["GEOID", "date"])
    assert nys.GEOID.nunique() == 62 and nys.groupby("GEOID").size().eq(1036).all()
    nys["increment"] = nys.groupby("GEOID").cumulative_number_of_positives.diff()
    first = ~nys.GEOID.duplicated()
    nys.loc[first, "increment"] = nys.loc[first, "cumulative_number_of_positives"]
    assert np.array_equal(nys.increment, nys.new_positives)
    nyc = {"36005", "36047", "36061", "36081", "36085"}
    pieces = []
    for year in [2020, 2021, 2022]:
        frame = pd.read_csv(
            ROOT / f"data/raw/us_covid19_nyt/us-counties-{year}.csv",
            dtype={"fips": str},
        )
        pieces.append(frame[frame.state == "New York"])
    nyt = pd.concat(pieces).sort_values(["county", "date"])
    nyt.date = pd.to_datetime(nyt.date)
    nyt["daily"] = nyt.groupby("county").cases.diff()
    first = ~nyt.county.duplicated()
    nyt.loc[first, "daily"] = nyt.loc[first, "cases"]
    nyt.daily = nyt.daily.clip(lower=0)
    source = (
        nys[~nys.GEOID.isin(nyc)]
        .pivot(index="date", columns="GEOID", values="increment")
        .clip(lower=0)
    )
    target = (
        nyt[nyt.fips.notna()]
        .pivot(index="date", columns="fips", values="daily")
        .reindex(columns=source.columns)
    )
    assert len(source.columns) == 57
    phases = {
        "Lockdown": ("2020-03-15", "2020-06-15"),
        "Delta": ("2021-07-01", "2021-11-30"),
        "Omicron": ("2021-12-01", "2022-03-31"),
    }
    rows = []
    for phase, (lo, hi) in phases.items():
        for shift in range(-3, 4):
            shifted = source.copy()
            shifted.index += pd.Timedelta(days=shift)
            shifted = (
                shifted.reindex(target.index)
                .rolling(7, center=True, min_periods=1)
                .mean()
                .loc[lo:hi]
            )
            observed = target.rolling(7, center=True, min_periods=1).mean().loc[lo:hi]
            for county in source.columns:
                rows.append(
                    dict(
                        phase=phase,
                        county=county,
                        shift_nys_dates_days=shift,
                        correlation=shifted[county].corr(observed[county]),
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "nys_nyt_date_alignment.csv", index=False)
    summary = frame.groupby(["phase", "shift_nys_dates_days"]).correlation.agg(
        ["median", "min", "count"]
    )
    summary.to_csv(OUT / "nys_nyt_date_alignment_summary.csv")
    borough = (
        nys[nys.GEOID.isin(nyc)].groupby("date").cumulative_number_of_positives.sum()
    )
    borough.index += pd.Timedelta(days=1)
    city = nyt[nyt.county == "New York City"].set_index("date").cases
    comparison = pd.DataFrame(
        {"nys_borough_cumulative": borough, "nyt_city_cumulative": city}
    )
    comparison["difference_pct"] = 100 * (
        comparison.nys_borough_cumulative / comparison.nyt_city_cumulative - 1
    )
    comparison.to_csv(OUT / "nyc_city_total_comparison.csv")
    print(summary.xs(1, level="shift_nys_dates_days").to_string())


if __name__ == "__main__":
    main()
