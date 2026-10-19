"""Add five observed NYC county case series; preserve the original NYT panel."""

from pathlib import Path
import pandas as pd, numpy as np

R = Path(__file__).resolve().parents[1]
D = R / "data/processed"
O = R / "results/analysis"
n = pd.read_csv(R / "data/raw/nys/nys_county_cases_2020_2022.csv")
lookup = {
    "Bronx": "36005",
    "Kings": "36047",
    "New York": "36061",
    "Queens": "36081",
    "Richmond": "36085",
}
n = n[n.county.isin(lookup)].copy()
n["GEOID"] = n.county.map(lookup)
n["source_date"] = pd.to_datetime(n.test_date)
n["date"] = n.source_date + pd.Timedelta(days=1)
assert not n.duplicated(["GEOID", "date"]).any()
assert n.groupby("GEOID").size().eq(1036).all()
n = n.sort_values(["GEOID", "date"])
diff = n.groupby("GEOID").cumulative_number_of_positives.diff()
first = ~n.GEOID.duplicated()
diff.loc[first] = n.loc[first, "cumulative_number_of_positives"]
assert np.array_equal(diff, n.new_positives)
p = pd.read_parquet(
    D / "epidemic_panel_daily.parquet",
    columns=["GEOID", "date", "cases", "smoothed_cases"],
)
p.date = pd.to_datetime(p.date)
assert not set(lookup.values()) & set(p.GEOID)
n = n[n.date <= p.date.max()].copy()
n["cases"] = n.new_positives.clip(lower=0)
n["smoothed_cases"] = n.groupby("GEOID").cases.transform(
    lambda x: x.rolling(7, center=True, min_periods=1).mean()
)
z = pd.concat(
    [p, n[["GEOID", "date", "cases", "smoothed_cases"]]], ignore_index=True
).sort_values(["GEOID", "date"])
assert not z.duplicated(["GEOID", "date"]).any()
m = pd.read_parquet(D / "county_master_census2019.parquet")
assert set(m.GEOID) <= set(z.GEOID)
z.to_parquet(D / "epidemic_panel_daily_with_nyc.parquet", index=False)
n[["GEOID", "county", "source_date", "date", "cases", "smoothed_cases"]].to_csv(
    O / "nyc_source_alignment.csv", index=False
)
print(
    "Case coverage:",
    len(set(z.GEOID) & set(m.GEOID)),
    "continental county equivalents; NYC added",
    len(n),
    "daily rows",
    flush=True,
)
