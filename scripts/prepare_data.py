"""Build the Census master, daily cases and ordered G1/G2 graphs from raw inputs."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DATA = ROOT / "data/processed"
OUT = ROOT / "results/analysis"
EX = {"02", "15", "60", "66", "69", "72", "78", "00"}


def master():
    registry = pd.read_csv(
        ROOT / "data/reference/county_registry.csv",
        dtype={"GEOID": str, "state": str, "CSA": "string", "CBSA": "string"},
    )
    census = pd.read_csv(
        RAW / "census/co-est2019-alldata.csv",
        encoding="latin1",
        dtype={"STATE": str, "COUNTY": str},
    )
    census = census[census.SUMLEV == 50].copy()
    census["GEOID"] = census.STATE + census.COUNTY
    registry = registry.merge(
        census[["GEOID", "POPESTIMATE2019"]], on="GEOID", validate="one_to_one"
    )
    registry = registry.rename(columns={"POPESTIMATE2019": "population"})
    assert (
        len(registry) == 3108
        and registry.GEOID.is_unique
        and registry.population.gt(0).all()
    )
    registry.population = registry.population.astype(float)
    registry[["GEOID", "state", "CBSA", "CSA", "population", "lon", "lat"]].to_parquet(
        DATA / "county_master_census2019.parquet", index=False
    )
    return registry.set_index("GEOID")


def cases():
    pieces = []
    for year in [2020, 2021, 2022]:
        frame = pd.read_csv(
            RAW / f"us_covid19_nyt/us-counties-{year}.csv",
            usecols=["date", "fips", "cases"],
            dtype={"fips": str},
        )
        valid = frame.fips.str.fullmatch(r"\d{5}", na=False)
        # Preserve the full observed NYT panel here; continental restrictions
        # are applied against the declared 3,108-county master during analysis.
        frame = frame[valid]
        pieces.append(frame)
    frame = (
        pd.concat(pieces)
        .drop_duplicates(["fips", "date"], keep="last")
        .sort_values(["fips", "date"])
    )
    cumulative = frame.cases.copy()
    daily = frame.groupby("fips").cases.diff()
    first = ~frame.fips.duplicated()
    daily.loc[first] = cumulative.loc[first]
    frame["cases"] = daily.clip(lower=0).fillna(0)
    frame["smoothed_cases"] = frame.groupby("fips").cases.transform(
        lambda s: s.rolling(7, center=True, min_periods=1).mean()
    )
    frame = frame.rename(columns={"fips": "GEOID"})
    frame.date = pd.to_datetime(frame.date)
    assert not frame.duplicated(["GEOID", "date"]).any()
    frame.to_parquet(DATA / "epidemic_panel_daily.parquet", index=False)
    pd.DataFrame(
        [
            {
                "panel_rows": len(frame),
                "negative_increments_zeroed": int(daily.lt(0).sum()),
            }
        ]
    ).to_csv(OUT / "case_panel_validation.csv", index=False)
    print(f"Built {len(frame):,} daily case records.", flush=True)


def graphs(counties):
    ids = sorted(counties.index)
    lookup = {f: i for i, f in enumerate(ids)}
    pop = counties.population.reindex(ids).to_numpy()
    g1 = np.zeros((len(ids), len(ids)))
    g2 = np.zeros_like(g1)
    files = sorted((RAW / "safegraph/weekly_county").glob("*_2019_*.csv"))
    if len(files) != 52:
        raise ValueError(
            f"Expected 52 weekly 2019 flow files; found {len(files)}. Run python run.py download."
        )
    for number, path in enumerate(files, 1):
        frame = pd.read_csv(
            path,
            usecols=["geoid_o", "geoid_d", "pop_flows"],
            dtype={"geoid_o": str, "geoid_d": str},
        )
        i, j = frame.geoid_o.map(lookup), frame.geoid_d.map(lookup)
        valid = i.notna() & j.notna() & (i != j)
        ii, jj = i[valid].to_numpy(int), j[valid].to_numpy(int)
        weight = frame.loc[valid, "pop_flows"].to_numpy(float)
        np.add.at(g2, (ii, jj), weight)
        np.add.at(g1, (ii, jj), weight / pop[ii])
        if number % 10 == 0 or number == 52:
            print(f"Mobility weeks: {number}/52", flush=True)
    directory = DATA / "graphs/edge_tables"
    directory.mkdir(parents=True, exist_ok=True)
    for number, array in [(1, g1), (2, g2)]:
        # Preserve the original arithmetic order as well as its edge order.
        # G2 averaged each directed flow first, then added both directions.
        # The Census-rebuilt G1 normalized first and averaged after summing.
        if number == 2:
            directed_mean = array / 52
            array = directed_mean + directed_mean.T
        else:
            array = (array + array.T) / 52
        # Lexical upper-triangle order is the paper's edge order. It determines
        # igraph vertex order and therefore matters to fixed-seed clustering.
        i, j = np.where(np.triu(array, 1) > 0)
        name = "graph_1_census2019" if number == 1 else "graph_2"
        pd.DataFrame(
            {
                "geoid_i": np.array(ids)[i],
                "geoid_j": np.array(ids)[j],
                "weight": array[i, j],
            }
        ).to_parquet(directory / f"{name}.parquet", index=False)
        print(f"G{number}: {len(i):,} ordered edges", flush=True)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    counties = master()
    cases()
    graphs(counties)
