"""Rebuild the archived national mobility predictor (optional, ~10 GB raw inputs).

The historical denominator is deliberately retained: a positive constant
rescaling does not change an OLS fit with an intercept. Current Census
populations are used for all empirical partition compression calculations.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = pd.read_csv(ROOT / "data/reference/rolling_mobility.csv")
    population = pd.read_csv(
        ROOT / "data/reference/rolling_mobility_population.csv", dtype={"GEOID": str}
    )
    ids = set(population.GEOID)
    weeks = pd.date_range(
        reference.window_start.min(), reference.window_end.max(), freq="7D"
    )
    totals = {}
    for week in weeks:
        name = f"weekly_county2county_{week:%Y_%m_%d}.csv"
        path = ROOT / "data/raw/safegraph/weekly_county" / name
        frame = pd.read_csv(
            path,
            usecols=["geoid_o", "geoid_d", "pop_flows"],
            dtype={"geoid_o": str, "geoid_d": str},
        )
        valid = (
            frame.geoid_o.isin(ids)
            & frame.geoid_d.isin(ids)
            & (frame.geoid_o != frame.geoid_d)
            & np.isfinite(frame.pop_flows)
            & frame.pop_flows.gt(0)
        )
        totals[week] = frame.loc[valid, "pop_flows"].sum()
        print(name, flush=True)
    rebuilt = []
    for row in reference.itertuples():
        group = pd.date_range(row.window_start, row.window_end, freq="7D")
        rebuilt.append(
            np.mean([totals[d] for d in group]) / population.population.sum()
        )
    column = "national_mixing_rate_per_person_per_week"
    output = reference.copy()
    output[column] = rebuilt
    destination = ROOT / "results/analysis"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "rebuilt_rolling_mobility.csv", index=False)
    assert np.allclose(
        reference[column], rebuilt, rtol=1e-12, atol=1e-12
    ), "Archived predictor differs"
    print("All archived predictor values reproduced.")


if __name__ == "__main__":
    main()
