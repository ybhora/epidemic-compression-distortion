"""Test +/- one day around the externally validated NYC reporting-date alignment."""

import pandas as pd, pickle
from validate_pipeline import DATA, OUT, load_master, metrics

p = pd.read_parquet(DATA / "epidemic_panel_daily_with_nyc.parquet")
m = load_master()
p = p[p.GEOID.isin(m.index)]
parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
cb = m.CBSA.dropna()
u = {
    "States": m.state,
    "CSA": m.CSA.dropna(),
    "CBSA>=2": cb[cb.map(cb.value_counts()) >= 2],
    "Leiden core r33.47": parts[("core85", 33.471236)],
}
nyc = ["36005", "36047", "36061", "36081", "36085"]
rows = []
for extra in [-1, 0, 1]:
    q = p.copy()
    q.loc[q.GEOID.isin(nyc), "date"] += pd.Timedelta(days=extra)
    raw = q.pivot(index="GEOID", columns="date", values="smoothed_cases").fillna(0)
    for phase, start, end in [
        ("Lockdown", "2020-03-15", "2020-06-15"),
        ("Delta", "2021-07-01", "2021-11-30"),
        ("Omicron", "2021-12-01", "2022-03-31"),
    ]:
        x = raw.loc[:, (raw.columns >= start) & (raw.columns <= end)]
        for name, v in u.items():
            c, a = metrics(x, v)
            rows.append(
                dict(
                    nys_to_nyt_shift_days=1 + extra,
                    phase=phase,
                    partition=name,
                    inc=a.inc.mean(),
                    lag=a.lag.mean(),
                )
            )
pd.DataFrame(rows).to_csv(OUT / "nyc_date_sensitivity.csv", index=False)
print("NYC date-alignment sensitivity complete")
