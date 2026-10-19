from pathlib import Path
import numpy as np, pandas as pd

O = Path(__file__).resolve().parents[1] / "results/analysis"
R = O.parents[1]
y = pd.read_csv(O / "dynamic_summary.csv")
y = y[y.tolerance == 3].copy()
w = pd.read_csv(O / "corrected_dynamic.csv")[
    ["center", "start", "end"]
].drop_duplicates()
y = y.merge(w, on="center")
y["origin"] = pd.to_datetime(y.start) - pd.Timedelta(days=1)
y["available"] = pd.to_datetime(y.end) + pd.Timedelta(days=3)
y["log_scale"] = np.log(y.admissible)
m = pd.read_csv(R / "data/reference/rolling_mobility.csv")
# window_end is the last weekly file's Monday, not its Sunday coverage end.
# The +7-day assumption means availability the following Monday.
m["available"] = pd.to_datetime(m.window_end) + pd.Timedelta(days=7)
col = "national_mixing_rate_per_person_per_week"
rows = []
for _, r in y.iterrows():
    hist = m[m.available <= r.origin].sort_values("available")
    if len(hist):
        rows.append({**r.to_dict(), "mobility": hist.iloc[-1][col]})
a = pd.DataFrame(rows)
out = []
for i, r in a.iterrows():
    train = a[
        (a.available <= r.origin) & np.isfinite(a.log_scale) & np.isfinite(a.mobility)
    ]
    if len(train) < 30 or not np.isfinite(r.log_scale):
        continue
    X = np.c_[np.ones(len(train)), train.mobility]
    coef = np.linalg.lstsq(X, train.log_scale, rcond=None)[0]
    pred = np.array([1, r.mobility]) @ coef
    out.append(
        dict(
            center=r.center,
            n_train=len(train),
            observed=r.log_scale,
            mobility_prediction=pred,
            persistence_prediction=train.sort_values("available").iloc[-1].log_scale,
            mean_prediction=train.log_scale.mean(),
        )
    )
b = pd.DataFrame(out)
b.to_csv(O / "prediction_holdout.csv", index=False)
st = []
for model in ["mobility", "persistence", "mean"]:
    e = b[model + "_prediction"] - b.observed
    st.append(
        dict(
            model=model,
            n=len(b),
            MAE_log_scale=abs(e).mean(),
            RMSE_log_scale=np.sqrt((e * e).mean()),
        )
    )
pd.DataFrame(st).to_csv(O / "prediction_scores.csv", index=False)
print(pd.DataFrame(st).to_string(index=False))
