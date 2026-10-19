import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np, pandas as pd
from validate_pipeline import metrics

# Singleton fidelity, raw aggregation (not equal z-scored county aggregation),
# unit-first weights, constant-series exclusion, and affine invariance.
t = np.arange(100)
x = np.exp(-(((t - 35) / 8) ** 2))
y = np.exp(-(((t - 48) / 8) ** 2))
r = pd.DataFrame([x, 10 * y, x], index=["a", "b", "c"])
u = pd.Series(["A", "A", "B"], index=r.index)
c, v = metrics(r, u)
assert abs(v.loc["B", "inc"]) < 1e-12 and v.loc["B", "lag"] == 0
assert c.loc[c.geoid == "a", "lag"].iloc[0] > c.loc[c.geoid == "b", "lag"].iloc[0]
c2, v2 = metrics(r * 13 + 2, u)
assert np.allclose(v, v2)
_, constant = metrics(
    pd.DataFrame(np.zeros((1, 100)), index=["a"]), pd.Series(["A"], index=["a"])
)
assert constant.isna().all().all()
assert not np.isclose(c.inc.mean(), v.inc.mean())
print("Passed five scientific metric checks.")

# Correlation must not be inferred from fewer than three paired observations.
short = pd.DataFrame([[1.0, 2.0]], index=["a"])
_, v = metrics(short, pd.Series(["u"], index=["a"]), max_lag=0)
assert v.isna().all().all()
# Linear series have equal correlation at many shifts: tie resolution must select zero.
linear = pd.DataFrame([np.arange(100.0)], index=["a"])
_, tied = metrics(linear, pd.Series(["u"], index=["a"]))
assert tied.loc["u", "lag"] == 0
print("Smallest-absolute-lag tie check passed.")

# Undefined optimized correlation must stay missing in diagnostic exports.
c, _ = metrics(
    pd.DataFrame([[0.0] * 80], index=["constant"]), pd.Series(["u"], index=["constant"])
)
assert c[["inc", "lag", "optimized_inc"]].isna().all().all()
# Raw positive counts can be anticorrelated with their aggregate (incoherence 2).
v = np.sin(np.arange(80.0))
c, _ = metrics(
    pd.DataFrame([10 + v, 10 - 2 * v], index=["a", "b"]),
    pd.Series(["u", "u"], index=["a", "b"]),
)
assert c.loc[c.geoid == "a", "inc"].iloc[0] > 1.999999
print("Undefined-diagnostic and full incoherence-range checks passed.")
