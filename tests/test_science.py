"""Independent checks of model conservation, lag and interpolation conventions."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sir_model import network_sir_deriv
from topologies import (
    star_shaped,
    fully_connected,
    unidirectional_cycle,
    two_cluster_bridge,
)
from validate_pipeline import metrics
from summarize_results import interp, admissible

for topology, size, phi in [
    (star_shaped, 25, 0.02),
    (fully_connected, 5, 0.02),
    (unidirectional_cycle, 5, 0.02),
    (two_cluster_bridge, 10, 0.2),
]:
    matrix = topology(size, phi)
    assert (matrix >= 0).all() and np.allclose(matrix.sum(axis=1), 1)
    population = np.full(size, 1000.0)
    infected = np.zeros(size)
    infected[0] = 1
    result = solve_ivp(
        network_sir_deriv,
        (0, 450),
        np.r_[population - infected, infected, np.zeros(size)],
        args=(matrix, np.full(size, 0.3), np.full(size, 0.1), matrix.T @ population),
        rtol=1e-7,
        atol=1e-9,
    )
    assert result.success
    assert np.allclose(
        result.y.reshape(3, size, -1).sum(axis=0), population[:, None], atol=1e-6
    )
try:
    star_shaped(100, 0.02)
    raise AssertionError("Infeasible star accepted")
except ValueError:
    pass

# Scalar implementation independently checks overlapping Pearson correlations.
t = np.arange(100.0)
raw = pd.DataFrame(
    [np.exp(-(((t - 35) / 9) ** 2)), 4 * np.exp(-(((t - 52) / 8) ** 2))],
    index=["a", "b"],
)
county, _ = metrics(raw, pd.Series("unit", index=raw.index))
aggregate = raw.sum(axis=0).to_numpy()
for i, row in county.iterrows():
    x = raw.loc[row.geoid].to_numpy()
    options = []
    for lag in sorted(range(-28, 29), key=lambda k: (abs(k), k)):
        a, b = (
            (x[lag:], aggregate[:-lag])
            if lag > 0
            else ((x[:lag], aggregate[-lag:]) if lag < 0 else (x, aggregate))
        )
        options.append((lag, np.corrcoef(a, b)[0, 1]))
    best = max(value for _, value in options)
    expected = next(abs(lag) for lag, value in options if value > best - 1e-12)
    assert row.lag == expected
    assert abs(row.inc - (1 - np.corrcoef(x, aggregate)[0, 1])) < 1e-12

curve = pd.DataFrame({"compression": [1.0, 10.0, 100.0], "lag": [1.0, 5.0, 1.0]})
assert np.isclose(interp(curve, "lag", np.sqrt(10)), 3)
assert np.isnan(interp(curve, "lag", 101))
assert np.isclose(
    admissible(curve, 2), 100
)  # all feasible segments, no monotonic envelope
assert np.isnan(admissible(curve, 0.5))
print("Passed independent SIR, scalar lag and interpolation checks.")
