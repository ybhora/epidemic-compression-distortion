"""Export manuscript-facing numerical tables at full precision."""

import pickle
import numpy as np
import pandas as pd
from validate_pipeline import ROOT
from summarize_results import interp

O = ROOT / "results/paper"


def matched_interval(curve, metric, compression, draws, signal, phase):
    values = {}
    for row in curve.itertuples():
        key = f"{signal}|{phase}|{row.family}|{row.resolution}"
        values.setdefault(row.compression, []).append(
            draws[key][:, 0 if metric == "inc" else 1]
        )
    xs = np.array(sorted(values))
    if not xs[0] <= compression <= xs[-1]:
        return np.nan, np.nan
    hi = min(int(np.searchsorted(xs, compression)), len(xs) - 1)
    lo = max(0, hi - 1)
    upper = np.mean(values[xs[hi]], axis=0)
    if compression == xs[hi]:
        sample = upper
    else:
        weight = np.log(compression / xs[lo]) / np.log(xs[hi] / xs[lo])
        sample = (1 - weight) * np.mean(values[xs[lo]], axis=0) + weight * upper
    return np.quantile(sample, [0.025, 0.975])


def main():
    static = pd.read_csv(O / "static_frontiers.csv")
    draws = pickle.loads((O / "bootstrap_draws.pkl").read_bytes())
    matched, algorithms = [], []
    for (signal, phase), frame in static.groupby(["signal", "phase"], sort=False):
        core = frame[frame.family == "core85"]
        full = frame[frame.family == "full"]
        for marker in frame[frame.resolution.isna()].itertuples():
            for metric in ["lag", "inc"]:
                value = interp(core, metric, marker.compression)
                lo, hi = matched_interval(
                    core, metric, marker.compression, draws, signal, phase
                )
                matched.append(
                    dict(
                        signal=signal,
                        phase=phase,
                        partition=marker.family,
                        metric=metric,
                        compression=marker.compression,
                        marker=getattr(marker, metric),
                        marker_lo=getattr(marker, metric + "_lo"),
                        marker_hi=getattr(marker, metric + "_hi"),
                        core=value,
                        core_lo=lo,
                        core_hi=hi,
                        reduction_pct=100
                        * (getattr(marker, metric) - value)
                        / getattr(marker, metric),
                        full=interp(full, metric, marker.compression),
                    )
                )
            if signal == "cases" and marker.family in ["States", "CSA"]:
                algorithms.append(
                    dict(
                        phase=phase,
                        partition=marker.family,
                        compression=marker.compression,
                        Leiden=interp(core, "lag", marker.compression),
                        Louvain=interp(
                            frame[frame.family == "Louvain"], "lag", marker.compression
                        ),
                    )
                )
    pd.DataFrame(matched).to_csv(O / "matched_comparisons.csv", index=False)
    pd.DataFrame(algorithms).to_csv(O / "algorithm_comparison.csv", index=False)
    cases = static[static.signal == "cases"]
    cases.to_csv(O / "case_frontiers.csv", index=False)
    static[static.signal == "deaths"].to_csv(O / "death_frontiers.csv", index=False)
    retention = []
    for phase, frame in cases.groupby("phase", sort=False):
        for marker in frame[frame.resolution.isna()].itertuples():
            for family in ["core85", "core90", "core95"]:
                value = interp(frame[frame.family == family], "lag", marker.compression)
                retention.append(
                    dict(
                        phase=phase,
                        partition=marker.family,
                        family=family,
                        compression=marker.compression,
                        marker_lag=marker.lag,
                        core_lag=value,
                        gap=marker.lag - value,
                    )
                )
    pd.DataFrame(retention).to_csv(O / "retention_matched_lags.csv", index=False)
    county = pd.read_parquet(O / "county_metrics.parquet")
    negative = (
        county[county.signal == "cases"]
        .groupby(["phase", "family", "resolution"], dropna=False)
        .inc.agg(
            valid_counties="count",
            incoherence_above_one=lambda values: values.gt(1).sum(),
        )
    )
    negative["percentage"] = (
        100 * negative.incoherence_above_one / negative.valid_counties
    )
    negative.to_csv(O / "incoherence_above_one.csv")
    bound = pd.read_csv(O / "S-I_lag_bound_frontiers.csv")
    rows = []
    for phase in cases.phase.unique():
        for maximum in [14, 28, 42]:
            frame = (
                cases[cases.phase == phase]
                if maximum == 28
                else bound[(bound.phase == phase) & (bound.max_lag == maximum)]
            )
            core = frame[frame.family == "core85"]
            for marker in frame[
                frame.family.isin(["States", "CSA", "CBSA>=2"])
            ].itertuples():
                rows.append(
                    dict(
                        phase=phase,
                        partition=marker.family,
                        max_lag=maximum,
                        marker=marker.lag,
                        core=interp(core, "lag", marker.compression),
                    )
                )
    pd.DataFrame(rows).to_csv(O / "matched_lag_bounds.csv", index=False)
    saturation = cases[
        ["phase", "family", "resolution", "n_valid_lag", "bound_pct"]
    ].copy()
    saturation["n_at_bound"] = np.rint(
        saturation.n_valid_lag * saturation.bound_pct / 100
    ).astype(int)
    saturation.to_csv(O / "lag_bound_saturation.csv", index=False)
    dynamic = pd.read_csv(O / "M6_dynamic.csv")
    dynamic.groupby("phase")[["lag_at_csa", "inc_at_csa"]].mean().to_csv(
        O / "dynamic_phase_means.csv"
    )
    admissible = pd.read_csv(ROOT / "results/analysis/dynamic_summary.csv")
    admissible["below_csa"] = admissible.admissible < admissible.csa_reference
    admissible.groupby("tolerance").below_csa.agg(
        n_windows="count", percentage=lambda values: 100 * values.mean()
    ).to_csv(O / "admissible_below_csa.csv")
    print("Full-precision paper tables written to results/paper.")


if __name__ == "__main__":
    main()
