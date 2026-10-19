"""Shared county metrics and declared-support conventions (Supplement S1)."""

from pathlib import Path
import numpy as np, pandas as pd, json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results/analysis"
EX = {"02", "15", "60", "66", "69", "72", "78", "00"}


def load_master():
    return pd.read_parquet(DATA / "county_master_census2019.parquet").set_index("GEOID")


def select_coarse_partition(parts, master):
    """Choose full-support Figure 3 candidate nearest States in log compression.

    Selection uses only population and membership, never epidemic distortion.
    """
    candidates = [(r, u) for (family, r), u in parts.items() if family == "full"]

    def mismatch(candidate):
        r, u = candidate
        ids = u.index.intersection(master.index)
        population = master.loc[ids, "population"].sum()
        compression = population / u.loc[ids].nunique()
        target = population / master.loc[ids, "state"].nunique()
        return abs(np.log(compression / target)), r

    return min(candidates, key=mismatch)


def metrics(raw, unit, max_lag=28):
    """Return county and unit summaries on the declared common membership support.

    Rows of raw are county identifiers and columns are ordered, equally spaced days.
    County correlation is computed against the sum of unstandardized unit counts.
    The caller supplies smoothing and handles missing panel entries explicitly.
    Undefined correlations stay missing; units are equally weighted by callers.
    """
    ids = raw.index.intersection(unit.dropna().index)
    x = raw.loc[ids].to_numpy(float)
    u = unit.loc[ids].astype(str)
    agg = raw.loc[ids].groupby(u).sum(min_count=1)
    y = agg.loc[u].to_numpy(float)

    def corr(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        a = np.where(valid, a, np.nan)
        b = np.where(valid, b, np.nan)
        a = a - np.nanmean(a, axis=1)[:, None]
        b = b - np.nanmean(b, axis=1)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.clip(
                np.nansum(a * b, axis=1)
                / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1)),
                -1,
                1,
            )
        return np.where(valid.sum(axis=1) >= 3, r, np.nan)

    inc = 1 - corr(x, y)
    best = np.full(len(x), -np.inf)
    lags = np.full(len(x), np.nan)
    for lag in sorted(range(-max_lag, max_lag + 1), key=lambda k: (abs(k), k)):
        a, b = (
            (x[:, lag:], y[:, :-lag])
            if lag > 0
            else ((x[:, :lag], y[:, -lag:]) if lag < 0 else (x, y))
        )
        c = corr(a, b)
        sel = np.isfinite(c) & (c > best + 1e-12)
        best[sel] = c[sel]
        lags[sel] = abs(lag)
    cm = pd.DataFrame(
        {
            "geoid": ids,
            "unit": u.values,
            "inc": inc,
            "lag": lags,
            "optimized_inc": np.where(np.isfinite(best), 1 - best, np.nan),
        }
    )
    um = cm.groupby("unit")[["inc", "lag"]].mean()
    return cm, um
