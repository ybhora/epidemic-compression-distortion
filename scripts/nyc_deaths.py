"""Official NYC borough COVID-19 deaths, by date of death (not NYT report date).

Raw daily total counts are used, not provider trailing averages. No city-total
allocation, scaling or inferred report-date adjustment is performed.
"""

import pandas as pd
import numpy as np
from validate_pipeline import ROOT

BOROUGHS = {"BX": "36005", "BK": "36047", "MN": "36061", "QN": "36081", "SI": "36085"}


def borough_deaths(columns, shift_days=0):
    d = pd.read_csv(ROOT / "data/raw/nyc_doh/deaths-by-day.csv")
    dates = pd.to_datetime(d.date_of_interest, format="%m/%d/%Y")
    assert dates.is_unique and dates.is_monotonic_increasing
    assert dates.equals(pd.Series(pd.date_range(dates.min(), dates.max())))
    a = d[[f"{b}_DEATH_COUNT" for b in BOROUGHS]].apply(pd.to_numeric, errors="raise")
    assert (
        a.notna().all().all()
        and (a >= 0).all().all()
        and np.equal(a, np.floor(a)).all().all()
    )
    a.index = dates + pd.Timedelta(days=shift_days)
    a.columns = list(BOROUGHS.values())
    requested = pd.DatetimeIndex(pd.to_datetime(columns))
    # Source explicitly starts at outbreak onset; prior dates contain no NYC deaths.
    assert requested.max() <= a.index.max()
    a = a.reindex(
        pd.date_range(min(requested.min(), a.index.min()), a.index.max()), fill_value=0
    )
    a = a.rolling(7, center=True, min_periods=1).mean().reindex(requested).T
    a.columns = columns
    assert a.notna().all().all()
    return a


def append_borough_deaths(deaths, shift_days=0):
    assert not set(BOROUGHS.values()) & set(
        deaths.index
    ), "Do not duplicate NYC counties"
    return pd.concat([deaths, borough_deaths(deaths.columns, shift_days)]).sort_index()
