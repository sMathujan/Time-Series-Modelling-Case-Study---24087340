"""Feature engineering: calendar, temperature, and supervised ML tables.

All lag / rolling features are built with shift() so a row for time t uses only
information available up to t-1 (see tests/test_features.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import BASE_HEAT, BASE_COOL

LAGS = [1, 2, 52]
ROLL_WINDOWS = [4, 8]


def _to_naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    return idx.tz_localize(None) if idx.tz is not None else idx


# ----------------------------------------------------------------------
# Temperature -> weekly features
# ----------------------------------------------------------------------

def build_weekly_temperature_features(temp_daily: pd.DataFrame,
                                      weekly_index: pd.DatetimeIndex,
                                      base_heat=BASE_HEAT,
                                      base_cool=BASE_COOL) -> pd.DataFrame:
    t = temp_daily["temperature_2m_mean"].copy()
    t.index = _to_naive(t.index)
    feats = pd.DataFrame({
        "temp_mean": t.resample("W").mean(),
        "temp_min": t.resample("W").min(),
        "temp_max": t.resample("W").max(),
        "heating_degree": np.maximum(base_heat - t, 0).resample("W").sum(),
        "cooling_degree": np.maximum(t - base_cool, 0).resample("W").sum(),
    })
    feats = feats.reindex(_to_naive(weekly_index)).interpolate("time")
    feats.index = weekly_index
    return feats


def assemble_feature_df(weekly_load: pd.Series,
                        temp_weekly: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"load_gw": weekly_load}).join(temp_weekly)
    return df.interpolate("time").dropna()


# ----------------------------------------------------------------------
# Calendar features
# ----------------------------------------------------------------------

def make_weekly_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    week = idx.isocalendar().week.astype(int).to_numpy()
    return pd.DataFrame({
        "wk_sin": np.sin(2 * np.pi * week / 52.0),
        "wk_cos": np.cos(2 * np.pi * week / 52.0),
        "month": idx.month.to_numpy(),
    }, index=index)


def make_hourly_features(load: pd.Series) -> pd.DataFrame:
    idx = load.index
    hour, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    df = pd.DataFrame({"load": load.to_numpy()}, index=idx)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


# ----------------------------------------------------------------------
# Supervised ML table (leakage-safe)
# ----------------------------------------------------------------------

def make_supervised_table(feature_df: pd.DataFrame,
                          lags=LAGS, roll_windows=ROLL_WINDOWS) -> pd.DataFrame:
    """Lag/rolling features via shift(); rolling stats on shift(1)."""
    df = feature_df.copy().join(make_weekly_calendar_features(feature_df.index))
    for L in lags:
        df[f"lag_{L}"] = df["load_gw"].shift(L)
    past = df["load_gw"].shift(1)
    for W in roll_windows:
        df[f"roll_mean_{W}"] = past.rolling(W).mean()
        df[f"roll_std_{W}"] = past.rolling(W).std()
    return df.dropna()


def feature_columns(table: pd.DataFrame, temp_cols) -> list:
    cal = ["wk_sin", "wk_cos", "month"]
    lagcols = [c for c in table.columns if c.startswith(("lag_", "roll_"))]
    return cal + list(temp_cols) + lagcols
