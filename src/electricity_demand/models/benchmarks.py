"""Benchmark forecasters (fixed forecast origin, no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import SEASONALITY


def mean_forecast(train, horizon, index) -> pd.Series:
    return pd.Series(train.mean(), index=index, name="mean")


def naive_forecast(train, horizon, index) -> pd.Series:
    return pd.Series(train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(train, horizon, seasonality=SEASONALITY,
                            index=None) -> pd.Series:
    last_season = train.iloc[-seasonality:].to_numpy()
    n_tiles = int(np.ceil(horizon / seasonality))
    values = np.tile(last_season, n_tiles)[:horizon]
    return pd.Series(values, index=index, name="seasonal_naive")


def drift_forecast(train, horizon, index) -> pd.Series:
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    values = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    return pd.Series(values, index=index, name="drift")


def all_benchmarks(train, test) -> dict:
    idx, h = test.index, len(test)
    return {
        "mean": mean_forecast(train, h, idx),
        "naive": naive_forecast(train, h, idx),
        "seasonal_naive": seasonal_naive_forecast(train, h, SEASONALITY, idx),
        "drift": drift_forecast(train, h, idx),
    }
