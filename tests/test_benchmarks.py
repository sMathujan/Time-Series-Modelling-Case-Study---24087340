import numpy as np
import pandas as pd
from electricity_demand.models.benchmarks import (
    seasonal_naive_forecast, naive_forecast, all_benchmarks,
)


def _series(n=160):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    return pd.Series(50 + 5 * np.cos(2 * np.pi * np.arange(n) / 52), index=idx)


def test_forecast_length_matches_horizon():
    y = _series()
    train, test = y.iloc[:-52], y.iloc[-52:]
    fc = all_benchmarks(train, test)
    for name, pred in fc.items():
        assert len(pred) == len(test)
        assert (pred.index == test.index).all()


def test_seasonal_naive_repeats_last_season():
    y = _series()
    train, test = y.iloc[:-52], y.iloc[-52:]
    sn = seasonal_naive_forecast(train, len(test), 52, test.index)
    assert np.allclose(sn.to_numpy(), train.iloc[-52:].to_numpy())


def test_naive_is_constant():
    y = _series()
    train, test = y.iloc[:-10], y.iloc[-10:]
    nf = naive_forecast(train, len(test), test.index)
    assert (nf == train.iloc[-1]).all()
