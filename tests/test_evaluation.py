import numpy as np
import pandas as pd
from electricity_demand.evaluation import evaluate_forecast, mase


def _series(n=160):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    return pd.Series(50 + 5 * np.cos(2 * np.pi * np.arange(n) / 52), index=idx)


def test_mase_zero_for_perfect_forecast():
    y = _series()
    train, test = y.iloc[:-52], y.iloc[-52:]
    m = evaluate_forecast("perfect", test, test.copy(), train)
    assert abs(m["MASE"]) < 1e-9
    assert abs(m["RMSE"]) < 1e-9
    assert abs(m["Bias"]) < 1e-9


def test_bias_sign():
    y = _series()
    train, test = y.iloc[:-52], y.iloc[-52:]
    over = test + 2.0
    m = evaluate_forecast("over", test, over, train)
    assert m["Bias"] > 0
