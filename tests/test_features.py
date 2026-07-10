import numpy as np
import pandas as pd
from electricity_demand.features import make_supervised_table


def _feature_df(n=160):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    load = 50 + 5 * np.cos(2 * np.pi * np.arange(n) / 52)
    return pd.DataFrame({"load_gw": load, "temp_mean": np.arange(n) % 20}, index=idx)


def test_lag_features_do_not_use_future():
    df = _feature_df()
    table = make_supervised_table(df)
    # lag_1 at row t must equal load at t-1
    manual = df["load_gw"].shift(1).loc[table.index]
    assert np.allclose(table["lag_1"].to_numpy(), manual.to_numpy())


def test_no_missing_target():
    df = _feature_df()
    table = make_supervised_table(df)
    assert table["load_gw"].isna().sum() == 0
    # lag_52 forces at least 52 rows to be dropped
    assert len(table) <= len(df) - 52
