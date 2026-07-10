"""Feature-based ML models (Random Forest / Gradient Boosting).

Forecasts the test window recursively so that target-lag features are never
read from actual future values (fair multi-step, no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from electricity_demand.config import RANDOM_STATE
from electricity_demand.features import (
    make_supervised_table, feature_columns, make_weekly_calendar_features,
    LAGS, ROLL_WINDOWS,
)


def make_ml_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    return make_supervised_table(feature_df)


def fit_gradient_boosting(X_train, y_train):
    return GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE).fit(X_train, y_train)


def fit_random_forest(X_train, y_train):
    return RandomForestRegressor(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)


def _feature_row(history_load, exog_row, cal_row, lags, roll_windows):
    feat = {}
    for L in lags:
        feat[f"lag_{L}"] = history_load.iloc[-L]
    for W in roll_windows:
        w = history_load.iloc[-W:]
        feat[f"roll_mean_{W}"] = w.mean()
        feat[f"roll_std_{W}"] = w.std()
    feat.update(exog_row)
    feat.update(cal_row)
    return feat


def recursive_forecast(model, train_load, future_df, temp_cols, feat_cols,
                       lags=LAGS, roll_windows=ROLL_WINDOWS) -> pd.Series:
    """Recursive multi-step forecast; predictions feed back as future lags."""
    history = train_load.copy()
    cal = make_weekly_calendar_features(future_df.index)
    preds = []
    for ts in future_df.index:
        exog_row = {c: future_df.at[ts, c] for c in temp_cols}
        cal_row = cal.loc[ts].to_dict()
        row = _feature_row(history, exog_row, cal_row, lags, roll_windows)
        x = pd.DataFrame([row])[feat_cols]
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history = pd.concat([history, pd.Series([yhat], index=[ts])])
    return pd.Series(preds, index=future_df.index)


def predict_feature_model(model, X_test, index=None) -> pd.Series:
    """Direct (non-recursive) prediction, for pipeline convenience."""
    pred = model.predict(X_test)
    return pd.Series(pred, index=index if index is not None else X_test.index)
