"""SARIMA / SARIMAX modelling."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

from electricity_demand.config import (
    DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER, SEASONALITY,
)


def _fit_one(y, order, seasonal_order, exog=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)


def grid_search(y_train, p_range=range(0, 7), d_range=range(0, 3),
                q_range=range(0, 7), seasonal_order=DEFAULT_SEASONAL_ORDER,
                exog=None, verbose=False) -> pd.DataFrame:
    """AIC grid search over non-seasonal (p,d,q) with fixed seasonal order."""
    rows = []
    combos = list(itertools.product(p_range, d_range, q_range))
    for i, (p, d, q) in enumerate(combos, 1):
        try:
            fit = _fit_one(y_train, (p, d, q), seasonal_order, exog)
            aic, bic = fit.aic, fit.bic
        except Exception:
            aic, bic = np.nan, np.nan
        rows.append({"order": (p, d, q), "p": p, "d": d, "q": q,
                     "aic": aic, "bic": bic})
        if verbose and i % 10 == 0:
            print(f"  {i}/{len(combos)}")
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def fit_sarimax(y_train, X_train=None, order=DEFAULT_ORDER,
                seasonal_order=DEFAULT_SEASONAL_ORDER):
    return _fit_one(y_train, order, seasonal_order, exog=X_train)


def forecast_sarimax(model_fit, horizon, X_test=None, index=None, alpha=0.05):
    fc = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    if index is not None:
        mean.index = index
        ci.index = index
    ci.columns = ["lower", "upper"]
    return mean, ci


def residual_diagnostics(fit, burn=SEASONALITY) -> dict:
    resid = pd.Series(fit.resid).iloc[burn:]
    lb = acorr_ljungbox(resid, lags=[12, 24, 52], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(resid)[:2]
    return {
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std()),
        "ljung_box": lb,
        "jarque_bera_p": float(jb_p),
    }
