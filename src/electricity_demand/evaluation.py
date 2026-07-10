"""Evaluation metrics for the forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from electricity_demand.config import SEASONALITY


def mase(y_true, y_pred, y_train, seasonality: int = SEASONALITY) -> float:
    """Mean Absolute Scaled Error, scaled by in-sample seasonal-naive MAE."""
    yt = np.asarray(y_train)
    scale = np.mean(np.abs(yt[seasonality:] - yt[:-seasonality]))
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(err) / scale)


def evaluate_forecast(name, y_true, y_pred, y_train,
                      seasonality: int = SEASONALITY) -> dict:
    y_pred = pd.Series(y_pred).reindex(y_true.index)
    err = y_pred.to_numpy() - y_true.to_numpy()
    return {
        "model": name,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MASE": mase(y_true, y_pred, y_train, seasonality),
        "Bias": float(np.mean(err)),
    }


def coverage(ci: pd.DataFrame, y_true) -> float:
    inside = (y_true.to_numpy() >= ci["lower"].to_numpy()) & \
             (y_true.to_numpy() <= ci["upper"].to_numpy())
    return float(inside.mean())
