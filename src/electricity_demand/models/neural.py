"""LSTM forecaster on the hourly series.

Predicts the next 168 hours from the previous 168 hours plus calendar features,
rolled across the test window and aggregated to weekly. Scaling is fitted on the
training hours only. TensorFlow is imported lazily so the rest of the package
does not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from electricity_demand.config import HOURS_PER_WEEK, RANDOM_STATE
from electricity_demand.features import make_hourly_features


def _keras():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    return tf, Sequential, LSTM, Dense, Dropout, Input, EarlyStopping


def scale_train_only(features: pd.DataFrame, n_train_hours: int):
    scaler = MinMaxScaler().fit(features.iloc[:n_train_hours])
    scaled = pd.DataFrame(scaler.transform(features),
                          index=features.index, columns=features.columns)
    return scaled, scaler


def make_windows(scaled: np.ndarray, lookback, horizon, stride, target_col=0):
    X, y = [], []
    n = len(scaled)
    for s in range(0, n - lookback - horizon + 1, stride):
        X.append(scaled[s:s + lookback])
        y.append(scaled[s + lookback:s + lookback + horizon, target_col])
    return np.asarray(X), np.asarray(y)


def build_lstm(n_features, horizon, units=64, n_layers=1, dropout=0.1, lr=1e-3):
    tf, Sequential, LSTM, Dense, Dropout, Input, _ = _keras()
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    model = Sequential([Input(shape=(None, n_features))])
    for i in range(n_layers):
        model.add(LSTM(units, return_sequences=(i < n_layers - 1)))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(horizon))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def rolling_weekly_forecast(model, scaled, scaler, n_test_weeks, lookback,
                            target_col=0):
    values = scaled.to_numpy()
    n = len(values)
    test_start = n - n_test_weeks * HOURS_PER_WEEK
    lo, hi = scaler.data_min_[target_col], scaler.data_max_[target_col]
    inv = lambda x: x * (hi - lo) + lo

    preds, trues, idxs = [], [], []
    for b in range(n_test_weeks):
        origin = test_start + b * HOURS_PER_WEEK
        window = values[origin - lookback:origin][np.newaxis, ...]
        preds.append(inv(model.predict(window, verbose=0)[0]))
        trues.append(inv(values[origin:origin + HOURS_PER_WEEK, target_col]))
        idxs.append(scaled.index[origin:origin + HOURS_PER_WEEK])

    idx = pd.DatetimeIndex(np.concatenate([i.to_numpy() for i in idxs]))
    hourly = pd.DataFrame({"actual": np.concatenate(trues),
                           "pred": np.concatenate(preds)}, index=idx)
    weekly = hourly.resample("W").mean()
    return hourly, weekly


def run_lstm(hourly_load, test_weeks, lookback=HOURS_PER_WEEK,
             horizon=HOURS_PER_WEEK, stride=24, units=64, n_layers=1,
             dropout=0.1, epochs=30, batch_size=64):
    """Train and produce the rolling weekly LSTM forecast."""
    _, _, _, _, _, _, EarlyStopping = _keras()
    feats = make_hourly_features(hourly_load)
    n_train = len(feats) - test_weeks * HOURS_PER_WEEK
    scaled, scaler = scale_train_only(feats, n_train)

    X, y = make_windows(scaled.iloc[:n_train].to_numpy(), lookback, horizon, stride)
    n_val = max(1, int(0.15 * len(X)))
    model = build_lstm(X.shape[2], horizon, units, n_layers, dropout)
    model.fit(X[:-n_val], y[:-n_val], validation_data=(X[-n_val:], y[-n_val:]),
              epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    hourly, weekly = rolling_weekly_forecast(model, scaled, scaler,
                                             test_weeks, lookback)
    return model, hourly, weekly
