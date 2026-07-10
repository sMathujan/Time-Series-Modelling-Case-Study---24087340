"""End-to-end weekly forecasting pipeline."""

from __future__ import annotations

import pandas as pd

from electricity_demand.config import (
    PROCESSED_DATA_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR,
    TEST_WEEKS, SEASONALITY, DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER, DEFAULT_EXOG,
    HOURS_PER_WEEK,
)
from electricity_demand.data import load_processed_data
from electricity_demand.evaluation import evaluate_forecast
from electricity_demand.plotting import plot_forecasts, plot_error_diagnostics

from electricity_demand.models.benchmarks import all_benchmarks
from electricity_demand.models.sarimax import fit_sarimax, forecast_sarimax
from electricity_demand.models.feature_models import (
    make_ml_table, fit_random_forest, recursive_forecast,
)
from electricity_demand.features import feature_columns


def run_pipeline(test_weeks=TEST_WEEKS, include_sarimax=True,
                 include_feature_model=True, include_neural=False,
                 include_bayesian=False, hourly_load=None):
    """Load processed data, fit models, evaluate, and save outputs."""
    data = load_processed_data()
    y = data["load_gw"]
    train, test = y.iloc[:-test_weeks], y.iloc[-test_weeks:]
    horizon = len(test)

    forecasts = dict(all_benchmarks(train, test))

    # --- SARIMAX (with temperature exog if present) ---
    if include_sarimax:
        exog_cols = [c for c in DEFAULT_EXOG if c in data.columns]
        if exog_cols:
            X = data[exog_cols]
            X_train, X_test = X.iloc[:-test_weeks], X.iloc[-test_weeks:]
        else:
            X_train = X_test = None
        fit = fit_sarimax(train, X_train, DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER)
        mean, _ = forecast_sarimax(fit, horizon, X_test, test.index)
        forecasts["sarimax"] = mean

    # --- feature-based model (recursive) ---
    if include_feature_model:
        table = make_ml_table(data)
        temp_cols = [c for c in DEFAULT_EXOG if c in data.columns]
        feat_cols = feature_columns(table, temp_cols)
        y_tbl = table["load_gw"]
        X_tbl = table[feat_cols]
        model = fit_random_forest(X_tbl.iloc[:-test_weeks], y_tbl.iloc[:-test_weeks])
        train_load = data["load_gw"].loc[:table.index[-test_weeks - 1]]
        future_df = data.loc[table.index[-test_weeks:]]
        forecasts["feature_model"] = recursive_forecast(
            model, train_load, future_df, temp_cols, feat_cols)

    # --- optional neural (LSTM on hourly) ---
    if include_neural:
        if hourly_load is None:
            raise ValueError("include_neural=True requires hourly_load (GW series).")
        from electricity_demand.models.neural import run_lstm
        _, _, weekly = run_lstm(hourly_load, test_weeks)
        forecasts["neural"] = weekly["pred"].reindex(test.index)

    if include_bayesian:
        from electricity_demand.models.bayesian import fit_bayesian
        fit_bayesian()  # raises NotImplementedError

    # --- evaluate ---
    metrics = [evaluate_forecast(name, test, pred.reindex(test.index), train)
               for name, pred in forecasts.items()]
    metrics_df = pd.DataFrame(metrics).sort_values("MASE").reset_index(drop=True)

    # --- save ---
    for d in (FORECAST_DIR, METRICS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)
    forecast_df.to_csv(FORECAST_DIR / "all_forecasts.csv")
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    plot_forecasts(train, test, forecasts, FIGURE_DIR / "forecast_comparison.png")
    plot_error_diagnostics(test, forecasts, FIGURE_DIR / "error_diagnostics.png")

    return metrics_df, forecast_df
