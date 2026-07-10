"""Recompute the model comparison table from saved forecasts."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from electricity_demand.config import FORECAST_DIR, METRICS_DIR, PROCESSED_DATA_DIR, TEST_WEEKS
from electricity_demand.evaluation import evaluate_forecast


def main():
    fc = pd.read_csv(FORECAST_DIR / "all_forecasts.csv", index_col=0, parse_dates=True)
    weekly = pd.read_csv(PROCESSED_DATA_DIR / "weekly_load.csv",
                         index_col=0, parse_dates=True).squeeze("columns")
    train = weekly.iloc[:-TEST_WEEKS]
    actual = fc["actual"]

    rows = [evaluate_forecast(c, actual, fc[c], train)
            for c in fc.columns if c != "actual"]
    table = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
