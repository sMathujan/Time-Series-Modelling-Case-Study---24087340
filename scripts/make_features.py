"""Build the processed modelling dataset (weekly load + temperature features)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from electricity_demand.config import PROCESSED_DATA_DIR
from electricity_demand.data import to_weekly_gw, get_open_meteo_temperature
from electricity_demand.features import (
    build_weekly_temperature_features, assemble_feature_df,
)


def main():
    hourly = pd.read_csv(PROCESSED_DATA_DIR / "hourly_load.csv",
                         index_col=0, parse_dates=True).squeeze("columns")
    # hourly is already GW; resample to weekly mean
    weekly = hourly.resample("W").mean()
    weekly.name = "load_gw"
    weekly.to_csv(PROCESSED_DATA_DIR / "weekly_load.csv")

    temp_daily = get_open_meteo_temperature(
        start_date=str(hourly.index.min().date()),
        end_date=str(hourly.index.max().date()))
    temp_weekly = build_weekly_temperature_features(temp_daily, weekly.index)

    feature_df = assemble_feature_df(weekly, temp_weekly)
    feature_df.to_csv(PROCESSED_DATA_DIR / "feature_df.csv")
    print(f"feature_df: {feature_df.shape[0]} weeks, cols={list(feature_df.columns)}")


if __name__ == "__main__":
    main()
