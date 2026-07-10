"""Data acquisition and preparation."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from electricity_demand.config import (
    OPSD_URL, LOAD_COLUMN, START_DATE,
    OPEN_METEO_URL, BERLIN_LAT, BERLIN_LON,
    PROCESSED_DATA_DIR,
)


# ----------------------------------------------------------------------
# Electricity load
# ----------------------------------------------------------------------

def load_hourly_load(url: str = OPSD_URL, column: str = LOAD_COLUMN) -> pd.Series:
    """Download the OPSD 60-min file; return German load (MW) from START_DATE."""
    df = pd.read_csv(url, usecols=["utc_timestamp", column],
                     parse_dates=["utc_timestamp"])
    df = df.rename(columns={"utc_timestamp": "date", column: "load_mw"})
    load = df.set_index("date").sort_index()["load_mw"].astype(float)
    load = load[load.notna()].loc[START_DATE:]
    load.name = "load_mw"
    return load


def to_daily_gw(hourly_mw: pd.Series) -> pd.Series:
    daily = hourly_mw.resample("D").mean() / 1000.0
    daily = daily.asfreq("D").interpolate("time")
    daily.name = "load_gw"
    return daily


def to_weekly_gw(hourly_mw: pd.Series) -> pd.Series:
    weekly = hourly_mw.resample("W").mean() / 1000.0
    weekly = weekly.asfreq("W").interpolate("time")
    weekly.name = "load_gw"
    return weekly


def to_hourly_gw(hourly_mw: pd.Series) -> pd.Series:
    s = (hourly_mw / 1000.0).asfreq("h").interpolate("time")
    s.name = "load_gw"
    return s


def load_processed_data(path=None) -> pd.DataFrame:
    """Load the processed weekly modelling frame (load_gw + covariates)."""
    path = path or (PROCESSED_DATA_DIR / "feature_df.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ----------------------------------------------------------------------
# Temperature (Open-Meteo archive API)
# ----------------------------------------------------------------------

def get_open_meteo_temperature(start_date: str, end_date: str,
                               latitude=BERLIN_LAT, longitude=BERLIN_LON,
                               cache_path=None, max_retries=3) -> pd.DataFrame:
    """Daily mean 2 m temperature; cached to CSV; retries with backoff."""
    if cache_path is None:
        cache_path = PROCESSED_DATA_DIR / "berlin_temp_daily.csv"
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean", "timezone": "Europe/Berlin",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            resp.raise_for_status()
            daily = resp.json()["daily"]
            temp = pd.DataFrame(
                {"temperature_2m_mean": daily["temperature_2m_mean"]},
                index=pd.to_datetime(daily["time"]))
            temp.index.name = "date"
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                temp.to_csv(cache_path)
            return temp
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed: {last_err}")
