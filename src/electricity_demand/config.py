"""Central configuration: paths and shared constants."""

from pathlib import Path

# --- paths (repo-root relative) ---------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

# --- data source ------------------------------------------------------
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
START_DATE = "2015-01-01"

# Berlin, representative for German temperature (Open-Meteo archive API)
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
BERLIN_LAT = 52.52
BERLIN_LON = 13.41

# --- modelling constants ----------------------------------------------
TEST_WEEKS = 104          # 2-year evaluation horizon
SEASONALITY = 52          # weeks per year
HOURS_PER_WEEK = 168
RANDOM_STATE = 42

# heating / cooling degree-day base temperatures (deg C)
BASE_HEAT = 15.5
BASE_COOL = 22.0

# SARIMA(X) defaults
DEFAULT_ORDER = (2, 1, 6)
DEFAULT_SEASONAL_ORDER = (0, 1, 1, 52)
DEFAULT_EXOG = ["temp_mean", "heating_degree", "cooling_degree"]
