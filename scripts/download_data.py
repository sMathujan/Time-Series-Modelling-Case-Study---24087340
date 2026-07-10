"""Download raw data: OPSD hourly load and Berlin temperature."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electricity_demand.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from electricity_demand.data import (
    load_hourly_load, to_hourly_gw, get_open_meteo_temperature,
)


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading OPSD hourly load ...")
    hourly_mw = load_hourly_load()
    hourly_gw = to_hourly_gw(hourly_mw)
    hourly_gw.to_csv(PROCESSED_DATA_DIR / "hourly_load.csv")
    print(f"  saved {len(hourly_gw)} hourly obs")

    print("Downloading Berlin temperature ...")
    start = str(hourly_gw.index.min().date())
    end = str(hourly_gw.index.max().date())
    get_open_meteo_temperature(start_date=start, end_date=end)
    print("  temperature cached")


if __name__ == "__main__":
    main()
