"""Run the full weekly forecasting pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electricity_demand.pipeline import run_pipeline


def main():
    metrics_df, _ = run_pipeline(
        include_sarimax=True,
        include_feature_model=True,
        include_neural=False,   # set True to add the LSTM (needs hourly data + TF)
        include_bayesian=False,
    )
    print("\nModel comparison (sorted by MASE):")
    print(metrics_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
