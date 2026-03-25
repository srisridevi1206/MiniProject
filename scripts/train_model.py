from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crime_risk.model import SelfExcitingCrimeModel
from crime_risk.pipeline import load_incident_data


def main() -> None:
    input_path = ROOT / "data/raw/crime_incidents.csv"
    model_path = ROOT / "models/self_exciting_model.joblib"
    grid_path = ROOT / "data/processed/current_risk_grid.csv"
    forecast_path = ROOT / "data/processed/risk_forecast.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data not found at {input_path}. Run scripts/generate_sample_data.py first."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.parent.mkdir(parents=True, exist_ok=True)

    incidents = load_incident_data(input_path)
    model = SelfExcitingCrimeModel().fit(incidents)
    model.save(str(model_path))

    ref_time = incidents["timestamp"].max()
    risk_grid = model.score_grid(ref_time, grid_steps=45)
    forecast = model.forecast_top_zones(horizon_hours=96, step_hours=12)

    risk_grid.to_csv(grid_path, index=False)
    forecast.to_csv(forecast_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"Risk grid saved to {grid_path} ({len(risk_grid)} rows)")
    print(f"Forecast saved to {forecast_path} ({len(forecast)} rows)")


if __name__ == "__main__":
    main()
