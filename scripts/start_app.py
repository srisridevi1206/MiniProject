from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/self_exciting_model.joblib"
RAW_PATH = ROOT / "data/raw/crime_incidents.csv"


def run_step(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_artifacts() -> None:
    if not RAW_PATH.exists():
        print("Generating synthetic incident dataset...")
        run_step([sys.executable, "scripts/generate_sample_data.py"])

    if not MODEL_PATH.exists():
        print("Training model and generating risk outputs...")
        run_step([sys.executable, "scripts/train_model.py"])


def main() -> None:
    ensure_artifacts()
    print("Starting app at http://127.0.0.1:8000")
    run_step([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"])


if __name__ == "__main__":
    main()
