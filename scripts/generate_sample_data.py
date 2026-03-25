from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crime_risk.data import generate_synthetic_crime_data


def main() -> None:
    output_path = ROOT / "data/raw/crime_incidents.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_crime_data(n_days=90, seed=7)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
