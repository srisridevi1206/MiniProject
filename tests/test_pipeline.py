from pathlib import Path

import pandas as pd

from crime_risk.pipeline import load_incident_data


def test_load_incident_data_filters_invalid_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "incidents.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "2025-01-01 12:00:00",
                "2025-01-01 13:00:00",
                "bad-date",
            ],
            "latitude": [17.4, 17.4, 999, 17.41, 17.42, 17.5],
            "longitude": [78.4, 78.4, 78.5, 78.41, 78.42, 78.6],
        }
    ).to_csv(p, index=False)

    out = load_incident_data(p)
    assert len(out) == 3
    assert out.iloc[0]["latitude"] == 17.4
    assert out.attrs["quality_stats"]["dropped_rows"] == 3
