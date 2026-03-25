from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, RAW_DATA_PATH


client = TestClient(app)


def test_health_endpoint() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_retrain_and_evaluate_endpoints(tmp_path: Path) -> None:
    backup = None
    if RAW_DATA_PATH.exists():
        backup = RAW_DATA_PATH.read_text(encoding="utf-8")

    try:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=60, freq="h"),
                "latitude": [17.38 + (i % 5) * 0.001 for i in range(60)],
                "longitude": [78.48 + (i % 5) * 0.001 for i in range(60)],
            }
        )
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            retrain = client.post("/retrain", files={"file": ("sample.csv", f, "text/csv")})
        assert retrain.status_code == 200
        assert retrain.json()["incident_count"] >= 50

        evaluate = client.get("/evaluate?sample_size=20&seed=4")
        assert evaluate.status_code == 200
        payload = evaluate.json()
        assert "self_exciting" in payload
        assert "baseline_static" in payload
    finally:
        if backup is not None:
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_DATA_PATH.write_text(backup, encoding="utf-8")
