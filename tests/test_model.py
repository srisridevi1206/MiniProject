from datetime import datetime, timedelta

import pandas as pd

from crime_risk.model import SelfExcitingCrimeModel


def test_intensity_higher_near_recent_event() -> None:
    base_time = datetime(2025, 1, 1, 12, 0)
    df = pd.DataFrame(
        {
            "timestamp": [
                base_time - timedelta(hours=6),
                base_time - timedelta(hours=3),
                base_time - timedelta(hours=1),
            ],
            "latitude": [17.400, 17.401, 17.402],
            "longitude": [78.490, 78.491, 78.492],
        }
    )

    model = SelfExcitingCrimeModel().fit(df)
    near = model.predict_intensity(17.4021, 78.4921, base_time)
    far = model.predict_intensity(17.35, 78.42, base_time)

    assert near > far
