from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_crime_data(
    n_days: int = 120,
    seed: int = 42,
    center_lat: float = 17.385,
    center_lon: float = 78.486,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_time = datetime(2025, 1, 1)
    hours = n_days * 24
    max_recent_triggers = 300
    max_offspring_per_hour = 120

    hotspot_centers = np.array(
        [
            [center_lat + 0.03, center_lon + 0.02],
            [center_lat - 0.02, center_lon + 0.04],
            [center_lat + 0.01, center_lon - 0.03],
            [center_lat - 0.03, center_lon - 0.02],
        ]
    )

    events: list[dict] = []
    recent_events: list[dict] = []

    for hour in range(hours):
        current_time = start_time + timedelta(hours=hour)

        recent_events = [
            ev for ev in recent_events if (current_time - ev["timestamp"]).total_seconds() / 3600 <= 72
        ]

        base_n = rng.poisson(1.6)
        for _ in range(base_n):
            center = hotspot_centers[rng.integers(0, len(hotspot_centers))]
            lat = center[0] + rng.normal(0, 0.006)
            lon = center[1] + rng.normal(0, 0.006)
            event = {
                "timestamp": current_time + timedelta(minutes=int(rng.integers(0, 60))),
                "latitude": lat,
                "longitude": lon,
            }
            events.append(event)
            recent_events.append(event)

        offspring: list[dict] = []
        trigger_pool = recent_events[-max_recent_triggers:]
        for trigger in trigger_pool:
            dt_hours = (current_time - trigger["timestamp"]).total_seconds() / 3600
            if dt_hours <= 0:
                continue
            lam = 0.18 * np.exp(-0.05 * dt_hours)
            n_off = rng.poisson(lam)
            for _ in range(n_off):
                if len(offspring) >= max_offspring_per_hour:
                    break
                event = {
                    "timestamp": current_time + timedelta(minutes=int(rng.integers(0, 60))),
                    "latitude": trigger["latitude"] + rng.normal(0, 0.003),
                    "longitude": trigger["longitude"] + rng.normal(0, 0.003),
                }
                offspring.append(event)
            if len(offspring) >= max_offspring_per_hour:
                break

        events.extend(offspring)
        recent_events.extend(offspring)

    df = pd.DataFrame(events)
    crime_types = ["Burglary", "Robbery", "Assault", "Vehicle Theft", "Drug Offense"]
    probs = [0.25, 0.2, 0.25, 0.15, 0.15]
    df["crime_type"] = rng.choice(crime_types, size=len(df), p=probs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
