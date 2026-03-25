from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from .config import ModelConfig


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    )
    return float(2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def haversine_km_vector(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    radius = 6371.0
    d_lat = np.radians(lats - lat)
    d_lon = np.radians(lons - lon)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(d_lon / 2) ** 2
    return 2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class SelfExcitingCrimeModel:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.is_fitted = False

    def fit(self, incidents: pd.DataFrame) -> "SelfExcitingCrimeModel":
        df = incidents.sort_values("timestamp").reset_index(drop=True).copy()
        if df.empty:
            raise ValueError("Cannot fit model with empty incident dataset")

        self.events = df[["timestamp", "latitude", "longitude"]].copy()
        self.min_lat = float(df["latitude"].min())
        self.max_lat = float(df["latitude"].max())
        self.min_lon = float(df["longitude"].min())
        self.max_lon = float(df["longitude"].max())

        self.lat_edges = np.linspace(self.min_lat, self.max_lat, self.config.grid_size + 1)
        self.lon_edges = np.linspace(self.min_lon, self.max_lon, self.config.grid_size + 1)

        hist, _, _ = np.histogram2d(
            df["latitude"].values,
            df["longitude"].values,
            bins=[self.lat_edges, self.lon_edges],
        )

        total_hours = max(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600.0,
            1.0,
        )
        self.background = (hist + self.config.epsilon) / total_hours
        self.alpha = self._estimate_alpha(df)
        self.beta = np.log(2) / self.config.decay_half_life_hours
        self.sigma_km = self.config.spatial_sigma_km
        self.last_event_time = pd.Timestamp(df["timestamp"].max())
        self.event_ts_hours = (self.events["timestamp"].astype("int64") / 3.6e12).to_numpy()
        self.event_lats = self.events["latitude"].to_numpy()
        self.event_lons = self.events["longitude"].to_numpy()
        self.is_fitted = True
        return self

    def _estimate_alpha(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.35

        ts_hours = (df["timestamp"].astype("int64") / 3.6e12).to_numpy()
        lats = df["latitude"].to_numpy()
        lons = df["longitude"].to_numpy()

        max_eval = min(2500, len(df) - 1)
        if len(df) - 1 > max_eval:
            eval_indices = np.linspace(1, len(df) - 1, num=max_eval, dtype=int)
        else:
            eval_indices = np.arange(1, len(df), dtype=int)

        near_repeats = 0
        checked = 0
        history_window = 350

        for i in eval_indices:
            start = max(0, i - history_window)
            h_ts = ts_hours[start:i]
            if h_ts.size == 0:
                continue

            dt_hours = ts_hours[i] - h_ts
            valid = (dt_hours > 0) & (dt_hours <= self.config.near_repeat_hours)
            if not np.any(valid):
                continue

            checked += 1
            dists = haversine_km_vector(lats[i], lons[i], lats[start:i][valid], lons[start:i][valid])
            if np.any(dists <= self.config.near_repeat_km):
                near_repeats += 1

        if checked == 0:
            return 0.35
        ratio = near_repeats / checked
        return float(np.clip(0.15 + ratio * 0.9, 0.15, 1.8))

    def _background_intensity(self, lat: float, lon: float) -> float:
        lat_idx = np.clip(np.digitize(lat, self.lat_edges) - 1, 0, self.background.shape[0] - 1)
        lon_idx = np.clip(np.digitize(lon, self.lon_edges) - 1, 0, self.background.shape[1] - 1)
        return float(self.background[lat_idx, lon_idx])

    def predict_intensity(self, lat: float, lon: float, timestamp: datetime | pd.Timestamp) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        ts = pd.Timestamp(timestamp)
        mu = self._background_intensity(lat, lon)

        ts_hours = ts.value / 3.6e12
        dt_hours = ts_hours - self.event_ts_hours
        mask = (dt_hours > 0) & (dt_hours <= self.config.lookback_hours)
        if not np.any(mask):
            return mu

        dts = dt_hours[mask]
        dists = haversine_km_vector(lat, lon, self.event_lats[mask], self.event_lons[mask])
        temporal = np.exp(-self.beta * dts)
        spatial = np.exp(-(dists**2) / (2 * self.sigma_km**2))
        trigger_sum = self.alpha * float(np.sum(temporal * spatial))
        return float(mu + trigger_sum)

    def explain_prediction(
        self, lat: float, lon: float, timestamp: datetime | pd.Timestamp, top_n: int = 5
    ) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        ts = pd.Timestamp(timestamp)
        mu = self._background_intensity(lat, lon)

        ts_hours = ts.value / 3.6e12
        dt_hours = ts_hours - self.event_ts_hours
        mask = (dt_hours > 0) & (dt_hours <= self.config.lookback_hours)

        if not np.any(mask):
            return {
                "total_risk": float(mu),
                "background_risk": float(mu),
                "trigger_risk": 0.0,
                "top_triggers": [],
            }

        dts = dt_hours[mask]
        lats = self.event_lats[mask]
        lons = self.event_lons[mask]
        dists = haversine_km_vector(lat, lon, lats, lons)
        temporal = np.exp(-self.beta * dts)
        spatial = np.exp(-(dists**2) / (2 * self.sigma_km**2))
        contributions = self.alpha * temporal * spatial

        trigger_sum = float(np.sum(contributions))
        total = float(mu + trigger_sum)

        hist_idx = np.where(mask)[0]
        top_idx = np.argsort(contributions)[::-1][:top_n]
        top_triggers = []
        for i in top_idx:
            ev = self.events.iloc[hist_idx[i]]
            top_triggers.append(
                {
                    "timestamp": pd.Timestamp(ev["timestamp"]).isoformat(),
                    "latitude": float(ev["latitude"]),
                    "longitude": float(ev["longitude"]),
                    "distance_km": float(dists[i]),
                    "hours_ago": float(dts[i]),
                    "contribution": float(contributions[i]),
                }
            )

        return {
            "total_risk": total,
            "background_risk": float(mu),
            "trigger_risk": trigger_sum,
            "top_triggers": top_triggers,
        }

    def score_grid(self, reference_time: datetime | pd.Timestamp, grid_steps: int = 40) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        ref = pd.Timestamp(reference_time)
        lats = np.linspace(self.min_lat, self.max_lat, grid_steps)
        lons = np.linspace(self.min_lon, self.max_lon, grid_steps)

        rows: list[dict] = []
        for lat in lats:
            for lon in lons:
                rows.append(
                    {
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "risk_score": self.predict_intensity(float(lat), float(lon), ref),
                    }
                )
        out = pd.DataFrame(rows)
        out["risk_rank"] = out["risk_score"].rank(method="dense", ascending=False)
        return out

    def forecast_top_zones(self, horizon_hours: int = 72, step_hours: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        rows: list[dict] = []
        for h in range(step_hours, horizon_hours + step_hours, step_hours):
            t = self.last_event_time + timedelta(hours=h)
            grid = self.score_grid(t, grid_steps=20)
            top = grid.nlargest(20, "risk_score")
            std_top = float(top["risk_score"].std(ddof=0)) if len(top) > 1 else 0.0
            ci_margin = 1.28 * std_top / np.sqrt(max(len(top), 1))
            mean_top = float(top["risk_score"].mean())
            rows.append(
                {
                    "forecast_time": t,
                    "mean_top20_risk": mean_top,
                    "max_risk": float(top["risk_score"].max()),
                    "lower_ci": float(max(mean_top - ci_margin, 0.0)),
                    "upper_ci": float(mean_top + ci_margin),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self.config),
            "is_fitted": self.is_fitted,
            "events": self.events,
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
            "lat_edges": self.lat_edges,
            "lon_edges": self.lon_edges,
            "background": self.background,
            "alpha": self.alpha,
            "beta": self.beta,
            "sigma_km": self.sigma_km,
            "last_event_time": self.last_event_time,
            "event_ts_hours": self.event_ts_hours,
            "event_lats": self.event_lats,
            "event_lons": self.event_lons,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "SelfExcitingCrimeModel":
        payload = joblib.load(path)
        obj = cls(config=ModelConfig(**payload["config"]))
        obj.is_fitted = payload["is_fitted"]
        obj.events = payload["events"]
        obj.min_lat = payload["min_lat"]
        obj.max_lat = payload["max_lat"]
        obj.min_lon = payload["min_lon"]
        obj.max_lon = payload["max_lon"]
        obj.lat_edges = payload["lat_edges"]
        obj.lon_edges = payload["lon_edges"]
        obj.background = payload["background"]
        obj.alpha = payload["alpha"]
        obj.beta = payload["beta"]
        obj.sigma_km = payload["sigma_km"]
        obj.last_event_time = payload["last_event_time"]
        obj.event_ts_hours = payload["event_ts_hours"]
        obj.event_lats = payload["event_lats"]
        obj.event_lons = payload["event_lons"]
        return obj
