from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "latitude", "longitude"]


def load_incident_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Keep only physically valid coordinates.
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()

    # Remove exact duplicates which can bias temporal contagion estimates.
    df = df.drop_duplicates(subset=["timestamp", "latitude", "longitude"], keep="first")

    if df.empty:
        raise ValueError("No valid incidents remain after data quality filtering")

    if len(df) < 3:
        raise ValueError("Need at least 3 valid incidents for training")

    out = df.sort_values("timestamp").reset_index(drop=True)
    out.attrs["quality_stats"] = {
        "input_rows": int(before),
        "valid_rows": int(len(out)),
        "dropped_rows": int(before - len(out)),
    }
    return out


def summarize_quality(df: pd.DataFrame) -> dict:
    stats = df.attrs.get("quality_stats", {})
    return {
        "input_rows": int(stats.get("input_rows", len(df))),
        "valid_rows": int(stats.get("valid_rows", len(df))),
        "dropped_rows": int(stats.get("dropped_rows", 0)),
    }
