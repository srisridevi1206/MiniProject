from datetime import datetime
from io import StringIO
from pathlib import Path
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crime_risk.model import SelfExcitingCrimeModel
from crime_risk.pipeline import load_incident_data, summarize_quality
from crime_risk.data import generate_synthetic_crime_data

app = FastAPI(title="Crime High-Risk Zone Prediction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ROOT / "models/self_exciting_model.joblib"
RAW_DATA_PATH = ROOT / "data/raw/crime_incidents.csv"
GRID_PATH = ROOT / "data/processed/current_risk_grid.csv"
FORECAST_PATH = ROOT / "data/processed/risk_forecast.csv"
WEB_DIR = ROOT / "web"
ROOT_INDEX_PATH = ROOT / "index.html"


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime | None = None


class PredictResponse(BaseModel):
    risk_score: float
    used_timestamp: datetime


def _pairwise_win_rate(pos_scores: list[float], neg_scores: list[float]) -> float:
    if not pos_scores or not neg_scores:
        return 0.0
    wins = 0
    total = min(len(pos_scores), len(neg_scores))
    for i in range(total):
        if pos_scores[i] > neg_scores[i]:
            wins += 1
    return float(wins / total)


def load_model() -> SelfExcitingCrimeModel:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model artifact missing. Train model first.")
    return SelfExcitingCrimeModel.load(str(MODEL_PATH))


def rebuild_artifacts(incidents: pd.DataFrame) -> dict:
    model = SelfExcitingCrimeModel().fit(incidents)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(MODEL_PATH))
    ref_time = incidents["timestamp"].max()
    grid_df = model.score_grid(ref_time, grid_steps=35)
    forecast_df = model.forecast_top_zones(horizon_hours=72, step_hours=12)

    grid_df.to_csv(GRID_PATH, index=False)
    forecast_df.to_csv(FORECAST_PATH, index=False)

    return {
        "incident_count": int(len(incidents)),
        "grid_cells": int(len(grid_df)),
        "forecast_rows": int(len(forecast_df)),
        "latest_incident": pd.Timestamp(ref_time).isoformat(),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = load_model()
    ts = request.timestamp or datetime.utcnow()
    risk = model.predict_intensity(request.latitude, request.longitude, ts)
    return PredictResponse(risk_score=risk, used_timestamp=ts)


@app.post("/explain")
def explain(request: PredictRequest) -> dict:
    model = load_model()
    ts = request.timestamp or datetime.utcnow()
    explanation = model.explain_prediction(request.latitude, request.longitude, ts, top_n=6)
    explanation["used_timestamp"] = ts.isoformat()
    return explanation


@app.get("/grid")
def grid(reference_time: datetime | None = None, grid_steps: int = 30) -> dict:
    model = load_model()
    ref = reference_time or datetime.utcnow()
    out = model.score_grid(ref, grid_steps=grid_steps)
    return {"reference_time": ref.isoformat(), "rows": out.to_dict(orient="records")}


@app.get("/forecast")
def forecast(horizon_hours: int = 72, step_hours: int = 12) -> dict:
    model = load_model()
    out = model.forecast_top_zones(horizon_hours=horizon_hours, step_hours=step_hours)
    return {"rows": out.to_dict(orient="records")}


@app.get("/evaluate")
def evaluate(sample_size: int = 300, seed: int = 42) -> dict:
    if not RAW_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Raw data missing. Generate or upload data first.")

    df = load_incident_data(RAW_DATA_PATH)
    if len(df) < 10:
        return {
            "ready": False,
            "detail": "Not enough incidents for evaluation; need at least 10 rows",
            "available_rows": int(len(df)),
        }

    split_idx = max(int(len(df) * 0.8), 20)
    if split_idx >= len(df):
        split_idx = len(df) - 1
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty:
        return {
            "ready": False,
            "detail": "Evaluation split has no holdout rows",
            "available_rows": int(len(df)),
        }

    model = SelfExcitingCrimeModel().fit(train_df)
    rng = random.Random(seed)

    min_lat, max_lat = float(train_df["latitude"].min()), float(train_df["latitude"].max())
    min_lon, max_lon = float(train_df["longitude"].min()), float(train_df["longitude"].max())

    holdout = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed)

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pos_baseline: list[float] = []
    neg_baseline: list[float] = []

    for _, row in holdout.iterrows():
        ts = row["timestamp"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        p = model.predict_intensity(lat, lon, ts)
        pos_scores.append(float(p))
        pos_baseline.append(float(model._background_intensity(lat, lon)))

        n_lat = rng.uniform(min_lat, max_lat)
        n_lon = rng.uniform(min_lon, max_lon)
        n = model.predict_intensity(n_lat, n_lon, ts)
        neg_scores.append(float(n))
        neg_baseline.append(float(model._background_intensity(n_lat, n_lon)))

    mean_pos = float(pd.Series(pos_scores).mean())
    mean_neg = float(pd.Series(neg_scores).mean())
    mean_pos_base = float(pd.Series(pos_baseline).mean())
    mean_neg_base = float(pd.Series(neg_baseline).mean())

    eps = 1e-8
    return {
        "ready": True,
        "sample_size": int(len(holdout)),
        "self_exciting": {
            "mean_positive_risk": mean_pos,
            "mean_negative_risk": mean_neg,
            "separation_ratio": float((mean_pos + eps) / (mean_neg + eps)),
            "pairwise_win_rate": _pairwise_win_rate(pos_scores, neg_scores),
        },
        "baseline_static": {
            "mean_positive_risk": mean_pos_base,
            "mean_negative_risk": mean_neg_base,
            "separation_ratio": float((mean_pos_base + eps) / (mean_neg_base + eps)),
            "pairwise_win_rate": _pairwise_win_rate(pos_baseline, neg_baseline),
        },
    }


@app.post("/bootstrap-demo")
def bootstrap_demo(days: int = 90, seed: int = 7) -> dict:
    if days < 30 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 30 and 365")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_crime_data(n_days=days, seed=seed)
    df.to_csv(RAW_DATA_PATH, index=False)

    incidents_df = load_incident_data(RAW_DATA_PATH)
    result = rebuild_artifacts(incidents_df)
    quality = summarize_quality(incidents_df)

    return {
        "message": "Demo dataset generated and model retrained",
        "quality": quality,
        **result,
    }


@app.get("/incidents")
def incidents(limit: int = 2000) -> dict:
    if not RAW_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Raw data missing. Generate data first.")

    df = pd.read_csv(RAW_DATA_PATH)
    required = {"timestamp", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise HTTPException(status_code=400, detail="Raw data does not contain required columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    out = df.sort_values("timestamp", ascending=False).head(limit)
    return {"rows": out.to_dict(orient="records")}


@app.post("/retrain")
async def retrain(file: UploadFile | None = File(default=None)) -> dict:
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload must be a CSV file")

        content = await file.read()
        try:
            uploaded_df = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

        required = {"timestamp", "latitude", "longitude"}
        if not required.issubset(uploaded_df.columns):
            raise HTTPException(status_code=400, detail="CSV must include timestamp, latitude, longitude columns")

        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        uploaded_df.to_csv(RAW_DATA_PATH, index=False)

    if not RAW_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="No raw dataset found. Upload a CSV first.")

    try:
        incidents_df = load_incident_data(RAW_DATA_PATH)
        result = rebuild_artifacts(incidents_df)
        quality = summarize_quality(incidents_df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Retrain failed: {exc}") from exc

    return {
        "message": "Retraining complete",
        "used_uploaded_file": file is not None,
        "quality": quality,
        **result,
    }


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/")
def landing_page() -> FileResponse:
    if ROOT_INDEX_PATH.exists():
        return FileResponse(ROOT_INDEX_PATH)
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")
