from datetime import datetime
from io import StringIO
from pathlib import Path
import sys

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
from crime_risk.pipeline import load_incident_data

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


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime | None = None


class PredictResponse(BaseModel):
    risk_score: float
    used_timestamp: datetime


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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Retrain failed: {exc}") from exc

    return {
        "message": "Retraining complete",
        "used_uploaded_file": file is not None,
        **result,
    }


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def landing_page() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
