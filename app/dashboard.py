from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from crime_risk.model import SelfExcitingCrimeModel

st.set_page_config(page_title="Crime Risk Intelligence Dashboard", layout="wide")
st.title("Crime Pattern Analysis and High-Risk Zone Prediction")
st.caption("Self-exciting spatio-temporal modeling for dynamic hotspot intelligence")

RAW_PATH = ROOT / "data/raw/crime_incidents.csv"
GRID_PATH = ROOT / "data/processed/current_risk_grid.csv"
FORECAST_PATH = ROOT / "data/processed/risk_forecast.csv"
MODEL_PATH = ROOT / "models/self_exciting_model.joblib"

if not RAW_PATH.exists():
    st.error("No dataset found. Run scripts/generate_sample_data.py and scripts/train_model.py first.")
    st.stop()

incidents = pd.read_csv(RAW_PATH)
incidents["timestamp"] = pd.to_datetime(incidents["timestamp"])

col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", f"{len(incidents):,}")
col2.metric("Date Range", f"{incidents['timestamp'].min().date()} to {incidents['timestamp'].max().date()}")
col3.metric("Crime Categories", str(incidents.get("crime_type", pd.Series()).nunique() or 0))

st.subheader("Temporal Analysis")
incidents["hour"] = incidents["timestamp"].dt.hour
incidents["day_of_week"] = incidents["timestamp"].dt.day_name()

hourly_counts = incidents.groupby("hour").size().reset_index(name="count")
day_counts = incidents.groupby("day_of_week").size().reset_index(name="count")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts["day_of_week"] = pd.Categorical(day_counts["day_of_week"], categories=day_order, ordered=True)
day_counts = day_counts.sort_values("day_of_week")

hcol1, hcol2 = st.columns(2)
fig_hour = px.bar(hourly_counts, x="hour", y="count", title="Incidents by Hour")
fig_day = px.bar(day_counts, x="day_of_week", y="count", title="Incidents by Day of Week")
hcol1.plotly_chart(fig_hour, use_container_width=True)
hcol2.plotly_chart(fig_day, use_container_width=True)

st.subheader("Hotspot Identification")
map_view = pdk.ViewState(
    latitude=float(incidents["latitude"].mean()),
    longitude=float(incidents["longitude"].mean()),
    zoom=11,
    pitch=35,
)

scatter = pdk.Layer(
    "ScatterplotLayer",
    data=incidents,
    get_position="[longitude, latitude]",
    get_fill_color="[225, 70, 57, 90]",
    get_radius=50,
    pickable=True,
)

st.pydeck_chart(pdk.Deck(initial_view_state=map_view, layers=[scatter]))

if GRID_PATH.exists():
    st.subheader("High-Risk Zone Predictive Map")
    risk_grid = pd.read_csv(GRID_PATH)
    heat = pdk.Layer(
        "HeatmapLayer",
        data=risk_grid,
        get_position="[longitude, latitude]",
        get_weight="risk_score",
        radiusPixels=45,
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=map_view, layers=[heat]))

st.subheader("Risk Zone Forecast")
if FORECAST_PATH.exists():
    forecast = pd.read_csv(FORECAST_PATH)
    forecast["forecast_time"] = pd.to_datetime(forecast["forecast_time"])
    fig_forecast = px.line(
        forecast,
        x="forecast_time",
        y=["mean_top20_risk", "max_risk"],
        title="Forecasted Risk Evolution",
        markers=True,
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("Run scripts/train_model.py to generate forecast outputs.")

st.subheader("Single Point Risk Query")
lat = st.number_input("Latitude", value=float(incidents["latitude"].mean()), format="%.6f")
lon = st.number_input("Longitude", value=float(incidents["longitude"].mean()), format="%.6f")

if st.button("Predict Risk"):
    if not MODEL_PATH.exists():
        st.error("Model file missing. Train model first.")
    else:
        model = SelfExcitingCrimeModel.load(str(MODEL_PATH))
        ts = incidents["timestamp"].max()
        score = model.predict_intensity(lat, lon, ts)
        st.success(f"Predicted risk score at {ts}: {score:.4f}")
