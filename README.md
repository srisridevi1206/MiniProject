# Crime Pattern Analysis and High-Risk Zone Prediction

This project implements a full geo-spatial crime analytics workflow inspired by a self-exciting point process (Hawkes-style) approach:

- Historical crime data ingestion
- Temporal pattern analysis
- Dynamic hotspot identification
- High-risk zone forecasting
- Model explainability for risk predictions
- Baseline-vs-model evaluation metrics
- Data quality guardrails during training/retraining
- Browser landing page and API inference
- Optional Streamlit analytics dashboard

## Project Structure

```text
MiniProject/
  api/
    main.py
  app/
    dashboard.py
  data/
    raw/
    processed/
  models/
  scripts/
    generate_sample_data.py
    start_app.py
    train_model.py
  web/
    index.html
  src/
    crime_risk/
      config.py
      data.py
      model.py
      pipeline.py
  tests/
    test_model.py
  requirements.txt
  pyproject.toml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Single command (recommended):

```bash
python scripts/start_app.py
```

Then open `http://127.0.0.1:8000`.

If you open `web/index.html` from VS Code Live Server on port 5500, the frontend is configured to call the API at `http://127.0.0.1:8000`.

Manual steps:

1. Generate synthetic historical incidents:

```bash
python scripts/generate_sample_data.py
```

2. Train model and export risk outputs:

```bash
python scripts/train_model.py
```

3. Run API:

```bash
uvicorn api.main:app --reload
```

4. Open landing page:

```text
http://127.0.0.1:8000
```

5. Optional: Run Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

## API Endpoints

- `GET /health` : service and model availability
- `POST /predict` : point-level risk score
- `GET /grid` : full-area risk grid for a reference timestamp
- `GET /forecast` : future high-risk trend summary
- `GET /incidents` : recent incident points for map rendering
- `POST /retrain` : retrain model using current raw dataset, or upload a CSV file
- `POST /explain` : decomposition of risk into background and top trigger events
- `GET /evaluate` : holdout evaluation and baseline comparison metrics
- `POST /bootstrap-demo` : generate synthetic demo dataset and retrain model

### Retraining From Browser

Use the landing page to retrain in two ways:

- Upload CSV and retrain: choose a file with `timestamp`, `latitude`, `longitude` columns and click **Upload + Retrain**.
- Retrain current data: click **Retrain with Current Data** to rebuild artifacts from `data/raw/crime_incidents.csv`.

Retrain response now includes quality stats:

- `input_rows`
- `valid_rows`
- `dropped_rows`

Example `POST /predict` payload:

```json
{
  "latitude": 17.4031,
  "longitude": 78.4923,
  "timestamp": "2025-05-12T12:00:00"
}
```

## Model Summary

The model uses:

- Background spatial intensity from 2D incident histogram
- Self-exciting trigger term with time decay
- Gaussian spatial contagion kernel
- Near-repeat calibration for contagion strength estimation
- Forecast uncertainty bands (`lower_ci`, `upper_ci`)

## Evaluation

Use `GET /evaluate` to compare:

- Self-exciting model discrimination
- Static baseline (background-only) discrimination

Reported metrics:

- Mean positive risk
- Mean negative risk
- Separation ratio
- Pairwise win rate

## Docker (One-Command Demo)

Run the complete app in Docker:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

This mounts local `data/` and `models/` folders so retraining outputs persist.

Intensity form:

$$
\lambda(t, x) = \mu(x) + \sum_{t_i < t} \alpha e^{-\beta (t-t_i)} e^{-\frac{d(x,x_i)^2}{2\sigma^2}}
$$

## Notes

- This implementation is a practical engineering version of the Hawkes concept for deployment prototyping.
- Replace synthetic data with real geocoded incident records in `data/raw/crime_incidents.csv` to use in production workflows.
