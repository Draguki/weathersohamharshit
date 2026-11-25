# Large-Scale Weather Data Analysis of Indian Cities
**Apache Spark (PySpark) · Spark MLlib · SQLite · FastAPI · React live dashboard**

1.8 million hourly weather records (Open-Meteo ERA5 reanalysis, 30 Indian cities across 9 regions, 2019-01-01 → 2025-10-31) are processed with Spark, analysed, modelled with Spark MLlib, stored in a SQLite results database, and served live to a dashboard that also pulls the current weather and runs the Spark models on it every 10 minutes.

## Run it (demo day)
```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```
Opens http://127.0.0.1:5173. Analytics appear immediately; the Spark models train in-process in ~2 min, then live predictions, the what-if simulator and the Retrain button switch on.

Re-run the big-data pipeline first: `.\run_all.ps1 -Pipeline` (Spark ETL ~1 min + training ~2 min).

## Pipeline
| Step | Code | Output |
|---|---|---|
| 1. Acquire | `spark_jobs/fetch_data.py` | `data/raw/hourly_<city>.csv` (30 files, 1,797,120 rows, ~107 MB) |
| 2. Spark ETL | `spark_jobs/etl.py` | cleaning → daily / monthly / yearly aggregates, seasonal climatology, IMD-style extreme events, warming trends, correlations → `db/weather.db` |
| 3. Spark MLlib | `spark_jobs/ml.py`, `train.py` | GBT + linear next-day max temp, Random Forest next-day rain, K-Means climate zones → metrics, feature importance, test predictions in DB |
| 4. Live API | `backend/` (FastAPI :8000) | analytics endpoints, live Open-Meteo feed + Spark predictions logged to `live_predictions`, what-if `/api/predict`, `/api/retrain` |
| 5. Dashboard | `dashboard/` (React + ECharts + Leaflet :5173) | Live overview, Climate analytics, City explorer, ML models, Extreme events, Pipeline & DB |

## Results (test period 2024-01-01 → 2025-10-31)
| Model | Metric |
|---|---|
| GBT next-day max temp | RMSE 1.42 °C · MAE 1.03 °C · R² 0.961 |
| Linear baseline | RMSE 1.44 °C · R² 0.960 |
| Random Forest rain tomorrow | AUC 0.928 · accuracy 0.855 · F1 0.825 |
| K-Means climate zones (k=4) | silhouette 0.567 |

## Deliverables
- Presentation: `presentation/Weather_BigData_Analysis.pptx` (rebuild: `python presentation\build_deck.py`; screenshots: `python presentation\capture_screenshots.py`)
- Report: `report/PBL_Report_Weather_BigData.docx`
- Design notes: `docs/DESIGN_BRIEF.md`, architecture/API contract: `docs/CONTRACT.md`

## Notes
- Windows: Spark disk writes need Hadoop `winutils.exe`, which isn't installed, so Spark results are collected with `toPandas()` and written to SQLite, and the backend trains the models in memory at startup. On Linux/HDFS/YARN the same Spark code runs unchanged at cluster scale.
- Python venv: `C:\Users\harsh\.venvs\wbd` (Python 3.10, PySpark 3.5.3, Java 8).
