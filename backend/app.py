"""FastAPI app: analytics (SQLite), live layer (Open-Meteo + Spark MLlib), what-if, retrain.

Run from the project root:
  C:\\Users\\harsh\\.venvs\\wbd\\Scripts\\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import db, live  # noqa: E402
from .engine import FAKE, engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
DIST = ROOT / "dashboard" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="Indian Cities Weather Analysis API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(db.DBMissing)
async def _db_missing(_req: Request, exc: db.DBMissing):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def not_ready():
    msg = {
        "starting": "Spark is starting",
        "starting_spark": "Spark is starting (about 20 s)",
        "waiting_for_db": "waiting for the ETL to populate daily_weather",
        "training": "Spark MLlib models are being trained (1-3 min)",
        "error": f"model initialisation failed: {engine.error}",
        "recovering": "Spark JVM died and is being restarted; models will retrain (1-3 min)",
    }.get(engine.status, engine.status)
    raise HTTPException(status_code=503, detail=f"Models not ready: {msg}. Try again shortly.")


def resolve_city(city: str) -> str:
    row = db.query_one("SELECT city FROM cities WHERE lower(city)=lower(?)", (city,)) if db.table_exists("cities") else None
    if not row:
        raise HTTPException(status_code=404, detail=f"unknown city '{city}'")
    return row["city"]


def latest_metrics_rows() -> list[dict]:
    if not db.table_exists("model_metrics"):
        return []
    return db.query(
        """SELECT m.* FROM model_metrics m
           JOIN (SELECT model_name, max(trained_at) AS t FROM model_metrics GROUP BY model_name) x
             ON m.model_name = x.model_name AND m.trained_at IS x.t
           ORDER BY m.model_name, m.metric""")


def current_metrics() -> list[dict]:
    if engine.ready and getattr(engine.bundle, "metrics", None):
        return db.clean(list(engine.bundle.metrics))
    return latest_metrics_rows()


def group_models(metrics: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for m in metrics:
        e = out.setdefault(m["model_name"], {"model_name": m["model_name"], "task": m.get("task"),
                                             "target": m.get("target"), "trained_at": m.get("trained_at"),
                                             "train_rows": m.get("train_rows"), "test_rows": m.get("test_rows"),
                                             "metrics": {}})
        e["metrics"][m["metric"]] = m.get("value")
    return list(out.values())


# ============================================================== health / overview
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "spark_ready": engine.ready,
        "model_version": engine.version,
        "db_path": str(db.DB_PATH),
        "db_exists": db.db_exists(),
        "engine_status": engine.status,
        "engine_error": engine.error,
        "fake_model": FAKE,
        "live_last_refresh": engine.live_meta["last_refresh"],
        "live_last_error": engine.live_meta["last_error"],
        "live_cities_cached": len(engine.live),
    }


@app.get("/api/overview")
def overview():
    if not db.db_exists():
        raise db.DBMissing(f"database not found at {db.DB_PATH} (run the ETL first)")
    has = db.table_exists
    res: dict = {}
    res["rows_processed"] = db.scalar("SELECT sum(rows_processed) FROM city_summary") if has("city_summary") else None
    if not res["rows_processed"] and has("pipeline_runs"):
        res["rows_processed"] = db.scalar("SELECT max(rows_in) FROM pipeline_runs WHERE job LIKE '%etl%'")
    res["raw_files"] = len(list((ROOT / "data" / "raw").glob("hourly_*.csv")))
    res["cities"] = db.scalar("SELECT count(*) FROM cities") if has("cities") else 0
    if has("daily_weather"):
        r = db.query_one("SELECT min(date) AS s, max(date) AS e, count(DISTINCT substr(date,1,4)) AS y, count(*) AS n FROM daily_weather")
        res.update(date_start=r["s"], date_end=r["e"], years=r["y"], daily_rows=r["n"])
    else:
        res.update(date_start=None, date_end=None, years=0, daily_rows=0)
    res["db_size_mb"] = round(db.DB_PATH.stat().st_size / 1e6, 2)

    for key, col, order in [("hottest_city", "avg_temp", "DESC"), ("coldest_city", "avg_temp", "ASC"),
                            ("wettest_city", "avg_annual_precip", "DESC"), ("driest_city", "avg_annual_precip", "ASC")]:
        r = db.query_one(f"SELECT city, {col} AS v FROM city_summary WHERE {col} IS NOT NULL ORDER BY {col} {order} LIMIT 1") if has("city_summary") else None
        res[key] = r["city"] if r else None
        res[key + "_value"] = r["v"] if r else None
    for key, col, dcol, order in [("record_high", "record_high", "record_high_date", "DESC"),
                                  ("record_low", "record_low", "record_low_date", "ASC")]:
        r = db.query_one(f"SELECT city, {col} AS value, {dcol} AS date FROM city_summary WHERE {col} IS NOT NULL ORDER BY {col} {order} LIMIT 1") if has("city_summary") else None
        res[key] = r
    counts = {"heatwave": 0, "cold_wave": 0, "heavy_rain": 0}
    if has("extreme_events"):
        for r in db.query("SELECT event_type, count(*) AS n FROM extreme_events GROUP BY event_type"):
            counts[r["event_type"]] = r["n"]
    res["extreme_event_counts"] = counts
    res["pipeline_runs"] = db.query("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 50") if has("pipeline_runs") else []
    res["models"] = group_models(current_metrics())
    res["spark_ready"] = engine.ready
    res["model_version"] = engine.version
    return res


# ============================================================== cities
def _live_brief(city: str) -> dict | None:
    e = engine.live.get(city)
    if not e:
        return None
    o = e["observation"]
    return {"temperature": o.get("temperature"), "humidity": o.get("humidity"),
            "weather_code": o.get("weather_code"), "fetched_at": o.get("fetched_at")}


@app.get("/api/cities")
def cities():
    if db.table_exists("city_summary"):
        rows = db.query("""SELECT c.city, c.state, c.region, c.lat, c.lon, c.climate_cluster,
                                  s.avg_temp, s.avg_annual_precip
                           FROM cities c LEFT JOIN city_summary s ON s.city = c.city ORDER BY c.city""")
    else:
        rows = db.query("SELECT city, state, region, lat, lon, climate_cluster, NULL AS avg_temp, "
                        "NULL AS avg_annual_precip FROM cities ORDER BY city")
    for r in rows:
        r["live"] = _live_brief(r["city"])
    return rows


@app.get("/api/city/{city}/summary")
def city_summary(city: str):
    c = resolve_city(city)
    base = db.query_one("SELECT * FROM cities WHERE city=?", (c,)) or {}
    s = db.query_one("SELECT * FROM city_summary WHERE city=?", (c,)) if db.table_exists("city_summary") else None
    return {**(s or {}), **base}


@app.get("/api/city/{city}/yearly")
def city_yearly(city: str):
    return db.query("SELECT * FROM yearly_stats WHERE city=? ORDER BY year", (resolve_city(city),))


@app.get("/api/city/{city}/monthly")
def city_monthly(city: str):
    return db.query("SELECT * FROM monthly_stats WHERE city=? ORDER BY year, month", (resolve_city(city),))


@app.get("/api/city/{city}/seasonal")
def city_seasonal(city: str):
    return db.query("SELECT * FROM seasonal_profile WHERE city=? ORDER BY month", (resolve_city(city),))


@app.get("/api/city/{city}/daily")
def city_daily(city: str, start: Optional[str] = None, end: Optional[str] = None):
    c = resolve_city(city)
    if not end:
        end = db.scalar("SELECT max(date) FROM daily_weather WHERE city=?", (c,))
        if not end:
            return []
    if not start:
        try:
            start = (date.fromisoformat(end[:10]) - timedelta(days=365)).isoformat()
        except ValueError:
            raise HTTPException(400, "end must be YYYY-MM-DD")
    return db.query("SELECT * FROM daily_weather WHERE city=? AND date>=? AND date<=? ORDER BY date", (c, start, end))


# ============================================================== comparisons
YEARLY_METRICS = {"temp_mean", "precip_total", "heatwave_days", "rainy_days", "temp_max_abs", "temp_min_abs"}


@app.get("/api/compare/yearly")
def compare_yearly(metric: str = "temp_mean", cities: Optional[str] = None):
    if metric not in YEARLY_METRICS:
        raise HTTPException(400, f"metric must be one of {sorted(YEARLY_METRICS)}")
    if cities:
        names = [resolve_city(x.strip()) for x in cities.split(",") if x.strip()]
    else:
        names = [r["city"] for r in db.query("SELECT city FROM cities ORDER BY city")]
    if not names:
        return {"metric": metric, "years": [], "series": {}}
    ph = ",".join("?" * len(names))
    rows = db.query(f"SELECT city, year, {metric} AS v FROM yearly_stats WHERE city IN ({ph}) ORDER BY year", names)
    years = sorted({r["year"] for r in rows})
    idx = {y: i for i, y in enumerate(years)}
    series = {n: [None] * len(years) for n in names}
    for r in rows:
        series[r["city"]][idx[r["year"]]] = r["v"]
    return {"metric": metric, "years": years, "series": series}


HEATMAP_COLS = {"temp_mean": "temp_mean", "precip_total": "precip_mean", "precip_mean": "precip_mean",
                "temp_max_mean": "temp_max_mean", "temp_min_mean": "temp_min_mean", "humidity_mean": "humidity_mean"}


@app.get("/api/heatmap")
def heatmap(metric: str = "temp_mean"):
    col = HEATMAP_COLS.get(metric)
    if not col:
        raise HTTPException(400, f"metric must be one of {sorted(HEATMAP_COLS)}")
    rows = db.query(f"SELECT city, month, {col} AS v FROM seasonal_profile ORDER BY city, month")
    names = sorted({r["city"] for r in rows})
    ci = {n: i for i, n in enumerate(names)}
    values = [[None] * 12 for _ in names]
    for r in rows:
        if r["month"] and 1 <= int(r["month"]) <= 12:
            values[ci[r["city"]]][int(r["month"]) - 1] = r["v"]
    return {"metric": metric, "cities": names, "months": list(range(1, 13)), "values": values}


@app.get("/api/extremes")
def extremes(city: Optional[str] = None, type: Optional[str] = None, limit: int = Query(50, ge=1, le=5000)):
    where, params = [], []
    if city:
        where.append("city=?")
        params.append(resolve_city(city))
    if type:
        where.append("event_type=?")
        params.append(type)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    events = db.query(f"SELECT * FROM extreme_events{w} ORDER BY date DESC, city LIMIT ?", params + [limit])
    total = db.scalar(f"SELECT count(*) FROM extreme_events{w}", params)
    by_year: dict[int, dict] = {}
    for r in db.query(f"SELECT CAST(substr(date,1,4) AS INTEGER) AS year, event_type, count(*) AS n "
                      f"FROM extreme_events{w} GROUP BY year, event_type ORDER BY year", params):
        e = by_year.setdefault(r["year"], {"year": r["year"], "heatwave": 0, "cold_wave": 0, "heavy_rain": 0})
        e[r["event_type"]] = r["n"]
    # "events" and "rows" carry the same list (either key may be used by clients)
    return {"events": events, "rows": events, "total": total, "counts_by_year": list(by_year.values())}


@app.get("/api/correlations")
def correlations():
    return db.query("SELECT * FROM correlations")


# ============================================================== models
@app.get("/api/models")
def models():
    fi: dict[str, list] = {}
    if db.table_exists("feature_importance"):
        for r in db.query("SELECT model_name, feature, importance FROM feature_importance ORDER BY model_name, importance DESC"):
            fi.setdefault(r["model_name"], []).append({"feature": r["feature"], "importance": r["importance"]})
    clusters = db.query("SELECT city, climate_cluster FROM cities ORDER BY climate_cluster, city") if db.table_exists("cities") else []
    metrics = current_metrics()
    version = engine.version or (max((m.get("trained_at") or "" for m in metrics), default=None) or None)
    return {"version": version, "spark_ready": engine.ready, "metrics": metrics, "models": group_models(metrics),
            "feature_importance": fi, "clusters": clusters}


@app.get("/api/models/{model_name}/predictions")
def model_predictions(model_name: str, city: Optional[str] = None, limit: int = Query(5000, ge=1, le=100000)):
    if city:
        return db.query("SELECT * FROM test_predictions WHERE model_name=? AND city=? ORDER BY date LIMIT ?",
                        (model_name, resolve_city(city), limit))
    return db.query("SELECT * FROM test_predictions WHERE model_name=? ORDER BY city, date LIMIT ?", (model_name, limit))


# ============================================================== live
@app.get("/api/live/history")
def live_history(city: Optional[str] = None, limit: int = Query(100, ge=1, le=10000)):
    if not db.table_exists("live_predictions"):
        return []
    if city:
        return db.query("SELECT * FROM live_predictions WHERE lower(city)=lower(?) ORDER BY id DESC LIMIT ?", (city, limit))
    return db.query("SELECT * FROM live_predictions ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/live")
def live_all():
    # While Spark warms up, observations are served with prediction = null.
    incomplete = len(engine.live) < len(engine.cities()) and not engine.live_meta["refreshing"]
    if not engine.live or incomplete or (engine.ready and not any(e.get("prediction") for e in engine.live.values())):
        try:
            engine.refresh()
        except Exception as e:
            raise HTTPException(502, f"live fetch from Open-Meteo failed: {e}")
    return [engine.public(engine.live[k]) for k in sorted(engine.live)]


@app.get("/api/live/{city}")
def live_city(city: str):
    # Never 503: observation/hourly_forecast/past_days are served immediately; prediction is
    # null until the Spark models are ready.
    try:
        entry = engine.live_city(city)
    except KeyError:
        raise HTTPException(404, f"unknown city '{city}'")
    except Exception as e:
        raise HTTPException(502, f"live fetch from Open-Meteo failed: {e}")
    if entry is None:
        raise HTTPException(502, "live fetch returned no data")
    pdf = entry["hourly"]
    today = live.local_today(entry["data"]).isoformat()
    daily = live.daily_from_hourly(pdf)
    past = daily[daily["date"] < today].tail(7)
    past_days = [db.clean(r) for r in past.astype(object).where(past.notna(), None).to_dict(orient="records")]
    return {
        "city": entry["city"],
        "observation": entry["observation"],
        "hourly_forecast": [db.clean(r) for r in live.next_hours(pdf, entry["observation"].get("time"), 48)],
        "past_days": past_days,
        "prediction": entry["prediction"],
        "model_status": "ready" if engine.ready else engine.status,
    }


# ============================================================== what-if / retrain
class PredictIn(BaseModel):
    city: str
    temp_max: Optional[float] = None
    temp_min: Optional[float] = None
    humidity_mean: Optional[float] = None
    pressure_mean: Optional[float] = None
    cloud_mean: Optional[float] = None
    precip_sum: Optional[float] = None
    wind_mean: Optional[float] = None


@app.post("/api/predict")
def predict(body: PredictIn):
    if not engine.ready:
        not_ready()
    city = resolve_city(body.city) if db.table_exists("cities") else body.city
    feats = {k: v for k, v in body.model_dump().items() if k != "city" and v is not None}
    if not engine.spark_lock.acquire(timeout=60):
        raise HTTPException(503, "Spark is busy (retraining). Try again in a minute.")
    try:
        res = engine.bundle.predict_from_features(city, feats)
    except KeyError as e:
        raise HTTPException(404, f"unknown city: {e}")
    except Exception as e:
        if engine.is_jvm_dead(e):
            engine.recover(e)
            raise HTTPException(503, "Spark JVM restarted; models are retraining. Try again in ~2 min.")
        raise HTTPException(400, f"prediction failed: {type(e).__name__}: {str(e)[:300]}")
    finally:
        engine.spark_lock.release()
    return db.clean({"city": city, "pred_temp_max": res.get("pred_temp_max"), "pred_rain_prob": res.get("pred_rain_prob"),
                     "pred_rain": res.get("pred_rain"), "model_version": engine.version, "inputs": feats})


@app.post("/api/retrain")
def retrain():
    r = engine.start_retrain()
    if r["status"] == "unavailable":
        raise HTTPException(503, r["detail"])
    return r


@app.get("/api/retrain/status")
def retrain_status():
    return db.clean(dict(engine.retrain))


@app.post("/api/live/refresh")
def live_refresh():
    """Force a live refresh now (handy during a demo)."""
    try:
        r = engine.refresh()
    except Exception as e:
        raise HTTPException(502, f"live fetch failed: {e}")
    return {"status": "ok", **r}


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
def api_404(rest: str):
    raise HTTPException(404, f"no such endpoint /api/{rest}")


# ============================================================== dashboard (SPA)
@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    index = DIST / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "dashboard not built (run `npm run build` in dashboard/). API docs at /docs"}, status_code=404)
    if full_path:
        target = (DIST / full_path).resolve()
        try:
            target.relative_to(DIST.resolve())
        except ValueError:
            raise HTTPException(404)
        if target.is_file():
            return FileResponse(target)
    return FileResponse(index)
