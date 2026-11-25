"""Open-Meteo Forecast API client (multi-location in one request)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "pressure_msl",
    "cloud_cover", "wind_speed_10m", "weather_code",
]
CURRENT_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "pressure_msl",
    "cloud_cover", "weather_code",
]
RAW_COLUMNS = ["city", "time"] + HOURLY_VARS
PAST_DAYS = 8
FORECAST_DAYS = 3  # contract says 2; 3 lets /api/live/{city} show a full 48 h ahead
TIMEOUT = httpx.Timeout(45.0, connect=15.0)
CHUNK = 15  # locations per request


def _params(lats: list[float], lons: list[float]) -> dict[str, str]:
    return {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": ",".join(HOURLY_VARS),
        "current": ",".join(CURRENT_VARS),
        "past_days": str(PAST_DAYS),
        "forecast_days": str(FORECAST_DAYS),
        "timezone": "Asia/Kolkata",
    }


def fetch_raw(cities: list[dict]) -> dict[str, dict]:
    """cities: [{city, lat, lon}] -> {city: open-meteo json}. Raises on total failure."""
    out: dict[str, dict] = {}
    errors: list[str] = []
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": "wbd-weather-backend/1.0"}) as client:
        for i in range(0, len(cities), CHUNK):
            chunk = cities[i:i + CHUNK]
            try:
                r = client.get(FORECAST_URL, params=_params([c["lat"] for c in chunk], [c["lon"] for c in chunk]))
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    data = [data]
                for c, d in zip(chunk, data):
                    out[c["city"]] = d
            except Exception as e:  # fall back to one request per city
                errors.append(f"chunk {i}: {e}")
                for c in chunk:
                    try:
                        r = client.get(FORECAST_URL, params=_params([c["lat"]], [c["lon"]]))
                        r.raise_for_status()
                        out[c["city"]] = r.json()
                    except Exception as e2:
                        errors.append(f"{c['city']}: {e2}")
    if not out and errors:
        raise RuntimeError("Open-Meteo fetch failed: " + "; ".join(errors[:5]))
    return out


def hourly_frame(city: str, data: dict) -> pd.DataFrame:
    h = data.get("hourly") or {}
    pdf = pd.DataFrame({"time": h.get("time", [])})
    for v in HOURLY_VARS:
        vals = h.get(v)
        pdf[v] = pd.to_numeric(pd.Series(vals if vals is not None else [None] * len(pdf)), errors="coerce")
    pdf.insert(0, "city", city)
    return pdf[RAW_COLUMNS]


def current_obs(data: dict) -> dict[str, Any]:
    c = data.get("current") or {}
    return {
        "time": c.get("time"),
        "temperature": c.get("temperature_2m"),
        "humidity": c.get("relative_humidity_2m"),
        "precipitation": c.get("precipitation"),
        "wind_speed": c.get("wind_speed_10m"),
        "pressure": c.get("pressure_msl"),
        "cloud_cover": c.get("cloud_cover"),
        "weather_code": c.get("weather_code"),
    }


def local_today(data: dict) -> date:
    t = (data.get("current") or {}).get("time")
    if t:
        return datetime.fromisoformat(t).date()
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def model_input(pdf: pd.DataFrame, today: date) -> pd.DataFrame:
    """Hourly rows up to the end of local today (forecast hours of later days dropped).
    spark_jobs.ml additionally ignores the current IST date, so "today" for the model is
    yesterday's fully observed day and target_date is the current date."""
    cutoff = (today + timedelta(days=1)).isoformat()
    sub = pdf[pdf["time"] < cutoff].copy()
    # forward/back fill tiny gaps so the feature pipeline never sees NaN
    num = HOURLY_VARS
    sub[num] = sub[num].ffill().bfill()
    return sub.reset_index(drop=True)


def daily_from_hourly(pdf: pd.DataFrame) -> pd.DataFrame:
    d = pdf.copy()
    d["date"] = d["time"].str[:10]
    g = d.groupby("date")
    out = pd.DataFrame({
        "temp_max": g["temperature_2m"].max(),
        "temp_min": g["temperature_2m"].min(),
        "temp_mean": g["temperature_2m"].mean(),
        "humidity_mean": g["relative_humidity_2m"].mean(),
        "dew_point_mean": g["dew_point_2m"].mean(),
        "precip_sum": g["precipitation"].sum(min_count=1),
        "pressure_mean": g["pressure_msl"].mean(),
        "cloud_mean": g["cloud_cover"].mean(),
        "wind_max": g["wind_speed_10m"].max(),
        "wind_mean": g["wind_speed_10m"].mean(),
        "hours": g["time"].count(),
    }).reset_index()
    return out.round(2)


def openmeteo_day(pdf: pd.DataFrame, day: str) -> tuple[float | None, float | None]:
    sub = pdf[pdf["time"].str.startswith(day)]
    if sub.empty:
        return None, None
    tmax = sub["temperature_2m"].max()
    psum = sub["precipitation"].sum(min_count=1)
    return (None if pd.isna(tmax) else round(float(tmax), 2), None if pd.isna(psum) else round(float(psum), 2))


def next_hours(pdf: pd.DataFrame, now_iso: str | None, hours: int = 48) -> list[dict]:
    start = (now_iso or "")[:13]  # 'YYYY-MM-DDTHH'
    sub = pdf[pdf["time"].str[:13] >= start] if start else pdf
    sub = sub.head(hours).drop(columns=["city"])
    return sub.astype(object).where(sub.notna(), None).to_dict(orient="records")
