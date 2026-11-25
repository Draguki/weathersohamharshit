"""SQLite helpers: per-call connections, NaN-safe JSON rows, live-table DDL."""
from __future__ import annotations

import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("WBD_DB_PATH") or (ROOT / "db" / "weather.db")).resolve()

_write_lock = threading.Lock()

LIVE_DDL = [
    """CREATE TABLE IF NOT EXISTS live_observations (
        city TEXT, fetched_at TEXT, temperature REAL, humidity REAL, precipitation REAL,
        wind_speed REAL, pressure REAL, cloud_cover REAL, weather_code INTEGER)""",
    """CREATE TABLE IF NOT EXISTS live_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, city TEXT, predicted_at TEXT, target_date TEXT,
        pred_temp_max REAL, pred_rain_prob REAL, pred_rain INTEGER,
        openmeteo_temp_max REAL, openmeteo_precip REAL, model_version TEXT)""",
]

OTHER_DDL = [
    """CREATE TABLE IF NOT EXISTS model_metrics (
        model_name TEXT, task TEXT, target TEXT, metric TEXT, value REAL,
        trained_at TEXT, train_rows INTEGER, test_rows INTEGER)""",
    """CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT, job TEXT, started_at TEXT, finished_at TEXT, rows_in INTEGER,
        rows_out INTEGER, duration_s REAL, status TEXT)""",
]


class DBMissing(Exception):
    pass


def clean(v: Any) -> Any:
    """Make a value JSON safe (NaN/inf -> None, bytes -> str, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    # numpy scalars
    item = getattr(v, "item", None)
    if callable(item) and type(v).__module__ == "numpy":
        try:
            return clean(item())
        except Exception:
            return str(v)
    return v


def db_exists() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def connect() -> sqlite3.Connection:
    if not db_exists():
        raise DBMissing(f"database not found at {DB_PATH} (run the ETL first)")
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    conn = connect()
    try:
        cur = conn.execute(sql, tuple(params))
        return [{k: clean(r[k]) for k in r.keys()} for r in cur.fetchall()]
    finally:
        conn.close()


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def scalar(sql: str, params: Iterable[Any] = ()) -> Any:
    conn = connect()
    try:
        r = conn.execute(sql, tuple(params)).fetchone()
        return clean(r[0]) if r else None
    finally:
        conn.close()


def table_exists(name: str) -> bool:
    try:
        return bool(scalar("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)))
    except DBMissing:
        return False


def columns(table: str) -> list[str]:
    try:
        return [r["name"] for r in query(f"PRAGMA table_info({table})")]
    except DBMissing:
        return []


def execute_many(statements: list[tuple[str, list[tuple]]], ddl: list[str] | None = None) -> bool:
    """Run write statements in one transaction. Skips silently (returns False) if DB missing."""
    if not db_exists():
        return False
    with _write_lock:
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
        try:
            for d in ddl or []:
                conn.execute(d)
            for sql, rows in statements:
                if rows:
                    conn.executemany(sql, rows)
            conn.commit()
            return True
        finally:
            conn.close()
