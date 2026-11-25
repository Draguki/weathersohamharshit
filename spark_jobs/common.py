"""
common.py - shared helpers for all Spark jobs.

* Project paths (data/, db/, results/)
* get_spark()  -> one SparkSession per process (singleton, local mode)
* load_cities() / CITY_INDEX -> fixed city -> integer mapping from cities.json order
* sqlite helpers (connect, log a pipeline run)

IMPORTANT: on this Windows machine there is no winutils, so Spark can READ files
but cannot WRITE to disk.  Pattern used everywhere: aggregate in Spark ->
.toPandas() the (small) result -> write it to SQLite with pandas/sqlite3.
"""
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.environ.get("WBD_RAW_DIR", os.path.join(DATA_DIR, "raw"))   # env override for tests
CITIES_JSON = os.path.join(DATA_DIR, "cities.json")
DB_DIR = os.path.join(ROOT, "db")
DB_PATH = os.environ.get("WBD_DB_PATH", os.path.join(DB_DIR, "weather.db"))   # env override for tests
RESULTS_DIR = os.path.join(ROOT, "results")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "summary.json")

# Spark configuration (shown in the presentation / summary.json)
SPARK_CORES = os.cpu_count() or 4
SHUFFLE_PARTITIONS = 32
DRIVER_MEMORY = "4g"

JAVA_HOME_DEFAULT = r"C:\Program Files\Java\jre1.8.0_461"

# Must be set BEFORE pyspark is imported anywhere.
if not os.environ.get("JAVA_HOME") or not os.path.isdir(os.environ.get("JAVA_HOME", "")):
    if os.path.isdir(JAVA_HOME_DEFAULT):
        os.environ["JAVA_HOME"] = JAVA_HOME_DEFAULT
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# --------------------------------------------------------------------------
# Cities
# --------------------------------------------------------------------------
def load_cities():
    """List of {city, state, region, lat, lon} in the fixed cities.json order."""
    with open(CITIES_JSON, encoding="utf-8") as f:
        return json.load(f)


CITIES = load_cities()
# Fixed integer id per city (order of cities.json) - identical at training and
# inference time, so a city can never be "unseen" by the model.
CITY_INDEX = {c["city"]: i for i, c in enumerate(CITIES)}
CITY_LATLON = {c["city"]: (c["lat"], c["lon"]) for c in CITIES}


def slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --------------------------------------------------------------------------
# SparkSession singleton
# --------------------------------------------------------------------------
_spark = None
_spark_lock = threading.Lock()


def get_spark(app_name: str = "IndianWeatherBigData"):
    """Return the process-wide SparkSession (created on first call, ~20 s)."""
    global _spark
    with _spark_lock:
        if _spark is None:
            from pyspark.sql import SparkSession
            _spark = (
                SparkSession.builder
                .appName(app_name)
                .master(f"local[{SPARK_CORES}]")
                .config("spark.driver.memory", DRIVER_MEMORY)
                .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
                .config("spark.sql.session.timeZone", "Asia/Kolkata")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
                .config("spark.ui.showConsoleProgress", "false")
                .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=ERROR,console")
                .getOrCreate()
            )
            _spark.sparkContext.setLogLevel("ERROR")
        return _spark


def spark_config_info():
    """Small dict describing the Spark setup (for summary.json / slides)."""
    return {
        "master": f"local[{SPARK_CORES}]",
        "cores": SPARK_CORES,
        "shuffle_partitions": SHUFFLE_PARTITIONS,
        "driver_memory": DRIVER_MEMORY,
        "spark_version": _spark.version if _spark is not None else None,
    }


# --------------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------------
def connect_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")   # backend can read while we write
    return con


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT, job TEXT, started_at TEXT, finished_at TEXT,
    rows_in INTEGER, rows_out INTEGER, duration_s REAL, status TEXT)"""


def log_run(con, run_id, job, started_ts, rows_in, rows_out, status="success"):
    """Insert one row into pipeline_runs (started_ts = time.time() at job start)."""
    con.execute(PIPELINE_RUNS_DDL)
    finished = time.time()
    con.execute(
        "INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?,?,?)",
        (run_id, job,
         datetime.fromtimestamp(started_ts).strftime("%Y-%m-%dT%H:%M:%S"),
         datetime.fromtimestamp(finished).strftime("%Y-%m-%dT%H:%M:%S"),
         int(rows_in) if rows_in is not None else None,
         int(rows_out) if rows_out is not None else None,
         round(finished - started_ts, 2), status),
    )
    con.commit()
    print(f"  [{job}] rows_in={rows_in} rows_out={rows_out} "
          f"{finished - started_ts:.1f}s {status}", flush=True)


def new_run_id(prefix):
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
