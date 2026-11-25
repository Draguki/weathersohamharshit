"""
etl.py - Big-data ETL: raw hourly CSVs -> Spark aggregations -> SQLite (db/weather.db).

Steps (each logged to the pipeline_runs table):
  1. load_clean     one glob read of ALL data/raw/hourly_*.csv, drop nulls / duplicates,
                    physical range checks
  2. daily          hourly -> daily_weather (shared function from features.py)
  3. monthly        monthly_stats
  4. yearly         yearly_stats (incl. heatwave_days)
  5. seasonal       seasonal_profile (monthly climatology)
  6. city_summary   records, warming trend, monsoon share, rainy days
  7. extremes       extreme_events (IMD-style heatwave / cold wave / heavy rain)
  8. correlations   Pearson correlation matrix over the daily table
  9. cities         cities table (keeps any existing climate_cluster)

Spark cannot write to disk on this machine, so every result is .toPandas()'d and
written with sqlite3.  Run:  python spark_jobs/etl.py   (from the project root)
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from spark_jobs.common import (CITIES, RAW_DIR, connect_db, get_spark, log_run,  # noqa: E402
                               new_run_id, PIPELINE_RUNS_DDL)
from spark_jobs.features import DAILY_VARS, RAW_SCHEMA, hourly_to_daily  # noqa: E402

from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.stat import Correlation  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

RAINY_DAY_MM = 2.5          # IMD definition of a rainy day
HEAVY_RAIN_MM = 64.5        # IMD "heavy rain"
MIN_HOURS_PER_DAY = 18      # a day needs >= 18 hourly readings to be kept
COMPLETE_YEAR_DAYS = 360    # years with fewer days (e.g. 2025 Jan-Oct) excluded from trends

# --------------------------------------------------------------------------
# SQLite schema (docs/CONTRACT.md)
# --------------------------------------------------------------------------
DDL = {
    "cities": """CREATE TABLE cities (city TEXT PRIMARY KEY, state TEXT, region TEXT,
        lat REAL, lon REAL, climate_cluster INTEGER)""",
    "daily_weather": """CREATE TABLE daily_weather (city TEXT, date TEXT, temp_max REAL,
        temp_min REAL, temp_mean REAL, humidity_mean REAL, dew_point_mean REAL,
        precip_sum REAL, pressure_mean REAL, cloud_mean REAL, wind_max REAL, wind_mean REAL,
        PRIMARY KEY (city, date))""",
    "monthly_stats": """CREATE TABLE monthly_stats (city TEXT, year INTEGER, month INTEGER,
        temp_mean REAL, temp_max REAL, temp_min REAL, precip_total REAL, humidity_mean REAL,
        rainy_days INTEGER)""",
    "yearly_stats": """CREATE TABLE yearly_stats (city TEXT, year INTEGER, temp_mean REAL,
        temp_max_abs REAL, temp_min_abs REAL, precip_total REAL, rainy_days INTEGER,
        heatwave_days INTEGER)""",
    "seasonal_profile": """CREATE TABLE seasonal_profile (city TEXT, month INTEGER,
        temp_mean REAL, temp_max_mean REAL, temp_min_mean REAL, precip_mean REAL,
        humidity_mean REAL)""",
    "city_summary": """CREATE TABLE city_summary (city TEXT PRIMARY KEY, record_high REAL,
        record_high_date TEXT, record_low REAL, record_low_date TEXT, avg_temp REAL,
        avg_annual_precip REAL, avg_humidity REAL, warming_trend_c_per_decade REAL,
        monsoon_share_pct REAL, rainy_days_per_year REAL, rows_processed INTEGER)""",
    "extreme_events": """CREATE TABLE extreme_events (city TEXT, date TEXT, event_type TEXT,
        value REAL)""",
    "correlations": """CREATE TABLE correlations (var_x TEXT, var_y TEXT, corr REAL)""",
}
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_monthly_city ON monthly_stats(city)",
    "CREATE INDEX IF NOT EXISTS ix_yearly_city ON yearly_stats(city)",
    "CREATE INDEX IF NOT EXISTS ix_seasonal_city ON seasonal_profile(city)",
    "CREATE INDEX IF NOT EXISTS ix_extreme_city ON extreme_events(city, date)",
]
# Tables owned by the backend - created if missing, never dropped here
LIVE_DDL = [
    """CREATE TABLE IF NOT EXISTS live_observations (city TEXT, fetched_at TEXT,
        temperature REAL, humidity REAL, precipitation REAL, wind_speed REAL, pressure REAL,
        cloud_cover REAL, weather_code INTEGER)""",
    """CREATE TABLE IF NOT EXISTS live_predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT, predicted_at TEXT, target_date TEXT, pred_temp_max REAL,
        pred_rain_prob REAL, pred_rain INTEGER, openmeteo_temp_max REAL,
        openmeteo_precip REAL, model_version TEXT)""",
]


def write_table(con, name, pdf):
    """Drop + recreate table with the contract schema, then bulk insert."""
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(DDL[name])
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
    pdf = pdf[cols]
    pdf = pdf.astype(object).where(pd.notna(pdf), None)
    con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(cols))})",
                    pdf.itertuples(index=False, name=None))
    con.commit()
    return len(pdf)


def r2(col, nd=2):
    return F.round(col, nd)


def main():
    t_all = time.time()
    run_id = new_run_id("etl")
    spark = get_spark()
    con = connect_db()
    con.execute(PIPELINE_RUNS_DDL)
    for ddl in LIVE_DDL:
        con.execute(ddl)
    print(f"Spark {spark.version} ready; run_id={run_id}", flush=True)

    # ---------------------------------------------------------------- 1. load + clean
    t = time.time()
    # One Spark read over ALL city files.  The file list is expanded in Python:
    # Hadoop's own glob/directory listing needs winutils (missing on this Windows box)
    # and hangs, whereas reading an explicit list of files works fine.
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "hourly_*.csv")))
    if not raw_files:
        raise SystemExit(f"No raw files in {RAW_DIR} - run spark_jobs/fetch_data.py first")
    n_files = len(raw_files)
    raw = spark.read.csv(raw_files, header=True, schema=RAW_SCHEMA)
    n_raw = raw.count()

    def in_range(c, lo, hi):
        """Out-of-range values become null (row is kept, bad value discarded)."""
        return F.when(F.col(c).between(lo, hi), F.col(c))

    clean = (
        raw.dropna(subset=["city", "time", "temperature_2m"])
           .dropDuplicates(["city", "time"])
           .filter(F.col("temperature_2m").between(-60, 60))
           .withColumn("relative_humidity_2m", in_range("relative_humidity_2m", 0, 100))
           .withColumn("dew_point_2m", in_range("dew_point_2m", -70, 45))
           .withColumn("precipitation", F.coalesce(in_range("precipitation", 0, 500), F.lit(0.0)))
           .withColumn("pressure_msl", in_range("pressure_msl", 850, 1100))
           .withColumn("cloud_cover", in_range("cloud_cover", 0, 100))
           .withColumn("wind_speed_10m", in_range("wind_speed_10m", 0, 300))
           .repartition(32, "city")
    ).cache()
    n_clean = clean.count()
    log_run(con, run_id, "etl_load_clean", t, n_raw, n_clean)
    print(f"  raw files={n_files}, raw rows={n_raw:,}, clean rows={n_clean:,}", flush=True)

    # ---------------------------------------------------------------- 2. daily
    t = time.time()
    daily_all = hourly_to_daily(clean).cache()
    rows_per_city = {r["city"]: int(r["n"]) for r in
                     daily_all.groupBy("city").agg(F.sum("n_hours").alias("n")).collect()}
    daily = (daily_all.filter(F.col("n_hours") >= MIN_HOURS_PER_DAY)
             .withColumn("year", F.year("date")).withColumn("month", F.month("date"))
             .cache())
    daily_pdf = daily.select(
        "city", F.date_format("date", "yyyy-MM-dd").alias("date"),
        *[r2(F.col(c)).alias(c) for c in DAILY_VARS]).toPandas()
    n_daily = write_table(con, "daily_weather", daily_pdf)
    log_run(con, run_id, "etl_daily", t, n_clean, n_daily)

    # ---------------------------------------------------------------- 3. monthly
    t = time.time()
    monthly = daily.groupBy("city", "year", "month").agg(
        r2(F.avg("temp_mean")).alias("temp_mean"),
        r2(F.max("temp_max")).alias("temp_max"),
        r2(F.min("temp_min")).alias("temp_min"),
        r2(F.sum("precip_sum")).alias("precip_total"),
        r2(F.avg("humidity_mean")).alias("humidity_mean"),
        F.sum((F.col("precip_sum") >= RAINY_DAY_MM).cast("int")).alias("rainy_days"),
    ).cache()
    n_monthly = write_table(con, "monthly_stats", monthly.orderBy("city", "year", "month").toPandas())
    log_run(con, run_id, "etl_monthly", t, n_daily, n_monthly)

    # ---------------------------------------------------------------- 5a. climatology (needed by extremes)
    clim = daily.groupBy("city", "month").agg(
        F.avg("temp_mean").alias("temp_mean"),
        F.avg("temp_max").alias("clim_max"),
        F.avg("temp_min").alias("clim_min"),
        F.avg("humidity_mean").alias("humidity_mean"),
    ).cache()

    # ---------------------------------------------------------------- 7. extremes
    t = time.time()
    d = daily.join(F.broadcast(clim.select("city", "month", "clim_max", "clim_min")),
                   ["city", "month"])
    heat = d.filter(((F.col("temp_max") >= 40) & (F.col("temp_max") - F.col("clim_max") >= 4.5))
                    | (F.col("temp_max") >= 45)) \
            .select("city", "date", F.lit("heatwave").alias("event_type"),
                    F.col("temp_max").alias("value"))
    cold = d.filter((F.col("temp_min") <= 10) & (F.col("clim_min") - F.col("temp_min") >= 4.5)) \
            .select("city", "date", F.lit("cold_wave").alias("event_type"),
                    F.col("temp_min").alias("value"))
    rain = d.filter(F.col("precip_sum") >= HEAVY_RAIN_MM) \
            .select("city", "date", F.lit("heavy_rain").alias("event_type"),
                    F.col("precip_sum").alias("value"))
    extremes = heat.unionByName(cold).unionByName(rain).cache()
    extremes_pdf = extremes.select(
        "city", F.date_format("date", "yyyy-MM-dd").alias("date"), "event_type",
        r2(F.col("value")).alias("value")).orderBy("date", "city").toPandas()
    n_ext = write_table(con, "extreme_events", extremes_pdf)
    log_run(con, run_id, "etl_extremes", t, n_daily, n_ext)

    # ---------------------------------------------------------------- 4. yearly
    t = time.time()
    hw_per_year = extremes.filter(F.col("event_type") == "heatwave") \
        .groupBy("city", F.year("date").alias("year")).agg(F.count("*").alias("heatwave_days"))
    yearly = daily.groupBy("city", "year").agg(
        F.avg("temp_mean").alias("temp_mean_raw"),
        r2(F.max("temp_max")).alias("temp_max_abs"),
        r2(F.min("temp_min")).alias("temp_min_abs"),
        r2(F.sum("precip_sum")).alias("precip_total"),
        F.sum((F.col("precip_sum") >= RAINY_DAY_MM).cast("int")).alias("rainy_days"),
        F.count("*").alias("n_days"),
    ).join(hw_per_year, ["city", "year"], "left") \
     .withColumn("heatwave_days", F.coalesce("heatwave_days", F.lit(0))) \
     .withColumn("temp_mean", r2(F.col("temp_mean_raw"))).cache()
    n_yearly = write_table(con, "yearly_stats", yearly.orderBy("city", "year").toPandas())
    log_run(con, run_id, "etl_yearly", t, n_daily, n_yearly)

    # ---------------------------------------------------------------- 5. seasonal profile
    t = time.time()
    precip_clim = monthly.groupBy("city", "month").agg(F.avg("precip_total").alias("precip_mean"))
    seasonal = clim.join(precip_clim, ["city", "month"]).select(
        "city", "month", r2(F.col("temp_mean")).alias("temp_mean"),
        r2(F.col("clim_max")).alias("temp_max_mean"), r2(F.col("clim_min")).alias("temp_min_mean"),
        r2(F.col("precip_mean")).alias("precip_mean"), r2(F.col("humidity_mean")).alias("humidity_mean"))
    n_seasonal = write_table(con, "seasonal_profile", seasonal.orderBy("city", "month").toPandas())
    log_run(con, run_id, "etl_seasonal", t, n_monthly, n_seasonal)

    # ---------------------------------------------------------------- 6. city summary
    t = time.time()
    complete = yearly.filter(F.col("n_days") >= COMPLETE_YEAR_DAYS)
    # warming trend = OLS slope of yearly mean temperature vs year, x10 => degC / decade
    trend = complete.groupBy("city").agg(
        (F.covar_pop("year", "temp_mean_raw") / F.var_pop("year") * 10).alias("warming_trend_c_per_decade"),
        F.avg("precip_total").alias("avg_annual_precip"),
        F.avg("rainy_days").alias("rainy_days_per_year"))
    complete_years = complete.select("city", "year")
    monsoon = daily.join(complete_years, ["city", "year"]).groupBy("city").agg(
        (F.sum(F.when(F.col("month").between(6, 9), F.col("precip_sum")).otherwise(0.0))
         / F.sum("precip_sum") * 100).alias("monsoon_share_pct"))
    records = daily.groupBy("city").agg(
        F.max("temp_max").alias("record_high"),
        F.max_by("date", "temp_max").alias("record_high_date"),
        F.min("temp_min").alias("record_low"),
        F.min_by("date", "temp_min").alias("record_low_date"),
        F.avg("temp_mean").alias("avg_temp"),
        F.avg("humidity_mean").alias("avg_humidity"))
    summary_pdf = records.join(trend, "city").join(monsoon, "city").toPandas()
    for c in ("record_high_date", "record_low_date"):
        summary_pdf[c] = pd.to_datetime(summary_pdf[c]).dt.strftime("%Y-%m-%d")
    for c in ("record_high", "record_low", "avg_temp", "avg_annual_precip", "avg_humidity",
              "monsoon_share_pct", "rainy_days_per_year"):
        summary_pdf[c] = summary_pdf[c].astype(float).round(2)
    summary_pdf["warming_trend_c_per_decade"] = summary_pdf["warming_trend_c_per_decade"].astype(float).round(3)
    summary_pdf["rows_processed"] = summary_pdf["city"].map(rows_per_city).astype(int)
    n_summary = write_table(con, "city_summary", summary_pdf.sort_values("city"))
    log_run(con, run_id, "etl_city_summary", t, n_daily, n_summary)

    # ---------------------------------------------------------------- 8. correlations
    t = time.time()
    vec = VectorAssembler(inputCols=DAILY_VARS, outputCol="v", handleInvalid="skip") \
        .transform(daily.select(*DAILY_VARS))
    matrix = Correlation.corr(vec, "v", "pearson").first()[0].toArray()
    corr_rows = [(x, y, round(float(matrix[i][j]), 4))
                 for i, x in enumerate(DAILY_VARS) for j, y in enumerate(DAILY_VARS)]
    n_corr = write_table(con, "correlations", pd.DataFrame(corr_rows, columns=["var_x", "var_y", "corr"]))
    log_run(con, run_id, "etl_correlations", t, n_daily, n_corr)

    # ---------------------------------------------------------------- 9. cities
    t = time.time()
    old_clusters = {}
    try:
        old_clusters = dict(con.execute("SELECT city, climate_cluster FROM cities").fetchall())
    except Exception:
        pass
    cities_pdf = pd.DataFrame(CITIES)
    cities_pdf["climate_cluster"] = cities_pdf["city"].map(old_clusters)
    n_cities = write_table(con, "cities", cities_pdf)
    log_run(con, run_id, "etl_cities", t, len(CITIES), n_cities)

    for ix in INDEXES:
        con.execute(ix)
    con.commit()
    total_out = n_daily + n_monthly + n_yearly + n_seasonal + n_summary + n_ext + n_corr + n_cities
    log_run(con, run_id, "etl_total", t_all, n_raw, total_out)
    con.close()
    print(f"ETL finished in {time.time() - t_all:.1f}s: daily={n_daily:,} monthly={n_monthly:,} "
          f"yearly={n_yearly:,} extremes={n_ext:,}", flush=True)


if __name__ == "__main__":
    main()
