import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from spark_jobs.common import (DB_PATH, RAW_DIR, RESULTS_DIR, SUMMARY_JSON,  # noqa: E402
                               connect_db, get_spark, log_run, new_run_id, now_iso,
                               spark_config_info)
from spark_jobs.ml import train_models  # noqa: E402

TEST_PRED_CAP = 65000   

DDL = {
    "model_metrics": """CREATE TABLE model_metrics (model_name TEXT, task TEXT, target TEXT,
        metric TEXT, value REAL, trained_at TEXT, train_rows INTEGER, test_rows INTEGER)""",
    "feature_importance": """CREATE TABLE feature_importance (model_name TEXT, feature TEXT,
        importance REAL)""",
    "test_predictions": """CREATE TABLE test_predictions (model_name TEXT, city TEXT, date TEXT,
        actual REAL, predicted REAL)""",
}


def write_table(con, name, pdf):
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(DDL[name])
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
    pdf = pdf[cols].astype(object).where(pd.notna(pdf[cols]), None)
    con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(cols))})",
                    pdf.itertuples(index=False, name=None))
    con.commit()
    return len(pdf)


def build_summary(con, bundle, n_test_pred):
    q = lambda sql: pd.read_sql_query(sql, con)  # noqa: E731
    cs = q("SELECT * FROM city_summary")
    cities = q("SELECT * FROM cities")
    rng = q("SELECT MIN(date) AS s, MAX(date) AS e, COUNT(*) AS n FROM daily_weather").iloc[0]
    runs = q("SELECT job, duration_s, rows_in, rows_out, finished_at FROM pipeline_runs "
             "WHERE job IN ('etl_total','train_total','etl_load_clean') ORDER BY finished_at")
    last = {r.job: r for r in runs.itertuples()}
    events = dict(q("SELECT event_type, COUNT(*) AS n FROM extreme_events GROUP BY event_type")
                  .itertuples(index=False, name=None))

    def city_val(df, col, largest=True):
        r = df.sort_values(col, ascending=not largest).iloc[0]
        return {"city": r["city"], "value": round(float(r[col]), 2)}

    hi = cs.sort_values("record_high", ascending=False).iloc[0]
    lo = cs.sort_values("record_low").iloc[0]
    metrics = {}
    for m in bundle.metrics:
        metrics.setdefault(m["model_name"], {})[m["metric"]] = m["value"]
    fi = pd.DataFrame(bundle.feature_importance)
    top_fi = {name: g.sort_values("importance", ascending=False).head(8)
              [["feature", "importance"]].to_dict("records")
              for name, g in fi.groupby("model_name")}
    raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith("hourly_") and f.endswith(".csv")]
    raw_bytes = sum(os.path.getsize(os.path.join(RAW_DIR, f)) for f in raw_files)

    return {
        "generated_at": now_iso(),
        "dataset": {
            "source": "Open-Meteo Historical Weather API (ERA5 reanalysis), hourly",
            "total_raw_rows": int(last["etl_load_clean"].rows_in) if "etl_load_clean" in last else None,
            "clean_rows": int(last["etl_load_clean"].rows_out) if "etl_load_clean" in last else None,
            "raw_files": len(raw_files),
            "raw_size_mb": round(raw_bytes / 1e6, 1),
            "cities": int(len(cities)),
            "regions": sorted(cities["region"].unique().tolist()),
            "date_start": rng["s"], "date_end": rng["e"],
            "daily_rows": int(rng["n"]),
            "db_size_mb": round(os.path.getsize(DB_PATH) / 1e6, 1),
        },
        "models": metrics,
        "train_rows": bundle.train_rows, "test_rows": bundle.test_rows,
        "test_predictions_rows": n_test_pred,
        "top_feature_importance": top_fi,
        "climate_clusters": bundle.clusters,
        "climate_cluster_members": {
            str(k): sorted(c for c, v in bundle.clusters.items() if v == k)
            for k in sorted(set(bundle.clusters.values()))},
        "hottest_city": city_val(cs, "avg_temp"),
        "coldest_city": city_val(cs, "avg_temp", largest=False),
        "wettest_city": city_val(cs, "avg_annual_precip"),
        "driest_city": city_val(cs, "avg_annual_precip", largest=False),
        "record_high": {"city": hi["city"], "value": float(hi["record_high"]), "date": hi["record_high_date"]},
        "record_low": {"city": lo["city"], "value": float(lo["record_low"]), "date": lo["record_low_date"]},
        "extreme_event_counts": {k: int(events.get(k, 0)) for k in ("heatwave", "cold_wave", "heavy_rain")},
        "warming_trend_c_per_decade": dict(zip(cs["city"], cs["warming_trend_c_per_decade"].round(3))),
        "avg_warming_trend_c_per_decade": round(float(cs["warming_trend_c_per_decade"].mean()), 3),
        "monsoon_share_pct": dict(zip(cs["city"], cs["monsoon_share_pct"].round(1))),
        "durations_s": {
            "etl_total": float(last["etl_total"].duration_s) if "etl_total" in last else None,
            "training": bundle.training_seconds,
            "train_total": float(last["train_total"].duration_s) if "train_total" in last else None,
        },
        "spark": spark_config_info(),
    }


def main():
    t_all = time.time()
    run_id = new_run_id("train")
    spark = get_spark()
    con = connect_db()
    print(f"Spark {spark.version} ready; run_id={run_id}", flush=True)

    t = time.time()
    bundle = train_models(spark)          # reads daily_weather from SQLite
    log_run(con, run_id, "train_models", t, bundle.train_rows + bundle.test_rows, len(bundle.metrics))

    t = time.time()
    n_met = write_table(con, "model_metrics", pd.DataFrame(bundle.metrics))
    n_fi = write_table(con, "feature_importance", pd.DataFrame(bundle.feature_importance))
    log_run(con, run_id, "train_write_metrics", t, len(bundle.metrics), n_met + n_fi)

    t = time.time()
    tp = bundle.test_predictions_pdf()
    if len(tp) > TEST_PRED_CAP:   # stratified sample per model if the test set is huge
        tp = tp.groupby("model_name").sample(frac=TEST_PRED_CAP / len(tp), random_state=1)
    tp["actual"] = tp["actual"].astype(float).round(3)
    tp["predicted"] = tp["predicted"].astype(float).round(3)
    tp = tp.sort_values(["model_name", "city", "date"])
    n_tp = write_table(con, "test_predictions", tp)
    con.execute("CREATE INDEX IF NOT EXISTS ix_testpred ON test_predictions(model_name, city)")
    log_run(con, run_id, "train_test_predictions", t, len(tp), n_tp)


    t = time.time()
    con.executemany("UPDATE cities SET climate_cluster=? WHERE city=?",
                    [(int(v), c) for c, v in bundle.clusters.items()])
    con.commit()
    log_run(con, run_id, "train_update_clusters", t, len(bundle.clusters), len(bundle.clusters))
    log_run(con, run_id, "train_total", t_all, bundle.train_rows + bundle.test_rows,
            n_met + n_fi + n_tp)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = build_summary(con, bundle, n_tp)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    con.close()

    print("\nModel metrics (test period 2024+):")
    for name, vals in summary["models"].items():
        print(f"  {name:<15} " + "  ".join(f"{k}={v}" for k, v in vals.items()))
    print(f"Training finished in {time.time() - t_all:.1f}s; summary -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
