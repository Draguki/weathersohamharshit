"""Spark/ML state, live refresher and retrain worker (all background threads)."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Any

from . import db, live

ROOT = db.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("wbd.engine")
FAKE = os.environ.get("WBD_FAKE_MODEL") == "1"
REFRESH_S = int(os.environ.get("WBD_REFRESH_S", "600"))
CACHE_TTL_S = 600


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class _ArrowSparkProxy:
    """Wraps a SparkSession so `createDataFrame(list_of_tuples, schema)` goes through pandas +
    Arrow instead of a Python RDD. On Windows a Python-RDD job spawns fresh python.exe workers
    for every task (no fork daemon), which made each live prediction take 15-45 s; the Arrow path
    needs no Python workers at all. Everything else is delegated untouched."""

    def __init__(self, spark):
        object.__setattr__(self, "_spark", spark)

    def createDataFrame(self, data, schema=None, *args, **kwargs):
        try:
            from pyspark.sql.types import StructType
            if isinstance(data, list) and isinstance(schema, StructType) and data and not args and not kwargs:
                import pandas as pd
                pdf = pd.DataFrame([tuple(r) for r in data], columns=schema.names)
                return self._spark.createDataFrame(pdf, schema=schema)
        except Exception as e:  # fall back to the original path
            log.debug("arrow createDataFrame fallback: %s", e)
        return self._spark.createDataFrame(data, schema, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._spark, name)


def _speed_up(bundle):
    if os.environ.get("WBD_NO_ARROW_PROXY") == "1" or bundle is None:
        return bundle
    sp = getattr(bundle, "spark", None)
    if sp is not None and not isinstance(sp, _ArrowSparkProxy):
        try:
            bundle.spark = _ArrowSparkProxy(sp)
        except Exception:
            pass
    return bundle


class Engine:
    def __init__(self):
        self.spark = None
        self.bundle = None
        self.spark_lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.status = "starting"
        self.error: str | None = None
        self.retrain = {"state": "idle", "started_at": None, "finished_at": None, "metrics": None, "error": None}
        # live cache: city -> {"city","observation","prediction","hourly","fetched_ts"}
        self.live: dict[str, dict] = {}
        self.live_meta = {"last_refresh": None, "last_error": None, "refreshing": False}
        self._refresh_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ startup
    @property
    def ready(self) -> bool:
        return self.bundle is not None

    @property
    def version(self) -> str | None:
        return getattr(self.bundle, "version", None)

    def start(self):
        threading.Thread(target=self._init_models, name="wbd-spark-init", daemon=True).start()
        threading.Thread(target=self._refresher, name="wbd-live-refresh", daemon=True).start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _train(self):
        if FAKE:
            from .fake_model import fake_train
            return fake_train(None)
        from spark_jobs.ml import train_models
        daily_sdf = None
        if os.environ.get("WBD_DB_PATH"):  # testing against a non-default DB: feed its daily table
            pdf = __import__("pandas").DataFrame(db.query("SELECT * FROM daily_weather"))
            daily_sdf = self.spark.createDataFrame(pdf)
        return _speed_up(train_models(self.spark, daily_sdf=daily_sdf))

    def _init_models(self):
        try:
            if not FAKE:
                self.status = "starting_spark"
                from spark_jobs.common import get_spark
                with self.spark_lock:
                    self.spark = get_spark()
            # wait for the ETL to have produced the daily table
            waited = 0
            while not db.table_exists("daily_weather") or not db.scalar("SELECT count(*) FROM daily_weather"):
                self.status = "waiting_for_db"
                if self._stop.wait(15):
                    return
                waited += 15
            self.status = "training"
            t0 = time.time()
            with self.spark_lock:
                bundle = self._train()
            self.bundle = bundle
            self.status = "ready"
            self.error = None
            log.info("models ready (version %s) in %.1fs", self.version, time.time() - t0)
            self._wake.set()  # trigger an immediate live refresh with predictions
        except Exception as e:
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            log.error("model init failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------ JVM recovery
    @staticmethod
    def is_jvm_dead(exc: BaseException) -> bool:
        txt = f"{type(exc).__name__}: {exc}"
        return any(s in txt for s in ("Py4JNetworkError", "ConnectionRefusedError", "ConnectionResetError",
                                      "Answer from Java side is empty", "Error while sending or receiving",
                                      "SparkContext was shut down", "Cannot call methods on a stopped SparkContext"))

    def recover(self, exc: BaseException):
        """The Java side died (e.g. java.exe killed): drop the session and re-init in background."""
        with self.state_lock:
            if self.status in ("starting_spark", "training", "recovering"):
                return
            self.status = "recovering"
        log.error("Spark JVM appears dead (%s); restarting Spark and retraining", exc)
        self.bundle = None
        try:
            from pyspark import SparkContext
            from pyspark.sql import SparkSession
            try:
                if self.spark is not None:
                    self.spark.stop()
            except Exception:
                pass
            SparkContext._active_spark_context = None
            SparkContext._gateway = None
            SparkContext._jvm = None
            SparkSession._instantiatedSession = None
            SparkSession._activeSession = None
        except Exception:
            pass
        self.spark = None
        threading.Thread(target=self._init_models, name="wbd-spark-reinit", daemon=True).start()

    # ------------------------------------------------------------------ cities
    def cities(self) -> list[dict]:
        rows: list[dict] = []
        try:
            if db.table_exists("cities"):
                rows = db.query("SELECT city, lat, lon FROM cities ORDER BY city")
        except Exception:
            rows = []
        if not rows:
            p = ROOT / "data" / "cities.json"
            if p.exists():
                try:
                    rows = [{"city": c["city"], "lat": c["lat"], "lon": c["lon"]} for c in json.loads(p.read_text("utf-8"))]
                except Exception:
                    rows = []
        return [r for r in rows if r.get("lat") is not None and r.get("lon") is not None]

    # ------------------------------------------------------------------ live
    def _predict_city(self, city: str, data: dict, pdf) -> dict | None:
        bundle = self.bundle
        if bundle is None:
            return None
        today = live.local_today(data)
        inp = live.model_input(pdf, today)
        with self.spark_lock:
            res = bundle.predict_next_day(city, inp)
        target = res.get("target_date")
        om_t, om_p = live.openmeteo_day(pdf, target) if target else (None, None)
        return db.clean({
            "target_date": target,
            "pred_temp_max": res.get("pred_temp_max"),
            "pred_rain_prob": res.get("pred_rain_prob"),
            "pred_rain": res.get("pred_rain"),
            "based_on_date": res.get("based_on_date"),
            "openmeteo_temp_max": om_t,
            "openmeteo_precip": om_p,
            "model_version": getattr(bundle, "version", None),
            "predicted_at": now_iso(),
        })

    def refresh(self, cities: list[dict] | None = None) -> dict:
        """Fetch Open-Meteo for the given (default all) cities, predict, cache and persist."""
        cities = cities if cities is not None else self.cities()
        if not cities:
            raise RuntimeError("no cities available (cities table / data/cities.json missing)")
        with self._refresh_lock:
            t_start = time.time()
            self.live_meta["refreshing"] = True
            try:
                raw = live.fetch_raw(cities)
                fetched_at = now_iso()
                obs_rows, pred_rows, errors = [], [], []
                for c in cities:
                    name = c["city"]
                    data = raw.get(name)
                    if data is None:
                        errors.append(f"{name}: no data")
                        continue
                    pdf = live.hourly_frame(name, data)
                    obs = db.clean(live.current_obs(data))
                    obs["fetched_at"] = fetched_at
                    pred = None
                    try:
                        pred = self._predict_city(name, data, pdf)
                    except Exception as e:
                        errors.append(f"{name}: predict failed: {type(e).__name__}: {str(e)[:200]}")
                        if self.is_jvm_dead(e):
                            self.recover(e)  # bundle -> None, so remaining cities skip prediction
                        else:
                            log.warning("predict failed for %s: %s", name, traceback.format_exc())
                    old = self.live.get(name) or {}
                    entry = {
                        "city": name, "observation": obs,
                        # keep the previous prediction if this round could not make one
                        "prediction": pred if pred is not None else old.get("prediction"),
                        "data": data, "hourly": pdf, "fetched_ts": time.time(),
                    }
                    self.live[name] = entry
                    obs_rows.append((name, fetched_at, obs["temperature"], obs["humidity"], obs["precipitation"],
                                     obs["wind_speed"], obs["pressure"], obs["cloud_cover"], obs["weather_code"]))
                    if pred is not None:
                        pred_rows.append((name, pred["predicted_at"], pred["target_date"], pred["pred_temp_max"],
                                          pred["pred_rain_prob"], pred["pred_rain"], pred["openmeteo_temp_max"],
                                          pred["openmeteo_precip"], pred["model_version"]))
                try:
                    db.execute_many([
                        ("INSERT INTO live_observations (city,fetched_at,temperature,humidity,precipitation,"
                         "wind_speed,pressure,cloud_cover,weather_code) VALUES (?,?,?,?,?,?,?,?,?)", obs_rows),
                        ("INSERT INTO live_predictions (city,predicted_at,target_date,pred_temp_max,pred_rain_prob,"
                         "pred_rain,openmeteo_temp_max,openmeteo_precip,model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                         pred_rows),
                    ], ddl=db.LIVE_DDL)
                except Exception as e:
                    errors.append(f"db write failed: {e}")
                self.live_meta["last_refresh"] = fetched_at
                self.live_meta["last_error"] = "; ".join(errors[:10]) or None
                return {"cities": len(obs_rows), "predictions": len(pred_rows),
                        "duration_s": round(time.time() - t_start, 1), "errors": errors}
            finally:
                self.live_meta["refreshing"] = False

    def _refresher(self):
        time.sleep(2)
        while not self._stop.is_set():
            self._wake.clear()
            try:
                r = self.refresh()
                log.info("live refresh: %s", {k: v for k, v in r.items() if k != "errors"})
            except Exception as e:
                self.live_meta["last_error"] = f"{type(e).__name__}: {e}"
                log.warning("live refresh failed: %s", e)
            # sleep until the next interval or until woken (models became ready / retrained)
            self._wake.wait(REFRESH_S if self.live else 60)

    def live_city(self, city: str) -> dict:
        entry = self.live.get(city)
        stale = (entry is None or (self.ready and entry.get("prediction") is None)
                 or time.time() - entry["fetched_ts"] > CACHE_TTL_S)
        if stale and entry is not None and self.live_meta["refreshing"]:
            return entry  # a full refresh is running; don't block on it, serve the cached data
        if stale:
            match = [c for c in self.cities() if c["city"].lower() == city.lower()]
            if not match:
                raise KeyError(city)
            self.refresh(match)
            entry = self.live.get(match[0]["city"])
        return entry

    @staticmethod
    def public(entry: dict) -> dict:
        return {"city": entry["city"], "observation": entry["observation"], "prediction": entry["prediction"]}

    # ------------------------------------------------------------------ retrain
    def start_retrain(self) -> dict:
        with self.state_lock:
            if self.retrain["state"] == "running":
                return {"status": "already_running", "started_at": self.retrain["started_at"]}
            if not FAKE and self.spark is None:
                return {"status": "unavailable", "detail": "Spark is not started yet"}
            self.retrain = {"state": "running", "started_at": now_iso(), "finished_at": None, "metrics": None, "error": None}
        threading.Thread(target=self._retrain_worker, name="wbd-retrain", daemon=True).start()
        return {"status": "started", "started_at": self.retrain["started_at"]}

    def _retrain_worker(self):
        started = self.retrain["started_at"]
        t0 = time.time()
        run_id = "retrain_api_" + datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            with self.spark_lock:
                bundle = self._train()
            self.bundle = bundle
            metrics = db.clean(list(getattr(bundle, "metrics", []) or []))
            finished = now_iso()
            dur = round(time.time() - t0, 2)
            mrows = [(m.get("model_name"), m.get("task"), m.get("target"), m.get("metric"), m.get("value"),
                      m.get("trained_at") or finished, m.get("train_rows"), m.get("test_rows")) for m in metrics]
            rows_in = max([m.get("train_rows") or 0 for m in metrics] + [0]) + max([m.get("test_rows") or 0 for m in metrics] + [0])
            fi = [(r.get("model_name"), r.get("feature"), r.get("importance"))
                  for r in db.clean(list(getattr(bundle, "feature_importance", None) or []))]
            clusters = [(int(v), k) for k, v in (getattr(bundle, "clusters", None) or {}).items()]
            extra = []
            if fi and db.table_exists("feature_importance"):
                extra.append(("DELETE FROM feature_importance", [()]))
                extra.append(("INSERT INTO feature_importance (model_name,feature,importance) VALUES (?,?,?)", fi))
            if clusters and "climate_cluster" in db.columns("cities"):
                extra.append(("UPDATE cities SET climate_cluster=? WHERE city=?", clusters))
            try:
                db.execute_many(extra + [
                    ("INSERT INTO model_metrics (model_name,task,target,metric,value,trained_at,train_rows,test_rows) "
                     "VALUES (?,?,?,?,?,?,?,?)", mrows),
                    ("INSERT INTO pipeline_runs (run_id,job,started_at,finished_at,rows_in,rows_out,duration_s,status) "
                     "VALUES (?,?,?,?,?,?,?,?)", [(run_id, "retrain_api", started, finished, rows_in, len(mrows), dur, "success")]),
                ], ddl=db.OTHER_DDL)
            except Exception as e:
                log.warning("could not persist retrain metrics: %s", e)
            self.retrain = {"state": "done", "started_at": started, "finished_at": finished, "metrics": metrics,
                            "error": None, "model_version": getattr(bundle, "version", None), "duration_s": dur}
            self._wake.set()  # refresh live predictions with the new models
        except Exception as e:
            finished = now_iso()
            log.error("retrain failed:\n%s", traceback.format_exc())
            try:
                db.execute_many([
                    ("INSERT INTO pipeline_runs (run_id,job,started_at,finished_at,rows_in,rows_out,duration_s,status) "
                     "VALUES (?,?,?,?,?,?,?,?)", [(run_id, "retrain_api", started, finished, 0, 0, round(time.time() - t0, 2), "error")]),
                ], ddl=db.OTHER_DDL)
            except Exception:
                pass
            self.retrain = {"state": "error", "started_at": started, "finished_at": finished, "metrics": None,
                            "error": f"{type(e).__name__}: {e}"}
            if self.is_jvm_dead(e):
                self.recover(e)


engine = Engine()
