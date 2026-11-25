import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, ClusteringEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import SQLTransformer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from spark_jobs.common import CITY_INDEX, DB_PATH, now_iso
from spark_jobs.features import (DAILY_VARS, FEATURE_COLS, HOURLY_VARS, RAW_SCHEMA,
                                 add_features, build_training_frame, hourly_to_daily)

TEST_START = "2024-01-01"
IST = timezone(timedelta(hours=5, minutes=30))
CLUSTER_FEATURES = ["avg_temp", "temp_range", "annual_precip", "avg_humidity", "monsoon_share"]

# Single-row feature schema used for inference
_FEATURE_SCHEMA = StructType([StructField("city", StringType())]
                             + [StructField(c, DoubleType()) for c in FEATURE_COLS])

def _read_daily_from_sqlite(spark):
    """Load the daily_weather table (written by etl.py) into a Spark DataFrame."""
    con = sqlite3.connect(DB_PATH)
    pdf = pd.read_sql_query("SELECT * FROM daily_weather", con)
    con.close()
    if pdf.empty:
        raise RuntimeError("daily_weather is empty - run spark_jobs/etl.py first")
    for c in DAILY_VARS:
        pdf[c] = pdf[c].astype("float64")
    return spark.createDataFrame(pdf[["city", "date"] + DAILY_VARS])


def _regression_metrics(pred_sdf, label, pred_col):
    out = {}
    for m in ("rmse", "mae", "r2"):
        ev = RegressionEvaluator(labelCol=label, predictionCol=pred_col, metricName=m)
        out[m] = float(ev.evaluate(pred_sdf))
    return out


def _classification_metrics(pred_sdf, label, raw_col, pred_col):
    auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol=raw_col,
                                        metricName="areaUnderROC").evaluate(pred_sdf)
    row = pred_sdf.agg(
        F.sum(((F.col(pred_col) == 1) & (F.col(label) == 1)).cast("int")).alias("tp"),
        F.sum(((F.col(pred_col) == 1) & (F.col(label) == 0)).cast("int")).alias("fp"),
        F.sum(((F.col(pred_col) == 0) & (F.col(label) == 1)).cast("int")).alias("fn"),
        F.sum(((F.col(pred_col) == 0) & (F.col(label) == 0)).cast("int")).alias("tn"),
    ).first()
    tp, fp, fn, tn = (row["tp"] or 0), (row["fp"] or 0), (row["fn"] or 0), (row["tn"] or 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"auc": float(auc), "accuracy": accuracy, "f1": f1,
            "precision": precision, "recall": recall}


def _city_climatology(daily_sdf):
    """Per-city climate descriptors used for KMeans clustering."""
    d = daily_sdf.withColumn("date", F.to_date("date")) \
                 .withColumn("year", F.year("date")).withColumn("month", F.month("date"))
    monthly = d.groupBy("city", "year", "month").agg(
        F.avg("temp_mean").alias("t"), F.sum("precip_sum").alias("p"))
    clim_month = monthly.groupBy("city", "month").agg(F.avg("t").alias("t"), F.avg("p").alias("p"))
    per_city = clim_month.groupBy("city").agg(
        (F.max("t") - F.min("t")).alias("temp_range"),          # seasonal amplitude
        F.sum("p").alias("annual_precip"),                       # climatological annual total
        (F.sum(F.when(F.col("month").between(6, 9), F.col("p")).otherwise(0.0))
         / F.sum("p") * 100).alias("monsoon_share"),
    )
    means = d.groupBy("city").agg(F.avg("temp_mean").alias("avg_temp"),
                                  F.avg("humidity_mean").alias("avg_humidity"))
    return means.join(per_city, "city")


class ModelBundle:
    """Holds the fitted Spark models + metadata, and serves predictions."""

    def __init__(self, spark, version, models, inference_model, metrics,
                 feature_importance, clusters, latest_features, test_pred_sdf,
                 train_rows, test_rows):
        self.spark = spark
        self.version = version                  
        self.models = models                      
        self._inference = inference_model        
        self.metrics = metrics                    
        self.feature_importance = feature_importance   
        self.clusters = clusters                   
        self.latest_features = latest_features     
        self._test_pred_sdf = test_pred_sdf
        self.train_rows = train_rows
        self.test_rows = test_rows

    def _to_spark(self, pdf, schema):
        """pandas -> Spark via Apache Arrow, in ONE partition.

        Arrow conversion happens in the JVM, so no Python worker processes are
        started (spawning them on Windows costs ~1 s each and made a single
        prediction take ~20 s).  One partition = minimal task overhead.
        """
        return self.spark.createDataFrame(pdf, schema=schema).coalesce(1)

    def _predict_rows(self, rows):
        """rows: list of (city, {feature: value}) -> list of prediction dicts."""
        pdf = pd.DataFrame([{"city": city, **{c: float(f[c]) for c in FEATURE_COLS}}
                            for city, f in rows])[_FEATURE_SCHEMA.fieldNames()]
        sdf = self._to_spark(pdf, _FEATURE_SCHEMA)
        out = self._inference.transform(sdf).select(
            "pred_gbt", "pred_lr", vector_to_array("prob_rf")[1].alias("p_rain")).collect()
        res = []
        for r in out:
            p = float(r["p_rain"])
            res.append({"pred_temp_max": round(float(r["pred_gbt"]), 2),
                        "pred_temp_max_lr": round(float(r["pred_lr"]), 2),
                        "pred_rain_prob": round(p, 4),
                        "pred_rain": int(p >= 0.5)})
        return res

    def _fill_missing(self, city, feats):
        """Fill missing/NaN features from the city's latest historical daily row."""
        base = self.latest_features.get(city, {})
        out = {}
        for c in FEATURE_COLS:
            v = feats.get(c)
            if v is None or (isinstance(v, float) and v != v):
                v = base.get(c, 0.0)
            out[c] = float(v)
        # location / id features are always the fixed values for the city
        out["city_idx"] = float(CITY_INDEX[city])
        return out

    def predict_next_day(self, city: str, hourly_pdf: pd.DataFrame) -> dict:
        if city not in CITY_INDEX:
            raise KeyError(f"Unknown city: {city}")
        pdf = hourly_pdf.copy()
        pdf["time"] = pd.to_datetime(pdf["time"]).dt.strftime("%Y-%m-%dT%H:%M")
        pdf["city"] = city
        for c in HOURLY_VARS:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float64") \
                if c in pdf.columns else float("nan")
        pdf = pdf[RAW_SCHEMA.fieldNames()]

        today_ist = datetime.now(IST).date().isoformat()
        hourly_sdf = self._to_spark(pdf, RAW_SCHEMA)       # NaN -> null via Arrow
        daily = hourly_to_daily(hourly_sdf) \
            .filter((F.col("n_hours") >= 24) & (F.col("date") < F.lit(today_ist).cast("date")))
        feat = add_features(daily).orderBy(F.col("date").desc()).limit(1).collect()
        if not feat:
            raise ValueError("hourly_pdf contains no complete day")
        row = feat[0].asDict()
        feats = self._fill_missing(city, row)
        pred = self._predict_rows([(city, feats)])[0]
        return {
            "target_date": (row["date"] + timedelta(days=1)).isoformat(),
            "pred_temp_max": pred["pred_temp_max"],
            "pred_rain_prob": pred["pred_rain_prob"],
            "pred_rain": pred["pred_rain"],
            "features": {k: round(v, 4) for k, v in feats.items()},
            "based_on_date": row["date"].isoformat(),
            "model_version": self.version,
        }

    def predict_from_features(self, city: str, feats: dict) -> dict:
        """What-if prediction: override any features; rest from the city's latest day."""
        if city not in CITY_INDEX:
            raise KeyError(f"Unknown city: {city}")
        base = dict(self.latest_features.get(city, {}))
        overrides = {k: v for k, v in (feats or {}).items()
                     if k in FEATURE_COLS and v is not None}
        merged = {**base, **overrides}
        # keep the pressure tendency consistent if the user changed today's pressure
        if "pressure_mean" in overrides and "pressure_delta1" not in overrides and base:
            prev_pressure = base["pressure_mean"] - base["pressure_delta1"]
            merged["pressure_delta1"] = float(overrides["pressure_mean"]) - prev_pressure
        pred = self._predict_rows([(city, self._fill_missing(city, merged))])[0]
        return {"pred_temp_max": pred["pred_temp_max"],
                "pred_rain_prob": pred["pred_rain_prob"],
                "pred_rain": pred["pred_rain"],
                "model_version": self.version}

    def test_predictions_pdf(self) -> pd.DataFrame:
        """Long-format test-set predictions: model_name, city, date, actual, predicted."""
        s = self._test_pred_sdf
        date_s = F.date_format("date", "yyyy-MM-dd")
        parts = [
            s.select(F.lit("gbt_temp_max").alias("model_name"), "city", date_s.alias("date"),
                     F.col("label_temp_max").alias("actual"), F.col("pred_gbt").alias("predicted")),
            s.select(F.lit("lr_temp_max").alias("model_name"), "city", date_s.alias("date"),
                     F.col("label_temp_max").alias("actual"), F.col("pred_lr").alias("predicted")),
            s.select(F.lit("rf_rain").alias("model_name"), "city", date_s.alias("date"),
                     F.col("label_rain").alias("actual"),
                     vector_to_array("prob_rf")[1].alias("predicted")),
        ]
        out = parts[0].unionByName(parts[1]).unionByName(parts[2])
        return out.toPandas()

def train_models(spark, daily_sdf=None) -> ModelBundle:
    """Train all models on the daily table and return a ModelBundle."""
    t0 = time.time()
    trained_at = now_iso()
    if daily_sdf is None:
        daily_sdf = _read_daily_from_sqlite(spark)
    daily_sdf = daily_sdf.select("city", F.to_date("date").alias("date"), *DAILY_VARS).cache()

    # label_delta = tomorrow's max - today's max (used by the GBT, see below)
    data = build_training_frame(daily_sdf)         .withColumn("label_delta", F.col("label_temp_max") - F.col("temp_max")).cache()
    train = data.filter(F.col("date") < F.lit(TEST_START).cast("date")).cache()
    test = data.filter(F.col("date") >= F.lit(TEST_START).cast("date")).cache()
    n_train, n_test = train.count(), test.count()
    print(f"  training rows={n_train:,}  test rows={n_test:,}", flush=True)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

    gbt = GBTRegressor(featuresCol="features", labelCol="label_delta",
                       predictionCol="pred_gbt_delta", maxIter=50, maxDepth=5,
                       stepSize=0.1, subsamplingRate=0.8, seed=42)
    add_back = SQLTransformer(
        statement="SELECT *, temp_max + pred_gbt_delta AS pred_gbt FROM __THIS__")
    gbt_model = Pipeline(stages=[assembler, gbt, add_back]).fit(train)
    print(f"  gbt_temp_max trained ({time.time() - t0:.0f}s)", flush=True)

    lr = LinearRegression(featuresCol="features", labelCol="label_temp_max",
                          predictionCol="pred_lr", regParam=0.01, elasticNetParam=0.0)
    lr_model = Pipeline(stages=[assembler, lr]).fit(train)
    print(f"  lr_temp_max trained ({time.time() - t0:.0f}s)", flush=True)

    pos = train.filter(F.col("label_rain") == 1).count()
    neg = n_train - pos
    w_pos, w_neg = n_train / (2.0 * max(pos, 1)), n_train / (2.0 * max(neg, 1))
    train_w = train.withColumn(
        "class_weight", F.when(F.col("label_rain") == 1, w_pos).otherwise(w_neg))
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_rain",
                                weightCol="class_weight", predictionCol="pred_rf",
                                probabilityCol="prob_rf", rawPredictionCol="raw_rf",
                                numTrees=60, maxDepth=10, seed=42)
    rf_model = Pipeline(stages=[assembler, rf]).fit(train_w)
    print(f"  rf_rain trained ({time.time() - t0:.0f}s), rain share={pos / n_train:.2%}",
          flush=True)

    inference = PipelineModel(stages=[assembler, gbt_model.stages[1], gbt_model.stages[2],
                                      lr_model.stages[-1], rf_model.stages[-1]])
    test_pred = inference.transform(test).cache()

    metrics = []

    def add(model_name, task, target, vals, tr=n_train, te=n_test):
        for k, v in vals.items():
            metrics.append({"model_name": model_name, "task": task, "target": target,
                            "metric": k, "value": round(float(v), 4), "trained_at": trained_at,
                            "train_rows": int(tr), "test_rows": int(te)})

    add("gbt_temp_max", "regression", "next_day_temp_max",
        _regression_metrics(test_pred, "label_temp_max", "pred_gbt"))
    add("lr_temp_max", "regression", "next_day_temp_max",
        _regression_metrics(test_pred, "label_temp_max", "pred_lr"))
    add("rf_rain", "classification", "next_day_rain",
        _classification_metrics(test_pred, "label_rain", "raw_rf", "pred_rf"))

    importance = []
    for name, est in (("gbt_temp_max", gbt_model.stages[1]), ("rf_rain", rf_model.stages[-1])):
        imp = est.featureImportances.toArray()
        importance += [{"model_name": name, "feature": f, "importance": round(float(v), 5)}
                       for f, v in zip(FEATURE_COLS, imp)]

    stds = train.select([F.stddev(c).alias(c) for c in FEATURE_COLS]).first().asDict()
    coefs = lr_model.stages[-1].coefficients.toArray()
    eff = [abs(c) * (stds[f] or 0.0) for f, c in zip(FEATURE_COLS, coefs)]
    tot = sum(eff) or 1.0
    importance += [{"model_name": "lr_temp_max", "feature": f, "importance": round(e / tot, 5)}
                   for f, e in zip(FEATURE_COLS, eff)]

    clim = _city_climatology(daily_sdf).cache()
    km_pipe = Pipeline(stages=[
        VectorAssembler(inputCols=CLUSTER_FEATURES, outputCol="clim_raw"),
        StandardScaler(inputCol="clim_raw", outputCol="clim_features",
                       withMean=True, withStd=True),
        KMeans(featuresCol="clim_features", predictionCol="cluster", k=4, seed=7, maxIter=50),
    ])
    km_model = km_pipe.fit(clim)
    clustered = km_model.transform(clim)
    silhouette = ClusteringEvaluator(featuresCol="clim_features", predictionCol="cluster",
                                     metricName="silhouette").evaluate(clustered)
    clusters = {r["city"]: int(r["cluster"]) for r in clustered.select("city", "cluster").collect()}
    n_cities = len(clusters)
    add("kmeans_climate", "clustering", "climate_zone", {"silhouette": silhouette},
        tr=n_cities, te=0)
    print(f"  kmeans_climate trained ({time.time() - t0:.0f}s)", flush=True)


    feats_all = add_features(daily_sdf)
    w_last = feats_all.groupBy("city").agg(F.max("date").alias("date"))
    latest_pdf = feats_all.join(w_last, ["city", "date"]).toPandas()
    latest = {r["city"]: {c: float(r[c]) for c in FEATURE_COLS if pd.notna(r[c])}
              for _, r in latest_pdf.iterrows()}

    bundle = ModelBundle(
        spark=spark, version=trained_at,
        models={"gbt_temp_max": gbt_model, "lr_temp_max": lr_model,
                "rf_rain": rf_model, "kmeans_climate": km_model},
        inference_model=inference, metrics=metrics, feature_importance=importance,
        clusters=clusters, latest_features=latest, test_pred_sdf=test_pred,
        train_rows=n_train, test_rows=n_test)
    bundle.climatology_pdf = clustered.select("city", "cluster", *CLUSTER_FEATURES).toPandas()
    bundle.training_seconds = round(time.time() - t0, 1)
    print(f"  all models trained in {bundle.training_seconds}s", flush=True)
    return bundle
