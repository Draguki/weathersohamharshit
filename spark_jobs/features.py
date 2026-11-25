import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from spark_jobs.common import CITIES


HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "pressure_msl", "cloud_cover", "wind_speed_10m", "weather_code",
]
RAW_COLUMNS = ["city", "time"] + HOURLY_VARS


RAW_SCHEMA = StructType(
    [StructField("city", StringType()), StructField("time", StringType())]
    + [StructField(c, DoubleType()) for c in HOURLY_VARS]
)


DAILY_VARS = [
    "temp_max", "temp_min", "temp_mean", "humidity_mean", "dew_point_mean",
    "precip_sum", "pressure_mean", "cloud_mean", "wind_max", "wind_mean",
]


FEATURE_COLS = DAILY_VARS + [
    "temp_max_lag1", "temp_max_lag2", "temp_max_lag3", "precip_lag1",
    "temp_max_roll7", "precip_roll7", "pressure_delta1",
    "doy_sin", "doy_cos", "month", "lat", "lon", "city_idx",
]

RAIN_THRESHOLD_MM = 1.0      


def hourly_to_daily(hourly_sdf: DataFrame) -> DataFrame:
    date_col = F.to_date(F.substring(F.col("time").cast("string"), 1, 10))
    return (
        hourly_sdf
        .withColumn("date", date_col)
        .groupBy("city", "date")
        .agg(
            F.max("temperature_2m").alias("temp_max"),
            F.min("temperature_2m").alias("temp_min"),
            F.avg("temperature_2m").alias("temp_mean"),
            F.avg("relative_humidity_2m").alias("humidity_mean"),
            F.avg("dew_point_2m").alias("dew_point_mean"),
            F.sum("precipitation").alias("precip_sum"),
            F.avg("pressure_msl").alias("pressure_mean"),
            F.avg("cloud_cover").alias("cloud_mean"),
            F.max("wind_speed_10m").alias("wind_max"),
            F.avg("wind_speed_10m").alias("wind_mean"),
            F.count("temperature_2m").alias("n_hours"),
        )
    )


def _city_lookup_expr(field):
    """Literal Spark map {city -> value} built from cities.json (no join needed)."""
    pairs = []
    for i, c in enumerate(CITIES):
        val = float(i) if field == "city_idx" else float(c[field])
        pairs += [F.lit(c["city"]), F.lit(val)]
    return F.create_map(*pairs)


def add_features(daily_sdf: DataFrame) -> DataFrame:
    df = daily_sdf.withColumn("date", F.to_date("date"))
    w = Window.partitionBy("city").orderBy("date")
    w7 = w.rowsBetween(-6, 0)                         # last 7 days incl. today
    doy = F.dayofyear("date").cast("double")
    return (
        df
        .withColumn("temp_max_lag1", F.lag("temp_max", 1).over(w))
        .withColumn("temp_max_lag2", F.lag("temp_max", 2).over(w))
        .withColumn("temp_max_lag3", F.lag("temp_max", 3).over(w))
        .withColumn("precip_lag1", F.lag("precip_sum", 1).over(w))
        .withColumn("temp_max_roll7", F.avg("temp_max").over(w7))
        .withColumn("precip_roll7", F.sum("precip_sum").over(w7))
        .withColumn("pressure_delta1", F.col("pressure_mean") - F.lag("pressure_mean", 1).over(w))
        .withColumn("doy_sin", F.sin(2 * math.pi * doy / 365.25))
        .withColumn("doy_cos", F.cos(2 * math.pi * doy / 365.25))
        .withColumn("month", F.month("date").cast("double"))
        .withColumn("lat", _city_lookup_expr("lat")[F.col("city")])
        .withColumn("lon", _city_lookup_expr("lon")[F.col("city")])
        .withColumn("city_idx", _city_lookup_expr("city_idx")[F.col("city")])
    )


def add_labels(feat_sdf: DataFrame) -> DataFrame:
    """Add next-day labels. Rows whose next row is not exactly the next calendar
    day (gap in data) or the last day of a city get null labels."""
    w = Window.partitionBy("city").orderBy("date")
    next_date = F.lead("date", 1).over(w)
    next_tmax = F.lead("temp_max", 1).over(w)
    next_prcp = F.lead("precip_sum", 1).over(w)
    consecutive = F.datediff(next_date, F.col("date")) == 1
    return (
        feat_sdf
        .withColumn("target_date", F.date_add("date", 1))
        .withColumn("label_temp_max", F.when(consecutive, next_tmax))
        .withColumn("label_rain",
                    F.when(consecutive, (next_prcp >= RAIN_THRESHOLD_MM).cast("double")))
    )


def build_training_frame(daily_sdf: DataFrame) -> DataFrame:
    """daily -> features + labels, dropping rows with any missing feature/label."""
    return add_labels(add_features(daily_sdf)).dropna(
        subset=FEATURE_COLS + ["label_temp_max", "label_rain"]
    )
