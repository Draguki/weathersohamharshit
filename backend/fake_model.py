"""Test-only stand-in for spark_jobs.ml.ModelBundle (enabled with WBD_FAKE_MODEL=1).
Never used unless that env var is set."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd


class FakeBundle:
    def __init__(self):
        self.version = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "-fake"
        now = self.version[:19]
        self.metrics = [
            {"model_name": "gbt_temp_max", "task": "regression", "target": "temp_max_next", "metric": m,
             "value": v, "trained_at": now, "train_rows": 1000, "test_rows": 200}
            for m, v in [("rmse", 1.3), ("mae", 0.9), ("r2", 0.93)]
        ] + [
            {"model_name": "rf_rain", "task": "classification", "target": "rain_next", "metric": m,
             "value": v, "trained_at": now, "train_rows": 1000, "test_rows": 200}
            for m, v in [("auc", 0.88), ("accuracy", 0.84), ("f1", 0.7)]
        ]

    def predict_next_day(self, city: str, hourly_pdf: pd.DataFrame) -> dict:
        d = hourly_pdf.copy()
        d["date"] = d["time"].str[:10]
        counts = d.groupby("date")["time"].count()
        last = counts[counts >= 24].index.max()
        day = d[d["date"] == last]
        tmax = float(day["temperature_2m"].max())
        hum = float(day["relative_humidity_2m"].mean())
        target = (datetime.fromisoformat(last) + timedelta(days=1)).date().isoformat()
        p = min(max((hum - 40) / 60, 0.0), 1.0)
        return {"target_date": target, "pred_temp_max": round(tmax + 0.3, 2), "pred_rain_prob": round(p, 3),
                "pred_rain": int(p >= 0.5), "features": {"temp_max": tmax, "humidity_mean": hum}}

    def predict_from_features(self, city: str, feats: dict) -> dict:
        t = float(feats.get("temp_max", 30.0))
        h = float(feats.get("humidity_mean", 60.0))
        p = min(max((h - 40) / 60, 0.0), 1.0)
        return {"pred_temp_max": round(t + 0.2, 2), "pred_rain_prob": round(p, 3), "pred_rain": int(p >= 0.5)}


def fake_train(_spark=None, daily_sdf=None) -> FakeBundle:
    time.sleep(2)
    return FakeBundle()
