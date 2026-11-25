"""
fetch_data.py - Download raw hourly weather data for every city in data/cities.json.

Source : Open-Meteo Historical Weather API (ERA5 reanalysis), no API key needed
         https://archive-api.open-meteo.com/v1/archive
Output : data/raw/hourly_<slug>.csv  (one file per city, columns per docs/CONTRACT.md)

Design notes
------------
* The full range (2016-01-01 -> 2025-10-31, ~86k hours per city) is fetched in
  chunks of ~3 years.  Each chunk response is cached in data/cache/ so the script
  is fully RESUMABLE - re-running it never downloads a chunk twice.
* Chunks are fetched "recent first": the 2019-2025 chunks for ALL cities are
  fetched before the 2016-2018 back-fill.  If the free-tier quota runs out we
  still have every region covered for at least 2019-2025.
* Open-Meteo rate limits by *weighted* calls (long date ranges count as many
  calls).  HTTP 429 responses are handled: minutely limit -> wait ~65 s and retry;
  hourly/daily limit -> stop fetching (cannot wait that long) and assemble what we have.
* Final assembly uses a COMMON start date for all cities, so every city covers
  the same years (fair year-by-year comparisons).

Usage:  python spark_jobs/fetch_data.py [--deadline-min 25] [--assemble-only]
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CITIES_JSON = os.path.join(DATA_DIR, "cities.json")

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "pressure_msl", "cloud_cover", "wind_speed_10m", "weather_code",
]
RAW_COLUMNS = ["city", "time"] + HOURLY_VARS

# Chunks in priority order: recent years first, back-fill last.
CHUNKS = [
    ("2022-01-01", "2025-10-31"),
    ("2019-01-01", "2021-12-31"),
    ("2016-01-01", "2018-12-31"),
]


class QuotaExhausted(Exception):
    """Raised when the hourly/daily quota is hit (no point waiting)."""


def slugify(name: str) -> str:
    """'Port Blair' -> 'port_blair' (used in file names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_cities():
    with open(CITIES_JSON, encoding="utf-8") as f:
        return json.load(f)


def cache_path(slug, start, end):
    return os.path.join(CACHE_DIR, f"{slug}_{start}_{end}.csv")


def fetch_chunk(city, start, end, max_retries=8):
    """Fetch one date-range chunk for one city; returns a pandas DataFrame."""
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=120)
        except requests.RequestException as e:
            wait = 5 * attempt
            print(f"    network error ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
            continue

        if r.status_code == 200:
            hourly = r.json()["hourly"]
            df = pd.DataFrame(hourly)
            df.insert(0, "city", city["city"])
            return df[RAW_COLUMNS]

        # --- error handling -------------------------------------------------
        try:
            reason = r.json().get("reason", r.text)
        except ValueError:
            reason = r.text
        print(f"    HTTP {r.status_code}: {reason}", flush=True)
        if r.status_code == 429:
            low = reason.lower()
            if "minutely" in low:
                time.sleep(65)
                continue
            if "hourly" in low or "daily" in low:
                raise QuotaExhausted(reason)
            time.sleep(30)
            continue
        if 500 <= r.status_code < 600:
            time.sleep(10 * attempt)
            continue
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {reason}")
    raise RuntimeError(f"Giving up on {city['city']} {start}..{end}")


def download_all(cities, deadline_ts):
    """Download every (chunk, city) pair not yet in the cache, in priority order."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for start, end in CHUNKS:
        for city in cities:
            slug = slugify(city["city"])
            path = cache_path(slug, start, end)
            if os.path.exists(path):
                continue
            if time.time() > deadline_ts:
                print("Deadline reached - stopping downloads.", flush=True)
                return
            t0 = time.time()
            print(f"  fetching {city['city']:<20} {start} -> {end}", flush=True)
            try:
                df = fetch_chunk(city, start, end)
            except QuotaExhausted as e:
                print(f"Quota exhausted ({e}) - stopping downloads.", flush=True)
                return
            tmp = path + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)          # atomic: no half-written cache files
            print(f"    ok: {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)


def assemble(cities):
    """Combine cached chunks into data/raw/hourly_<slug>.csv with a common date range."""
    os.makedirs(RAW_DIR, exist_ok=True)
    # A chunk is used only if EVERY city has it -> all cities share the same years.
    usable = [c for c in CHUNKS
              if all(os.path.exists(cache_path(slugify(ct["city"]), *c)) for ct in cities)]
    if not usable:
        print("No chunk is complete for all cities yet - nothing assembled.")
        return 0
    usable.sort()
    print("Chunks used for all cities:", usable)
    total = 0
    for city in cities:
        slug = slugify(city["city"])
        parts = [pd.read_csv(cache_path(slug, *c)) for c in usable]
        df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
        df = df[RAW_COLUMNS]
        df.to_csv(os.path.join(RAW_DIR, f"hourly_{slug}.csv"), index=False)
        total += len(df)
    print(f"Assembled {len(cities)} files, {total:,} hourly rows total, "
          f"range {usable[0][0]} -> {usable[-1][1]}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-min", type=float, default=25.0,
                    help="stop downloading after this many minutes")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    cities = load_cities()
    print(f"{len(cities)} cities, {len(CHUNKS)} chunks each", flush=True)
    if not args.assemble_only:
        download_all(cities, time.time() + args.deadline_min * 60)
    assemble(cities)


if __name__ == "__main__":
    main()
