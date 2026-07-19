import os
import time
import requests
import pandas as pd

# =====================================================
# Config
# =====================================================
#
# india_weather_fetched.csv already has temp / pressure / wind / rainfall
# but NOT humidity or dew_point, which train_openmeteo.py now expects.
#
# This script fetches relative_humidity_2m + dew_point_2m for every station
# from the same Open-Meteo archive, aggregates to daily means, merges them
# onto the existing rows (by station + date), and writes the file the
# trainer reads: india_weather_openmeteo.csv
#
# Humidity / dew point are independent measurements — they cannot be derived
# from the temp/pressure/wind columns already present, so they must be fetched.

INPUT_CSV = "india_weather_fetched.csv"
OUTPUT_CSV = "india_weather_openmeteo.csv"          # <-- what train_openmeteo.py loads
CACHE_CSV = "humidity_dew_cache.csv"                # per-station fetch cache (resume)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Politeness / rate-limit handling for the free API (mirrors fetch_data.py).
REQUEST_DELAY = 3        # seconds between successful requests
MAX_RETRIES = 6          # retries on HTTP 429
BASE_BACKOFF = 30        # starting backoff (seconds), doubles each retry

HOURLY_VARS = [
    "relative_humidity_2m",
    "dew_point_2m",
]

session = requests.Session()


def get_with_retry(params):
    """GET that backs off and retries on 429 Too Many Requests."""
    backoff = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(ARCHIVE_URL, params=params, timeout=120)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else backoff
            print(f"    429 rate limited; waiting {wait}s (attempt {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
            backoff = min(backoff * 2, 600)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("gave up after repeated 429s")


def fetch_station(station, start_date, end_date):
    """Fetch hourly humidity + dew point for one station, aggregate to daily means."""
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    resp = get_with_retry(params)
    data = resp.json()

    h = pd.DataFrame(data["hourly"])
    if h.empty:
        return pd.DataFrame(columns=["date_of_record", "humidity", "dew_point"])

    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date

    daily = h.groupby("date").agg(
        humidity=("relative_humidity_2m", "mean"),
        dew_point=("dew_point_2m", "mean"),
    ).reset_index()

    daily["date_of_record"] = pd.to_datetime(daily["date"])
    daily["station_name"] = station["station_name"]
    daily["humidity"] = daily["humidity"].round(1)
    daily["dew_point"] = daily["dew_point"].round(1)

    return daily[["date_of_record", "station_name", "humidity", "dew_point"]]


# =====================================================
# Load existing rows + build station catalog
# =====================================================

print(f"Loading {INPUT_CSV} ...")
base = pd.read_csv(INPUT_CSV)
base["date_of_record"] = pd.to_datetime(base["date_of_record"])
print(f"  rows: {len(base)}  stations: {base['station_name'].nunique()}")

# One coordinate + date range per station (fetch exactly the span we have data for).
catalog = (
    base.groupby("station_name")
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        start_date=("date_of_record", "min"),
        end_date=("date_of_record", "max"),
    )
    .reset_index()
)

# =====================================================
# Resume: skip stations already saved in the cache
# =====================================================

done = set()
write_header = True
if os.path.exists(CACHE_CSV):
    prev = pd.read_csv(CACHE_CSV, usecols=["station_name"])
    done = set(prev["station_name"].astype(str))
    write_header = False
    print(f"  resuming: {len(done)} stations already cached in {CACHE_CSV}")

todo = [s for s in catalog["station_name"].astype(str) if s not in done]
print(f"  stations to fetch: {len(todo)}")

# =====================================================
# Fetch each station; append to cache incrementally so a
# rate-limit interruption never loses progress.
# =====================================================

for _, station in catalog.iterrows():
    name = str(station["station_name"])
    if name in done:
        continue

    start = station["start_date"].date()
    end = station["end_date"].date()
    print(f"Fetching {name} ({start} -> {end}) ...", flush=True)
    try:
        d = fetch_station(station, str(start), str(end))
    except Exception as e:
        print(f"    ! failed for {name}: {e}  (rerun to resume)")
        break

    if len(d):
        d.to_csv(CACHE_CSV, mode="a", header=write_header, index=False)
        write_header = False
        print(f"    +{len(d)} rows cached")

    time.sleep(REQUEST_DELAY)

# =====================================================
# Merge cached humidity/dew_point onto the base rows
# =====================================================

if not os.path.exists(CACHE_CSV):
    raise SystemExit("No humidity/dew data fetched yet; nothing to merge.")

hd = pd.read_csv(CACHE_CSV)
hd["date_of_record"] = pd.to_datetime(hd["date_of_record"])
hd = hd.drop_duplicates(subset=["station_name", "date_of_record"], keep="first")

merged = base.merge(hd, on=["station_name", "date_of_record"], how="left")

matched = merged["humidity"].notna().sum()
print(f"\nMerged humidity/dew_point onto {matched}/{len(merged)} rows")

missing_stations = sorted(set(catalog["station_name"].astype(str)) - set(hd["station_name"].astype(str)))
if missing_stations:
    print(f"  still missing (rerun to fetch): {len(missing_stations)} stations")

merged.to_csv(OUTPUT_CSV, index=False)
print(f"Saved -> {OUTPUT_CSV}  ({len(merged)} rows)")
print("\nColumns:", ", ".join(merged.columns))
