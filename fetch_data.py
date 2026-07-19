import os
import time
import requests
import pandas as pd

# =====================================================
# Config
# =====================================================

DATASET = "india_weather_rainfall_data.xlsx"   # source of stations + coordinates
STATE_MAP = "state_mapping.csv"
STATION_MAP = "station_name_mapping.csv"

OUTPUT_CSV = "india_weather_fetched.csv"

# Full date range to (re)build over.
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Politeness / rate-limit handling for the free API.
REQUEST_DELAY = 3        # seconds between successful requests
MAX_RETRIES = 6          # retries on HTTP 429
BASE_BACKOFF = 30        # starting backoff (seconds), doubles each retry

HOURLY_VARS = [
    "temperature_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
]

SCHEMA = [
    "date_of_record", "month", "season", "station_name", "state", "district",
    "avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure",
    "elevation", "latitude", "longitude", "rainfall",
]

session = requests.Session()


def indian_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


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


def fetch_station(station):
    """Fetch hourly archive data for one station, aggregate to daily schema rows."""
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    resp = get_with_retry(params)
    data = resp.json()

    h = pd.DataFrame(data["hourly"])
    if h.empty:
        return pd.DataFrame(columns=SCHEMA)

    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date

    daily = h.groupby("date").agg(
        avg_temp=("temperature_2m", "mean"),
        min_temp=("temperature_2m", "min"),
        max_temp=("temperature_2m", "max"),
        wind_speed=("wind_speed_10m", "mean"),
        air_pressure=("pressure_msl", "mean"),
        rainfall=("precipitation", "sum"),
    ).reset_index()

    daily["date_of_record"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date_of_record"].dt.strftime("%B")
    daily["season"] = daily["date_of_record"].dt.month.map(indian_season)
    daily["station_name"] = station["station_name"]
    daily["state"] = station["state"]
    daily["district"] = station["district"]
    daily["elevation"] = int(round(station["elevation"]))
    daily["latitude"] = station["latitude"]
    daily["longitude"] = station["longitude"]

    for c in ["avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure", "rainfall"]:
        daily[c] = daily[c].round(1)

    return daily[SCHEMA]


# =====================================================
# Build station catalog from the existing dataset,
# validated against the mapping files.
# =====================================================

print(f"Loading stations from {DATASET} ...")
existing = pd.read_excel(DATASET)

valid_states = set(pd.read_csv(STATE_MAP)["Original"].astype(str))
valid_stations = set(pd.read_csv(STATION_MAP)["Original"].astype(str))

catalog = (
    existing.groupby("station_name")
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        state=("state", "first"),
        district=("district", "first"),
        elevation=("elevation", "first"),
    )
    .reset_index()
)

before = len(catalog)
catalog = catalog[
    catalog["station_name"].astype(str).isin(valid_stations)
    & catalog["state"].astype(str).isin(valid_states)
]
skipped = before - len(catalog)

# Resume: skip stations already saved in a previous (interrupted) run.
done = set()
write_header = True
if os.path.exists(OUTPUT_CSV):
    prev = pd.read_csv(OUTPUT_CSV, usecols=["station_name"])
    done = set(prev["station_name"].astype(str))
    write_header = False
    print(f"  resuming: {len(done)} stations already in {OUTPUT_CSV}")

print(f"  stations found   : {before}")
if skipped:
    print(f"  skipped (not in mapping): {skipped}")
print(f"  stations to fetch: {len(catalog) - len(done & set(catalog['station_name']))}")
print(f"  date range       : {START_DATE} -> {END_DATE}")

# =====================================================
# Fetch each station; append incrementally so a
# rate-limit interruption never loses progress.
# =====================================================

for _, station in catalog.iterrows():
    name = station["station_name"]
    if str(name) in done:
        continue

    print(f"Fetching {name} ...", flush=True)
    try:
        d = fetch_station(station)
    except Exception as e:
        print(f"    ! failed for {name}: {e}  (rerun to resume)")
        break

    if len(d):
        d.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        write_header = False
        print(f"    +{len(d)} rows saved")

    time.sleep(REQUEST_DELAY)

# =====================================================
# Report
# =====================================================

if os.path.exists(OUTPUT_CSV):
    final = pd.read_csv(OUTPUT_CSV)
    print(f"\n{OUTPUT_CSV}: {len(final)} rows across {final['station_name'].nunique()} stations")
    print("Schema matches india_weather_rainfall_data.xlsx")
