import time
import shutil
import requests
import pandas as pd
from datetime import datetime, timedelta

# =====================================================
# Config
# =====================================================

DATASET = "india_weather_rainfall_data.xlsx"
BACKUP = "india_weather_rainfall_data.backup.xlsx"

STATE_MAP = "state_mapping.csv"
STATION_MAP = "station_name_mapping.csv"

# Open-Meteo archive lags real time by a few days.
ARCHIVE_LAG_DAYS = 7

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
]

# Exact column order of the existing dataset.
SCHEMA = [
    "date_of_record", "month", "season", "station_name", "state", "district",
    "avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure",
    "elevation", "latitude", "longitude", "rainfall",
]


def indian_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def fetch_station(station, start_date, end_date):
    """Fetch hourly archive data for one station and aggregate to daily schema rows."""
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
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
# Load existing dataset + mappings
# =====================================================

print(f"Loading {DATASET} ...")
existing = pd.read_excel(DATASET)
existing["date_of_record"] = pd.to_datetime(existing["date_of_record"])
print(f"  existing rows: {len(existing)}")

valid_states = set(pd.read_csv(STATE_MAP)["Original"].astype(str))
valid_stations = set(pd.read_csv(STATION_MAP)["Original"].astype(str))
print(f"  valid states in mapping   : {len(valid_states)}")
print(f"  valid stations in mapping : {len(valid_stations)}")

# =====================================================
# Build the station catalog dynamically from the data.
# (No assumption about how many stations exist.)
# Coordinates/metadata come from the dataset; labels are
# validated against the mapping files.
# =====================================================

catalog = (
    existing.groupby("station_name")
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        state=("state", "first"),
        district=("district", "first"),
        elevation=("elevation", "first"),
        last_date=("date_of_record", "max"),
    )
    .reset_index()
)

# Keep only stations/states that exist in the encoder mappings.
before = len(catalog)
catalog = catalog[
    catalog["station_name"].astype(str).isin(valid_stations)
    & catalog["state"].astype(str).isin(valid_states)
]
skipped = before - len(catalog)
print(f"  stations found in dataset : {before}")
if skipped:
    print(f"  stations skipped (not in mapping): {skipped}")
print(f"  stations to update        : {len(catalog)}")

# =====================================================
# Fetch new dates per station (extend forward to present)
# =====================================================

end_date = (datetime.now() - timedelta(days=ARCHIVE_LAG_DAYS)).date()

frames = []
for _, station in catalog.iterrows():
    start = (station["last_date"] + timedelta(days=1)).date()
    if start > end_date:
        print(f"  {station['station_name']}: already up to date, skipping")
        continue

    print(f"Fetching {station['station_name']} ({start} -> {end_date}) ...", flush=True)
    try:
        d = fetch_station(station, str(start), str(end_date))
        if len(d):
            frames.append(d)
            print(f"    +{len(d)} rows")
    except Exception as e:
        print(f"    ! failed for {station['station_name']}: {e}")

    time.sleep(1)   # be polite to the free API

if not frames:
    print("\nNothing new to add. Dataset already up to date.")
    raise SystemExit(0)

new_data = pd.concat(frames, ignore_index=True)
print(f"\nFetched {len(new_data)} new rows across {new_data['station_name'].nunique()} stations")

# =====================================================
# Back up, merge, de-duplicate, save
# =====================================================

shutil.copyfile(DATASET, BACKUP)
print(f"Backup written -> {BACKUP}")

combined = pd.concat([existing, new_data], ignore_index=True)

before = len(combined)
combined = combined.drop_duplicates(subset=["date_of_record", "station_name"], keep="first")
removed = before - len(combined)

combined = combined.sort_values(["station_name", "date_of_record"]).reset_index(drop=True)
combined.to_excel(DATASET, index=False)

print("\n==============================")
print(f"Old rows   : {len(existing)}")
print(f"Added rows : {len(new_data)}  (duplicate date/station removed: {removed})")
print(f"New total  : {len(combined)}")
print(f"Saved -> {DATASET}   (original backed up at {BACKUP})")
print("==============================")
