import os
import requests
import joblib
import pandas as pd
from datetime import datetime
import dotenv
dotenv.load_dotenv()    
# =====================================================
# Google Weather variant of pred_rain.py
# =====================================================
#
# Same model + feature contract as pred_rain.py, but pulls live conditions
# from the Google Maps Platform Weather API instead of Open-Meteo.
#
# Two endpoints are used:
#   - currentConditions:lookup  -> temp, humidity, dew point, pressure, wind
#   - forecast/days:lookup      -> today's max/min temp (for temp_range)
#
# Requires a Google Maps Platform API key with the Weather API enabled.
# Set it in the environment (do NOT hardcode secrets):
#
#     export GOOGLE_MAPS_API_KEY="your-key-here"
#
# NOTE: Google's Weather API is a live/short-forecast service — it does NOT
# provide the multi-year hourly archive the training scripts need, so the
# dataset build (add_humidity_dew.py / fetch_data.py) still uses Open-Meteo.

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise SystemExit(
        "GOOGLE_MAPS_API_KEY is not set.\n"
        "  export GOOGLE_MAPS_API_KEY='your-key-here'"
    )

CURRENT_URL = "https://weather.googleapis.com/v1/currentConditions:lookup"
FORECAST_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"

# =====================================================
# Load Model
# =====================================================

model = joblib.load("rainfall_random_forest.pkl")
encoders = joblib.load("label_encoders.pkl")

# =====================================================
# Bangalore Information
# =====================================================

LATITUDE = 12.9716
LONGITUDE = 77.5946
ELEVATION = 920

STATE = "KA"
DISTRICT = "Bengaluru Rural"
SEASON = "Monsoon"

# =====================================================
# Fetch Weather (Google Maps Platform Weather API)
# =====================================================

common = {
    "key": API_KEY,
    "location.latitude": LATITUDE,
    "location.longitude": LONGITUDE,
    "unitsSystem": "METRIC",
}

# Current conditions
cur_resp = requests.get(CURRENT_URL, params=common, timeout=60)
cur_resp.raise_for_status()
current = cur_resp.json()

# Today's forecast (for max/min temp)
fc_resp = requests.get(FORECAST_URL, params={**common, "days": 1}, timeout=60)
fc_resp.raise_for_status()
forecast = fc_resp.json()

# --- current conditions ---
avg_temp = current["temperature"]["degrees"]
humidity = current["relativeHumidity"]
dew_point = current["dewPoint"]["degrees"]
air_pressure = current["airPressure"]["meanSeaLevelMillibars"]
wind_speed = current["wind"]["speed"]["value"]

# --- today's max / min from the daily forecast ---
today_fc = forecast["forecastDays"][0]
max_temp = today_fc["maxTemperature"]["degrees"]
min_temp = today_fc["minTemperature"]["degrees"]
temp_range = max_temp - min_temp

# =====================================================
# Date Features
# =====================================================

today = datetime.now()

year = today.year
month_num = today.month
day_of_year = today.timetuple().tm_yday

# =====================================================
# Encode categorical values
# =====================================================

season = encoders["season"].transform([SEASON])[0]
state = encoders["state"].transform([STATE])[0]
district = encoders["district"].transform([DISTRICT])[0]

# =====================================================
# Build Feature Vector  (same order as train_openmeteo.py)
# =====================================================

X = pd.DataFrame([{
    "avg_temp": avg_temp,
    "min_temp": min_temp,
    "max_temp": max_temp,
    "temp_range": temp_range,
    "humidity": humidity,
    "dew_point": dew_point,
    "wind_speed": wind_speed,
    "air_pressure": air_pressure,
    "elevation": ELEVATION,
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "season": season,
    "state": state,
    "district": district,
    "year": year,
    "month_num": month_num,
    "day_of_year": day_of_year
}])

# =====================================================
# Predict
# =====================================================

prediction = model.predict(X)[0]

print("\nToday's Weather (Google Weather)")
print("-" * 30)
print(f"Average Temp : {avg_temp:.1f} °C")
print(f"Max Temp     : {max_temp:.1f} °C")
print(f"Min Temp     : {min_temp:.1f} °C")
print(f"Humidity     : {humidity:.1f} %")
print(f"Dew Point    : {dew_point:.1f} °C")
print(f"Wind Speed   : {wind_speed:.1f} km/h")
print(f"Pressure     : {air_pressure:.1f} hPa")

print("\nPredicted Rainfall")
print("-" * 30)
print(f"{prediction:.2f} mm")

if prediction > 0:
    print("Prediction: Rain Expected")
else:
    print("Prediction: No Rain")
