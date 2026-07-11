import requests
import joblib
import pandas as pd
from datetime import datetime

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
# Fetch Weather
# =====================================================

url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&current="
    "temperature_2m,"
    "pressure_msl,"
    "wind_speed_10m"
    "&daily="
    "temperature_2m_max,"
    "temperature_2m_min"
    "&timezone=auto"
)

response = requests.get(url)
response.raise_for_status()

data = response.json()

current = data["current"]
daily = data["daily"]

avg_temp = current["temperature_2m"]
air_pressure = current["pressure_msl"]
wind_speed = current["wind_speed_10m"]

max_temp = daily["temperature_2m_max"][0]
min_temp = daily["temperature_2m_min"][0]
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
# Build Feature Vector
# =====================================================

X = pd.DataFrame([{
    "avg_temp": avg_temp,
    "min_temp": min_temp,
    "max_temp": max_temp,
    "temp_range": temp_range,
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

print("\nToday's Weather")
print("-" * 30)
print(f"Average Temp : {avg_temp:.1f} °C")
print(f"Max Temp     : {max_temp:.1f} °C")
print(f"Min Temp     : {min_temp:.1f} °C")
print(f"Wind Speed   : {wind_speed:.1f} km/h")
print(f"Pressure     : {air_pressure:.1f} hPa")

print("\nPredicted Rainfall")
print("-" * 30)
print(f"{prediction:.2f} mm")

if prediction > 0:
    print("Prediction: Rain Expected")
else:
    print("Prediction: No Rain")
    