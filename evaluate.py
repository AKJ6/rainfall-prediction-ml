import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# Load Saved Model + Encoders
# =====================================================

model = joblib.load("rainfall_random_forest.pkl")
encoders = joblib.load("label_encoders.pkl")

# =====================================================
# Load Dataset (same steps as train_openmeteo.py)
# =====================================================

df = pd.read_csv("india_weather_openmeteo.csv")
df = df.dropna()

# =====================================================
# Date Features
# =====================================================

df["date_of_record"] = pd.to_datetime(df["date_of_record"])

df["year"] = df["date_of_record"].dt.year
df["month_num"] = df["date_of_record"].dt.month
df["day"] = df["date_of_record"].dt.day
df["day_of_year"] = df["date_of_record"].dt.dayofyear

# =====================================================
# Temperature Feature
# =====================================================

df["temp_range"] = df["max_temp"] - df["min_temp"]

# =====================================================
# Encode Categorical Columns (reuse saved encoders)
# =====================================================

categorical_columns = ["season", "state", "district"]

for col in categorical_columns:
    df[col] = encoders[col].transform(df[col].astype(str))

# =====================================================
# Features / Target
# =====================================================

features = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "temp_range",
    "humidity",
    "dew_point",
    "wind_speed",
    "air_pressure",
    "elevation",
    "latitude",
    "longitude",
    "season",
    "state",
    "district",
    "year",
    "month_num",
    "day_of_year"
]

target = "rainfall"

X = df[features]
y = df[target]

# =====================================================
# Same Train/Test Split as train.py
# (random_state=42 -> identical rows in each split)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Predict
# =====================================================

test_predictions = model.predict(X_test)
train_predictions = model.predict(X_train)

# =====================================================
# R Squared + Metrics
# =====================================================

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("\n==============================")
print("R Squared (R²)")
print("==============================")
print(f"Train R² : {train_r2:.4f}")
print(f"Test  R² : {test_r2:.4f}")
print(f"Gap      : {train_r2 - test_r2:.4f}  (smaller = less overfitting)")

print("\n==============================")
print("Other Test Metrics")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
