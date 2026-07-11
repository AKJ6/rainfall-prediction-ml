import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_excel("india_weather_rainfall_data.xlsx")

# =====================================================
# Remove rows containing NaN
# (Keeps rainfall = 0)
# =====================================================

df = df.dropna()

print(f"Dataset size after removing NaNs: {len(df)} rows")

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
# Encode Categorical Columns
# =====================================================

encoders = {}

categorical_columns = [
    "season",
    "state",
    "district"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

# =====================================================
# Features
# =====================================================

features = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "temp_range",
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
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Train Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# =====================================================
# Predict
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# Metrics
# =====================================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n==============================")
print("Evaluation")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# Feature Importance
# =====================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n==============================")
print("Feature Importance")
print("==============================")
print(importance)

# =====================================================
# Save Model
# =====================================================

joblib.dump(model, "rainfall_random_forest.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("\nModel saved as rainfall_random_forest.pkl")
print("Encoders saved as label_encoders.pkl")