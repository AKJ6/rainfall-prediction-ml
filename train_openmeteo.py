import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# Load Dataset (produced by fetch_data.py)
# =====================================================

df = pd.read_csv("india_weather_openmeteo.csv")
df = df.dropna()

print(f"Dataset size: {len(df)} rows")

# =====================================================
# Date Features
# =====================================================

df["date_of_record"] = pd.to_datetime(df["date_of_record"])

df["year"] = df["date_of_record"].dt.year
df["month_num"] = df["date_of_record"].dt.month
df["day"] = df["date_of_record"].dt.day
df["day_of_year"] = df["date_of_record"].dt.dayofyear

# temp_range already computed in fetch_data.py, but recompute defensively
df["temp_range"] = df["max_temp"] - df["min_temp"]

# =====================================================
# Encode Categorical Columns
# =====================================================

encoders = {}
categorical_columns = ["season", "state", "district"]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

# =====================================================
# Features  (NOTE: humidity + dew_point are the new signal)
# =====================================================

features = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "temp_range",
    "humidity",        # <-- new
    "dew_point",       # <-- new
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
    "day_of_year",
]

target = "rainfall"

X = df[features]
y = df[target]

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# Train Model (pruned -> small + generalizes)
# =====================================================

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=16,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

print("Training model...")
model.fit(X_train, y_train)

# =====================================================
# Evaluate
# =====================================================

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)

print("\n==============================")
print("Evaluation")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"Train R² : {train_r2:.4f}")
print(f"Test  R² : {test_r2:.4f}")
print(f"Gap      : {train_r2 - test_r2:.4f}  (smaller = less overfitting)")

# =====================================================
# Feature Importance
# =====================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("\n==============================")
print("Feature Importance")
print("==============================")
print(importance.to_string(index=False))

# =====================================================
# Save
# =====================================================

joblib.dump(model, "rainfall_random_forest.pkl", compress=3)
joblib.dump(encoders, "label_encoders.pkl")

print("\nModel saved as rainfall_random_forest.pkl")
print("Encoders saved as label_encoders.pkl")
