import joblib
import pandas as pd

# Load trained model
model = joblib.load("rainfall_random_forest.pkl")

# Feature names used during training (same order as train_openmeteo.py)
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

# Get feature importances
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))