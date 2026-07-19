import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =====================================================
# Binary Rain / No-Rain Evaluation
# =====================================================
#
# The model is a regressor that predicts rainfall in mm.  This script does
# NOT judge how much rain — it collapses both the actual and predicted
# rainfall to a yes/no label:
#
#     rain    -> rainfall > 0   (1)
#     no rain -> rainfall == 0  (0)
#
# and reports R² on those 0/1 labels.  Same dataset, feature contract, and
# train/test split as train_openmeteo.py so the test rows line up exactly.

# threshold (mm) above which we call it "rain".
# 1.0 mm is the standard meteorological "rain day" cutoff (below = trace/drizzle).
RAIN_THRESHOLD = 1.0

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

df["temp_range"] = df["max_temp"] - df["min_temp"]

# =====================================================
# Encode Categorical Columns (reuse saved encoders)
# =====================================================

categorical_columns = ["season", "state", "district"]

for col in categorical_columns:
    df[col] = encoders[col].transform(df[col].astype(str))

# =====================================================
# Features / Target  (same order as train_openmeteo.py)
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
    "day_of_year",
]

target = "rainfall"

X = df[features]
y = df[target]

# =====================================================
# Same Train/Test Split as train_openmeteo.py
# (random_state=42 -> identical rows in each split)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# Predict (continuous mm) then collapse to rain / no-rain
# =====================================================

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

y_test_bin = (y_test.values > RAIN_THRESHOLD).astype(int)
y_train_bin = (y_train.values > RAIN_THRESHOLD).astype(int)

test_pred_bin = (test_pred > RAIN_THRESHOLD).astype(int)
train_pred_bin = (train_pred > RAIN_THRESHOLD).astype(int)

# =====================================================
# R Squared on the binary rain / no-rain labels
# =====================================================

train_r2 = r2_score(y_train_bin, train_pred_bin)
test_r2 = r2_score(y_test_bin, test_pred_bin)

print("\n==============================")
print("R Squared (R²) - Rain vs No Rain")
print("==============================")
print(f"Threshold : rainfall > {RAIN_THRESHOLD} mm  = rain")
print(f"Train R²  : {train_r2:.4f}")
print(f"Test  R²  : {test_r2:.4f}")
print(f"Gap       : {train_r2 - test_r2:.4f}  (smaller = less overfitting)")

# =====================================================
# Classification context (accuracy + confusion matrix)
# =====================================================

# precision / recall / F1 are reported for the "rain" class (label 1)
acc = accuracy_score(y_test_bin, test_pred_bin)
precision = precision_score(y_test_bin, test_pred_bin, pos_label=1, zero_division=0)
recall = recall_score(y_test_bin, test_pred_bin, pos_label=1, zero_division=0)
f1 = f1_score(y_test_bin, test_pred_bin, pos_label=1, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test_bin, test_pred_bin, labels=[0, 1]).ravel()

print("\n==============================")
print("Classification Context (Test)")
print("==============================")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}  (of predicted-rain days, how many actually rained)")
print(f"Recall    : {recall:.4f}  (of actual-rain days, how many were caught)")
print(f"F1 score  : {f1:.4f}  (harmonic mean of precision & recall)")
print(f"Actual rain days   : {y_test_bin.sum()} / {len(y_test_bin)}")
print(f"Predicted rain days: {test_pred_bin.sum()} / {len(test_pred_bin)}")
print("\nConfusion matrix:")
print(f"  True No-Rain (TN) : {tn}")
print(f"  False Rain   (FP) : {fp}")
print(f"  Missed Rain  (FN) : {fn}")
print(f"  True Rain    (TP) : {tp}")
