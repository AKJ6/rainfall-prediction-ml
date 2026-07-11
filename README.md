# Rainfall Prediction

A machine-learning project that predicts daily rainfall (in mm) from weather
and location features using a **Random Forest Regressor**. It trains on
historical India weather data and can produce a live prediction for a location
by pulling current conditions from the [Open-Meteo](https://open-meteo.com/)
API.

---

## Architecture

```
                       ┌──────────────────────────────┐
                       │  india_weather_rainfall_data  │
                       │           .xlsx               │
                       └───────────────┬──────────────┘
                                       │
                          (1) train.py │  clean → feature-engineer →
                                       │        label-encode → fit
                                       ▼
        ┌──────────────────────────────────────────────────────┐
        │  rainfall_random_forest.pkl   (trained model)         │
        │  label_encoders.pkl           (season/state/district) │
        └───────────────┬───────────────────────┬──────────────┘
                        │                       │
       (2) para.py      │                       │  (3) pred_rain.py
       feature          │                       │  fetch live weather
       importance       ▼                       ▼  (Open-Meteo API)
                    print table          build feature vector → predict
                                                 → print rainfall (mm)
```

### Components

| File | Role |
|------|------|
| `train.py` | Loads the Excel dataset, engineers features, label-encodes categoricals, trains the Random Forest, prints metrics + feature importance, and saves the model and encoders as `.pkl` files. |
| `pred_rain.py` | Loads the saved model + encoders, fetches **current** weather for a location from the Open-Meteo API, builds the feature vector, and prints the predicted rainfall. |
| `para.py` | Loads the saved model and prints the feature-importance table (a quick diagnostic tool). |
| `india_weather_rainfall_data.xlsx` | Training dataset (historical India weather + rainfall). |
| `rainfall_random_forest.pkl` | Serialized trained model (produced by `train.py`). |
| `label_encoders.pkl` | Serialized `LabelEncoder`s for `season`, `state`, `district`. |
| `*_mapping.csv` | Human-readable lookups of encoded category values (state, district, station name). |

### Prediction Architecture (`pred_rain.py`)

How a single live prediction is produced:

```
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │ rainfall_random_forest.pkl│   │     label_encoders.pkl    │
        └─────────────┬─────────────┘   └─────────────┬─────────────┘
                      │ joblib.load                   │ joblib.load
                      ▼                               ▼
              ┌───────────────┐              ┌──────────────────┐
              │  model (RF)   │              │ season / state / │
              │               │              │ district encoders│
              └───────┬───────┘              └────────┬─────────┘
                      │                               │
   Location config    │                               │
   LAT / LON / ELEV   │                               │
   STATE / DISTRICT / │                               │
   SEASON             │                               │
        │             │                               │
        ▼             │                               ▼
 ┌──────────────┐     │                    ┌────────────────────────┐
 │ Open-Meteo   │     │                    │ encode categoricals     │
 │ forecast API │     │                    │ SEASON→n  STATE→n        │
 │ (HTTP GET)   │     │                    │ DISTRICT→n               │
 └──────┬───────┘     │                    └───────────┬────────────┘
        │             │                                │
        ▼             │                                │
 ┌──────────────┐     │       ┌────────────────┐       │
 │ current +    │     │       │ date features  │       │
 │ daily weather│     │       │ year / month / │       │
 │ temp, wind,  │     │       │ day_of_year    │       │
 │ pressure     │     │       │ (datetime.now) │       │
 └──────┬───────┘     │       └────────┬───────┘       │
        │             │                │               │
        └─────────────┴────────┬───────┴───────────────┘
                               ▼
                 ┌──────────────────────────┐
                 │  build feature vector X   │
                 │  (15 features, 1 row)     │
                 └─────────────┬────────────┘
                               ▼
                    ┌────────────────────┐
                    │ model.predict(X)[0]│
                    └──────────┬─────────┘
                               ▼
                 ┌──────────────────────────┐
                 │ rainfall (mm)             │
                 │  > 0  → "Rain Expected"   │
                 │  ≤ 0  → "No Rain"         │
                 └──────────────────────────┘
```

**Step by step:**

1. **Load artifacts** — `model` and `encoders` are read from the `.pkl` files.
2. **Configure location** — `LATITUDE`, `LONGITUDE`, `ELEVATION`, `STATE`,
   `DISTRICT`, `SEASON` are set as constants at the top of the script.
3. **Fetch live weather** — an HTTP GET to the Open-Meteo forecast API returns
   `current` (temperature, pressure, wind) and `daily` (max/min temp) values;
   `temp_range = max_temp − min_temp` is derived.
4. **Derive date features** — `year`, `month_num`, `day_of_year` from
   `datetime.now()`.
5. **Encode categoricals** — `SEASON`, `STATE`, `DISTRICT` are transformed with
   the *same* encoders used in training (must be known categories).
6. **Assemble feature vector** — a single-row `DataFrame` with all 15 features
   in the training column order.
7. **Predict & report** — `model.predict(X)[0]` gives rainfall in mm; the script
   prints the weather summary and a "Rain Expected / No Rain" verdict.

### Model

- **Algorithm:** `sklearn.ensemble.RandomForestRegressor` (`n_estimators=300`, `random_state=42`, `n_jobs=-1`)
- **Target:** `rainfall` (mm)
- **Split:** 80% train / 20% test (`random_state=42`)
- **Metrics reported:** MAE, RMSE, R²

### Features (15)

`avg_temp`, `min_temp`, `max_temp`, `temp_range` (max − min), `wind_speed`,
`air_pressure`, `elevation`, `latitude`, `longitude`, `season`*, `state`*,
`district`*, `year`, `month_num`, `day_of_year`

\* Categorical — label-encoded at train time; the same encoders are reused at
prediction time.

---

## Setup

Requires **Python 3** and the following packages:

```
pandas  numpy  scikit-learn  joblib  openpyxl  requests
```

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv my
source my/bin/activate
pip install pandas numpy scikit-learn joblib openpyxl requests
```

> A pre-built virtual environment named `my/` is already present in this repo
> (git-ignored). If you use it, just run `source my/bin/activate`.

---

## How to Run

### 1. Train the model

Reads `india_weather_rainfall_data.xlsx`, trains the model, prints evaluation
metrics + feature importance, and writes `rainfall_random_forest.pkl` and
`label_encoders.pkl`:

```bash
python train.py
```

### 2. Predict rainfall (live weather)

Loads the saved model, fetches current weather from the Open-Meteo API for the
configured location, and prints today's conditions and the predicted rainfall:

```bash
python pred_rain.py
```

The target location is set at the top of `pred_rain.py` (defaults to
Bengaluru). Edit these constants to change it:

```python
LATITUDE  = 12.9716
LONGITUDE = 77.5946
ELEVATION = 920
STATE     = "KA"
DISTRICT  = "Bengaluru Rural"
SEASON    = "Monsoon"
```

> `STATE`, `DISTRICT`, and `SEASON` must be values the encoders were trained
> on — see the `*_mapping.csv` files for valid categories.

Example output:

```
Today's Weather
------------------------------
Average Temp : 24.3 °C
Max Temp     : 27.1 °C
Min Temp     : 21.0 °C
Wind Speed   : 11.4 km/h
Pressure     : 1012.0 hPa

Predicted Rainfall
------------------------------
3.45 mm
Prediction: Rain Expected
```

### 3. Inspect feature importance (optional)

```bash
python para.py
```

---

## Notes

- `pred_rain.py` and `para.py` require the `.pkl` files, so run `train.py` first
  (or use the committed model).
- Network access is required for `pred_rain.py` (Open-Meteo API).
- The dataset (`.xlsx`) and the virtual environment (`my/`) are git-ignored.
