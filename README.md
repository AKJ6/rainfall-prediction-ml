# Rainfall Prediction (India)

A machine-learning project that predicts **daily rainfall (mm)** from weather and
location features using a **Random Forest Regressor**. It trains on historical
India weather data (17 features, including **humidity** and **dew point**) and can
produce a live prediction for any location by pulling current conditions from the
[Open-Meteo](https://open-meteo.com/) API — or the Google Maps Platform Weather API.

---

## Contents

- [Pipeline overview](#pipeline-overview)
- [File reference](#file-reference)
- [Dataset schema](#dataset-schema)
- [The model](#the-model)
- [Scripts in detail](#scripts-in-detail)
- [Data collection (Open-Meteo)](#data-collection-open-meteo)
- [Mapping files](#mapping-files)
- [Version control](#version-control)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Accuracy & known limitations](#accuracy--known-limitations)

---

## Pipeline overview

```
   ┌───────────────────────┐
   │ fetch_data.py         │  build a full dataset from Open-Meteo
   └──────────┬────────────┘
              │ india_weather_fetched.csv  (15-col schema)
              ▼
   ┌───────────────────────┐
   │ add_humidity_dew.py   │  fetch + merge humidity + dew_point
   └──────────┬────────────┘
              │ india_weather_openmeteo.csv  (17 features)
              ▼
   train_openmeteo.py       clean → feature-engineer → label-encode → fit (pruned RF)
              │
              ▼
   ┌───────────────────────────────────────────────────┐
   │ rainfall_random_forest.pkl   (trained model)       │
   │ label_encoders.pkl           (season/state/district)│
   └──┬──────────────┬───────────────┬──────────────┬───┘
      │              │               │              │
 evaluate.py   eval_rain_binary  para.py      pred_rain.py /
 R²/MAE/RMSE   R² rain-vs-no     feature      pred_rain_google.py
 (amount)      -rain + acc.      importance   live weather → predict mm
```

---

## File reference

### Scripts

| File | Role |
|------|------|
| `train_openmeteo.py` | **Main trainer.** Trains on `india_weather_openmeteo.csv` (17 features, incl. `humidity` + `dew_point`). Cleans NaNs, engineers date + `temp_range` features, label-encodes `season`/`state`/`district`, fits a **pruned** Random Forest, prints metrics + feature importance, and saves the two `.pkl` artifacts (model saved with `compress=3`). |
| `train.py` | ⚠️ **Legacy trainer.** The original 15-feature variant trained on `india_weather_rainfall_data.xlsx` (no humidity/dew_point). Superseded by `train_openmeteo.py`; kept for reference. |
| `evaluate.py` | Loads the saved model + encoders and reports **Train R²**, **Test R²**, their gap (overfitting indicator), MAE and RMSE on the rainfall **amount (mm)** — without retraining. Reproduces `train_openmeteo.py`'s exact split (`random_state=42`). |
| `eval_rain_binary.py` | Same split/model as `evaluate.py`, but collapses actual and predicted rainfall to **rain / no-rain** (default cutoff `> 1.0 mm`) and reports **R² on those 0/1 labels** plus accuracy, **precision / recall / F1** (rain class), and a confusion matrix. Answers "does it get rain-vs-no-rain right", not "how much". |
| `pred_rain.py` | Loads the model + encoders, fetches **current** weather (incl. humidity + dew point) for a configured location from Open-Meteo, builds the 17-feature vector, and prints today's conditions + predicted rainfall. |
| `pred_rain_google.py` | Same model + feature contract as `pred_rain.py`, but pulls live conditions from the **Google Maps Platform Weather API** instead of Open-Meteo. Needs `GOOGLE_MAPS_API_KEY` in the environment. |
| `para.py` | Loads the model and prints the feature-importance table (quick diagnostic). |
| `add_data.py` | **Grows the dataset.** Derives every station from the `.xlsx` (coordinates + metadata), validates names against the mapping files, and fetches only the dates each station is *missing* (its last date → ~1 week ago) from Open-Meteo. Backs up the file, merges, de-duplicates on `(date, station)`, and saves back into the `.xlsx`. |
| `fetch_data.py` | **Builds a full dataset** over a fixed date range (`START_DATE`→`END_DATE`) for all validated stations, into `india_weather_fetched.csv` in the base 15-column schema. Handles 429 rate limits with backoff and **resumes** (skips stations already saved). |
| `add_humidity_dew.py` | Reads `india_weather_fetched.csv`, fetches hourly `relative_humidity_2m` + `dew_point_2m` from Open-Meteo, aggregates to daily means, and merges them on `(station, date)` to produce `india_weather_openmeteo.csv` — the file `train_openmeteo.py` trains on. |

### Data & artifacts

> These are all **git-ignored** (see [Version control](#version-control)) — regenerate
> them by running the scripts; they are not part of the repo.

| File | Tracked? | Description |
|------|----------|-------------|
| `india_weather_openmeteo.csv` | no | **Primary training dataset** — 17 features incl. humidity + dew point (built by `add_humidity_dew.py`). This is what `train_openmeteo.py` / `evaluate.py` / `eval_rain_binary.py` load. |
| `india_weather_rainfall_data.xlsx` | no | Legacy 15-feature dataset used by `train.py`. |
| `india_weather_fetched.csv` | no | Base dataset built by `fetch_data.py` from Open-Meteo (15-col schema); input to `add_humidity_dew.py`. |
| `humidity_dew_cache.csv` | no | Per-station fetch cache written by `add_humidity_dew.py` (enables resume after a 429). |
| `india_weather_rainfall_data.backup.xlsx` | no | Automatic backup written by `add_data.py` before it modifies the dataset. |
| `rainfall_random_forest.pkl` | no | Serialized trained model (produced by `train_openmeteo.py`). |
| `label_encoders.pkl` | no | Serialized `LabelEncoder`s for `season`, `state`, `district`. |
| `importance.txt` | **yes** | Saved feature-importance table from a training run (the one non-script file kept in the repo). |

### Mapping files

> Also **git-ignored** (`*.csv`). The fetch scripts read them to validate new rows;
> keep local copies alongside the scripts.

| File | Tracked? | Description |
|------|----------|-------------|
| `state_mapping.csv` | no | 32 state codes → encoded integer. |
| `district_mapping.csv` | no | District names → encoded integer. |
| `station_name_mapping.csv` | no | 406 station names → encoded integer. |

---

## Dataset schema

`india_weather_openmeteo.csv` (the current training dataset) adds `humidity` and
`dew_point` on top of the base schema. The base 15-column schema (used by the
`.xlsx` and `india_weather_fetched.csv`) is:

| Column | Type | Notes |
|--------|------|-------|
| `date_of_record` | datetime | The day of the record |
| `month` | str | Month name (e.g. `January`) |
| `season` | str | `Winter` / `Summer` / `Monsoon` / `Post-Monsoon` |
| `station_name` | str | Weather station / city |
| `state` | str | State code (e.g. `KA`) |
| `district` | str | District name |
| `avg_temp` | float | Daily mean temperature (°C) |
| `min_temp` | float | Daily min temperature (°C) |
| `max_temp` | float | Daily max temperature (°C) |
| `wind_speed` | float | Daily mean wind speed |
| `air_pressure` | float | Daily mean MSL pressure (hPa) |
| `elevation` | int | Station elevation (m) |
| `latitude` | float | |
| `longitude` | float | |
| `rainfall` | float | **Target** — daily rainfall (mm) |
| `humidity`\*\* | float | Daily mean relative humidity (%) — *openmeteo CSV only* |
| `dew_point`\*\* | float | Daily mean dew point (°C) — *openmeteo CSV only* |

\*\* Added by `add_humidity_dew.py`; present only in `india_weather_openmeteo.csv`.

`train_openmeteo.py`/`pred_rain.py` derive additional features at train/predict time:
`temp_range = max_temp − min_temp`, and `year` / `month_num` / `day_of_year`
from `date_of_record`. The **17 model features** are:

`avg_temp`, `min_temp`, `max_temp`, `temp_range`, `humidity`, `dew_point`,
`wind_speed`, `air_pressure`, `elevation`, `latitude`, `longitude`, `season`\*,
`state`\*, `district`\*, `year`, `month_num`, `day_of_year`

\* Categorical — label-encoded at train time; the same encoders are reused at prediction time.

---

## The model

- **Algorithm:** `sklearn.ensemble.RandomForestRegressor`
- **Hyperparameters (pruned for size + generalization):**

  ```python
  RandomForestRegressor(
      n_estimators=150,
      max_depth=16,           # cap tree depth (the key size fix)
      min_samples_leaf=20,    # no tiny memorized leaves
      max_features="sqrt",
      random_state=42,
      n_jobs=-1,
  )
  ```

- **Target:** `rainfall` (mm)
- **Features:** 17 (incl. `humidity` + `dew_point`)
- **Split:** 80% train / 20% test (`random_state=42`)
- **Saved with:** `joblib.dump(..., compress=3)`

> **History:** an earlier unpruned model (`n_estimators=300`, `max_depth=None`)
> produced a **6.3 GB** `.pkl` that needed ~10 GB RAM to load. Pruning shrank it
> to **~36 MB** with essentially no loss in test accuracy — the extra size was
> memorizing training noise, not signal.

---

## Scripts in detail

### `train_openmeteo.py`
Input: `india_weather_openmeteo.csv` → Output: `rainfall_random_forest.pkl`, `label_encoders.pkl`.
Trains the current 17-feature model and prints MAE, RMSE, Train R², Test R² (+ gap)
and the feature-importance table.

### `train.py`
Legacy 15-feature trainer on the `.xlsx` (no humidity/dew_point). Superseded by
`train_openmeteo.py`; kept for reference.

### `evaluate.py`
Input: the two `.pkl` files + `india_weather_openmeteo.csv`. Recomputes rainfall-amount
metrics (R²/MAE/RMSE) on the same held-out split **without retraining**. Use it to
sanity-check accuracy or the overfitting gap after any change.

### `eval_rain_binary.py`
Same model + split as `evaluate.py`, but evaluates **rain vs no-rain** instead of the
amount: both actual and predicted rainfall are collapsed to `1` (rain) / `0` (no rain)
using `RAIN_THRESHOLD` (default `1.0 mm`, the standard meteorological rain-day cutoff),
and R² is reported on those labels alongside **accuracy, precision, recall, and F1**
(for the rain class) and a confusion matrix. Change `RAIN_THRESHOLD` at the top of the
file to use a different cutoff — every metric follows it.

### `pred_rain.py`
Loads the model + encoders, fetches **current** weather (incl. `relative_humidity_2m`
and `dew_point_2m`) from the Open-Meteo *forecast* API for the location configured at
the top of the file, and prints a rainfall prediction. Location constants:

```python
LATITUDE  = 12.9716
LONGITUDE = 77.5946
ELEVATION = 920
STATE     = "KA"
DISTRICT  = "Bengaluru Rural"
SEASON    = "Monsoon"
```

`STATE`, `DISTRICT`, `SEASON` must be values the encoders were trained on — see the
`*_mapping.csv` files for valid categories.

### `pred_rain_google.py`
Same model, feature contract, and output as `pred_rain.py`, but sources live
conditions from the **Google Maps Platform Weather API** (`currentConditions:lookup`
for temp/humidity/dew/pressure/wind, `forecast/days:lookup` for today's max/min).
Requires a Weather-API-enabled key in `GOOGLE_MAPS_API_KEY` (never hardcode it). Note:
Google's Weather API is live/short-forecast only — the multi-year archive the dataset
build needs still comes from Open-Meteo.

### `para.py`
Loads the model and prints feature importances (no data needed).

### `add_data.py`
Extends the dataset forward in time. For each station already in the `.xlsx` it
fetches from its **last recorded date + 1** up to ~1 week ago (Open-Meteo's archive
lag), then merges + de-duplicates. Writes a `.backup.xlsx` first. Safe to re-run —
dates you already have are never re-added.

### `fetch_data.py`
Builds a full dataset over `START_DATE`→`END_DATE` for all validated stations into
`india_weather_fetched.csv`. See [Data collection](#data-collection-open-meteo) for
rate-limit/resume behavior.

---

## Data collection (Open-Meteo)

Both `fetch_data.py` and `add_data.py` pull **hourly** archive data
(`temperature_2m`, `pressure_msl`, `wind_speed_10m`, `precipitation`) and aggregate
to one daily row per station (means for temp/wind/pressure, **sum** for rainfall).
Station **coordinates come from the existing `.xlsx`** — no hardcoded station list,
no assumed station count. Only stations/states present in the mapping files are kept.

**Rate limits & resume:**

| | `fetch_data.py` | `add_data.py` |
|---|---|---|
| 429 handling | Backoff + retry (honors `Retry-After`) | Fails the run; just re-run |
| Skips already-collected | Per **station** (reads the output CSV) | Per **date** (dedup + forward-fill) |
| Survives mid-run interruption | ✅ appends per station, resumes | ❌ saves once at the end |

> The free Open-Meteo tier caps daily call volume. A full multi-year hourly pull for
> many stations may need several runs (resume makes this painless) or shortening the
> date range.

---

## Mapping files

`state_mapping.csv`, `district_mapping.csv`, and `station_name_mapping.csv` are the
canonical label→integer encodings. The fetch scripts use them to **validate** any new
row so the dataset never introduces a category the encoders don't know.

---

## Version control

`.gitignore` keeps all large / generated files out of the repo:

```
*my/      # the virtual environment
*.xlsx    # datasets + backups
*.pkl     # trained model + encoders
*.csv     # fetched datasets, caches, and the mapping files
*.env     # secrets (e.g. GOOGLE_MAPS_API_KEY)
```

So the **only tracked files** (per `git ls-files`) are the source and docs:

| Tracked | Files |
|---------|-------|
| Scripts | `train_openmeteo.py`, `train.py`, `evaluate.py`, `eval_rain_binary.py`, `pred_rain.py`, `pred_rain_google.py`, `para.py`, `add_data.py`, `fetch_data.py`, `add_humidity_dew.py` |
| Docs | `README.md`, `History.md` |
| Other | `importance.txt`, `.gitignore` |

Everything else — datasets (`*.csv`, `*.xlsx`), model artifacts (`*.pkl`), the mapping
CSVs, `*.env`, and the `my/` virtualenv — is **git-ignored** and must exist locally.
Regenerate datasets/artifacts with the fetch + train scripts; the mapping CSVs must be
kept alongside the scripts (the fetch scripts read them and won't run without them).

---

## Setup

Requires **Python 3** and:

```
pandas  numpy  scikit-learn  joblib  openpyxl  requests
```

```bash
python3 -m venv my
source my/bin/activate
pip install pandas numpy scikit-learn joblib openpyxl requests
```

> A pre-built virtual environment `my/` is already present (git-ignored).
> If you use it, just run `source my/bin/activate`.

Tested with: scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.1.

---

## How to run

**Typical workflow:**

```bash
# 1. build the dataset
python fetch_data.py        # build india_weather_fetched.csv from Open-Meteo
python add_humidity_dew.py  # add humidity + dew_point -> india_weather_openmeteo.csv
#   (or) top up the legacy .xlsx with recent dates:
python add_data.py

# 2. train the current 17-feature model
python train_openmeteo.py   # writes the two .pkl artifacts

# 3. check accuracy
python evaluate.py          # rainfall amount (R² / MAE / RMSE)
python eval_rain_binary.py  # rain vs no-rain (R² on 0/1 labels + accuracy)

# 4. predict live rainfall for the configured location
python pred_rain.py                 # Open-Meteo
GOOGLE_MAPS_API_KEY=... python pred_rain_google.py   # Google Weather API

# 5. (optional) inspect feature importance
python para.py
```

The prediction and eval scripts require the `.pkl` files, so run `train_openmeteo.py`
first (or use the committed model). `pred_rain*.py`, `add_data.py`, `fetch_data.py`,
and `add_humidity_dew.py` require network access; `pred_rain_google.py` also needs
`GOOGLE_MAPS_API_KEY`.

---

## Accuracy & known limitations

Latest run on the 17-feature model:

**Rainfall amount** (`evaluate.py`):

```
MAE      : 2.3925
RMSE     : 6.5969
Train R² : 0.6685
Test  R² : 0.6311   (gap 0.0374 — little overfitting)
```

**Rain vs no-rain** (`eval_rain_binary.py`, cutoff `> 1.0 mm`):

```
Test R²   : 0.3989      (R² on 0/1 rain labels)
Accuracy  : 0.8626      (86.3% of test days classified correctly)
Precision : 0.7390      (of predicted-rain days, how many actually rained)
Recall    : 0.9455      (of actual-rain days, how many were caught)
F1 score  : 0.8296      (harmonic mean of precision & recall)
```

Recall (0.95) ≫ precision (0.74): the model rarely **misses** rain but raises a fair
number of **false alarms** — i.e. it over-predicts rain. Note the earlier `> 0 mm`
cutoff was useless — the regressor almost never outputs exactly `0`, so *every* day
read as "rain"; the `1.0 mm` rain-day cutoff is what makes the binary metrics meaningful.

- Adding **humidity + dew point** lifted amount R² from ~0.46 → **0.63** — the single
  biggest accuracy gain, and the reason the pipeline moved to
  `india_weather_openmeteo.csv` + `train_openmeteo.py`. Both `pred_rain.py` variants
  now fetch humidity at inference, so train/serve stays consistent.
- **Remaining accuracy levers** (data, not hyperparameters):
  - **Lag features** — yesterday's rainfall, rolling averages (rainfall is strongly
    autocorrelated); usually the biggest remaining single gain.
  - **Gradient boosting** (`HistGradientBoostingRegressor`) often beats Random
    Forest on tabular data and stays small in memory.
- The datasets (`.xlsx`, the large `.csv`s) and the virtual environment (`my/`) are git-ignored.
```
