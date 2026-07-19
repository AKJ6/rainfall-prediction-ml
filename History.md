# Design History & Decision Log

> This document records **how the project evolved and why** — the previous
> approach, the new approach, and the reasoning behind each change. It is a
> running history and is meant to keep growing.
>
> For *what the project is today*, see [`README.md`](README.md). This file keeps
> the story; the README keeps the current facts.

---

## Timeline

| Change | Previous approach | New approach |
|--------|-------------------|--------------|
| Model size | Unpruned RF → 6.3 GB `.pkl`, ~10 GB RAM | Pruned RF → ~36 MB, low RAM |
| Evaluation | Metrics printed only during training | Dedicated `evaluate.py` (Train vs Test R²) |
| Dataset growth | Static `.xlsx` only | `add_data.py` + `fetch_data.py` (Open-Meteo) |
| Humidity | Not present | Fetched via `add_humidity_dew.py` → `india_weather_openmeteo.csv` |
| Main model | 15-feature `train.py` (xlsx) | **17-feature `train_openmeteo.py`** (humidity + dew_point) |
| Inference | Open-Meteo only, no humidity | Humidity fetched at inference; **Google Weather** variant added |
| Eval scope | Rainfall amount only | + **`eval_rain_binary.py`** (rain vs no-rain: R², precision/recall/F1) |
| Docs | Single (stale) `README.md` | Current `README.md` + this history log |

---

## 1. The RAM problem (6.3 GB model → 10 GB RAM)

### Previous approach
`train.py` trained a Random Forest with defaults that let the trees grow without
limit:

```python
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
#                     └ 300 trees      └ max_depth=None (unbounded)
```

On ~617k rows with no depth cap, each of the 300 trees grew until its leaves were
pure — millions of nodes per tree. The resulting `rainfall_random_forest.pkl` was
**6.3 GB on disk**, and deserializing it in `pred_rain.py` inflated it to **~10 GB
in RAM** (numpy node arrays + Python object overhead).

### New approach
Prune the trees so they stop before memorizing, and compress the saved file:

```python
RandomForestRegressor(
    n_estimators=150,        # fewer trees
    max_depth=16,            # cap depth — the main size fix
    min_samples_leaf=20,     # no tiny memorized leaves
    max_features="sqrt",     # standard RF regularization
    random_state=42,
    n_jobs=-1,
)
joblib.dump(model, "rainfall_random_forest.pkl", compress=3)
```

**Result:** `.pkl` dropped from **6.3 GB → ~36 MB** (~175×), RAM from ~10 GB to tens
of MB.

### Why this was safe
Pruning reduces overfitting; it only hurts accuracy if the model *isn't* overfitting.
The 6.3 GB model was clearly memorizing (a 15-feature problem doesn't need gigabytes
of trees). Measured Train vs Test R² after pruning:

```
Train R² : 0.5162
Test  R² : 0.4643   (gap 0.0518)
```

A small gap → the pruned model generalizes; the removed gigabytes were noise, not
signal. **Key insight:** pruning doesn't "add" accuracy — it closes the gap between
training and test performance.

---

## 2. Evaluation as a first-class step

### Previous approach
Accuracy was only visible as a side effect of running `train.py`; there was no way
to check a saved model without retraining.

### New approach
Added **`evaluate.py`** — it loads the saved `.pkl` + encoders, reproduces the exact
`train.py` split (`random_state=42`), and reports **Train R²**, **Test R²**, their
**gap**, MAE and RMSE. This makes the overfitting gap a visible, repeatable metric.

---

## 3. Growing the dataset (Open-Meteo)

### Previous approach
The model trained only on the static `india_weather_rainfall_data.xlsx`. No way to
add more stations or extend the time range.

### New approach — two complementary scripts

Both fetch **hourly** archive data from Open-Meteo and aggregate to daily rows in the
dataset's exact 15-column schema. Station coordinates come from the existing `.xlsx`,
and labels are validated against the mapping files.

- **`add_data.py`** — tops up *existing* stations: for each, fetches from its last
  recorded date forward to ~1 week ago, then merges + de-duplicates on
  `(date, station)`. Backs up the `.xlsx` first.
- **`fetch_data.py`** — builds a *full* dataset over a fixed `START_DATE`→`END_DATE`
  into `india_weather_fetched.csv`.

### Design decisions along the way

**a) No hardcoded station list.**
An early draft of `add_data.py`/`fetch_data.py` hardcoded ~18 cities. This was
rejected — the scripts now **derive every station dynamically** from the dataset and
validate against `state_mapping.csv` / `station_name_mapping.csv`. No assumption about
how many stations exist.

**b) Schema consistency over humidity (for now).**
Adding a `humidity` column would leave the ~617k existing rows blank in that column,
and `train.py`'s `df.dropna()` would then **delete every old row**. So the dataset-
growth scripts deliberately keep the original 15-column schema (no humidity). Humidity
is pursued separately (see §4), not mixed in.

**c) Rate-limit resilience.**
The free Open-Meteo API returns **429 Too Many Requests** under rapid load. Fixes:
- `fetch_data.py`: backoff + retry honoring `Retry-After`, 3s between requests, and
  **per-station resume** (skips stations already saved, appends incrementally) — so a
  429 never loses progress; just re-run.
- `add_data.py`: skips already-collected **dates** via forward-fill + dedup, but saves
  once at the end (no mid-run checkpoint).

---

## 4. Humidity / accuracy — wired in

### Context
With Test R² ≈ 0.46, the limiting factor is **signal, not overfitting**. The most
physically direct rainfall driver — **humidity / dew point** — is absent from the
dataset. Open-Meteo (which `pred_rain.py` already uses) exposes `relative_humidity_2m`
and `dew_point_2m`, so it's obtainable end-to-end.

### New approach — `add_humidity_dew.py`
`train_openmeteo.py` trains with `humidity` + `dew_point` features but expects them in
`india_weather_openmeteo.csv`, which `fetch_data.py` never produced (it deliberately
keeps the 15-column schema — see §3b). **`add_humidity_dew.py`** closes that gap:

- Reads `india_weather_fetched.csv` and builds a per-station catalog (coords + the
  exact date span each station covers).
- Fetches hourly `relative_humidity_2m` + `dew_point_2m` from the same Open-Meteo
  archive, aggregates to **daily means** (rounded to 1 decimal, matching the existing
  style).
- Merges them onto the existing rows by `(station_name, date_of_record)` and writes
  `india_weather_openmeteo.csv` — the exact file the trainer loads.

Humidity / dew point are **independent measurements** — they cannot be derived from the
temp/pressure/wind columns already present, so a fresh fetch is unavoidable.

### Design decisions
**a) New output file, original untouched.**
The script only *reads* `india_weather_fetched.csv`; it writes `india_weather_openmeteo.csv`
(merged output) and `humidity_dew_cache.csv` (fetch cache). The 15-column source CSV is
never modified or deleted — both files coexist afterward. This sidesteps the §3b
`dropna()` hazard: old rows aren't blanked in place, the merged file is a separate build.

**b) Rate-limit resilience (reuses §3c).**
Same 429 backoff/retry honoring `Retry-After`, 3s between requests, and **per-station
resume** via `humidity_dew_cache.csv` — a 429 mid-run never loses progress; just re-run.

### Now resolved
`train_openmeteo.py` is now the **main trainer** (the committed model is 17-feature),
and **`pred_rain.py` fetches `relative_humidity_2m` + `dew_point_2m` at inference**, so
train/serve is consistent. Measured lift on the amount model: Test R² **0.46 → 0.63**.

### Other accuracy levers noted (not yet done)
- **Lag features** — yesterday's rainfall, rolling averages (rainfall is strongly
  autocorrelated); usually the biggest single gain.
- **Gradient boosting** (`HistGradientBoostingRegressor`) — often beats RF on tabular
  data and stays memory-light.

---

## 5. Google Weather variant (`pred_rain_google.py`)

### Previous approach
Live inference (`pred_rain.py`) pulled current conditions only from Open-Meteo.

### New approach
Added **`pred_rain_google.py`** — identical model, feature contract, and output, but
sources live conditions from the **Google Maps Platform Weather API**
(`currentConditions:lookup` for temp/humidity/dew/pressure/wind,
`forecast/days:lookup` for today's max/min). The key is read from the
`GOOGLE_MAPS_API_KEY` environment variable — **never hardcoded**.

### Why the dataset still uses Open-Meteo
Google's Weather API is a **live / short-forecast** service — it doesn't expose the
multi-year hourly archive the dataset build needs. So `add_humidity_dew.py` /
`fetch_data.py` stay on Open-Meteo; only live prediction has a Google option.

---

## 6. Rain vs no-rain evaluation (`eval_rain_binary.py`)

### Motivation
`evaluate.py` scores the rainfall **amount** (R²/MAE/RMSE). A separate question is
"does the model get **rain vs no-rain** right?" — a classification view of the same
regressor.

### New approach
Added **`eval_rain_binary.py`** — same model, dataset, and `random_state=42` split as
`evaluate.py`, but it collapses both actual and predicted rainfall to `1` (rain) / `0`
(no rain) via a `RAIN_THRESHOLD`, then reports **R² on those 0/1 labels** plus accuracy,
**precision / recall / F1** (rain class), and a confusion matrix.

### Design decision — the threshold matters
The first cut used `> 0 mm` as "rain". That was useless: the regressor almost never
outputs exactly `0`, so **every** test day read as "rain" → predicted-rain 296624/296624,
accuracy 0.50, **R² ≈ −1.0** (worse than guessing the mean). Switching to the standard
meteorological rain-day cutoff of **`> 1.0 mm`** (below = trace/drizzle) made the metric
meaningful:

```
Test R²   : 0.3989      (R² on 0/1 rain labels)
Accuracy  : 0.8626
Precision : 0.7390
Recall    : 0.9455
F1 score  : 0.8296
```

### Precision / recall / F1 added
Accuracy and R² alone hide *how* the model errs. Adding **precision, recall, and F1**
(for the rain class, `pos_label=1`, `zero_division=0`) made the failure mode explicit:
recall **0.95** ≫ precision **0.74** — the model rarely misses rain but raises many
false alarms, i.e. it over-predicts rain. All metrics honor `RAIN_THRESHOLD` (a constant
at the top of the file), so retuning the cutoff moves them together.

### Also fixed here
`evaluate.py` was still stale — reading the legacy `.xlsx` with the old 15-feature
list (no `humidity`/`dew_point`), so it no longer matched the committed 17-feature
model. It was repointed to `india_weather_openmeteo.csv` with the humidity/dew_point
features, matching `train_openmeteo.py`.

---

## 7. Documentation

### Previous approach
A single `README.md` that only covered `train.py` / `pred_rain.py` / `para.py` and
became stale (it still described `n_estimators=300`).

### New approach
- `README.md` rewritten to describe the **current** state of everything (all scripts,
  17-feature schema, pruned model, rate-limit behavior, both eval scripts, accuracy).
- **This `History.md`** (renamed from `doc.md`) keeps the **history** — the previous vs
  new approach and the reasoning — so decisions aren't lost even as the README moves on.

---

## Open items

- [x] Build the humidity dataset — `add_humidity_dew.py` produces
      `india_weather_openmeteo.csv` for `train_openmeteo.py`.
- [x] Promote the 17-feature humidity model to main (`train_openmeteo.py`, committed model).
- [x] Make `pred_rain.py` fetch humidity/dew at inference for train/serve consistency.
- [x] Repoint `evaluate.py` to the 17-feature openmeteo dataset (was still on the xlsx).
- [x] Add a rain-vs-no-rain eval — `eval_rain_binary.py`.
- [ ] Consider lag/rolling features for a real accuracy gain.
- [ ] `add_data.py` has no mid-run checkpoint — add per-station resume if 429s
      interrupt large runs.
