# Predicting Urban PM2.5 Air Pollution from Weather and Satellite Data

A supervised machine-learning regression project for the [Zindi Urban Air Pollution Challenge](https://zindi.africa/). The goal is to predict daily **PM2.5 particulate matter concentration** in African cities using weather data from the Global Forecast System (GFS) and atmospheric pollutant measurements from the ESA Sentinel-5P satellite.

> **Best validated result:** Weighted Ensemble (Tuned LightGBM + ExtraTrees) — **RMSE 31.94** (GroupKFold, 5 folds)

---

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Workflow Overview](#workflow-overview)
- [Feature Engineering](#feature-engineering)
- [Validation Strategy](#validation-strategy)
- [Models and Results](#models-and-results)
- [Data Product Concept](#data-product-concept)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Background

PM2.5 refers to particulate matter with a diameter smaller than 2.5 micrometers — particles fine enough to penetrate deep into the lungs. It is one of the most harmful air pollutants, linked to respiratory disease, cardiovascular disease, and premature mortality.

Many African cities lack the dense ground-based sensor networks needed to monitor air quality in real time. This project demonstrates how satellite observations and weather data can be combined with machine learning to fill that monitoring gap at low cost.

The long-term vision is a **mobile app** that warns citizens, schools, clinics, and city authorities about high-pollution days — especially in regions where ground-based infrastructure is limited.

---

## Problem Statement

This is a **supervised regression** task. The model learns the mapping:

```
weather features + Sentinel-5P satellite pollutants + date/context features → daily PM2.5
```

The key challenge is that the **test set contains locations not seen during training**, so the model must generalize to new environments rather than memorize location-specific pollution levels.

---

## Dataset

The dataset is sourced from the Zindi Urban Air Pollution Challenge and is **not included in this repository** due to Zindi's data usage terms. Place the raw files under `data/` before running the notebooks.

| File | Description |
|---|---|
| `data/Train.csv` | Training observations with ground-sensor PM2.5 target |
| `data/Test.csv` | Test observations (no target; different locations from train) |
| `data/SampleSubmission.csv` | Submission template (optional) |

### Feature groups

**Ground sensor (target — training only)**

| Column | Meaning |
|---|---|
| `target` | Daily mean PM2.5 — the prediction target |
| `target_min / max / variance / count` | Sensor statistics — not used in modeling (unavailable at test time) |

**Weather (GFS)**

| Column | Physical meaning |
|---|---|
| `temperature_2m_above_ground` | Near-surface air temperature |
| `relative_humidity_2m_above_ground` | Relative humidity near surface |
| `specific_humidity_2m_above_ground` | Actual water vapor content |
| `precipitable_water_entire_atmosphere` | Total atmospheric water vapor |
| `u_component_of_wind_10m_above_ground` | East–west wind component |
| `v_component_of_wind_10m_above_ground` | North–south wind component |

**Sentinel-5P satellite pollutants**

| Prefix | Pollutant | PM2.5 connection |
|---|---|---|
| `L3_NO2_` | Nitrogen dioxide | Traffic and combustion signal |
| `L3_CO_` | Carbon monoxide | Fires, biomass burning, incomplete combustion |
| `L3_SO2_` | Sulfur dioxide | Industry; forms sulfate aerosols |
| `L3_O3_` | Ozone | Photochemical air pollution indicator |
| `L3_HCHO_` | Formaldehyde | VOC chemistry; secondary aerosol formation |
| `L3_CH4_` | Methane | Greenhouse gas; weak direct PM2.5 link |
| `L3_AER_AI_` | Absorbing Aerosol Index | Dust, smoke, and soot — highly relevant |
| `L3_CLOUD_` | Cloud properties | Atmospheric context and satellite data reliability |

---

## Project Structure

```
ML-project/
│
├── EDA_ML.ipynb                          # Main modeling notebook
├── EDA-and-modeling.ipynb                # Alternative combined notebook
├── requirements.txt                      # Python package dependencies
├── Makefile                              # Environment setup automation
│
├── data/                                 # (not tracked) Raw CSV files from Zindi
│
├── models/                               # (not tracked) Saved model artifacts
│
├── submission.csv                        # Latest prediction submission
│
├── urban_air_pollution_project_plan.md   # Full ML project plan and methodology
├── notebook_optimization_notes.md        # Changelog for notebook improvements
├── pre_optimization_result_snapshot.md  # Baseline vs. optimized RMSE history
│
├── Urban Air Pollution Challenge         # Feature and physical principle reference
│   Feature and Physical Principle Notes.md
│
├── ML Project - Sumac Stacks.pdf         # Stakeholder slide deck
│
├── example_files/                        # Reference examples
└── images/                               # Images for documentation
```

---

## Environment Setup

The project uses **Python 3.11.3** managed with `pyenv` and a virtual environment.

### Automated setup (recommended)

```bash
make setup
```

This will:
1. Install Python 3.11.3 via pyenv (if not already installed)
2. Set the local pyenv version
3. Create a `.venv` virtual environment
4. Install all dependencies from `requirements.txt`

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.2.2 | Core ML models and pipelines |
| `lightgbm` | 4.6.0 | Best standalone model |
| `xgboost` | 3.2.0 | Ensemble component |
| `pandas` | 2.0.1 | Data manipulation |
| `numpy` | 1.24.3 | Numerical operations |
| `matplotlib` / `seaborn` | 3.7.1 / 0.12.2 | Visualization |
| `statsmodels` | 0.13.5 | Statistical analysis |
| `jupyterlab` | 3.6.3 | Notebook environment |
| `pytest` | 7.3.1 | Testing |

> **GPU support:** XGBoost and LightGBM GPU acceleration is detected automatically at runtime. The notebook falls back to CPU if a GPU probe fails. No separate GPU builds are required.

---

## Workflow Overview

The main notebook (`EDA_ML.ipynb`) follows this structure:

```
1.  Problem statement and domain knowledge
2.  Data loading and inspection
3.  Exploratory Data Analysis (target distribution, time trends, location patterns)
4.  Feature engineering
5.  Feature selection (remove high-missingness columns)
6.  Validation strategy setup (GroupKFold by Place_ID)
7.  Baseline models
8.  Model comparison
9.  Hyperparameter tuning
10. Final model selection and training
11. Test set prediction and submission
12. Feature importance analysis
13. Error analysis
14. Data product concept slide
```

### Execution controls

The following flags at the top of the notebook let you trade speed for quality:

| Flag | Default | Effect when changed |
|---|---|---|
| `FAST_MODE` | `False` | Reduces search iterations and tree counts for quick reruns |
| `CV_N_SPLITS` | `5` | Number of GroupKFold folds |
| `BASELINE_HIGH_MISSING_THRESHOLD` | `0.80` | Drops columns with more than this fraction missing |
| `USE_FULL_FEATURE_SET_FOR_NATIVE_MISSING_MODELS` | `True` | Lets boosters see sparse columns instead of dropping them |
| `ENABLE_PHYSICAL_PROXY_FEATURES` | `True` | Adds engineered physical proxies |
| `RUN_FEATURE_GROUP_ABLATION` | `False` | Runs ablation analysis (adds runtime) |

---

## Feature Engineering

Beyond the raw columns, the notebook creates several physically motivated features:

**Date and seasonality**
- `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`, `is_weekend`
- Cyclical encodings: `dayofyear_sin / cos`, `month_sin / cos`

**Wind**
- `wind_speed` = √(u² + v²)
- `wind_dir_sin`, `wind_dir_cos`
- `low_wind_day` flag (below 25th percentile)

**Atmospheric conditions**
- `stagnation_proxy` = humidity / (wind speed + 1) — accumulation risk under calm, humid conditions
- `dryness_proxy` = temperature / (humidity + 1) — dust and aridity signal
- `dust_risk_proxy` = dryness × aerosol index
- `stagnant_humid_day` = low wind AND high humidity flag

**Emission proxies**
- `combustion_proxy` = NO₂ + CO + HCHO — traffic, fires, biomass burning
- `industrial_proxy` = SO₂ + NO₂
- `industrial_aerosol_proxy` = industrial proxy × aerosol index

**Data quality and uncertainty**
- `missing_satellite_count` — number of missing satellite columns per row
- `missing_weather_count`
- `has_missing_satellite` flag
- `satellite_reliability_risk` — cloud fraction + zenith angle uncertainty

**Threshold-based regime flags** are computed from training data statistics only and then applied to the test set to avoid test-set peeking.

---

## Validation Strategy

Because the test locations are **different** from the training locations, standard random cross-validation would be optimistic. The project uses **GroupKFold** with `Place_ID` as the grouping variable:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
scores = cv.split(X, y, groups=train["Place_ID"])
```

This ensures that all observations from the same location appear in only one fold, giving a realistic estimate of how well the model generalizes to unseen locations.

> **Important:** `Place_ID` is used **only** for cross-validation grouping, never as a model feature. Using it as a feature would cause the model to memorize location-specific averages rather than learning transferable environmental patterns.

---

## Models and Results

### Baseline comparison

| Model | Mean RMSE (5-fold GroupKFold) |
|---|---:|
| Dummy Regressor (mean) | 46.83 |
| Ridge Regression | 38.56 |
| Random Forest | 34.55 |
| HistGradientBoosting (baseline) | 34.30 |

### Optimization progression

| Experiment | RMSE (pre-optimization) | RMSE (post-optimization) |
|---|---:|---:|
| HistGradientBoosting baseline | 34.30 | 33.32 |
| Tuned HistGradientBoosting | 33.68 | 32.66 |
| HGB + missingness indicators | 33.61 | 32.65 |
| HGB with native missing handling | 33.52 | 32.65 |
| ExtraTreesRegressor | 33.68 | 33.42 |
| LightGBM baseline | 33.34 | 32.35 |
| XGBoost baseline | 33.72 | 32.89 |
| **Tuned LightGBM** | **33.04** | **32.01** |
| **Weighted Ensemble (LightGBM + ExtraTrees)** | **32.81** | **31.94** |

The post-optimization improvement was driven by:
- Physically motivated engineered features (stagnation, dryness, combustion, and industrial proxies)
- Explicit seasonality features (`month_sin / cos`)
- Missingness and satellite-reliability features as uncertainty signals
- Expanded LightGBM hyperparameter search space around previous boundary values
- Aligning tuned-model cells with `best_estimator_` to prevent result drift between reruns

### Final ensemble

The best model is a **weighted ensemble** of:

| Component | Weight |
|---|---:|
| Tuned LightGBM | 0.65 |
| ExtraTrees Regressor | 0.35 |

Ensemble weights are selected by grid search and stored automatically so reruns stay consistent with the latest validated result.

### Hyperparameter tuning

Tuning uses `RandomizedSearchCV` with `GroupKFold` passed as the cross-validator — this is critical to ensure that hyperparameter selection is evaluated on held-out locations, not held-out time points.

```python
search = RandomizedSearchCV(
    estimator=lgbm_pipe,
    param_distributions=lgbm_param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=GroupKFold(n_splits=5),
    random_state=42,
    n_jobs=1,   # Serial for Windows stability
)
search.fit(X, y, groups=train["Place_ID"])
```

---

## Data Product Concept

The notebook and slide deck include a proposed data product built on top of the model:

**AirAware Africa** — a mobile app providing daily PM2.5 risk forecasts for African cities where ground sensors are sparse.

```
User opens app
→ app reads location and date
→ backend retrieves GFS weather and Sentinel-5P satellite features
→ ML model predicts PM2.5
→ app displays risk level and health recommendation
```

| Predicted PM2.5 | Risk level | Recommendation |
|---:|---|---|
| 0–12 μg/m³ | Good | Normal outdoor activity |
| 12–35 μg/m³ | Moderate | Sensitive groups take care |
| 35–55 μg/m³ | Unhealthy for sensitive groups | Reduce outdoor activity |
| 55–150 μg/m³ | Unhealthy | Avoid extended outdoor exercise |
| > 150 μg/m³ | Very unhealthy | Stay indoors if possible |

**Target users:** citizens, parents, schools, clinics and hospitals, city authorities, NGOs, and environmental agencies.

---

## Limitations and Future Work

**Current limitations**
- The dataset does not include latitude/longitude, so geographic distance between locations cannot be used as a feature.
- Test locations are anonymous and entirely different from training locations; the model cannot rely on any location-specific memory.
- Sentinel-5P columns have significant missingness (particularly CH4), which limits how much satellite signal is available for some observations.
- The model tends to underpredict extreme pollution events because very high PM2.5 days are rare in the training data and may be driven by local phenomena not captured by the available features.

**Possible extensions**
- Add location cluster features derived from environmental profiles (mean temperature, aerosol index, wind, precipitation patterns) using Gaussian Mixture Models — without using the PM2.5 target in the clustering step.
- Incorporate temporal lag features (previous-day PM2.5) where past observations are reliably available.
- Add geographic features (latitude, longitude, land use type, population density, distance to major roads or industry) if they become available.
- Use SHAP values for feature-level explainability, which would also support the data product's per-day risk explanation ("risk is elevated because wind is low and aerosol index is high").
- Deploy the final model as an inference endpoint connected to automated daily GFS and Sentinel-5P data pipelines.

---

## Citation

Data source: [Zindi Urban Air Pollution Challenge](https://zindi.africa/competitions/urban-air-pollution-challenge)  
Satellite data: ESA Sentinel-5P (Copernicus Programme)  
Weather data: NOAA Global Forecast System (GFS)
