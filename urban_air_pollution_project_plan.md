# Urban Air Pollution Challenge — Full ML Project Plan

## Project Title

**Predicting Urban PM2.5 Air Pollution from Weather and Satellite Data**

## Main Project Story

We build a machine-learning model that predicts daily **PM2.5 air-pollution levels** using weather and Sentinel-5P satellite pollutant observations. The long-term data-product idea is a **mobile app** that warns users, schools, clinics, and city authorities about high-pollution days, especially in African cities where ground-based sensors are limited.

---

# 1. Project Objective

The objective of the Urban Air Pollution Challenge is to predict **daily PM2.5 particulate matter concentration** for each city/location.

PM2.5 means particulate matter with a diameter smaller than **2.5 micrometers**. It is a dangerous air pollutant because the particles are small enough to enter deep into the lungs.

This is a **supervised regression problem**.

The model learns:

```text
weather + satellite pollutants + date/context features -> PM2.5
```

Mathematically:

```text
PM2.5_hat = f(weather, satellite pollutants, date, atmospheric context)
```

where:

- `PM2.5_hat` = predicted PM2.5 concentration,
- `f(...)` = machine-learning model,
- features = weather, pollutants, clouds, aerosols, date information.

---

# 2. Course Deliverables

The machine-learning project requires:

1. **Slide deck PDF**  
   - For non-technical stakeholders.
   - 10-minute presentation.

2. **Jupyter notebook**  
   - For technical/data-science audience.
   - Should be clean, readable, and follow PEP8.

3. **Optional Python script**  
   - Train model from terminal.
   - Run model on test data.
   - Save predictions.

4. **One slide about a potential data product**  
   - Explain how the predictions could be used in practice.

5. **Modeling requirements**
   - Try at least **three machine-learning algorithms**.
   - Use cross-validation.
   - Use hyperparameter tuning.
   - Discuss the right performance metric.
   - Check data imbalance/skewness.

---

# 3. Business Case

## Product Idea

A practical business case is:

> A mobile app that predicts daily PM2.5 air-pollution risk for a user's location and gives simple health recommendations.

Possible product names:

- **AirAware Africa**
- **CleanAir Alert**
- **Urban Air Monitor**
- **PM2.5 Risk App**

## Why This Is Useful in Africa

Many African cities have limited air-quality monitoring infrastructure. Expensive ground-based sensor networks are difficult to install and maintain everywhere.

A practical app could combine:

- satellite data,
- weather data,
- limited ground-sensor data,
- machine-learning predictions,

to provide low-cost air-quality warnings.

## Target Users

| User group | Practical use |
|---|---|
| Citizens | Know whether it is safe to exercise outdoors |
| Parents | Protect children on bad air-quality days |
| Schools | Decide whether to reduce outdoor activity |
| Clinics/hospitals | Prepare for respiratory-risk days |
| City authorities | Identify hotspots and plan interventions |
| NGOs | Target clean-air programs |
| Environmental agencies | Monitor pollution without dense sensors |

## App Workflow

```text
User opens app
-> app gets location/date
-> backend retrieves weather and satellite features
-> ML model predicts PM2.5
-> app shows risk level and advice
```

## Example App Output

| Predicted PM2.5 | Risk level | App recommendation |
|---:|---|---|
| 0-12 | Good | Normal outdoor activity |
| 12-35 | Moderate | Sensitive people should be careful |
| 35-55 | Unhealthy for sensitive groups | Reduce outdoor activity |
| 55-150 | Unhealthy | Avoid long outdoor exercise |
| >150 | Very unhealthy | Stay indoors if possible |

---

# 4. Domain Knowledge

## 4.1 What Is PM2.5?

PM2.5 is not one chemical. It is a **size category**.

```text
PM2.5 = tiny particles smaller than 2.5 micrometers
```

PM2.5 can include:

- soot,
- smoke,
- dust,
- sulfate particles,
- nitrate particles,
- organic particles,
- sea salt,
- industrial particles.

## 4.2 Primary vs Secondary PM2.5

### Primary PM2.5

Emitted directly as particles.

Examples:

```text
diesel exhaust -> soot particles -> PM2.5
wood burning -> smoke particles -> PM2.5
dust storm -> dust particles -> PM2.5
```

### Secondary PM2.5

Formed later in the atmosphere from gases.

Examples:

```text
NO2 -> nitrate particles -> PM2.5
SO2 -> sulfate particles -> PM2.5
VOCs -> organic aerosol particles -> PM2.5
```

## 4.3 Relationship Between Pollutants and PM2.5

| Pollutant/feature | Type | Relation to PM2.5 |
|---|---|---|
| PM2.5 | tiny particles | target variable |
| NO2 | gas | traffic/combustion signal; can form nitrate particles |
| CO | gas | combustion/fire indicator |
| CO2 | gas | greenhouse gas; not direct PM2.5 precursor |
| SO2 | gas | industry/fuel signal; can form sulfate particles |
| O3 | gas | photochemical air-pollution indicator |
| HCHO | gas | VOC chemistry; related to secondary organic aerosols |
| CH4 | gas | mostly greenhouse gas; usually indirect |
| AER_AI | aerosol index | dust/smoke/aerosol indicator |

## 4.4 Main Sources of PM2.5 in Africa

The dominant sources vary by region, city, season, and climate zone.

Important sources include:

| Source | Why important |
|---|---|
| Household solid fuels | wood, charcoal, coal, dung, crop residues for cooking/heating |
| Desert dust | especially North Africa, West Africa, Sahel |
| Traffic | old diesel vehicles, buses, trucks, minibuses |
| Open waste burning | common in many urban areas |
| Industry and generators | factories, power generation, diesel generators |
| Fires/agricultural burning | seasonal smoke and biomass burning |

Simple domain summary:

```text
PM2.5 = emissions + atmospheric formation - dispersion/removal
```

Where:

- emissions = traffic, industry, burning, dust,
- atmospheric formation = chemical reactions from gases,
- dispersion/removal = wind, rain, mixing.

---

# 5. Dataset Overview

The dataset contains three main sources.

## 5.1 Ground-Based Air Quality Sensors

These provide the PM2.5 target in the training set.

Columns:

```text
target
target_min
target_max
target_variance
target_count
```

Meaning:

| Column | Meaning |
|---|---|
| `target` | daily mean PM2.5 concentration |
| `target_min` | minimum PM2.5 reading that day |
| `target_max` | maximum PM2.5 reading that day |
| `target_variance` | variability of PM2.5 readings that day |
| `target_count` | number of readings used |

Important:

> `target_min`, `target_max`, `target_variance`, and `target_count` should **not** be used in the final model because they are target-related and not available in the test set.

## 5.2 Weather Data

Weather data comes from the Global Forecast System.

Important columns:

```text
precipitable_water_entire_atmosphere
relative_humidity_2m_above_ground
specific_humidity_2m_above_ground
temperature_2m_above_ground
u_component_of_wind_10m_above_ground
v_component_of_wind_10m_above_ground
```

## 5.3 Sentinel-5P Satellite Data

Sentinel-5P provides atmospheric pollutant measurements.

Pollutant groups:

```text
L3_NO2
L3_O3
L3_CO
L3_HCHO
L3_CLOUD
L3_AER_AI
L3_SO2
L3_CH4
```

The most important satellite features are usually:

```text
column_number_density
tropospheric_X_column_number_density
absorbing_aerosol_index
cloud_fraction
```

---

# 6. Explanation of Main Column Types

## 6.1 ID Columns

| Column | Meaning | Use |
|---|---|---|
| `Place_ID X Date` | unique row ID | use for submission only |
| `Date` | observation date | convert to date features |
| `Place_ID` | anonymous location ID | use for EDA and GroupKFold, not as model feature |

## 6.2 Weather Columns

| Column | Meaning | PM2.5 relevance |
|---|---|---|
| `precipitable_water_entire_atmosphere` | total water vapor in atmospheric column | moisture/cloud context |
| `relative_humidity_2m_above_ground` | relative humidity near surface | particles can absorb water and grow |
| `specific_humidity_2m_above_ground` | actual water vapor content | particle growth and atmospheric moisture |
| `temperature_2m_above_ground` | near-surface temperature | chemistry and atmospheric stability |
| `u_component_of_wind_10m_above_ground` | east-west wind component | used for wind speed/direction |
| `v_component_of_wind_10m_above_ground` | north-south wind component | used for wind speed/direction |

Wind speed:

```text
wind_speed = sqrt(u^2 + v^2)
```

## 6.3 Satellite Measurement Terms

| Term | Meaning |
|---|---|
| `column_number_density` | amount of pollutant in vertical air column |
| `tropospheric_X_column_number_density` | amount of pollutant in lower atmosphere |
| `slant_column_number_density` | amount measured along satellite viewing path |
| `cloud_fraction` | fraction of satellite pixel covered by clouds |
| `sensor_altitude` | satellite altitude |
| `sensor_azimuth_angle` | satellite viewing direction |
| `sensor_zenith_angle` | viewing angle away from vertical |
| `solar_azimuth_angle` | sun direction |
| `solar_zenith_angle` | sun angle away from vertical |
| `amf` | air mass factor used in satellite retrieval correction |

## 6.4 Important Pollutant Columns

### NO2

Important columns:

```text
L3_NO2_NO2_column_number_density
L3_NO2_tropospheric_NO2_column_number_density
L3_NO2_absorbing_aerosol_index
L3_NO2_cloud_fraction
```

Meaning:

```text
NO2 -> traffic/combustion signal -> possible secondary PM2.5
```

### O3

Important columns:

```text
L3_O3_O3_column_number_density
L3_O3_cloud_fraction
```

Meaning:

```text
O3 -> photochemical air-pollution indicator
```

### CO

Important columns:

```text
L3_CO_CO_column_number_density
L3_CO_H2O_column_number_density
L3_CO_cloud_height
```

Meaning:

```text
CO -> combustion/fire/biomass-burning indicator
```

### HCHO

Important columns:

```text
L3_HCHO_tropospheric_HCHO_column_number_density
L3_HCHO_cloud_fraction
```

Meaning:

```text
HCHO -> VOC chemistry -> secondary organic aerosol potential
```

### AER_AI

Important column:

```text
L3_AER_AI_absorbing_aerosol_index
```

Meaning:

```text
AER_AI -> dust/smoke/aerosol signal
```

This is likely very relevant for PM2.5.

### SO2

Important columns:

```text
L3_SO2_SO2_column_number_density
L3_SO2_absorbing_aerosol_index
L3_SO2_cloud_fraction
```

Meaning:

```text
SO2 -> industry/fuel burning -> sulfate particles -> PM2.5
```

### CH4

Important columns:

```text
L3_CH4_CH4_column_volume_mixing_ratio_dry_air
L3_CH4_aerosol_height
L3_CH4_aerosol_optical_depth
```

Meaning:

```text
CH4 -> mostly greenhouse gas, weak direct relation to PM2.5
```

Keep CH4 only if missingness is acceptable.

---

# 7. Final Feature Selection Strategy

## 7.1 Final Verdict About Location

Use `Place_ID` for validation grouping, but not as a model feature.

Final decision:

```text
Do NOT include raw Place_ID as a model feature.
DO use GroupKFold(groups=train["Place_ID"]) for validation.
```

Why?

- `Place_ID` is an anonymous label.
- It does not contain real geography.
- The test set contains different locations from the training set.
- If included as a feature, the model may memorize location-specific PM2.5 levels that do not generalize.

## 7.2 Columns to Exclude

```python
exclude_cols = [
    "Place_ID X Date",
    "Date",
    "Place_ID",
    "target",
    "target_min",
    "target_max",
    "target_variance",
    "target_count",
]
```

## 7.3 Weather Features

```python
weather_features = [
    "precipitable_water_entire_atmosphere",
    "relative_humidity_2m_above_ground",
    "specific_humidity_2m_above_ground",
    "temperature_2m_above_ground",
    "u_component_of_wind_10m_above_ground",
    "v_component_of_wind_10m_above_ground",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
]
```

## 7.4 Date Features

```python
date_features = [
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "dayofyear_sin",
    "dayofyear_cos",
]
```

## 7.5 Main Satellite Pollutant Features

```python
pollutant_features = [
    "L3_NO2_NO2_column_number_density",
    "L3_NO2_tropospheric_NO2_column_number_density",
    "L3_NO2_absorbing_aerosol_index",

    "L3_O3_O3_column_number_density",

    "L3_CO_CO_column_number_density",
    "L3_CO_H2O_column_number_density",

    "L3_HCHO_tropospheric_HCHO_column_number_density",

    "L3_AER_AI_absorbing_aerosol_index",

    "L3_SO2_SO2_column_number_density",
    "L3_SO2_absorbing_aerosol_index",
]
```

## 7.6 Cloud and Context Features

```python
cloud_features = [
    "L3_NO2_cloud_fraction",
    "L3_O3_cloud_fraction",
    "L3_HCHO_cloud_fraction",
    "L3_SO2_cloud_fraction",

    "L3_CLOUD_cloud_base_height",
    "L3_CLOUD_cloud_base_pressure",
    "L3_CLOUD_cloud_fraction",
    "L3_CLOUD_cloud_optical_depth",
    "L3_CLOUD_cloud_top_height",
    "L3_CLOUD_cloud_top_pressure",
    "L3_CLOUD_surface_albedo",
]
```

## 7.7 Optional CH4 Features

```python
ch4_features = [
    "L3_CH4_CH4_column_volume_mixing_ratio_dry_air",
    "L3_CH4_aerosol_height",
    "L3_CH4_aerosol_optical_depth",
]
```

Drop CH4 columns if they have too much missing data.

## 7.8 Candidate Features

```python
candidate_features = (
    weather_features
    + date_features
    + pollutant_features
    + cloud_features
    + ch4_features
)
```

Then remove high-missing columns:

```python
missing_ratio = train[candidate_features].isna().mean()
features = missing_ratio[missing_ratio < 0.80].index.tolist()
```

---

# 8. Validation Strategy

Use:

```python
from sklearn.model_selection import GroupKFold

groups = train["Place_ID"]
cv = GroupKFold(n_splits=5)
```

Meaning:

> All rows from the same location stay together in either training or validation.

Example:

| Fold | Training locations | Validation locations |
|---|---|---|
| 1 | A, B, C, D | E |
| 2 | A, B, C, E | D |
| 3 | A, B, D, E | C |
| 4 | A, C, D, E | B |
| 5 | B, C, D, E | A |

This tests:

```text
Can the model predict PM2.5 for unseen locations?
```

This is important because the challenge test locations are different from the training locations.

---

# 9. Data Loading

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sample_submission = pd.read_csv("data/SampleSubmission.csv")
```

Initial checks:

```python
print(train.shape)
print(test.shape)

train.head()
test.head()

train.info()
test.info()

train["target"].describe()
```

Missing values:

```python
missing_train = train.isna().mean().sort_values(ascending=False)
missing_test = test.isna().mean().sort_values(ascending=False)

missing_train.head(20)
missing_test.head(20)
```

---

# 10. Exploratory Data Analysis

## 10.1 Target Distribution

```python
import matplotlib.pyplot as plt

train["target"].hist(bins=50)
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.title("Distribution of PM2.5 target")
plt.show()
```

Check skewness:

```python
train["target"].skew()
```

## 10.2 PM2.5 Over Time

```python
train["Date"] = pd.to_datetime(train["Date"])

daily_pm25 = train.groupby("Date")["target"].mean()

daily_pm25.plot(figsize=(12, 4))
plt.ylabel("Mean PM2.5")
plt.title("Average PM2.5 over time")
plt.show()
```

## 10.3 PM2.5 by Location

```python
place_stats = train.groupby("Place_ID")["target"].agg(
    ["mean", "std", "min", "max", "count"]
).sort_values("mean", ascending=False)

place_stats.head(10)
```

## 10.4 Correlation With Important Features

```python
important_cols = [
    "target",
    "relative_humidity_2m_above_ground",
    "specific_humidity_2m_above_ground",
    "temperature_2m_above_ground",
    "precipitable_water_entire_atmosphere",
    "L3_NO2_tropospheric_NO2_column_number_density",
    "L3_NO2_NO2_column_number_density",
    "L3_CO_CO_column_number_density",
    "L3_SO2_SO2_column_number_density",
    "L3_O3_O3_column_number_density",
    "L3_HCHO_tropospheric_HCHO_column_number_density",
    "L3_AER_AI_absorbing_aerosol_index",
]

train[important_cols].corr()["target"].sort_values(ascending=False)
```

---

# 11. Feature Engineering

## 11.1 Date Features

```python
def add_date_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df

train = add_date_features(train)
test = add_date_features(test)
```

## 11.2 Wind Features

```python
def add_wind_features(df):
    df = df.copy()

    u = df["u_component_of_wind_10m_above_ground"]
    v = df["v_component_of_wind_10m_above_ground"]

    df["wind_speed"] = np.sqrt(u**2 + v**2)
    df["wind_direction"] = np.arctan2(v, u)

    df["wind_dir_sin"] = np.sin(df["wind_direction"])
    df["wind_dir_cos"] = np.cos(df["wind_direction"])

    return df

train = add_wind_features(train)
test = add_wind_features(test)
```

## 11.3 Missingness Indicators

```python
def add_missing_indicators(train_df, test_df, cols, threshold=0.05):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cols:
        if train_df[col].isna().mean() > threshold or test_df[col].isna().mean() > threshold:
            train_df[col + "_missing"] = train_df[col].isna().astype(int)
            test_df[col + "_missing"] = test_df[col].isna().astype(int)

    return train_df, test_df
```

---

# 12. Preprocessing Pipeline

For tree-based models:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

numeric_features = features

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)
```

For linear models, use scaling:

```python
from sklearn.preprocessing import StandardScaler

linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)
```

---

# 13. Models to Try

The project requires at least three models. Recommended:

1. Dummy baseline
2. Ridge Regression
3. Random Forest Regressor
4. HistGradientBoostingRegressor

## 13.1 Dummy Baseline

```python
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
```

Purpose:

> Any useful model should beat the dummy baseline.

## 13.2 Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
```

Purpose:

> Simple interpretable linear baseline.

## 13.3 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

Purpose:

> Nonlinear model that captures interactions.

## 13.4 HistGradientBoosting

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    random_state=42
)
```

Purpose:

> Strong tabular regression model using scikit-learn only.

---

# 14. Evaluation Metric

Use RMSE:

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Code:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

RMSE is appropriate because:

- the competition uses RMSE,
- large errors on dangerous high-pollution days are strongly penalized.

---

# 15. Cross-Validation Function

```python
from sklearn.model_selection import GroupKFold

def cross_validate_model(model, X, y, groups, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        fold_rmse = rmse(y_val, preds)
        scores.append(fold_rmse)

        print(f"Fold {fold}: RMSE = {fold_rmse:.4f}")

    print(f"Mean RMSE: {np.mean(scores):.4f}")
    print(f"Std RMSE:  {np.std(scores):.4f}")

    return scores
```

---

# 16. Model Pipelines

```python
from sklearn.pipeline import Pipeline

dummy_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="mean")),
    ]
)

ridge_pipe = Pipeline(
    steps=[
        ("preprocessor", linear_preprocessor),
        ("model", Ridge(alpha=1.0)),
    ]
)

rf_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

hgb_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            random_state=42
        )),
    ]
)
```

---

# 17. Model Comparison

```python
X = train[features]
y = train["target"]
groups = train["Place_ID"]

models = {
    "Dummy Mean": dummy_pipe,
    "Ridge Regression": ridge_pipe,
    "Random Forest": rf_pipe,
    "HistGradientBoosting": hgb_pipe,
}

results = []

for name, model in models.items():
    print(f"\n{name}")
    scores = cross_validate_model(model, X, y, groups)
    results.append(
        {
            "model": name,
            "mean_rmse": np.mean(scores),
            "std_rmse": np.std(scores),
        }
    )

results_df = pd.DataFrame(results).sort_values("mean_rmse")
results_df
```

---

# 18. Hyperparameter Tuning

Tune the best two models only.

## 18.1 Random Forest Grid

```python
rf_param_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [10, 20, None],
    "model__min_samples_leaf": [1, 3, 5],
}
```

## 18.2 HistGradientBoosting Grid

```python
hgb_param_grid = {
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_iter": [300, 500, 800],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}
```

Use `RandomizedSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV

cv = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=hgb_pipe,
    param_distributions=hgb_param_grid,
    n_iter=15,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
)

search.fit(X, y, groups=groups)

print(search.best_score_)
print(search.best_params_)
```

Important:

```text
Always pass groups=groups during tuning.
```

---

# 19. Optional Log-Target Experiment

If PM2.5 is highly skewed, test:

```text
target_log = log(1 + target)
```

Using:

```python
from sklearn.compose import TransformedTargetRegressor

log_model = TransformedTargetRegressor(
    regressor=hgb_pipe,
    func=np.log1p,
    inverse_func=np.expm1,
)
```

Evaluate with GroupKFold.

Use log-target only if it improves RMSE.

---

# 20. Final Model Training

After choosing the best model:

```python
final_model = search.best_estimator_
final_model.fit(X, y)
```

Train on the full training data only after model selection is complete.

---

# 21. Test Prediction and Submission

```python
X_test = test[features]

test_preds = final_model.predict(X_test)

# PM2.5 cannot be negative
test_preds = np.clip(test_preds, 0, None)
```

Create submission:

```python
submission = sample_submission.copy()
submission["target"] = test_preds

submission.to_csv("submissions/submission.csv", index=False)
```

Check:

```python
submission.head()
submission.shape
submission["target"].describe()
```

---

# 22. Feature Importance

For tree-based models with `feature_importances_`:

```python
importances = final_model.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame(
    {
        "feature": features,
        "importance": importances,
    }
).sort_values("importance", ascending=False)

feature_importance.head(20)
```

For models without `feature_importances_`, use permutation importance:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    final_model,
    X,
    y,
    n_repeats=5,
    random_state=42,
    scoring="neg_root_mean_squared_error",
)

importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": result.importances_mean,
    }
).sort_values("importance", ascending=False)

importance_df.head(20)
```

Interpret likely important features:

| Feature | Expected meaning |
|---|---|
| `L3_AER_AI_absorbing_aerosol_index` | dust/smoke/aerosols |
| `L3_NO2_tropospheric_NO2_column_number_density` | traffic/combustion |
| `L3_CO_CO_column_number_density` | fires/combustion |
| `wind_speed` | dispersion |
| `relative_humidity_2m_above_ground` | particle growth |
| `L3_SO2_SO2_column_number_density` | industry/sulfate precursor |

---

# 23. Error Analysis

Generate out-of-fold predictions:

```python
from sklearn.model_selection import cross_val_predict

oof_preds = cross_val_predict(
    final_model,
    X,
    y,
    groups=groups,
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
)

train["oof_pred"] = oof_preds
train["error"] = train["target"] - train["oof_pred"]
train["abs_error"] = train["error"].abs()
```

Worst errors:

```python
train.sort_values("abs_error", ascending=False)[
    ["Place_ID", "Date", "target", "oof_pred", "error"]
].head(20)
```

Predicted vs actual:

```python
plt.scatter(train["target"], train["oof_pred"], alpha=0.3)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.show()
```

Residual plot:

```python
plt.scatter(train["oof_pred"], train["error"], alpha=0.3)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Error")
plt.title("Residuals")
plt.show()
```

Possible conclusion:

> The model may underpredict extreme pollution events because very high PM2.5 days are rare and may be caused by local events not fully captured by the available weather and satellite features.

---

# 24. Notebook Structure

Use this structure:

```text
1. Introduction and problem statement
2. Dataset overview
3. Domain knowledge: PM2.5 and pollution sources
4. Data loading
5. Initial inspection
6. Exploratory data analysis
7. Feature engineering
8. Feature selection
9. Validation strategy
10. Baseline model
11. Model comparison
12. Hyperparameter tuning
13. Final model
14. Test prediction and submission
15. Feature importance
16. Error analysis
17. Data product idea
18. Conclusion and future work
```

---

# 25. Slide Deck Structure

Use around 10-11 slides.

## Slide 1 — Title

**Predicting Urban PM2.5 Air Pollution from Weather and Satellite Data**

## Slide 2 — Problem

Air pollution is harmful, but many cities lack dense ground-sensor networks.

## Slide 3 — Objective

Predict daily PM2.5 for each location.

```text
Inputs: weather + satellite pollutants
Output: PM2.5
```

## Slide 4 — Data

Three data sources:

- ground-based PM2.5 sensors,
- GFS weather data,
- Sentinel-5P satellite pollutant data.

## Slide 5 — Domain Knowledge

```text
NO2 -> traffic/combustion
CO -> combustion/fires
SO2 -> industry/fuel burning
AER_AI -> dust/smoke
weather -> dispersion and particle growth
```

## Slide 6 — ML Workflow

```text
Raw data
-> cleaning
-> feature engineering
-> GroupKFold validation
-> model comparison
-> tuning
-> final prediction
```

## Slide 7 — Validation Strategy

Key message:

> Because test locations are different from training locations, we used GroupKFold by `Place_ID` to evaluate performance on unseen locations.

## Slide 8 — Model Comparison

Example table:

| Model | RMSE |
|---|---:|
| Dummy baseline | ... |
| Ridge Regression | ... |
| Random Forest | ... |
| HistGradientBoosting | ... |

## Slide 9 — Results and Interpretation

Show:

- best model,
- best RMSE,
- important features,
- one example prediction.

## Slide 10 — Data Product: Mobile App

Show:

```text
AirAware Africa
Daily PM2.5 prediction
Risk level
Health recommendation
```

## Slide 11 — Limitations and Future Work

Mention:

- no latitude/longitude in dataset,
- test locations are unseen,
- satellite data has missing values,
- external data not allowed for challenge,
- future app could use GPS, traffic, land use, and low-cost sensors.

---

# 26. GitHub Repository Structure

Recommended:

```text
urban-air-pollution/
│
├── notebooks/
│   └── urban_air_pollution_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── train_model.py
│
├── reports/
│   └── presentation.pdf
│
├── submissions/
│   └── submission.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

Important:

> Do not upload the Zindi data publicly to GitHub.

`.gitignore`:

```text
data/
*.csv
*.zip
models/
submissions/
__pycache__/
.ipynb_checkpoints/
```

---

# 27. Final Technical Choices

| Decision | Final choice |
|---|---|
| Problem type | Regression |
| Target | `target` = daily PM2.5 |
| Metric | RMSE |
| Main validation | `GroupKFold(groups=train["Place_ID"])` |
| Use raw `Place_ID` as feature? | No |
| Use date? | Yes, transformed into date features |
| Use target min/max/variance/count? | No |
| Missing values | median imputation + missing indicators |
| Main models | Dummy, Ridge, Random Forest, HistGradientBoosting |
| Final likely model | HistGradientBoosting or Random Forest |
| Feature focus | weather + pollutant column densities + aerosol/cloud context |
| Business case | PM2.5 mobile warning app for African cities |
| GitHub data upload | No, data must stay private |

---

# 28. Minimum Viable Project

To finish safely, complete these steps first:

1. Load train/test/sample.
2. Create date features.
3. Create wind features.
4. Drop leakage columns.
5. Select weather + pollutant + aerosol + cloud features.
6. Median impute missing values.
7. Use `GroupKFold(groups=train["Place_ID"])`.
8. Train:
   - Dummy baseline,
   - Ridge Regression,
   - Random Forest,
   - HistGradientBoosting.
9. Compare RMSE.
10. Choose best model.
11. Train final model on all training data.
12. Predict test set.
13. Create submission.
14. Prepare stakeholder slides.
15. Include mobile-app product slide.

---

# 29. Stronger Version If Time Allows

Add:

- missingness indicators,
- log-target experiment,
- hyperparameter tuning,
- feature importance,
- error analysis,
- mobile-app mockup slide,
- stakeholder recommendations.

---

# 30. Final A-to-Z Checklist

```text
A. Create GitHub repo
B. Add .gitignore so data is not uploaded
C. Load Train.csv, Test.csv, SampleSubmission.csv
D. Understand target: PM2.5 regression
E. Check shapes, columns, missing values
F. Analyze target distribution
G. Analyze PM2.5 by date
H. Analyze PM2.5 by Place_ID
I. Explain domain knowledge
J. Create date features
K. Create wind speed and wind direction features
L. Remove leakage columns
M. Select weather, pollutant, aerosol, and cloud features
N. Remove columns with very high missingness
O. Add missingness indicators
P. Build preprocessing pipeline
Q. Use GroupKFold with Place_ID
R. Train dummy baseline
S. Train Ridge Regression
T. Train Random Forest
U. Train HistGradientBoosting
V. Compare RMSE
W. Tune best model
X. Train final model on all training data
Y. Predict test set and create submission
Z. Prepare stakeholder slides and mobile-app product slide
```

---

# 31. Important Paragraph for the Notebook

Use this paragraph in your notebook:

> This project predicts daily PM2.5 concentration using weather and Sentinel-5P satellite pollutant data. Because the test set contains locations that are different from the training locations, raw `Place_ID` was not used as a model feature. Instead, `Place_ID` was used for GroupKFold cross-validation, ensuring that all observations from the same location stay in the same fold. This gives a more realistic estimate of how well the model generalizes to unseen locations. The final model uses weather variables, engineered wind/date features, pollutant column densities, aerosol indicators, and cloud-context features to predict PM2.5 with RMSE as the evaluation metric.

---

# 32. Final Presentation Message

Use this as your closing statement:

> Our model is not only a prediction exercise. It demonstrates how satellite data, weather data, and machine learning can support low-cost PM2.5 air-quality monitoring in cities with limited ground sensors. The proposed mobile app turns model predictions into practical health-risk information for citizens, schools, clinics, NGOs, and city authorities.

