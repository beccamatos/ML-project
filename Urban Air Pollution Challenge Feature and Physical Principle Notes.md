# Urban Air Pollution Challenge — Feature and Physical Principle Notes

## 1. Purpose of This Document

This document focuses on the dataset features, their physical meaning, possible relationships with PM2.5, and feature engineering ideas.

It does not define the business case or project goal. Those are documented separately.

The main question here is:

```text
Which dataset features are scientifically plausible predictors of PM2.5, and how can they be transformed into useful model inputs?
```

---

# 2. Dataset Feature Groups

The dataset contains several types of features:

```text
1. ID and time features
2. Ground sensor target features
3. Weather features
4. Sentinel-5P satellite pollutant features
5. Satellite measurement quality features
6. Engineered physical proxy features
7. Location behavior cluster features
```

The important modeling principle is:

```text
Use features that describe physical or environmental conditions.
Avoid features that only memorize location identity or leak target information.
```

---

# 3. ID and Time Features

## `Place_ID X Date`

**Meaning:**
A unique identifier combining an anonymized location and a date.

```text
Place_ID X Date = anonymized location + day
```

**Use:**
Useful as an identifier for each prediction row.

**Modeling note:**
This column should not be used as a physical model feature.

---

## `Place_ID`

**Meaning:**
An anonymized location identifier.

**Potential issue:**
Using raw `Place_ID` directly can cause memorization.

The model may learn:

```text
Place_ID → typical PM2.5 level
```

instead of:

```text
environmental conditions → PM2.5 risk
```

**Recommended use:**
Use `Place_ID` only to build aggregated environmental profiles and location behavior clusters.

---

## `Date`

**Meaning:**
Date of the observation or prediction.

**Physical relevance:**
Date can represent seasonal and temporal patterns:

```text
dry season
rainy season
dust season
biomass burning periods
weekday / weekend activity patterns
seasonal weather cycles
```

**Feature engineering:**

```python
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek
df["dayofyear"] = df["Date"].dt.dayofyear
```

**Cyclical features:**

```python
df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
```

**Why cyclical features matter:**
They represent seasonal cycles smoothly. For example, December and January are close in time, even though their numeric month values are far apart.

---

# 4. Target and Ground Sensor Features

## `target`

**Meaning:**
Daily mean PM2.5 value.

**Role:**
This is the target variable to predict.

---

## `target_min`

**Meaning:**
Minimum PM2.5 value measured on that day.

**Use:**
Useful for EDA.

**Modeling warning:**
Should not be used as a model feature if unavailable in the test set.

---

## `target_max`

**Meaning:**
Maximum PM2.5 value measured on that day.

**Use:**
Useful for identifying peak pollution days during EDA.

**Modeling warning:**
Should not be used as a model feature if unavailable in the test set.

---

## `target_variance`

**Meaning:**
Variance of PM2.5 sensor readings during the day.

**Use:**
Useful for understanding whether PM2.5 was stable or had strong fluctuations.

**Modeling warning:**
Can create target leakage if used for prediction.

---

## `target_count`

**Meaning:**
Number of sensor readings used to calculate the daily PM2.5 value.

**Use:**
Useful as a measurement-quality indicator in EDA.

**Modeling warning:**
Should not be used for final prediction if unavailable at prediction time.

---

## Leakage Summary

```text
target_min
target_max
target_variance
target_count
```

These features are target-derived. They can be useful for understanding the data, but they should not be used for model training if they are not available in the test set or real prediction scenario.

---

# 5. Weather Features and Physical Meaning

Weather features are important because PM2.5 is affected by particle dispersion, accumulation, chemical transformation, aerosol growth and washout.

---

## Temperature

Column pattern:

```text
temperature_2m_above_ground
```

**Physical meaning:**
Air temperature near the surface.

**Possible relationship with PM2.5:**
Temperature can influence:

```text
atmospheric chemistry
photochemical reactions
vertical mixing
dryness
seasonal patterns
ozone formation
secondary aerosol formation
```

**Modeling interpretation:**
The relationship is often indirect and non-linear.

**Possible engineered features:**

```python
df["temp_squared"] = df["temperature_2m_above_ground"] ** 2

df["hot_day"] = (
    df["temperature_2m_above_ground"] >
    df["temperature_2m_above_ground"].quantile(0.75)
).astype(int)
```

---

## Relative Humidity

Column pattern:

```text
relative_humidity_2m_above_ground
```

**Physical meaning:**
Percentage of water vapor in the air relative to the maximum possible amount at that temperature.

**Possible relationship with PM2.5:**
Humidity can influence:

```text
aerosol growth
haze formation
secondary particle formation
particle water uptake
```

**Important nuance:**
High humidity can increase aerosol growth, but if it leads to rainfall, PM2.5 may decrease through washout.

**Possible engineered feature:**

```python
df["humid_day"] = (
    df["relative_humidity_2m_above_ground"] >
    df["relative_humidity_2m_above_ground"].quantile(0.75)
).astype(int)
```

---

## Specific Humidity

Column pattern:

```text
specific_humidity_2m_above_ground
```

**Physical meaning:**
Actual mass of water vapor per mass of air.

**Possible relationship with PM2.5:**
It can capture moisture conditions more directly than relative humidity in some cases.

**Use:**
Useful together with temperature and relative humidity.

---

## Precipitable Water

Column pattern:

```text
precipitable_water_entire_atmosphere
```

**Physical meaning:**
Total water vapor contained in the atmospheric column.

**Possible relationship with PM2.5:**
An indirect indicator of humid weather systems.

**Important nuance:**
It does not necessarily mean it is raining.

---

## Precipitation

Column pattern:

```text
total_precipitation_surface
```

**Physical meaning:**
Rainfall or precipitation at the surface.

**Possible relationship with PM2.5:**
Precipitation can remove particles from the air.

```text
precipitation → particle washout → lower PM2.5
```

**Possible engineered features:**

```python
df["rain_flag"] = (df["total_precipitation_surface"] > 0).astype(int)

df["washout_proxy"] = (
    df["total_precipitation_surface"] *
    df["wind_speed_10m"]
)
```

---

## Wind Components

Column patterns:

```text
u_component_of_wind_10m_above_ground
v_component_of_wind_10m_above_ground
```

**Physical meaning:**
Horizontal wind components at 10 meters above ground.

* `u_component`: east-west wind component
* `v_component`: north-south wind component

**Possible relationship with PM2.5:**
Wind affects pollution through:

```text
dispersion
dilution
transport
ventilation
accumulation under low-wind conditions
```

**Recommended engineered feature:**

```python
df["wind_speed_10m"] = np.sqrt(
    df["u_component_of_wind_10m_above_ground"]**2 +
    df["v_component_of_wind_10m_above_ground"]**2
)
```

**Possible interpretation:**

```text
low wind speed → PM2.5 can accumulate
high wind speed → PM2.5 can disperse or be transported
```

**Optional flag:**

```python
df["low_wind_day"] = (
    df["wind_speed_10m"] <
    df["wind_speed_10m"].quantile(0.25)
).astype(int)
```

---

## Cloud Cover

Column pattern:

```text
total_cloud_cover_entire_atmosphere
```

**Physical meaning:**
Total cloud cover in the atmospheric column.

**Possible relationship with PM2.5:**
Cloud cover can influence:

```text
radiation
temperature
atmospheric stability
humidity
satellite measurement quality
```

**Modeling interpretation:**
Clouds are not a direct pollution source, but they can affect both weather patterns and satellite data reliability.

---

# 6. Sentinel-5P Satellite Pollutant Features

Sentinel-5P pollutant features describe atmospheric column values. They are not identical to ground-level PM2.5, but they can be useful proxies for pollution sources and atmospheric conditions.

Common dataset prefixes:

```text
L3_NO2
L3_CO
L3_SO2
L3_O3
L3_HCHO
L3_CH4
L3_AER_AI
L3_CLOUD
```

---

## NO2 Features

Prefix:

```text
L3_NO2_...
```

**Meaning:**
Nitrogen dioxide.

**Typical sources:**

```text
traffic
combustion
diesel engines
industry
power generation
```

**Possible relationship with PM2.5:**
NO2 is a proxy for combustion-related activity. It can indicate areas or days with high urban emissions.

**Relevant variants:**

```text
NO2_column_number_density
tropospheric_NO2_column_number_density
```

**Interpretation:**

```text
high NO2 → combustion activity → potentially higher PM2.5
```

---

## CO Features

Prefix:

```text
L3_CO_...
```

**Meaning:**
Carbon monoxide.

**Typical sources:**

```text
incomplete combustion
traffic
fires
biomass burning
household fuels
diesel generators
```

**Possible relationship with PM2.5:**
CO can be a proxy for smoke and combustion emissions.

**Interpretation:**

```text
high CO → smoke / combustion → potentially higher PM2.5
```

---

## SO2 Features

Prefix:

```text
L3_SO2_...
```

**Meaning:**
Sulfur dioxide.

**Typical sources:**

```text
coal combustion
heavy industry
sulfur-containing fuels
power generation
volcanic activity
```

**Possible relationship with PM2.5:**
SO2 can contribute to secondary sulfate aerosols, which are part of PM2.5.

**Interpretation:**

```text
high SO2 → industrial emissions / sulfate aerosol formation → potentially higher PM2.5
```

---

## O3 Features

Prefix:

```text
L3_O3_...
```

**Meaning:**
Ozone.

**Possible relationship with PM2.5:**
Ozone indicates photochemical air pollution. It is not PM2.5, but it can be related to atmospheric chemistry.

**Important nuance:**
Ozone and PM2.5 do not always increase together. The relationship can be complex.

---

## HCHO Features

Prefix:

```text
L3_HCHO_...
```

**Meaning:**
Formaldehyde.

**Possible interpretation:**

```text
VOC proxy
biomass burning signal
photochemical activity indicator
```

**Possible relationship with PM2.5:**
HCHO can indicate chemical processes and emissions that may contribute to secondary aerosol formation.

---

## CH4 Features

Prefix:

```text
L3_CH4_...
```

**Meaning:**
Methane.

**Possible relationship with PM2.5:**
Methane is more relevant for climate and emissions context than short-term PM2.5 prediction.

**Modeling note:**
Likely less directly useful than NO2, CO, SO2 or Aerosol Index.

---

## Aerosol Index Features

Prefix:

```text
L3_AER_AI_...
```

Key column pattern:

```text
absorbing_aerosol_index
```

**Meaning:**
Indicator for absorbing aerosols in the atmosphere.

**Typical signals:**

```text
dust
smoke
soot
biomass burning
absorbing aerosols
```

**Possible relationship with PM2.5:**
This is one of the most directly relevant satellite indicators for particulate pollution.

**Interpretation:**

```text
high aerosol index → dust / smoke / soot → potentially higher PM2.5
```

---

# 7. Satellite Measurement Quality Features

Satellite features can be affected by clouds, viewing geometry and sunlight conditions.

These features may not describe pollution directly, but they can help the model understand measurement reliability.

---

## Cloud Fraction

Column pattern:

```text
L3_CLOUD_cloud_fraction
```

**Meaning:**
Cloud fraction during the satellite observation.

**Modeling interpretation:**

```text
high cloud fraction → satellite pollutant measurements may be less reliable
```

---

## Sensor and Solar Angles

Column patterns:

```text
sensor_azimuth_angle
sensor_zenith_angle
solar_azimuth_angle
solar_zenith_angle
```

**Meaning:**
Satellite observation geometry and sunlight angle.

**Modeling interpretation:**
These are not pollution sources. They may help explain measurement artifacts or satellite retrieval quality.

**Use with caution:**
They can improve model performance, but should not be interpreted as direct PM2.5 causes.

---

# 8. Engineered Physical Proxy Features

The raw features can be transformed into physically meaningful proxies.

The goal is to provide the model with interpretable signals.

---

## Wind Speed

```python
df["wind_speed_10m"] = np.sqrt(
    df["u_component_of_wind_10m_above_ground"]**2 +
    df["v_component_of_wind_10m_above_ground"]**2
)
```

**Physical meaning:**
Overall near-surface wind strength.

**Why useful:**
PM2.5 can accumulate under low-wind conditions.

---

## Stagnation Proxy

```python
df["stagnation_proxy"] = (
    df["relative_humidity_2m_above_ground"] /
    (df["wind_speed_10m"] + 1)
)
```

**Physical idea:**

```text
high humidity + low wind → possible accumulation / haze risk
```

**Usefulness:**
Can help capture conditions where particles remain near the ground.

---

## Washout Proxy

```python
df["rain_flag"] = (df["total_precipitation_surface"] > 0).astype(int)

df["washout_proxy"] = (
    df["total_precipitation_surface"] *
    df["wind_speed_10m"]
)
```

**Physical idea:**

```text
precipitation can remove particles from the air
```

**Usefulness:**
Can help identify days where PM2.5 may be reduced by rainfall.

---

## Dryness Proxy

```python
df["dryness_proxy"] = (
    df["temperature_2m_above_ground"] /
    (df["relative_humidity_2m_above_ground"] + 1)
)
```

**Physical idea:**

```text
hot + dry conditions → higher dust potential
```

---

## Dust Risk Proxy

```python
df["dust_risk_proxy"] = (
    df["dryness_proxy"] *
    df["L3_AER_AI_absorbing_aerosol_index"]
)
```

**Physical idea:**

```text
dry conditions + aerosol signal → possible dust / smoke PM2.5 risk
```

---

## Combustion Proxy

Possible input groups:

```text
NO2
CO
HCHO
```

Example:

```python
df["combustion_proxy"] = (
    df["L3_NO2_tropospheric_NO2_column_number_density"] +
    df["L3_CO_column_number_density"] +
    df["L3_HCHO_tropospheric_HCHO_column_number_density"]
)
```

**Physical idea:**

```text
traffic + generators + biomass burning + open fires → combustion-related PM2.5
```

---

## Industrial Proxy

Possible input groups:

```text
SO2
NO2
Aerosol Index
```

Example:

```python
df["industrial_proxy"] = (
    df["L3_SO2_SO2_column_number_density"] +
    df["L3_NO2_tropospheric_NO2_column_number_density"]
)
```

Optional:

```python
df["industrial_aerosol_proxy"] = (
    df["industrial_proxy"] *
    df["L3_AER_AI_absorbing_aerosol_index"]
)
```

**Physical idea:**

```text
industrial gases + aerosol signal → possible industrial PM2.5 contribution
```

---

## Satellite Reliability Risk

```python
df["missing_satellite_count"] = df[satellite_columns].isna().sum(axis=1)
```

Optional:

```python
df["satellite_reliability_risk"] = (
    df["L3_CLOUD_cloud_fraction"] +
    abs(df["L3_AER_AI_sensor_zenith_angle"]) / 90 +
    abs(df["L3_AER_AI_solar_zenith_angle"]) / 90
)
```

**Physical / technical idea:**

```text
clouds + difficult observation angles → less reliable satellite measurements
```

---

# 9. Location Behavior Clustering

## Why not use raw `Place_ID`

Raw `Place_ID` may cause the model to memorize location-specific average pollution.

Instead of using:

```text
Place_ID → PM2.5
```

use:

```text
environmental location profile → location cluster → PM2.5 risk
```

---

## Location Profile Features

Build profiles for each `Place_ID` using only non-target variables.

Possible profile features:

```text
mean temperature
temperature variability
mean relative humidity
humidity variability
mean wind speed
wind variability
mean precipitation
frequency of rain days
mean aerosol index
aerosol variability
mean NO2
mean CO
mean SO2
mean cloud fraction
missing satellite frequency
```

Do not use:

```text
target
target_min
target_max
target_variance
target_count
```

---

## Example Location Profile Code

```python
location_profile = (
    df.groupby("Place_ID")
    .agg(
        temp_mean=("temperature_2m_above_ground", "mean"),
        temp_std=("temperature_2m_above_ground", "std"),

        humidity_mean=("relative_humidity_2m_above_ground", "mean"),
        humidity_std=("relative_humidity_2m_above_ground", "std"),

        wind_mean=("wind_speed_10m", "mean"),
        wind_std=("wind_speed_10m", "std"),

        precipitation_mean=("total_precipitation_surface", "mean"),
        precipitation_days=("rain_flag", "mean"),

        aerosol_mean=("L3_AER_AI_absorbing_aerosol_index", "mean"),
        aerosol_std=("L3_AER_AI_absorbing_aerosol_index", "std"),

        cloud_fraction_mean=("L3_CLOUD_cloud_fraction", "mean"),
        missing_satellite_mean=("missing_satellite_count", "mean")
    )
    .reset_index()
)
```

---

## Scaling Before Clustering

```python
from sklearn.preprocessing import StandardScaler

cluster_features = location_profile.drop(columns=["Place_ID"])

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)
```

---

## Cluster Model Option A: Gaussian Mixture Model

A Gaussian Mixture Model can create soft cluster memberships.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(cluster_scaled)

cluster_probabilities = gmm.predict_proba(cluster_scaled)

location_profile["main_cluster"] = gmm.predict(cluster_scaled)
location_profile["cluster_confidence"] = cluster_probabilities.max(axis=1)
```

**Why useful:**
Some locations may be between clusters.

Example:

```text
Location A:
70% dry-dust cluster
20% combustion cluster
10% humid cluster
```

---

## Cluster Model Option B: KMeans With Intermediate Flag

```python
from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
location_profile["main_cluster"] = kmeans.fit_predict(cluster_scaled)

distances = kmeans.transform(cluster_scaled)
location_profile["distance_to_cluster"] = distances.min(axis=1)
```

Mark unclear locations:

```python
threshold = location_profile["distance_to_cluster"].quantile(0.75)

location_profile["cluster_type"] = np.where(
    location_profile["distance_to_cluster"] > threshold,
    "intermediate_or_uncertain",
    "clear_cluster"
)
```

---

## Recommended Cluster Types

After clustering, interpret each cluster by looking at average feature values.

Possible cluster interpretations:

```text
hot / dry / dust-prone
humid / low-wind / stagnation-prone
windy / well-ventilated
combustion-influenced urban
industrial / sulfur-influenced
cloudy / low satellite reliability
mixed / intermediate
```

---

## Merge Cluster Features Back

```python
df = df.merge(
    location_profile[["Place_ID", "main_cluster", "cluster_confidence"]],
    on="Place_ID",
    how="left"
)
```

Better than a hard label:

```text
cluster_0_probability
cluster_1_probability
cluster_2_probability
cluster_3_probability
cluster_4_probability
```

Cluster probabilities preserve uncertainty and avoid forcing unclear locations into one fixed group.

---

# 10. Additional Out-of-the-Box Feature Ideas

## Missingness as Information

Missing values may indicate cloud cover, retrieval issues or data quality limitations.

```python
df["missing_satellite_count"] = df[satellite_columns].isna().sum(axis=1)
df["missing_weather_count"] = df[weather_columns].isna().sum(axis=1)
```

---

## Weather Regime Features

Create interpretable day types:

```text
dry and hot
humid and low-wind
rainy
windy
cloudy
stagnant
```

Example:

```python
df["stagnant_humid_day"] = (
    (df["low_wind_day"] == 1) &
    (df["humid_day"] == 1)
).astype(int)
```

---

## Extreme Value Flags

```python
df["very_high_aerosol_day"] = (
    df["L3_AER_AI_absorbing_aerosol_index"] >
    df["L3_AER_AI_absorbing_aerosol_index"].quantile(0.90)
).astype(int)

df["very_low_wind_day"] = (
    df["wind_speed_10m"] <
    df["wind_speed_10m"].quantile(0.10)
).astype(int)
```

---

## Temporal Lag Features

PM2.5 can persist over several days.

```python
df = df.sort_values(["Place_ID", "Date"])

df["pm25_lag_1"] = df.groupby("Place_ID")["target"].shift(1)
df["pm25_lag_3"] = df.groupby("Place_ID")["target"].shift(3)
```

**Warning:**
Use only past values. Never use future values.

**Use only if:**
The real prediction scenario has access to past PM2.5 observations for the same location.

---

## Uncertainty Features

Possible uncertainty indicators:

```text
missing_satellite_count
satellite_reliability_risk
cluster_confidence
distance_to_cluster
model disagreement
prediction variance across ensemble models
```

---

## Explainability Features

Potential explanations:

```text
Risk is high because wind is low and humidity is high.
Risk is high because aerosol index is elevated.
Risk is lower because rainfall is expected.
Risk is uncertain because satellite observations are cloud-contaminated.
```

Useful methods:

```text
feature importance
permutation importance
SHAP values
cluster profile comparison
```

---

# 11. Modeling Notes Related to Features

## Recommended Feature Groups

```text
weather features
satellite pollutant proxies
engineered dispersion features
engineered washout features
dryness and dust proxies
combustion proxies
industrial proxies
satellite reliability features
missingness features
seasonality features
location behavior clusters
cluster confidence / intermediate cluster flags
```

---

## Compare Feature Strategies

Useful comparison:

```text
Model A: raw weather + satellite features
Model B: engineered physical proxy features added
Model C: engineered features + hard location clusters
Model D: engineered features + cluster probabilities / confidence
Model E: raw Place_ID encoding
```

Interpretation:

```text
If raw Place_ID performs suspiciously well, it may indicate memorization.
If cluster probabilities improve performance, environmental profiles may help generalization.
If engineered features improve high-risk detection, they are useful for the pollution warning use case.
```

---

# 12. Leakage and Bias Control

## Avoid

```text
raw Place_ID as direct shortcut
target_min
target_max
target_variance
target_count
target-based clusters
features using future target values
unexplained black-box embeddings
uncontrolled automatic feature generation
```

---

## Prefer

```text
environmental location clusters
cluster confidence
intermediate cluster flags
physically plausible feature engineering
weather regime features
satellite reliability indicators
missingness indicators
explainable model interpretation
```

---

# 13. Final Feature Principle

The model should not simply memorize which anonymized location had high pollution before.

Instead, it should learn environmental patterns that explain why PM2.5 becomes high.

```text
Do not let the model memorize where pollution was high.
Help the model learn why pollution becomes high.
```

Final feature logic:

```text
PM2.5 prediction
= weather features
+ satellite pollution proxies
+ environmental behavior clusters
+ engineered physical proxies
+ uncertainty indicators
```
