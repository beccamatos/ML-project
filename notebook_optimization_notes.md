# Notebook Optimization Notes

## Goal
This file documents the structural and methodological changes made to `EDA_ML.ipynb` so the notebook is easier to explain, less misleading after reruns, and faster to iterate on.

## Main fixes

### 1. Removed the unfinished scratch appendix
- Deleted the `# TO DO` / exploratory cells after the submission section.
- Reason: those cells reloaded `Train.csv`, mutated `train`, and contained saved runtime errors.
- Effect: `Run All` now ends on the intended submission workflow instead of drifting into an unfinished side branch.

### 2. Split feature usage into two explicit tracks
- Added a **baseline/imputed feature set**:
  - drops columns above the configured missing-value threshold
  - used by Ridge, RandomForest, ExtraTrees, and the imputed HistGradientBoosting pipelines
- Added a **native-missing feature set**:
  - keeps the full engineered feature list when enabled
  - used by native-missing models such as HistGradientBoosting, LightGBM, and XGBoost
- Reason: previously all later models inherited the same global feature drop, which prevented native-missing models from testing whether sparse columns still contained signal.

### 3. Made the execution path reproducible and easier to explain
- Centralized execution controls:
  - `FAST_MODE`
  - `CV_N_SPLITS`
  - `BASELINE_HIGH_MISSING_THRESHOLD`
  - `USE_FULL_FEATURE_SET_FOR_NATIVE_MISSING_MODELS`
- Reason: this makes the notebook easier to present because the important trade-offs are controlled from one place.

## Speed optimizations

### 4. Added a fast rerun mode
- `FAST_MODE = False` keeps the original high-quality search path.
- If set to `True`, the notebook automatically uses:
  - fewer `RandomizedSearchCV` iterations
  - a coarser ensemble-weight search grid
  - fewer trees for the expensive bagging models
- Reason: this speeds up exploratory reruns without changing the main workflow structure.

### 5. Reused shared configuration values
- Replaced repeated hard-coded values with configuration constants where useful.
- Examples:
  - CV fold count
  - HGB / LightGBM random-search iterations
  - ensemble weight step size
  - tree counts for RandomForest and ExtraTrees

## Consistency improvements

### 6. Aligned ensemble training with model comparison
- The ensemble now uses the same native feature view that was used when evaluating the native-missing models.
- Reason: this keeps the final training path consistent with the validation logic.

### 7. Cleared stale outputs for modified sections
- Cleared outputs and execution counts in changed code cells and downstream result cells.
- Reason: old outputs would no longer match the updated code and could mislead reviewers.

### 8. Made `SampleSubmission.csv` optional
- The notebook now checks whether `data/SampleSubmission.csv` exists before reading it.
- If the file is missing, the notebook prints a clear message and continues without failing during the data-loading section.
- The sample-submission preview cell is also guarded so it does not raise a `NoneType` error.
- Reason: the local project copy currently contains `Train.csv` and `Test.csv`, but not always the sample submission template.

## Dependency updates

### 9. Added the native-missing boosting libraries to environment requirements
- The notebook imports and uses:
  - `lightgbm==4.6.0`
  - `xgboost==3.2.0`
- These packages were missing from `requirements.txt` and have now been added.
- Reason: without pinned entries, another machine could open the notebook but fail as soon as the LightGBM or XGBoost blocks are executed.

## Stability fixes

### 10. Stabilized LightGBM tuning on Windows
- The LightGBM `RandomizedSearchCV` block now runs with:
  - `LGBMRegressor(..., n_jobs=1)`
  - `RandomizedSearchCV(..., n_jobs=1)`
- Reason: the previous parallel setup triggered `joblib/loky` worker failures on Windows, including:
  - `BrokenProcessPool`
  - `PermissionError: [WinError 5] Zugriff verweigert`
- Effect: the LightGBM search is slower, but it is much more stable and reproducible in the local environment.

### 11. Fixed the LightGBM notebook execution order
- Removed premature LightGBM follow-up cells that appeared before the actual `## Tune LightGBM` section.
- Added explicit guards in the tuned-model and results-table blocks so they fail with a clear message if `lgbm_search.fit(...)` has not been executed yet.
- Reason: those follow-up cells previously referenced `lgbm_search.best_score_` before `lgbm_search` existed, which caused:
  - `NameError: name 'lgbm_search' is not defined`
- Effect: the notebook now reflects the real dependency order:
  1. tune LightGBM
  2. inspect best score and parameters
  3. build the tuned LightGBM pipeline
  4. update the results table

## New modeling improvements

### 12. Added physically motivated proxy features from the project notes
- Extended the feature engineering section to add new deterministic feature groups based on `Urban Air Pollution Challenge Feature and Physical Principle Notes.md`.
- New engineered features now include:
  - seasonal month cycles: `month_sin`, `month_cos`
  - dispersion / accumulation proxies: `stagnation_proxy`, `low_wind_day`, `stagnant_humid_day`
  - dryness / dust proxies: `temp_squared`, `dryness_proxy`, `dust_risk_proxy`
  - emission proxies: `combustion_proxy`, `industrial_proxy`, `industrial_aerosol_proxy`
  - data-quality / uncertainty proxies: `missing_weather_count`, `missing_satellite_count`, `missing_total_measurement_count`, `has_missing_satellite`, `satellite_reliability_risk`
- Threshold-based regime flags are built from training data only and then reused for the test set.
- Reason: this keeps the logic physically explainable while avoiding hidden heuristics or test-set peeking.

### 13. Added explicit feature toggles for explainability
- Added top-level switches for:
  - `ENABLE_PHYSICAL_PROXY_FEATURES`
  - `ENABLE_WEATHER_REGIME_FLAGS`
  - `ENABLE_MISSINGNESS_PROXY_FEATURES`
  - `ENABLE_SATELLITE_RELIABILITY_FEATURES`
- Reason: this makes it easier to demonstrate which engineered feature groups are active and to rerun ablation experiments cleanly.

### 14. Corrected tuned-model handoff after hyperparameter search
- The tuned `HistGradientBoosting` and tuned `LightGBM` pipelines now use the actual `best_estimator_` returned by `RandomizedSearchCV`.
- Reason: the previous notebook copied parameter values manually into later cells, which was fragile if the search space or best result changed after a rerun.
- Effect: the final tuned-model blocks now always stay synchronized with the latest search result.

### 15. Expanded the LightGBM search space around the previous optimum
- The prior best run sat near the upper edge for several LightGBM hyperparameters.
- The tuning block now also explores:
  - larger `n_estimators`
  - explicit `max_depth`
  - explicit `min_split_gain`
  - slightly wider leaf and sampling choices
- Reason: when the previous best result sits on a boundary, a wider local search is a justified next step.

### 16. Tightened the feature pipeline around physically explainable signals
- The new run improved after combining several notebook changes instead of relying on one isolated tweak.
- Most relevant changes that likely contributed to the better validation score:
  - added physically motivated proxy features from weather and satellite variables
  - added explicit seasonality with `month_sin` and `month_cos`
  - added missingness and satellite-reliability features as uncertainty signals
  - reused train-derived thresholds for regime flags so the feature logic stayed deterministic
  - aligned tuned-model cells with `best_estimator_` so final evaluation matched the latest search result
  - expanded the LightGBM search around the former boundary values
- Reason: these changes make the model inputs richer without introducing target leakage or hidden test-set assumptions.

### 17. Added hardware-aware execution controls
- Added a startup hardware block that:
  - detects available CPU cores
  - checks whether `nvidia-smi` is present
  - runs a tiny LightGBM and XGBoost GPU probe before enabling GPU mode
- GPU is only enabled for supported models if the probe fit succeeds; otherwise the notebook falls back to CPU automatically.
- The new hardware config now controls:
  - `CPU_TRAIN_JOBS`
  - `SEARCH_N_JOBS`
  - `LIGHTGBM_DEVICE_TYPE`
  - `XGBOOST_DEVICE`
- If GPU acceleration is available, the notebook now also expands:
  - `LGBM_RANDOM_SEARCH_ITERATIONS`
  - the LightGBM hyperparameter search boundary
  - the ablation-model size
  - the ensemble-weight search resolution
- Reason: this allows faster model training on capable machines without making the notebook fragile on machines where GPU builds or CUDA support are unavailable.

### 18. Added optional feature-group ablation
- Added an optional analysis block controlled by:
  - `RUN_FEATURE_GROUP_ABLATION`
  - `ABLATION_USE_FAST_MODEL`
- The ablation block measures what happens when whole active feature groups are removed one at a time.
- It is off by default because it adds runtime, but it is useful when deciding whether a group should stay enabled or be excluded in later runs.
- Reason: this is a safer way to test redundancy than manually deleting features based only on intuition.

### 19. Added ensemble-diversity diagnostics
- Added a new post-OOF analysis block that reports:
  - prediction correlation between ensemble members
  - per-model RMSE and prediction spread
  - pairwise mean absolute disagreement between model predictions
- Reason: the ensemble gains strength when strong models make different mistakes. This block helps show whether the second and third ensemble members add real diversity or just duplicate the LightGBM signal.

### 20. Reused the best ensemble weights automatically
- The final ensemble block now copies the best weights found by the current weight search instead of keeping manually fixed values.
- Reason: this keeps the final prediction path aligned with the latest validated ensemble result after each rerun.

### 21. Restored `EDA_ML.ipynb` to the stable notebook logic
- `EDA_ML.ipynb` briefly diverged from the successfully executed `EDA_ML2.ipynb` after an experimental GPU-aware expansion of the LightGBM search and ablation logic.
- To avoid losing the working notebook path, the divergent config/search cells were synchronized back to the stable `EDA_ML2.ipynb` logic.
- Reason: the project needs one primary notebook that is both current and runnable, which is more valuable than keeping an unverified experimental branch inside the main file.

## Latest validated results

### 22. Current post-fix run snapshot
- Latest values recovered from the finished notebook run:
  - `HistGradientBoosting` baseline: `34.3040`
  - tuned `HistGradientBoosting`: `33.6825`
  - tuned `HistGradientBoosting` + missing indicators: `33.6056`
  - native-missing `HistGradientBoosting`: `33.5213`
  - `ExtraTrees`: `33.6849`
  - `LightGBM` baseline: `33.3345`
  - `XGBoost` baseline: `33.6688`
  - tuned `LightGBM`: `33.0009`
  - weighted ensemble (`Tuned LightGBM + ExtraTrees`): `32.7879`
- Current best documented model remains the weighted ensemble, with tuned LightGBM as the best standalone model.

### 23. New improved run after domain-feature expansion
- Latest rerun values after the new feature engineering and search-space changes:
  - `HistGradientBoosting` baseline: `33.3199`
  - tuned `HistGradientBoosting`: `32.6555`
  - tuned `HistGradientBoosting` + missing indicators: `32.6460`
  - native-missing `HistGradientBoosting`: `32.6455`
  - `ExtraTrees`: `33.4233`
  - `LightGBM` baseline: `32.3503`
  - `XGBoost` baseline: `32.8867`
  - tuned `LightGBM`: `32.0079`
  - weighted ensemble (`Tuned LightGBM + ExtraTrees`): `31.9383`
- Improvement versus the earlier best documented run:
  - tuned `LightGBM`: from `33.0009` to `32.0079`
  - best ensemble: from `32.7879` to `31.9383`
- Current best documented model is still the weighted ensemble, but now at a materially better validation score.

## What still needs rerunning
- The notebook has now been rerun successfully after the latest optimization pass.
- A fresh rerun is only needed if you change feature toggles, tuning ranges, or ensemble settings again.
- Recommended rerun order:
  1. data loading
  2. feature engineering
  3. feature selection
  4. baseline models
  5. tuning
  6. ensemble
  7. submission

## Expected presentation benefit
- The notebook is now easier to defend because:
  - the final path is cleaner
  - the native-missing experiments are methodologically more honest
  - the runtime/quality trade-off is explicit
  - stale side experiments no longer weaken the main story
