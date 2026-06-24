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

## What still needs rerunning
- The notebook structure is updated, but the changed modeling sections need to be rerun to produce fresh metrics and plots.
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
