# Pre-Optimization Result Snapshot

This file stores the model results that were documented in `EDA_ML.ipynb` before the notebook refactor and speed optimizations.

Note:
- These values were recovered from the notebook's explanatory markdown cells.
- Several code-cell outputs were cleared during the cleanup, so this snapshot is based on the written result summaries that were already present in the notebook.

## Baseline Models

| Model | Mean RMSE |
|---|---:|
| Dummy Regressor | 46.83 |
| Ridge Regression | 38.56 |
| Random Forest | 34.55 |
| HistGradientBoosting baseline | 34.30 |

## HistGradientBoosting Progression

| Experiment | Mean RMSE |
|---|---:|
| HistGradientBoosting baseline | 34.30 |
| Tuned HistGradientBoosting | 33.68 |
| Tuned HistGradientBoosting + missingness indicators | 33.61 |
| HistGradientBoosting with log target | 35.07 |
| HistGradientBoosting with native missing handling | 33.52 |

## Additional Model Experiments

| Model | Mean RMSE |
|---|---:|
| ExtraTreesRegressor | 33.68 |
| LightGBM baseline | 33.34 |
| XGBoost baseline | 33.72 |
| Tuned LightGBM | 33.04 |

## Final Ensemble

| Final model | Mean RMSE |
|---|---:|
| Weighted Ensemble: Tuned LightGBM + ExtraTrees | 32.81 |

## Ensemble Weights

| Component | Weight |
|---|---:|
| Tuned LightGBM | 0.65 |
| Native HistGradientBoosting | 0.00 |
| ExtraTrees | 0.35 |

## Summary of Best Known Progression

1. Best baseline model: `HistGradientBoosting` with RMSE `34.30`
2. Best sklearn-only standalone model after experiments: `HistGradientBoosting with native missing handling` with RMSE `33.52`
3. Best standalone boosted model: `Tuned LightGBM` with RMSE `33.04`
4. Best overall documented model before refactor: `Weighted Ensemble: Tuned LightGBM + ExtraTrees` with RMSE `32.81`

## Post-Optimization Run Snapshot

This section captures the newer validated results after the additional feature engineering, LightGBM search expansion, and tuned-model handoff fixes.

| Model | Mean RMSE |
|---|---:|
| HistGradientBoosting baseline | 33.32 |
| Tuned HistGradientBoosting | 32.66 |
| Tuned HistGradientBoosting + missingness indicators | 32.65 |
| HistGradientBoosting with log target | 34.00 |
| HistGradientBoosting with native missing handling | 32.65 |
| ExtraTreesRegressor | 33.42 |
| LightGBM baseline | 32.35 |
| XGBoost baseline | 32.89 |
| Tuned LightGBM | 32.01 |
| Weighted Ensemble: Tuned LightGBM + ExtraTrees | 31.94 |

## Improvement vs Earlier Best

| Comparison point | Earlier RMSE | New RMSE | Improvement |
|---|---:|---:|---:|
| Tuned LightGBM | 33.04 | 32.01 | 1.03 |
| Best overall ensemble | 32.81 | 31.94 | 0.87 |

## Updated Summary of Best Known Progression

1. Best baseline model in the new run: `LightGBM` with RMSE `32.35`
2. Best sklearn-only standalone model in the new run: `HistGradientBoosting with native missing handling` with RMSE `32.65`
3. Best standalone boosted model in the new run: `Tuned LightGBM` with RMSE `32.01`
4. Best overall documented model after the latest optimization run: `Weighted Ensemble: Tuned LightGBM + ExtraTrees` with RMSE `31.94`
