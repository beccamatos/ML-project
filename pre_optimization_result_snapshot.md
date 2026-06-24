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
