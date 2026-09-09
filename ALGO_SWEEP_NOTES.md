# Personalized No-CGM Algorithm Sweep

This follow-up experiment tested whether alternative estimators improve on ridge regression for personalized, wearable-only 30-minute glucose forecasting.

## Setup

- Training data: each user’s earlier 80% chronological window.
- Test data: that user’s final 20% window.
- Features: time, wearable signals, carbohydrate/insulin logs, and lagged or rolling features derived from previous observations.
- Excluded at prediction: CGM history.
- Metrics: weighted RMSE and MAE across 61,858 test observations from 25 users.

## Results

| Model | Weighted RMSE (mg/dL) | Weighted MAE (mg/dL) |
| --- | ---: | ---: |
| Extra trees | 53.111 | 42.885 |
| Histogram gradient boosting | 53.175 | 42.557 |
| Elastic net | 53.639 | 43.406 |
| Ridge | 53.652 | 43.411 |

The tree ensembles achieved a small RMSE improvement over ridge. The narrow gap suggests that the main constraint is the available input signal, rather than estimator selection alone.

Source files: [script](src/algorithm_sweep_no_cgm_30m_quick.py), [per-user results](results/tables/personalized_no_cgm_algorithm_sweep.csv), and [weighted summary](results/tables/personalized_no_cgm_top_models.json).
