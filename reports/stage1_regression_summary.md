# Stage 1 XGBoost Regression Summary — Cycle Life Prediction

## 1. Objective

Train an XGBoost regression model to predict battery `cycle_life` from early-cycle physical features
extracted from the original Severson LFP/graphite benchmark dataset.

The model is intended as a **proof-of-concept** for the FORGE diagnostic pipeline.
Results should be interpreted as benchmark validation, not final field performance.

---

## 2. Input Data

| Item | Value |
| --- | --- |
| Training set | `data/processed/stage1_regression_train.csv` |
| Test set | `data/processed/stage1_regression_test.csv` |
| Training rows | 110 |
| Test rows | 28 |
| Feature columns | 16 |
| Target | `cycle_life` (continuous) |

---

## 3. Feature Set

16 physical early-cycle features:

| Index | Feature |
| --- | --- |
| 0 | n_cycles_available |
| 1 | q_discharge_cycle_2 |
| 2 | q_discharge_cycle_100 |
| 3 | q_discharge_mean_2_100 |
| 4 | q_charge_mean_2_100 |
| 5 | capacity_fade_2_100 |
| 6 | capacity_fade_pct_2_100 |
| 7 | capacity_slope_2_100 |
| 8 | ir_mean_2_100 |
| 9 | ir_change_2_100 |
| 10 | tavg_mean_2_100 |
| 11 | tmax_mean_2_100 |
| 12 | tmin_mean_2_100 |
| 13 | chargetime_mean_2_100 |
| 14 | chargetime_change_2_100 |
| 15 | soh_cycle_100_proxy |

---

## 4. Model Configuration

| Parameter | Value |
| --- | --- |
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | reg:squarederror |
| random_state | 42 |
| n_jobs | -1 |

XGBoost is well-suited for this task because it handles nonlinear feature-target relationships
and is robust on small tabular datasets without requiring feature scaling.

---

## 5. Cross-Validation Results (5-Fold, Training Set Only)

| fold | MAE | RMSE | R² |
| --- | --- | --- | --- |
| 1 | 170.5506 | 262.3962 | 0.6446 |
| 2 | 150.7225 | 174.7371 | 0.5937 |
| 3 | 96.9288 | 123.229 | 0.8024 |
| 4 | 129.8587 | 194.1923 | 0.6493 |
| 5 | 100.1282 | 128.5311 | 0.8499 |
| mean | 129.6378 | 176.6171 | 0.708 |
| std | 28.4927 | 50.6617 | 0.0996 |

| Summary | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Mean | 129.6378 | 176.6171 | 0.708 |
| Std  | 28.4927  | 50.6617  | 0.0996 |

The test set was kept **untouched** during cross-validation.

---

## 6. Test Set Results

| Metric | Value |
| --- | --- |
| MAE | 120.2286 |
| RMSE | 171.5513 |
| R2 | 0.4719 |
| MAPE | 15.1491 |

---

## 7. Feature Importance (Top 10)

| Feature | Importance | Rank |
| --- | --- | --- |
| tmin_mean_2_100 | 0.225388 | 1 |
| ir_mean_2_100 | 0.164212 | 2 |
| chargetime_mean_2_100 | 0.115539 | 3 |
| capacity_fade_2_100 | 0.085967 | 4 |
| chargetime_change_2_100 | 0.077692 | 5 |
| capacity_fade_pct_2_100 | 0.046823 | 6 |
| capacity_slope_2_100 | 0.046609 | 7 |
| q_discharge_mean_2_100 | 0.043691 | 8 |
| ir_change_2_100 | 0.041927 | 9 |
| tavg_mean_2_100 | 0.035946 | 10 |

---

## 8. Generated Outputs

### Model
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\models\stage1\xgboost_cycle_life_regressor.pkl`

### Tables
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_regression_cv_metrics.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_regression_metrics.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_regression_predictions.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_regression_feature_importance.csv`

### Figures
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\regression_predicted_vs_actual.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\regression_residual_plot.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\regression_feature_importance.png`

---

## 9. Key Interpretation

- The model was trained on only 110 samples from a controlled laboratory benchmark.
  Performance metrics indicate feasibility, not readiness for field deployment.
- Features most predictive of cycle life (tmin_mean_2_100,
  ir_mean_2_100) are consistent with the electrochemical degradation
  literature: early capacity retention and internal resistance growth are known
  early-life predictors.
- MAPE (if finite) reflects relative prediction error, useful for comparing across cell
  life ranges.
- The test set contains only 28 cells, so test metrics carry high variance.
  Cross-validation MAE/RMSE on the training set provides a more stable performance estimate.

---

## 10. Recommended Next Step

Create and run:

```
python src/stage1_diagnostic/train_xgboost_classification.py
```

This will train a multi-class XGBoost classifier to predict the screening label
(Reuse / Conditional Reuse / Recycle) using the same feature set and train/test split.
Evaluate with accuracy, F1-score, and confusion matrix.
