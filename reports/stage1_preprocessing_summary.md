# Stage 1 Preprocessing Summary — Original Severson Features

## 1. Objective

This preprocessing step prepares clean, model-ready train/test CSV datasets from
the original Severson feature extraction output. It produces two task-specific dataset pairs:

- **XGBoost Regression** — predict `cycle_life` (continuous target)
- **XGBoost Classification** — predict `screening_label` (Reuse / Conditional Reuse / Recycle)

XGBoost does not require feature standardization, so no scaling was applied.

---

## 2. Input Dataset

| Item | Value |
| --- | --- |
| Input file | `data/processed/severson_original_features.csv` |
| Input shape | 140 rows × 22 columns |
| Source | 3 original Severson batch files (Stanford/MIT LFP benchmark) |

> This dataset is a **public laboratory benchmark** (Severson et al., 2019).
> It should be framed as benchmark validation for the FORGE diagnostic pipeline,
> not as representative of Indonesian EV battery field data.

---

## 3. Cleaning Step

| Action | Count |
| --- | --- |
| Input rows | 140 |
| Rows dropped (missing cycle_life) | 2 |
| Remaining model rows | 138 |

Rows with missing `cycle_life` were removed because supervised learning requires valid targets.

---

## 4. Feature Columns

**16 physical features** were selected. Excluded: `cell_index`, `cycle_life`,
`screening_label`, `screening_label_encoded`, and metadata columns.

| feature_name | feature_index |
| --- | --- |
| n_cycles_available | 0 |
| q_discharge_cycle_2 | 1 |
| q_discharge_cycle_100 | 2 |
| q_discharge_mean_2_100 | 3 |
| q_charge_mean_2_100 | 4 |
| capacity_fade_2_100 | 5 |
| capacity_fade_pct_2_100 | 6 |
| capacity_slope_2_100 | 7 |
| ir_mean_2_100 | 8 |
| ir_change_2_100 | 9 |
| tavg_mean_2_100 | 10 |
| tmax_mean_2_100 | 11 |
| tmin_mean_2_100 | 12 |
| chargetime_mean_2_100 | 13 |
| chargetime_change_2_100 | 14 |
| soh_cycle_100_proxy | 15 |

---

## 5. Target Definitions

| Task | Target Column | Type |
| --- | --- | --- |
| Regression | `cycle_life` | Continuous float |
| Classification | `screening_label_encoded` | Integer (0–2) |

---

## 6. Label Encoding

`sklearn.preprocessing.LabelEncoder` was applied to `screening_label`.

| screening_label | screening_label_encoded |
| --- | --- |
| Conditional Reuse | 0 |
| Recycle | 1 |
| Reuse | 2 |

Label mapping saved to:
- `outputs/tables/stage1/stage1_label_mapping.csv`
- `outputs/tables/stage1/stage1_label_mapping.json`

---

## 7. Train-Test Split

| Setting | Value |
| --- | --- |
| Split ratio | 80% train / 20% test |
| Stratified by | `screening_label` |
| Random state | 42 |
| Validation set | Not created (dataset is small; use cross-validation in training scripts) |

The **same stratified split** was used for both regression and classification, so both
task datasets contain identical cell_id rows in each partition.

| split | rows | Recycle | Conditional Reuse | Reuse | Recycle% | Cond.Reuse% | Reuse% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 110 | 28 | 55 | 27 | 25.5% | 50.0% | 24.5% |
| test | 28 | 7 | 14 | 7 | 25.0% | 50.0% | 25.0% |

---

## 8. Generated Outputs

### Model-Ready CSV Files
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\data\processed\stage1_regression_train.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\data\processed\stage1_regression_test.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\data\processed\stage1_classification_train.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\data\processed\stage1_classification_test.csv`

### Metadata Tables
- `outputs/tables/stage1/stage1_preprocessing_summary.csv`
- `outputs/tables/stage1/stage1_feature_list.csv`
- `outputs/tables/stage1/stage1_label_mapping.csv`
- `outputs/tables/stage1/stage1_train_test_distribution.csv`

---

## 9. Recommended Next Step

Create and run these training scripts in order:

1. `src/stage1_diagnostic/train_xgboost_regression.py`
   — Train XGBoost regressor on `stage1_regression_train.csv`, predict `cycle_life`.
   — Use k-fold cross-validation inside the training set.
   — Evaluate with RMSE, MAE, R².

2. `src/stage1_diagnostic/train_xgboost_classification.py`
   — Train XGBoost classifier on `stage1_classification_train.csv`, predict `screening_label_encoded`.
   — Use stratified k-fold cross-validation.
   — Evaluate with accuracy, F1, and confusion matrix.

3. `src/stage1_diagnostic/evaluate_stage1.py`
   — Consolidate evaluation metrics and generate final paper tables and figures.
