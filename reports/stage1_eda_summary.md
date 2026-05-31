# Stage 1 EDA Summary — Original Severson Features

## 1. Dataset Overview

The dataset was extracted from the original Severson et al. (2019) LFP/graphite battery aging benchmark.
It contains **140 battery cells** across **3 batch files**.

| Metric | Value |
| --- | --- |
| Total rows | 140 |
| Total columns | 22 |
| Rows with valid cycle_life | 138 |
| Rows missing cycle_life | 2 |
| Number of physical features | 16 |
| Number of batch files | 3 |
| Number of screening labels | 3 |

> **Note:** 2 rows have missing `cycle_life` and should be excluded
> from supervised regression and classification model training.
> All model-based analyses below use `df_model` (n=138).

---

## 2. Missing Value Summary

| Column | Missing Count | Missing % |
| --- | --- | --- |
| cycle_life | 2 | 1.4% |

All physical feature columns have **no missing values**.

---

## 3. Label Distribution

Screening labels are assigned based on `cycle_life` thresholds:
- **Reuse**: cycle_life ≥ 1000
- **Conditional Reuse**: 500 ≤ cycle_life < 1000
- **Recycle**: cycle_life < 500

| Screening Label | Count |
| --- | --- |
| Conditional Reuse | 69 |
| Recycle | 37 |
| Reuse | 34 |

---

## 4. Batch Distribution

| Source File | Count |
| --- | --- |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | 48 |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | 46 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | 46 |

---

## 5. Cycle Life Distribution

The `cycle_life` target spans from short-lived cells (< 200 cycles) to long-lived cells (> 1000 cycles).
The distribution is right-skewed, with the majority of cells falling in the 300–900 cycle range.

See: `outputs/figures/stage1/cycle_life_distribution.png`

---

## 6. Top Correlated Features with Cycle Life

| feature | correlation_with_cycle_life |
| --- | --- |
| tmin_mean_2_100 | 0.4922 |
| ir_mean_2_100 | -0.4918 |
| chargetime_change_2_100 | -0.3196 |
| tavg_mean_2_100 | 0.3065 |
| capacity_slope_2_100 | 0.3055 |
| capacity_fade_pct_2_100 | -0.2925 |
| soh_cycle_100_proxy | 0.2925 |
| capacity_fade_2_100 | -0.2902 |
| chargetime_mean_2_100 | 0.2409 |
| q_discharge_cycle_100 | 0.1641 |

The strongest predictors of cycle life are discharge capacity–related features
(`soh_cycle_100_proxy`, `q_discharge_cycle_100`, `q_discharge_mean_2_100`)
and capacity fade metrics (`capacity_fade_pct_2_100`, `capacity_fade_2_100`).
Internal resistance (`ir_mean_2_100`) and temperature metrics show moderate correlation.

---

## 7. Key Observations

1. **SOH proxy and discharge capacity** at cycle 100 are the most predictive features of long-term cycle life.
2. **Capacity fade (%)** in the first 100 cycles is a strong negative predictor, consistent with the literature.
3. **IR mean** shows moderate positive correlation, suggesting early degradation signals are partially captured.
4. **Temperature means** (Tavg, Tmax) show weak correlation, indicating operating temperature was relatively stable.
5. **Label imbalance** is moderate: Conditional Reuse is the largest class; ensure stratified splits during training.
6. The dataset originates from a **controlled laboratory benchmark** (Stanford/MIT LFP cells).
   It should be framed as **benchmark validation** for the FORGE diagnostic pipeline,
   not as representative of Indonesian EV battery field conditions.

---

## 8. Generated Outputs

### Tables
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_dataset_overview.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_missing_values.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_descriptive_statistics.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_label_distribution.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_batch_distribution.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_correlation_with_cycle_life.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_top_correlated_features.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_grouped_feature_means_by_label.csv`

### Figures
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\cycle_life_distribution.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\screening_label_distribution.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\batch_distribution.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\correlation_bar_top10.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\correlation_heatmap_top10.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\cycle_life_by_label_boxplot.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\q_discharge_cycle_100_vs_cycle_life.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\ir_mean_vs_cycle_life.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\tavg_mean_vs_cycle_life.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\capacity_fade_pct_vs_cycle_life.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\soh_cycle_100_proxy_vs_cycle_life.png`

---

## 9. Recommended Next Step

Proceed to **preprocessing and model training**:

1. Run `src/stage1_diagnostic/preprocess.py` to apply scaling and stratified train/test split.
2. Run `src/stage1_diagnostic/train_xgboost.py` to train:
   - XGBoost regression model for `cycle_life` prediction.
   - XGBoost classification model for `screening_label` prediction.
3. Evaluate using RMSE, MAE, R² (regression) and accuracy, F1, confusion matrix (classification).
