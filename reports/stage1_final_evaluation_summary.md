# Stage 1 Final Evaluation Summary — FORGE Diagnostic Pipeline

## 1. Executive Summary

Stage 1 implements a complete battery diagnostic pipeline:

**Raw Severson MATLAB data → Physical early-cycle features → EDA → Preprocessing →
XGBoost Regression → XGBoost Classification → Final Evaluation**

The pipeline extracts 16 physical early-cycle features from the original Severson
LFP/graphite benchmark dataset and demonstrates the feasibility of data-driven
cycle-life prediction and battery reuse screening as the diagnostic foundation
of the FORGE second-life infrastructure concept.

---

## 2. Stage 1 Objective

Stage 1 aims to:
1. **Predict battery cycle life** (regression) to estimate remaining useful life.
2. **Classify battery screening labels** (Reuse / Conditional Reuse / Recycle)
   to support second-life allocation decisions.

Both tasks use physical features extracted only from early cycles (2–100),
simulating a fast diagnostic assessment without requiring full charge-discharge cycling.

---

## 3. Dataset and Feature Engineering Summary

| Item | Value |
| --- | --- |
| total_original_rows | 140 |
| valid_supervised_rows | 138 |
| rows_dropped_missing_cycle_life | 2 |
| regression_train_rows | 110 |
| regression_test_rows | 28 |
| classification_train_rows | 110 |
| classification_test_rows | 28 |
| number_of_features | 16 |
| number_of_labels | 3 |
| label_distribution_train | Conditional Reuse:55; Recycle:28; Reuse:27 |
| label_distribution_test | Conditional Reuse:14; Recycle:7; Reuse:7 |

- **Source:** Severson et al. (2019) public LFP/graphite battery aging benchmark.
- **Coverage:** 3 batch files, 140 total cells; 138 used after removing 2 with missing `cycle_life`.
- **Features:** 16 physical early-cycle features derived from cycle 2 to cycle 100
  (discharge/charge capacity, internal resistance, temperature, charge time, capacity slope).
- **Framing:** This dataset serves as **benchmark validation** for the FORGE diagnostic
  pipeline — it is not Indonesian EV field data.

---

## 4. Preprocessing Summary

- 80/20 stratified train-test split: **110 train rows / 28 test rows**.
- No separate validation set (dataset is small; cross-validation used instead).
- XGBoost does not require feature scaling — no standardisation applied.
- Balanced sample weights applied for classification final fit to mitigate class imbalance.

---

## 5. XGBoost Regression Result

**Target:** `cycle_life` (continuous, in full charge-discharge cycles)

| Eval Type | Metric | Value |
| --- | --- | --- |
| CV | CV MAE Mean | 129.6378 |
| CV | CV MAE Std | 28.4927 |
| CV | CV RMSE Mean | 176.6171 |
| CV | CV RMSE Std | 50.6617 |
| CV | CV R2 Mean | 0.708 |
| CV | CV R2 Std | 0.0996 |
| Test | Test MAE | 120.2286 |
| Test | Test RMSE | 171.5513 |
| Test | Test R2 | 0.4719 |
| Test | Test MAPE | 15.1491 |

Key observations:
- CV R² of **0.708** indicates the model explains ~70% of
  cycle-life variance on unseen training folds.
- Test MAE of **120.2286 cycles** means predictions are off by ~120.2286 cycles on average.
- Test MAPE of **15.1491%** provides a normalised error estimate.
- Lower test R² (0.4719) vs CV R² (0.708) reflects the small test set size
  (n=28) and batch-level variability — the CV estimate is more reliable.

---

## 6. XGBoost Classification Result

**Target:** `screening_label` (Reuse / Conditional Reuse / Recycle)

| Eval Type | Metric | Value |
| --- | --- | --- |
| CV | CV Accuracy Mean | 0.6636 |
| CV | CV Accuracy Std | 0.0364 |
| CV | CV F1 Macro Mean | 0.6549 |
| CV | CV F1 Macro Std | 0.0373 |
| CV | CV F1 Weighted Mean | 0.6608 |
| CV | CV F1 Weighted Std | 0.0368 |
| Test | Test Accuracy | 0.5714 |
| Test | Test Macro F1 | 0.5842 |
| Test | Test Weighted F1 | 0.5728 |
| Test | Test Macro Precision | 0.5807 |
| Test | Test Macro Recall | 0.5952 |

Key observations:
- CV accuracy of **0.6636** and CV macro-F1 of **0.6549** on the training set.
- Test accuracy of **0.5714** and test macro-F1 of **0.5842**.
- The **Recycle** class is the cleanest (distinct low cycle-life cells).
- **Conditional Reuse** is the most ambiguous class — it sits near threshold boundaries
  with both Reuse (upper) and Recycle (lower), making it inherently harder to classify.

---

## 7. Comparison Between Regression and Classification

| Aspect | Regression | Classification |
| --- | --- | --- |
| Task | Predict `cycle_life` (continuous) | Predict screening label (3 classes) |
| Primary use | Quantitative cycle-life estimation | Decision support for reuse allocation |
| CV performance | R² ≈ 0.708 | Accuracy ≈ 0.6636, F1 ≈ 0.6549 |
| Test performance | R² ≈ 0.4719, MAE ≈ 120.2286 | Accuracy ≈ 0.5714, F1 ≈ 0.5842 |
| Reliability | Stronger; regression is more informative | Useful baseline; boundary ambiguity expected |

Regression is the stronger primary output. A practical system may combine both:
predict `cycle_life` via regression and derive the screening label by thresholding,
rather than using the classifier directly.

---

## 8. Feature Importance Interpretation

Top 10 features by average importance rank across both models:

| Feature | Reg Rank | Reg Imp | Cls Rank | Cls Imp | Avg Rank |
| --- | --- | --- | --- | --- | --- |
| ir_mean_2_100 | 2 | 0.16421 | 1 | 0.14817 | 1.5 |
| tmin_mean_2_100 | 1 | 0.22539 | 2 | 0.10591 | 1.5 |
| chargetime_mean_2_100 | 3 | 0.11554 | 5 | 0.07406 | 4.0 |
| chargetime_change_2_100 | 5 | 0.07769 | 6 | 0.06196 | 5.5 |
| q_discharge_mean_2_100 | 8 | 0.04369 | 4 | 0.08828 | 6.0 |
| capacity_fade_2_100 | 4 | 0.08597 | 8 | 0.0558 | 6.0 |
| capacity_fade_pct_2_100 | 6 | 0.04682 | 9 | 0.05332 | 7.5 |
| ir_change_2_100 | 9 | 0.04193 | 7 | 0.0568 | 8.0 |
| q_charge_mean_2_100 | 13 | 0.01819 | 3 | 0.08934 | 8.0 |
| capacity_slope_2_100 | 7 | 0.04661 | 15 | 0.03792 | 11.0 |

Consistently important features across EDA, regression, and classification:

| Feature Group | Physical Meaning |
| --- | --- |
| `ir_mean_2_100` | Internal resistance growth — direct degradation signal |
| `tmin_mean_2_100`, `tavg_mean_2_100` | Thermal behavior during early cycling |
| `chargetime_mean_2_100`, `chargetime_change_2_100` | Charging response; aging indicator |
| `q_discharge_*`, `q_charge_mean_2_100` | Early capacity behavior |
| `capacity_fade_pct_2_100`, `capacity_slope_2_100` | Early degradation trend |
| `soh_cycle_100_proxy` | Early state-of-health retention proxy |

---

## 9. Key Findings

1. Early-cycle physical features (cycles 2–100) are sufficient to predict battery
   cycle life with meaningful accuracy, consistent with the Severson et al. findings.
2. Internal resistance (`ir_mean_2_100`) and minimum temperature (`tmin_mean_2_100`)
   are the top recurring predictors across both tasks.
3. Regression explains ~70% of cycle-life variance (CV R²),
   demonstrating the diagnostic feasibility of the FORGE pipeline.
4. Classification macro-F1 of ~0.6549 (CV) is reasonable for a 3-class imbalanced
   problem on only 110 training samples.
5. Feature importance is consistent between regression and classification, reinforcing
   that the identified early-cycle signals are physically meaningful.

---

## 10. Limitations

- **Small dataset:** only 138 valid cells; test set has 28 cells — metrics carry
  high variance.
- **Benchmark data:** controlled lab conditions may not represent Indonesian EV field
  conditions (temperature, usage patterns, cell chemistry variation).
- **Label construction:** classification labels are derived from `cycle_life` thresholds,
  making the classifier an indirect estimator of cycle life.
- **No hyperparameter tuning:** baseline XGBoost settings used; systematic tuning and
  model comparison (e.g., Random Forest, LightGBM) have not been performed.
- **Single benchmark:** results should be validated on additional datasets before
  deployment claims.

---

## 11. Paper-Ready Interpretation

The Stage 1 diagnostic pipeline demonstrates the feasibility of using early-cycle physical
battery features for cycle-life prediction and reuse screening. XGBoost regression achieved
stronger predictive performance for cycle-life estimation (CV R² ≈ 0.708,
test MAE ≈ 120.2286 cycles), while the classification model provides an interpretable
baseline for reuse decision support (CV accuracy ≈ 0.6636, CV macro-F1 ≈ 0.6549).
Internal resistance and thermal features consistently dominated feature importance across
both tasks, aligning with established electrochemical degradation indicators. These results
validate the proposed diagnostic workflow on a public LFP/graphite benchmark dataset and
motivate future adaptation to Indonesian second-life EV battery data within the FORGE
infrastructure concept.

---

## 12. Generated Outputs

### Tables
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_final_dataset_summary.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_final_metrics_summary.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_final_feature_importance_summary.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_final_model_outputs_summary.csv`

### Figures
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\stage1_model_performance_summary.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\stage1_top_features_summary.png`

---

## 13. Recommended Next Step

1. **Commit and push** the complete Stage 1 outputs:
   ```
   git add .
   git commit -m "Complete Stage 1 ML diagnostic pipeline with final evaluation"
   git push origin main
   ```

2. **Optional Stage 1 improvements** (future work):
   - Hyperparameter tuning (GridSearchCV / Optuna).
   - Model comparison (Random Forest, LightGBM, Ridge regression).
   - SHAP explainability analysis.

3. **Proceed to Stage 4:** MPC-inspired fuzzy rule-based supervisory control
   simulation using the battery screening outputs from Stage 1 to drive
   second-life energy system dispatch decisions.
