# Stage 1 XGBoost Classification Summary — Battery Screening

## 1. Objective

Train an XGBoost multi-class classifier to assign each battery cell a screening label —
**Reuse**, **Conditional Reuse**, or **Recycle** — using early-cycle physical features
extracted from the original Severson LFP/graphite benchmark dataset.

The model is intended as a **proof-of-concept** for the FORGE diagnostic pipeline.
Results should be interpreted as benchmark validation, not final field performance.

---

## 2. Input Data

| Item | Value |
| --- | --- |
| Training set | `data/processed/stage1_classification_train.csv` |
| Test set | `data/processed/stage1_classification_test.csv` |
| Training rows | 110 |
| Test rows | 28 |
| Feature columns | 16 |
| Target | `screening_label_encoded` (0/1/2) |

---

## 3. Feature Set

16 physical early-cycle features (same as regression model).

---

## 4. Label Definition

Labels are derived from `cycle_life` thresholds:

| Threshold | Label |
| --- | --- |
| cycle_life ≥ 1000 | Reuse |
| 500 ≤ cycle_life < 1000 | Conditional Reuse |
| cycle_life < 500 | Recycle |

**Encoding:**

| Encoded | Label |
| --- | --- |
| 0 | Conditional Reuse |
| 1 | Recycle |
| 2 | Reuse |

**Training set distribution:**

| Label | Train Count |
| --- | --- |
| Conditional Reuse | 55 |
| Recycle | 28 |
| Reuse | 27 |

---

## 5. Model Configuration

| Parameter | Value |
| --- | --- |
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| num_class | 3 |
| eval_metric | mlogloss |
| random_state | 42 |
| n_jobs | -1 |

XGBoost is well-suited for this task because it handles nonlinear feature-label relationships
and is robust on small tabular datasets without requiring feature scaling.

---

## 6. Class Imbalance Handling

The training set has a moderate class imbalance (Conditional Reuse is the majority class).
`sklearn.utils.class_weight.compute_sample_weight` with `class_weight="balanced"` was used
to compute per-sample weights for the final model fit.
Cross-validation was performed without sample weights (standard CV).

---

## 7. Cross-Validation Results (5-Fold Stratified, Training Set Only)

| Fold | Accuracy | F1 Macro | F1 Weighted | Prec Macro | Recall Macro |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.6818 | 0.6667 | 0.6777 | 0.6646 | 0.6758 |
| 2 | 0.6818 | 0.6548 | 0.663 | 0.7032 | 0.6505 |
| 3 | 0.6818 | 0.694 | 0.6914 | 0.7333 | 0.701 |
| 4 | 0.5909 | 0.5847 | 0.5895 | 0.5944 | 0.5788 |
| 5 | 0.6818 | 0.6743 | 0.6825 | 0.6889 | 0.6646 |
| mean | 0.6636 | 0.6549 | 0.6608 | 0.6769 | 0.6541 |
| std | 0.0364 | 0.0373 | 0.0368 | 0.0468 | 0.0411 |

| Summary | Accuracy | F1 Macro | F1 Weighted | Prec Macro | Recall Macro |
| --- | --- | --- | --- | --- | --- |
| Mean | 0.6636 | 0.6549 | 0.6608 | 0.6769 | 0.6541 |
| Std  | 0.0364  | 0.0373  | 0.0368  | 0.0468  | 0.0411 |

The test set was kept **untouched** during cross-validation.

---

## 8. Test Set Results

| Metric | Value |
| --- | --- |
| accuracy | 0.5714 |
| f1_macro | 0.5842 |
| f1_weighted | 0.5728 |
| precision_macro | 0.5807 |
| recall_macro | 0.5952 |

### Per-Class Classification Report

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Conditional Reuse | 0.583 | 0.5 | 0.538 | 14 |
| Recycle | 0.714 | 0.714 | 0.714 | 7 |
| Reuse | 0.444 | 0.571 | 0.5 | 7 |

---

## 9. Confusion Matrix Interpretation

| Actual \ Predicted | Conditional Reuse | Recycle | Reuse |
| --- | --- | --- | --- |
| Conditional Reuse | 7 | 2 | 5 |
| Recycle | 2 | 5 | 0 |
| Reuse | 3 | 0 | 4 |

See: `outputs/figures/stage1/classification_confusion_matrix.png`

---

## 10. Feature Importance (Top 10)

| Feature | Importance | Rank |
| --- | --- | --- |
| ir_mean_2_100 | 0.148174 | 1 |
| tmin_mean_2_100 | 0.10591 | 2 |
| q_charge_mean_2_100 | 0.089339 | 3 |
| q_discharge_mean_2_100 | 0.088284 | 4 |
| chargetime_mean_2_100 | 0.074064 | 5 |
| chargetime_change_2_100 | 0.061955 | 6 |
| ir_change_2_100 | 0.056797 | 7 |
| capacity_fade_2_100 | 0.055805 | 8 |
| capacity_fade_pct_2_100 | 0.053316 | 9 |
| soh_cycle_100_proxy | 0.052603 | 10 |

---

## 11. Generated Outputs

### Model
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\models\stage1\xgboost_screening_classifier.pkl`

### Tables
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_cv_metrics.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_metrics.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_report.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_confusion_matrix.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_predictions.csv`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\tables\stage1\stage1_classification_feature_importance.csv`

### Figures
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\classification_confusion_matrix.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\classification_feature_importance.png`
- `D:\DocDante\Kuliah\Lomba\REPPC UI\FORGE_ML_Control\outputs\figures\stage1\classification_prediction_distribution.png`

---

## 12. Key Interpretation

- The model was trained on only 110 samples from a controlled laboratory benchmark.
  Performance metrics indicate feasibility for the FORGE diagnostic concept, not readiness
  for direct field deployment.
- Features most predictive of screening label (ir_mean_2_100,
  tmin_mean_2_100) align with those identified in the EDA correlation analysis.
- The test set contains only 28 cells, so per-class support is limited.
  Cross-validation macro-F1 on the training set provides a more stable performance estimate.
- Sample weighting was applied to partially compensate for class imbalance in the final model.

---

## 13. Recommended Next Step

Create and run a combined Stage 1 evaluation summary:

```
python src/stage1_diagnostic/evaluate_stage1.py
```

This should consolidate regression and classification results into a unified Stage 1
performance report for the FORGE paper, including:
- Side-by-side regression and classification metrics
- Feature importance overlap analysis
- Final paper-ready tables and figures
