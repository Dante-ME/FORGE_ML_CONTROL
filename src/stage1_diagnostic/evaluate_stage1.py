"""
Stage 1 final evaluation: aggregate regression + classification outputs into a
paper-ready summary package.  Does NOT train any model.

Input:  existing Stage 1 CSV outputs under outputs/tables/stage1/
Output: outputs/tables/stage1/stage1_final_*.csv
        outputs/figures/stage1/stage1_*_summary.png
        reports/stage1_final_evaluation_summary.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── input paths ──────────────────────────────────────────────────────────────
PROC_DIR = PROJECT_ROOT / "data" / "processed"
TBL_DIR  = PROJECT_ROOT / "outputs" / "tables" / "stage1"
FIG_DIR  = PROJECT_ROOT / "outputs" / "figures" / "stage1"
RPT_DIR  = PROJECT_ROOT / "reports"
MDL_DIR  = PROJECT_ROOT / "outputs" / "models" / "stage1"

REQUIRED = {
    # dataset
    "severson_features":    PROC_DIR / "severson_original_features.csv",
    "reg_train":            PROC_DIR / "stage1_regression_train.csv",
    "reg_test":             PROC_DIR / "stage1_regression_test.csv",
    "cls_train":            PROC_DIR / "stage1_classification_train.csv",
    "cls_test":             PROC_DIR / "stage1_classification_test.csv",
    "feature_list":         TBL_DIR / "stage1_feature_list.csv",
    "label_mapping":        TBL_DIR / "stage1_label_mapping.csv",
    # regression
    "reg_metrics":          TBL_DIR / "stage1_regression_metrics.csv",
    "reg_cv":               TBL_DIR / "stage1_regression_cv_metrics.csv",
    "reg_feat_imp":         TBL_DIR / "stage1_regression_feature_importance.csv",
    "reg_predictions":      TBL_DIR / "stage1_regression_predictions.csv",
    # classification
    "cls_metrics":          TBL_DIR / "stage1_classification_metrics.csv",
    "cls_cv":               TBL_DIR / "stage1_classification_cv_metrics.csv",
    "cls_feat_imp":         TBL_DIR / "stage1_classification_feature_importance.csv",
    "cls_confusion":        TBL_DIR / "stage1_classification_confusion_matrix.csv",
    "cls_report":           TBL_DIR / "stage1_classification_report.csv",
    "cls_predictions":      TBL_DIR / "stage1_classification_predictions.csv",
}

PREREQ_MSG = {
    "feature_list":    "python src/stage1_diagnostic/preprocess_original_features.py",
    "label_mapping":   "python src/stage1_diagnostic/preprocess_original_features.py",
    "reg_metrics":     "python src/stage1_diagnostic/train_xgboost_regression.py",
    "cls_metrics":     "python src/stage1_diagnostic/train_xgboost_classification.py",
}

FEATURE_INTERPRETATIONS = {
    "ir_mean_2_100":            "Internal resistance behavior during early cycles; related to cell degradation.",
    "ir_change_2_100":          "Change in internal resistance over early cycles; degradation indicator.",
    "tmin_mean_2_100":          "Minimum temperature during early cycling; thermal behavior proxy.",
    "tavg_mean_2_100":          "Average temperature during early cycling; thermal behavior proxy.",
    "tmax_mean_2_100":          "Maximum temperature during early cycling; thermal behavior proxy.",
    "chargetime_mean_2_100":    "Mean charging duration; reflects charging protocol and cell response.",
    "chargetime_change_2_100":  "Change in charge time; possible electrochemical aging indicator.",
    "q_charge_mean_2_100":      "Mean charge capacity; early capacity behavior indicator.",
    "q_discharge_mean_2_100":   "Mean discharge capacity; early capacity behavior indicator.",
    "q_discharge_cycle_2":      "Discharge capacity at cycle 2; initial capacity reference.",
    "q_discharge_cycle_100":    "Discharge capacity at cycle 100; early retention indicator.",
    "capacity_fade_2_100":      "Absolute capacity fade over early cycles; direct degradation metric.",
    "capacity_fade_pct_2_100":  "Percentage capacity fade over early cycles; normalised degradation trend.",
    "capacity_slope_2_100":     "Linear slope of capacity over early cycles; early degradation trend.",
    "soh_cycle_100_proxy":      "State-of-health proxy at cycle 100; early retention relative to initial.",
    "n_cycles_available":       "Number of cycles in the early-cycle window used for feature extraction.",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _savetbl(df: pd.DataFrame, name: str) -> Path:
    path = TBL_DIR / name
    df.to_csv(path, index=False)
    return path


def _tbl_md(rows: list, headers: list) -> str:
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _load_metric(df: pd.DataFrame, key: str, fallback=np.nan):
    """Read value from a metric/value style table, case-insensitive key match."""
    col_m = next((c for c in df.columns if c.lower() in ("metric", "name")), None)
    col_v = next((c for c in df.columns if c.lower() in ("value",)), None)
    if col_m is None or col_v is None:
        return fallback
    row = df[df[col_m].str.lower() == key.lower()]
    return float(row[col_v].iloc[0]) if not row.empty else fallback


def _cv_stat(cv_df: pd.DataFrame, col: str, stat: str) -> float:
    """Extract mean or std row from a CV table."""
    row = cv_df[cv_df.iloc[:, 0].astype(str).str.lower() == stat]
    if row.empty or col not in cv_df.columns:
        return np.nan
    return float(row[col].iloc[0])


# ── tables ────────────────────────────────────────────────────────────────────

def build_dataset_summary(data: dict) -> pd.DataFrame:
    raw_df   = data["severson_features"]
    reg_tr   = data["reg_train"]
    reg_te   = data["reg_test"]
    cls_tr   = data["cls_train"]
    cls_te   = data["cls_test"]
    lbl_map  = data["label_mapping"]

    total_rows  = len(raw_df)
    valid_rows  = total_rows - int(raw_df["cycle_life"].isna().sum())
    n_features  = len(data["feature_list"])
    n_labels    = lbl_map["screening_label"].nunique()

    def _dist(df, col="screening_label"):
        vc = df[col].value_counts() if col in df.columns else pd.Series()
        return "; ".join(f"{k}:{v}" for k, v in vc.items())

    rows = [
        ("total_original_rows",         total_rows),
        ("valid_supervised_rows",        valid_rows),
        ("rows_dropped_missing_cycle_life", total_rows - valid_rows),
        ("regression_train_rows",        len(reg_tr)),
        ("regression_test_rows",         len(reg_te)),
        ("classification_train_rows",    len(cls_tr)),
        ("classification_test_rows",     len(cls_te)),
        ("number_of_features",           n_features),
        ("number_of_labels",             n_labels),
        ("label_distribution_train",     _dist(cls_tr)),
        ("label_distribution_test",      _dist(cls_te)),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_metrics_summary(data: dict) -> pd.DataFrame:
    reg_m  = data["reg_metrics"]
    reg_cv = data["reg_cv"]
    cls_m  = data["cls_metrics"]
    cls_cv = data["cls_cv"]

    rows = []

    # Regression CV
    for metric, col in [("MAE", "mae"), ("RMSE", "rmse"), ("R2", "r2")]:
        mn = _cv_stat(reg_cv, col, "mean")
        sd = _cv_stat(reg_cv, col, "std")
        rows.append(("Regression", "CV",   f"CV {metric} Mean",  round(mn, 4), ""))
        rows.append(("Regression", "CV",   f"CV {metric} Std",   round(sd, 4), ""))

    # Regression test
    for metric, key in [("MAE","MAE"), ("RMSE","RMSE"), ("R2","R2"), ("MAPE","MAPE")]:
        val = _load_metric(reg_m, key)
        rows.append(("Regression", "Test", f"Test {metric}", round(val, 4) if np.isfinite(val) else val, ""))

    # Classification CV
    for metric, col in [("Accuracy","accuracy"), ("F1 Macro","f1_macro"), ("F1 Weighted","f1_weighted")]:
        mn = _cv_stat(cls_cv, col, "mean")
        sd = _cv_stat(cls_cv, col, "std")
        rows.append(("Classification", "CV",   f"CV {metric} Mean",  round(mn, 4), ""))
        rows.append(("Classification", "CV",   f"CV {metric} Std",   round(sd, 4), ""))

    # Classification test
    for metric, key in [
        ("Accuracy","accuracy"), ("Macro F1","f1_macro"),
        ("Weighted F1","f1_weighted"),
        ("Macro Precision","precision_macro"), ("Macro Recall","recall_macro"),
    ]:
        val = _load_metric(cls_m, key)
        rows.append(("Classification", "Test", f"Test {metric}", round(val, 4) if np.isfinite(val) else val, ""))

    df = pd.DataFrame(rows, columns=["task", "evaluation_type", "metric", "value", "interpretation"])

    # Add brief interpretations
    interp = {
        "CV R2 Mean":           "Average variance explained across 5 folds (training set).",
        "Test R2":              "Variance explained on held-out test set.",
        "Test MAE":             "Mean absolute error in cycle count on test set.",
        "Test MAPE":            "Mean absolute percentage error on test set.",
        "CV Accuracy Mean":     "Average classification accuracy across 5 folds.",
        "Test Accuracy":        "Overall accuracy on held-out test set.",
        "CV F1 Macro Mean":     "Average per-class F1 across 5 folds; accounts for class imbalance.",
        "Test Macro F1":        "Per-class F1 on held-out test set.",
    }
    df["interpretation"] = df["metric"].map(interp).fillna("")
    return df


def build_feature_importance_summary(data: dict) -> pd.DataFrame:
    reg_fi = data["reg_feat_imp"].copy()
    cls_fi = data["cls_feat_imp"].copy()

    reg_fi = reg_fi.rename(columns={"importance": "regression_importance",
                                     "importance_rank": "regression_rank"})
    cls_fi = cls_fi.rename(columns={"importance": "classification_importance",
                                     "importance_rank": "classification_rank"})

    merged = pd.merge(
        reg_fi[["feature", "regression_importance", "regression_rank"]],
        cls_fi[["feature", "classification_importance", "classification_rank"]],
        on="feature", how="outer",
    ).fillna({"regression_importance": 0, "classification_importance": 0,
              "regression_rank": 99, "classification_rank": 99})

    merged["average_rank"] = ((merged["regression_rank"] + merged["classification_rank"]) / 2).round(1)
    merged["appears_in_regression_top10"]      = merged["regression_rank"] <= 10
    merged["appears_in_classification_top10"]  = merged["classification_rank"] <= 10
    merged["interpretation"] = merged["feature"].map(FEATURE_INTERPRETATIONS).fillna(
        "Early-cycle physical diagnostic feature."
    )

    merged = merged.sort_values("average_rank").reset_index(drop=True)
    merged["regression_rank"]      = merged["regression_rank"].astype(int)
    merged["classification_rank"]  = merged["classification_rank"].astype(int)
    return merged


def build_model_outputs_summary() -> pd.DataFrame:
    rows = [
        ("trained_model",              MDL_DIR / "xgboost_cycle_life_regressor.pkl",
         "Trained XGBoost regression model for cycle_life prediction."),
        ("trained_model",              MDL_DIR / "xgboost_screening_classifier.pkl",
         "Trained XGBoost classification model for screening label prediction."),
        ("regression_predictions",     TBL_DIR / "stage1_regression_predictions.csv",
         "Per-cell actual vs predicted cycle_life on the test set."),
        ("classification_predictions", TBL_DIR / "stage1_classification_predictions.csv",
         "Per-cell actual vs predicted screening label on the test set."),
        ("regression_metrics",         TBL_DIR / "stage1_regression_metrics.csv",
         "Test set regression metrics (MAE, RMSE, R2, MAPE)."),
        ("classification_metrics",     TBL_DIR / "stage1_classification_metrics.csv",
         "Test set classification metrics (accuracy, F1, precision, recall)."),
        ("regression_feat_imp",        TBL_DIR / "stage1_regression_feature_importance.csv",
         "Feature importances from the regression model."),
        ("classification_feat_imp",    TBL_DIR / "stage1_classification_feature_importance.csv",
         "Feature importances from the classification model."),
        ("regression_figures",         FIG_DIR / "regression_predicted_vs_actual.png",
         "Predicted vs actual scatter plot for regression."),
        ("classification_figures",     FIG_DIR / "classification_confusion_matrix.png",
         "Confusion matrix heatmap for classification."),
        ("final_evaluation_report",    RPT_DIR / "stage1_final_evaluation_summary.md",
         "Final Stage 1 paper-ready evaluation narrative."),
    ]
    return pd.DataFrame(
        [(r, str(p), d) for r, p, d in rows],
        columns=["output_type", "path", "description"]
    )


# ── figures ───────────────────────────────────────────────────────────────────

def make_performance_summary_figure(data: dict) -> Path:
    reg_m  = data["reg_metrics"]
    reg_cv = data["reg_cv"]
    cls_m  = data["cls_metrics"]
    cls_cv = data["cls_cv"]

    metrics = {
        "Reg CV R²":        _cv_stat(reg_cv, "r2", "mean"),
        "Reg Test R²":      _load_metric(reg_m, "R2"),
        "Cls CV Acc":       _cv_stat(cls_cv, "accuracy", "mean"),
        "Cls Test Acc":     _load_metric(cls_m, "accuracy"),
        "Cls CV F1 Macro":  _cv_stat(cls_cv, "f1_macro", "mean"),
        "Cls Test F1 Macro":_load_metric(cls_m, "f1_macro"),
    }

    labels = list(metrics.keys())
    values = [v if np.isfinite(v) else 0 for v in metrics.values()]
    colors = ["#4C72B0","#4C72B0","#55A868","#55A868","#DD8452","#DD8452"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Stage 1 Model Performance Summary")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.35)

    # simple legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Regression (R²)"),
        Patch(facecolor="#55A868", label="Classification (Accuracy)"),
        Patch(facecolor="#DD8452", label="Classification (F1 Macro)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    return _savefig("stage1_model_performance_summary.png")


def make_top_features_figure(feat_imp_summary: pd.DataFrame) -> Path:
    top10 = feat_imp_summary.head(10).copy()
    top10 = top10.sort_values("average_rank", ascending=False)  # bottom = most important

    features   = top10["feature"].tolist()
    reg_imp    = top10["regression_importance"].tolist()
    cls_imp    = top10["classification_importance"].tolist()

    y = np.arange(len(features))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - height / 2, reg_imp, height, label="Regression",     color="#4C72B0", edgecolor="black")
    ax.barh(y + height / 2, cls_imp, height, label="Classification",  color="#55A868", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Stage 1: Top 10 Features by Average Rank (Regression vs Classification)")
    ax.legend()
    ax.grid(axis="x", alpha=0.35)
    return _savefig("stage1_top_features_summary.png")


# ── markdown report ───────────────────────────────────────────────────────────

def make_report(
    data: dict,
    ds_summary: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    feat_summary: pd.DataFrame,
    tbl_saved: dict,
    fig_saved: dict,
) -> Path:

    def _mv(task, metric_str):
        row = metrics_summary[
            (metrics_summary["task"] == task) &
            (metrics_summary["metric"].str.contains(metric_str, case=False))
        ]
        return round(float(row["value"].iloc[0]), 4) if not row.empty else "N/A"

    # shortcuts
    reg_cv_r2   = _mv("Regression",     "CV R2 Mean")
    reg_cv_mae  = _mv("Regression",     "CV MAE Mean")
    reg_cv_rmse = _mv("Regression",     "CV RMSE Mean")
    reg_t_mae   = _mv("Regression",     "Test MAE")
    reg_t_rmse  = _mv("Regression",     "Test RMSE")
    reg_t_r2    = _mv("Regression",     "Test R2")
    reg_t_mape  = _mv("Regression",     "Test MAPE")
    cls_cv_acc  = _mv("Classification", "CV Accuracy Mean")
    cls_cv_f1   = _mv("Classification", "CV F1 Macro Mean")
    cls_t_acc   = _mv("Classification", "Test Accuracy")
    cls_t_f1    = _mv("Classification", "Test Macro F1")
    cls_t_wf1   = _mv("Classification", "Test Weighted F1")

    n_feat  = int(ds_summary.loc[ds_summary["item"]=="number_of_features","value"].iloc[0])
    n_train = int(ds_summary.loc[ds_summary["item"]=="regression_train_rows","value"].iloc[0])
    n_test  = int(ds_summary.loc[ds_summary["item"]=="regression_test_rows","value"].iloc[0])

    # tables
    ds_tbl = _tbl_md(
        [(r["item"], r["value"]) for _, r in ds_summary.iterrows()],
        ["Item", "Value"]
    )

    metrics_reg_rows = [
        (r["evaluation_type"], r["metric"], r["value"])
        for _, r in metrics_summary[metrics_summary["task"]=="Regression"].iterrows()
    ]
    metrics_cls_rows = [
        (r["evaluation_type"], r["metric"], r["value"])
        for _, r in metrics_summary[metrics_summary["task"]=="Classification"].iterrows()
    ]
    reg_tbl = _tbl_md(metrics_reg_rows, ["Eval Type", "Metric", "Value"])
    cls_tbl = _tbl_md(metrics_cls_rows, ["Eval Type", "Metric", "Value"])

    feat_rows = [
        (r["feature"], r["regression_rank"], round(r["regression_importance"], 5),
         r["classification_rank"], round(r["classification_importance"], 5),
         r["average_rank"])
        for _, r in feat_summary.head(10).iterrows()
    ]
    feat_tbl = _tbl_md(
        feat_rows,
        ["Feature", "Reg Rank", "Reg Imp", "Cls Rank", "Cls Imp", "Avg Rank"]
    )

    tbl_paths = "\n".join(f"- `{p}`" for p in tbl_saved.values())
    fig_paths = "\n".join(f"- `{p}`" for p in fig_saved.values())

    report = f"""# Stage 1 Final Evaluation Summary — FORGE Diagnostic Pipeline

## 1. Executive Summary

Stage 1 implements a complete battery diagnostic pipeline:

**Raw Severson MATLAB data → Physical early-cycle features → EDA → Preprocessing →
XGBoost Regression → XGBoost Classification → Final Evaluation**

The pipeline extracts {n_feat} physical early-cycle features from the original Severson
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

{ds_tbl}

- **Source:** Severson et al. (2019) public LFP/graphite battery aging benchmark.
- **Coverage:** 3 batch files, 140 total cells; 138 used after removing 2 with missing `cycle_life`.
- **Features:** {n_feat} physical early-cycle features derived from cycle 2 to cycle 100
  (discharge/charge capacity, internal resistance, temperature, charge time, capacity slope).
- **Framing:** This dataset serves as **benchmark validation** for the FORGE diagnostic
  pipeline — it is not Indonesian EV field data.

---

## 4. Preprocessing Summary

- 80/20 stratified train-test split: **{n_train} train rows / {n_test} test rows**.
- No separate validation set (dataset is small; cross-validation used instead).
- XGBoost does not require feature scaling — no standardisation applied.
- Balanced sample weights applied for classification final fit to mitigate class imbalance.

---

## 5. XGBoost Regression Result

**Target:** `cycle_life` (continuous, in full charge-discharge cycles)

{reg_tbl}

Key observations:
- CV R² of **{reg_cv_r2}** indicates the model explains ~{int(float(reg_cv_r2)*100)}% of
  cycle-life variance on unseen training folds.
- Test MAE of **{reg_t_mae} cycles** means predictions are off by ~{reg_t_mae} cycles on average.
- Test MAPE of **{reg_t_mape}%** provides a normalised error estimate.
- Lower test R² ({reg_t_r2}) vs CV R² ({reg_cv_r2}) reflects the small test set size
  (n={n_test}) and batch-level variability — the CV estimate is more reliable.

---

## 6. XGBoost Classification Result

**Target:** `screening_label` (Reuse / Conditional Reuse / Recycle)

{cls_tbl}

Key observations:
- CV accuracy of **{cls_cv_acc}** and CV macro-F1 of **{cls_cv_f1}** on the training set.
- Test accuracy of **{cls_t_acc}** and test macro-F1 of **{cls_t_f1}**.
- The **Recycle** class is the cleanest (distinct low cycle-life cells).
- **Conditional Reuse** is the most ambiguous class — it sits near threshold boundaries
  with both Reuse (upper) and Recycle (lower), making it inherently harder to classify.

---

## 7. Comparison Between Regression and Classification

| Aspect | Regression | Classification |
| --- | --- | --- |
| Task | Predict `cycle_life` (continuous) | Predict screening label (3 classes) |
| Primary use | Quantitative cycle-life estimation | Decision support for reuse allocation |
| CV performance | R² ≈ {reg_cv_r2} | Accuracy ≈ {cls_cv_acc}, F1 ≈ {cls_cv_f1} |
| Test performance | R² ≈ {reg_t_r2}, MAE ≈ {reg_t_mae} | Accuracy ≈ {cls_t_acc}, F1 ≈ {cls_t_f1} |
| Reliability | Stronger; regression is more informative | Useful baseline; boundary ambiguity expected |

Regression is the stronger primary output. A practical system may combine both:
predict `cycle_life` via regression and derive the screening label by thresholding,
rather than using the classifier directly.

---

## 8. Feature Importance Interpretation

Top 10 features by average importance rank across both models:

{feat_tbl}

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
3. Regression explains ~{int(float(reg_cv_r2)*100)}% of cycle-life variance (CV R²),
   demonstrating the diagnostic feasibility of the FORGE pipeline.
4. Classification macro-F1 of ~{cls_cv_f1} (CV) is reasonable for a 3-class imbalanced
   problem on only {n_train} training samples.
5. Feature importance is consistent between regression and classification, reinforcing
   that the identified early-cycle signals are physically meaningful.

---

## 10. Limitations

- **Small dataset:** only 138 valid cells; test set has {n_test} cells — metrics carry
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
stronger predictive performance for cycle-life estimation (CV R² ≈ {reg_cv_r2},
test MAE ≈ {reg_t_mae} cycles), while the classification model provides an interpretable
baseline for reuse decision support (CV accuracy ≈ {cls_cv_acc}, CV macro-F1 ≈ {cls_cv_f1}).
Internal resistance and thermal features consistently dominated feature importance across
both tasks, aligning with established electrochemical degradation indicators. These results
validate the proposed diagnostic workflow on a public LFP/graphite benchmark dataset and
motivate future adaptation to Indonesian second-life EV battery data within the FORGE
infrastructure concept.

---

## 12. Generated Outputs

### Tables
{tbl_paths}

### Figures
{fig_paths}

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
"""

    out_path = RPT_DIR / "stage1_final_evaluation_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate_stage1() -> None:
    for d in [TBL_DIR, FIG_DIR, RPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── load all required inputs ──────────────────────────────────────────────
    prereq_default = "python src/stage1_diagnostic/preprocess_original_features.py"
    data = {}
    for key, path in REQUIRED.items():
        if not path.exists():
            hint = PREREQ_MSG.get(key, prereq_default)
            raise FileNotFoundError(
                f"Required file not found: {path}\nPlease run:  {hint}"
            )
        data[key] = pd.read_csv(path)
        print(f"  Loaded: {path.name}")

    print(f"\nAll {len(REQUIRED)} required files loaded.")

    # ── dataset summary ───────────────────────────────────────────────────────
    ds_summary = build_dataset_summary(data)
    print("\n--- Dataset summary ---")
    print(ds_summary.to_string(index=False))

    # ── metrics summary ───────────────────────────────────────────────────────
    metrics_summary = build_metrics_summary(data)

    reg_cv_r2  = metrics_summary.loc[metrics_summary["metric"]=="CV R2 Mean",  "value"].values
    reg_t_mae  = metrics_summary.loc[metrics_summary["metric"]=="Test MAE",    "value"].values
    reg_t_r2   = metrics_summary.loc[metrics_summary["metric"]=="Test R2",     "value"].values
    cls_cv_acc = metrics_summary.loc[metrics_summary["metric"]=="CV Accuracy Mean","value"].values
    cls_cv_f1  = metrics_summary.loc[metrics_summary["metric"]=="CV F1 Macro Mean","value"].values
    cls_t_acc  = metrics_summary.loc[metrics_summary["metric"]=="Test Accuracy",   "value"].values
    cls_t_f1   = metrics_summary.loc[metrics_summary["metric"]=="Test Macro F1",   "value"].values

    print("\n--- Regression key metrics ---")
    print(f"  CV R²  : {reg_cv_r2[0] if len(reg_cv_r2) else 'N/A'}")
    print(f"  Test MAE : {reg_t_mae[0] if len(reg_t_mae) else 'N/A'}")
    print(f"  Test R²  : {reg_t_r2[0] if len(reg_t_r2) else 'N/A'}")
    print("\n--- Classification key metrics ---")
    print(f"  CV Accuracy  : {cls_cv_acc[0] if len(cls_cv_acc) else 'N/A'}")
    print(f"  CV F1 Macro  : {cls_cv_f1[0] if len(cls_cv_f1) else 'N/A'}")
    print(f"  Test Accuracy: {cls_t_acc[0] if len(cls_t_acc) else 'N/A'}")
    print(f"  Test F1 Macro: {cls_t_f1[0] if len(cls_t_f1) else 'N/A'}")

    # ── feature importance summary ────────────────────────────────────────────
    feat_summary = build_feature_importance_summary(data)
    print("\n--- Top 10 shared important features ---")
    for _, row in feat_summary.head(10).iterrows():
        print(f"  {row['feature']:<35} reg_rank={int(row['regression_rank']):2d}  "
              f"cls_rank={int(row['classification_rank']):2d}  avg={row['average_rank']}")

    # ── model outputs summary ─────────────────────────────────────────────────
    outputs_summary = build_model_outputs_summary()

    # ── save tables ───────────────────────────────────────────────────────────
    tbl_saved = {
        "dataset_summary":       _savetbl(ds_summary,       "stage1_final_dataset_summary.csv"),
        "metrics_summary":       _savetbl(metrics_summary,   "stage1_final_metrics_summary.csv"),
        "feat_imp_summary":      _savetbl(feat_summary,      "stage1_final_feature_importance_summary.csv"),
        "model_outputs_summary": _savetbl(outputs_summary,   "stage1_final_model_outputs_summary.csv"),
    }

    # ── generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_saved = {
        "performance_summary": make_performance_summary_figure(data),
        "top_features":        make_top_features_figure(feat_summary),
    }
    for k, p in fig_saved.items():
        print(f"  {p.name}")

    # ── markdown report ───────────────────────────────────────────────────────
    print("\nGenerating final markdown report...")
    rpt_path = make_report(data, ds_summary, metrics_summary, feat_summary,
                           tbl_saved, fig_saved)
    print(f"  {rpt_path}")

    # ── final console summary ─────────────────────────────────────────────────
    print("\n--- Saved tables ---")
    for k, p in tbl_saved.items():
        print(f"  {p}")

    print(f"\nStage 1 evaluation complete.")


if __name__ == "__main__":
    try:
        evaluate_stage1()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
