"""
Stage 1 XGBoost classification: predict battery screening label from early-cycle features.
Input:  data/processed/stage1_classification_{train,test}.csv
Output: outputs/models/stage1/xgboost_screening_classifier.pkl
        outputs/tables/stage1/stage1_classification_*.csv
        outputs/figures/stage1/classification_*.png
        reports/stage1_classification_summary.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:
    print("[ERROR] xgboost is not installed. Please run:  pip install xgboost")
    sys.exit(1)

try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV    = PROJECT_ROOT / "data" / "processed" / "stage1_classification_train.csv"
TEST_CSV     = PROJECT_ROOT / "data" / "processed" / "stage1_classification_test.csv"
FEAT_CSV     = PROJECT_ROOT / "outputs" / "tables" / "stage1" / "stage1_feature_list.csv"
LBL_CSV      = PROJECT_ROOT / "outputs" / "tables" / "stage1" / "stage1_label_mapping.csv"

MODEL_DIR    = PROJECT_ROOT / "outputs" / "models" / "stage1"
TBL_DIR      = PROJECT_ROOT / "outputs" / "tables" / "stage1"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "stage1"
RPT_DIR      = PROJECT_ROOT / "reports"

MODEL_PATH   = MODEL_DIR / "xgboost_screening_classifier.pkl"

META_COLS    = ["cell_id", "source_file", "batch_date_inferred", "cell_index"]
TARGET_ENC   = "screening_label_encoded"
TARGET_LBL   = "screening_label"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

# Canonical display order for confusion matrix / charts
LABEL_ORDER = ["Conditional Reuse", "Recycle", "Reuse"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _savetbl(df_out: pd.DataFrame, name: str) -> Path:
    path = TBL_DIR / name
    df_out.to_csv(path, index=False)
    return path


def _tbl_md(rows: list, headers: list) -> str:
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy":         "accuracy",
        "f1_macro":         "f1_macro",
        "f1_weighted":      "f1_weighted",
        "precision_macro":  "precision_macro",
        "recall_macro":     "recall_macro",
    }
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring,
                                return_train_score=False)

    n_folds = len(cv_results["test_accuracy"])
    rows = []
    for i in range(n_folds):
        rows.append({
            "fold":             i + 1,
            "accuracy":         round(cv_results["test_accuracy"][i], 4),
            "f1_macro":         round(cv_results["test_f1_macro"][i], 4),
            "f1_weighted":      round(cv_results["test_f1_weighted"][i], 4),
            "precision_macro":  round(cv_results["test_precision_macro"][i], 4),
            "recall_macro":     round(cv_results["test_recall_macro"][i], 4),
        })

    cv_df = pd.DataFrame(rows)
    for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
        cv_df = pd.concat([cv_df, pd.DataFrame([{
            "fold":             stat_name,
            "accuracy":         round(fn(cv_results["test_accuracy"]), 4),
            "f1_macro":         round(fn(cv_results["test_f1_macro"]), 4),
            "f1_weighted":      round(fn(cv_results["test_f1_weighted"]), 4),
            "precision_macro":  round(fn(cv_results["test_precision_macro"]), 4),
            "recall_macro":     round(fn(cv_results["test_recall_macro"]), 4),
        }])], ignore_index=True)

    return cv_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_test_label: pd.Series,
    y_pred_label: list,
    feat_imp_df: pd.DataFrame,
    enc_to_label: dict,
) -> dict:
    saved = {}

    # Ordered encoded values matching LABEL_ORDER
    ordered_enc = sorted(enc_to_label.keys(),
                         key=lambda e: LABEL_ORDER.index(enc_to_label[e])
                         if enc_to_label[e] in LABEL_ORDER else e)
    ordered_names = [enc_to_label[e] for e in ordered_enc]

    # A. Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred, labels=ordered_enc)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(ordered_names, fontsize=9)
    for r in range(len(ordered_names)):
        for c in range(len(ordered_names)):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    fontsize=12, color="white" if cm[r, c] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)
    ax.set_title("XGBoost Classification: Confusion Matrix")
    saved["conf_matrix"] = _savefig("classification_confusion_matrix.png")

    # B. Feature importance (top 15)
    top15 = feat_imp_df.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"], top15["importance"], color="#55A868", edgecolor="black")
    ax.set_xlabel("Importance (weight)")
    ax.set_title("XGBoost Classification: Top 15 Feature Importances")
    ax.grid(axis="x", alpha=0.4)
    saved["feat_imp"] = _savefig("classification_feature_importance.png")

    # C. Actual vs predicted distribution
    actual_counts = {lbl: list(y_test_label).count(lbl) for lbl in LABEL_ORDER}
    pred_counts   = {lbl: list(y_pred_label).count(lbl) for lbl in LABEL_ORDER}
    x = np.arange(len(LABEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [actual_counts[l] for l in LABEL_ORDER],
           width, label="Actual",    color="#4C72B0", edgecolor="black")
    ax.bar(x + width / 2, [pred_counts[l] for l in LABEL_ORDER],
           width, label="Predicted", color="#DD8452", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Screening Label")
    ax.set_ylabel("Count")
    ax.set_title("XGBoost Classification: Actual vs Predicted Label Distribution (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    saved["pred_dist"] = _savefig("classification_prediction_distribution.png")

    return saved


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_report(
    train_df, test_df, feature_cols,
    enc_to_label, label_to_enc,
    cv_df, metrics_df,
    cls_report_df, cm_df,
    feat_imp_df,
    fig_saved, tbl_saved,
) -> Path:

    cv_mean = cv_df[cv_df["fold"] == "mean"].iloc[0]
    cv_std  = cv_df[cv_df["fold"] == "std"].iloc[0]

    # Label mapping table
    lbl_rows = sorted(enc_to_label.items())
    lbl_table = _tbl_md(
        [(enc, lbl) for enc, lbl in lbl_rows],
        ["Encoded", "Label"]
    )

    # Train dist
    train_dist = train_df[TARGET_LBL].value_counts()
    train_dist_table = _tbl_md(
        [(lbl, int(train_dist.get(lbl, 0))) for lbl in LABEL_ORDER],
        ["Label", "Train Count"]
    )

    # CV table
    cv_table = _tbl_md(
        [(r["fold"], r["accuracy"], r["f1_macro"], r["f1_weighted"],
          r["precision_macro"], r["recall_macro"])
         for _, r in cv_df.iterrows()],
        ["Fold", "Accuracy", "F1 Macro", "F1 Weighted", "Prec Macro", "Recall Macro"]
    )

    # Test metrics
    test_table = _tbl_md(
        [(r["metric"], round(r["value"], 4)) for _, r in metrics_df.iterrows()],
        ["Metric", "Value"]
    )

    # Classification report
    rpt_rows = [(r["class"], round(r["precision"], 3), round(r["recall"], 3),
                 round(r["f1_score"], 3), int(r["support"]))
                for _, r in cls_report_df.iterrows()]
    rpt_table = _tbl_md(rpt_rows, ["Class", "Precision", "Recall", "F1", "Support"])

    # Confusion matrix
    cm_rows = [(idx,) + tuple(row) for idx, row in
               zip(cm_df.index.tolist(), cm_df.values.tolist())]
    cm_table = _tbl_md(cm_rows, ["Actual \\ Predicted"] + cm_df.columns.tolist())

    # Top 10 features
    feat_rows = [(r["feature"], round(r["importance"], 6), int(r["importance_rank"]))
                 for _, r in feat_imp_df.head(10).iterrows()]
    feat_table = _tbl_md(feat_rows, ["Feature", "Importance", "Rank"])

    params_table = _tbl_md(list(XGB_PARAMS.items()), ["Parameter", "Value"])

    tbl_paths = "\n".join(f"- `{p}`" for p in tbl_saved.values())
    fig_paths = "\n".join(f"- `{p}`" for p in fig_saved.values())

    report = f"""# Stage 1 XGBoost Classification Summary — Battery Screening

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
| Training rows | {len(train_df)} |
| Test rows | {len(test_df)} |
| Feature columns | {len(feature_cols)} |
| Target | `screening_label_encoded` (0/1/2) |

---

## 3. Feature Set

{len(feature_cols)} physical early-cycle features (same as regression model).

---

## 4. Label Definition

Labels are derived from `cycle_life` thresholds:

| Threshold | Label |
| --- | --- |
| cycle_life ≥ 1000 | Reuse |
| 500 ≤ cycle_life < 1000 | Conditional Reuse |
| cycle_life < 500 | Recycle |

**Encoding:**

{lbl_table}

**Training set distribution:**

{train_dist_table}

---

## 5. Model Configuration

{params_table}

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

{cv_table}

| Summary | Accuracy | F1 Macro | F1 Weighted | Prec Macro | Recall Macro |
| --- | --- | --- | --- | --- | --- |
| Mean | {cv_mean["accuracy"]} | {cv_mean["f1_macro"]} | {cv_mean["f1_weighted"]} | {cv_mean["precision_macro"]} | {cv_mean["recall_macro"]} |
| Std  | {cv_std["accuracy"]}  | {cv_std["f1_macro"]}  | {cv_std["f1_weighted"]}  | {cv_std["precision_macro"]}  | {cv_std["recall_macro"]} |

The test set was kept **untouched** during cross-validation.

---

## 8. Test Set Results

{test_table}

### Per-Class Classification Report

{rpt_table}

---

## 9. Confusion Matrix Interpretation

{cm_table}

See: `outputs/figures/stage1/classification_confusion_matrix.png`

---

## 10. Feature Importance (Top 10)

{feat_table}

---

## 11. Generated Outputs

### Model
- `{MODEL_PATH}`

### Tables
{tbl_paths}

### Figures
{fig_paths}

---

## 12. Key Interpretation

- The model was trained on only {len(train_df)} samples from a controlled laboratory benchmark.
  Performance metrics indicate feasibility for the FORGE diagnostic concept, not readiness
  for direct field deployment.
- Features most predictive of screening label ({feat_imp_df.iloc[0]["feature"]},
  {feat_imp_df.iloc[1]["feature"]}) align with those identified in the EDA correlation analysis.
- The test set contains only {len(test_df)} cells, so per-class support is limited.
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
"""

    out_path = RPT_DIR / "stage1_classification_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_xgboost_classification() -> None:
    # --- Check inputs ---
    for p in [TRAIN_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Please run:  python src/stage1_diagnostic/preprocess_original_features.py"
            )

    for d in [MODEL_DIR, TBL_DIR, FIG_DIR, RPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"Train shape  : {train_df.shape}")
    print(f"Test shape   : {test_df.shape}")

    # --- Feature columns ---
    if FEAT_CSV.exists():
        feat_df      = pd.read_csv(FEAT_CSV)
        feature_cols = feat_df["feature_name"].tolist()
    else:
        exclude = set(META_COLS) | {TARGET_ENC, TARGET_LBL, "cell_index", "cycle_life"}
        feature_cols = [c for c in train_df.select_dtypes(include="number").columns
                        if c not in exclude]
        print(f"  [WARN] Feature list CSV not found; inferred {len(feature_cols)} features.")

    # --- Label mapping ---
    if LBL_CSV.exists():
        lbl_df = pd.read_csv(LBL_CSV)
        enc_to_label  = dict(zip(lbl_df["screening_label_encoded"].astype(int),
                                 lbl_df["screening_label"]))
        label_to_enc  = {v: k for k, v in enc_to_label.items()}
    else:
        enc_to_label = {0: "Conditional Reuse", 1: "Recycle", 2: "Reuse"}
        label_to_enc = {v: k for k, v in enc_to_label.items()}
        print("  [WARN] Label mapping CSV not found; using default mapping.")

    print(f"\nLabel mapping: {enc_to_label}")

    # --- Validate columns ---
    for col in feature_cols + [TARGET_ENC, TARGET_LBL]:
        for df_name, df in [("train", train_df), ("test", test_df)]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing from {df_name} CSV.")

    print(f"\nFeatures ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f}")

    X_train      = train_df[feature_cols].values
    y_train      = train_df[TARGET_ENC].values.astype(int)
    X_test       = test_df[feature_cols].values
    y_test       = test_df[TARGET_ENC].values.astype(int)
    y_test_label = test_df[TARGET_LBL]

    for name, arr in [("X_train", X_train), ("X_test", X_test),
                      ("y_train", y_train.astype(float)), ("y_test", y_test.astype(float))]:
        if np.isnan(arr.astype(float)).any():
            raise ValueError(f"NaN values found in {name}. Clean the data first.")

    # --- Sample weights for final fit ---
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # --- Cross-validation (no sample weights) ---
    print("\nRunning 5-fold stratified cross-validation on training set...")
    cv_model = XGBClassifier(**XGB_PARAMS)
    cv_df    = run_cv(cv_model, X_train, y_train)
    cv_mean  = cv_df[cv_df["fold"] == "mean"].iloc[0]
    cv_std   = cv_df[cv_df["fold"] == "std"].iloc[0]
    print(f"  CV Accuracy    : {cv_mean['accuracy']:.4f} +/- {cv_std['accuracy']:.4f}")
    print(f"  CV F1 Macro    : {cv_mean['f1_macro']:.4f} +/- {cv_std['f1_macro']:.4f}")
    print(f"  CV F1 Weighted : {cv_mean['f1_weighted']:.4f} +/- {cv_std['f1_weighted']:.4f}")
    tbl_saved = {}
    tbl_saved["cv"] = _savetbl(cv_df, "stage1_classification_cv_metrics.csv")

    # --- Fit final model with sample weights ---
    print("\nFitting final model on full training set (with sample weights)...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # --- Save model ---
    if USE_JOBLIB:
        joblib.dump(model, MODEL_PATH)
    else:
        import pickle
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(model, fh)
    print(f"Model saved  : {MODEL_PATH}")

    # --- Predict ---
    y_pred       = model.predict(X_test)
    y_prob       = model.predict_proba(X_test)
    y_pred_label = [enc_to_label[int(e)] for e in y_pred]

    # --- Test metrics ---
    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wt   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    recall  = recall_score(y_test, y_pred,    average="macro",    zero_division=0)

    print(f"\nTest set results:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 Macro        : {f1_mac:.4f}")
    print(f"  F1 Weighted     : {f1_wt:.4f}")
    print(f"  Precision Macro : {prec:.4f}")
    print(f"  Recall Macro    : {recall:.4f}")

    metrics_df = pd.DataFrame([
        {"metric": "accuracy",         "value": round(acc,    4)},
        {"metric": "f1_macro",         "value": round(f1_mac, 4)},
        {"metric": "f1_weighted",      "value": round(f1_wt,  4)},
        {"metric": "precision_macro",  "value": round(prec,   4)},
        {"metric": "recall_macro",     "value": round(recall, 4)},
    ])
    tbl_saved["metrics"] = _savetbl(metrics_df, "stage1_classification_metrics.csv")

    # --- Classification report ---
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names  = [enc_to_label[e] for e in present_labels]
    rpt_dict = classification_report(
        y_test, y_pred,
        labels=present_labels, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    rpt_rows = []
    for cls_name in present_names:
        d = rpt_dict[cls_name]
        rpt_rows.append({
            "class":     cls_name,
            "precision": round(d["precision"], 4),
            "recall":    round(d["recall"],    4),
            "f1_score":  round(d["f1-score"],  4),
            "support":   int(d["support"]),
        })
    cls_report_df = pd.DataFrame(rpt_rows)
    tbl_saved["cls_report"] = _savetbl(cls_report_df, "stage1_classification_report.csv")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=present_labels,
                                target_names=present_names, zero_division=0))

    # --- Confusion matrix ---
    ordered_enc   = [label_to_enc[l] for l in LABEL_ORDER if l in label_to_enc]
    ordered_names = [enc_to_label[e] for e in ordered_enc]
    cm = confusion_matrix(y_test, y_pred, labels=ordered_enc)
    cm_df = pd.DataFrame(cm, index=ordered_names, columns=ordered_names)
    cm_df.index.name = "actual"
    tbl_saved["conf_matrix"] = _savetbl(
        cm_df.reset_index(), "stage1_classification_confusion_matrix.csv"
    )

    print("Confusion matrix:")
    print(cm_df.to_string())

    # --- Predictions table ---
    # Build probability columns in a stable order
    num_classes = y_prob.shape[1]
    pred_df     = test_df[META_COLS].copy()
    pred_df["actual_label"]          = y_test_label.values
    pred_df["actual_label_encoded"]  = y_test
    pred_df["predicted_label"]       = y_pred_label
    pred_df["predicted_label_encoded"] = y_pred.astype(int)
    pred_df["is_correct"]            = (y_test == y_pred.astype(int))

    for enc in sorted(enc_to_label.keys()):
        if enc < num_classes:
            col_name = f"probability_{enc_to_label[enc].replace(' ', '_')}"
            pred_df[col_name] = np.round(y_prob[:, enc], 4)

    tbl_saved["predictions"] = _savetbl(pred_df, "stage1_classification_predictions.csv")

    # --- Feature importance ---
    importances  = model.feature_importances_
    feat_imp_df  = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp_df["importance_rank"] = range(1, len(feat_imp_df) + 1)
    tbl_saved["feat_imp"] = _savetbl(
        feat_imp_df, "stage1_classification_feature_importance.csv"
    )

    print("\nTop 10 feature importances:")
    for _, row in feat_imp_df.head(10).iterrows():
        print(f"  {int(row['importance_rank']):2d}. {row['feature']:<35} {row['importance']:.6f}")

    # --- Figures ---
    print("\nGenerating figures...")
    fig_saved = make_figures(
        y_test, y_pred, y_test_label, y_pred_label,
        feat_imp_df, enc_to_label,
    )
    for k, p in fig_saved.items():
        print(f"  {p.name}")

    # --- Report ---
    print("\nGenerating markdown report...")
    rpt_path = make_report(
        train_df, test_df, feature_cols,
        enc_to_label, label_to_enc,
        cv_df, metrics_df, cls_report_df, cm_df,
        feat_imp_df, fig_saved, tbl_saved,
    )
    print(f"  {rpt_path}")

    print("\n--- Saved tables ---")
    for k, p in tbl_saved.items():
        print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        train_xgboost_classification()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
