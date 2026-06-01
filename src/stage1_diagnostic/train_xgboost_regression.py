"""
Stage 1 XGBoost regression: predict battery cycle_life from early-cycle features.
Input:  data/processed/stage1_regression_{train,test}.csv
Output: outputs/models/stage1/xgboost_cycle_life_regressor.pkl
        outputs/tables/stage1/stage1_regression_*.csv
        outputs/figures/stage1/regression_*.png
        reports/stage1_regression_summary.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] xgboost is not installed. Please run:  pip install xgboost")
    sys.exit(1)

try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV    = PROJECT_ROOT / "data" / "processed" / "stage1_regression_train.csv"
TEST_CSV     = PROJECT_ROOT / "data" / "processed" / "stage1_regression_test.csv"
FEAT_CSV     = PROJECT_ROOT / "outputs" / "tables" / "stage1" / "stage1_feature_list.csv"

MODEL_DIR    = PROJECT_ROOT / "outputs" / "models" / "stage1"
TBL_DIR      = PROJECT_ROOT / "outputs" / "tables" / "stage1"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "stage1"
RPT_DIR      = PROJECT_ROOT / "reports"

MODEL_PATH   = MODEL_DIR / "xgboost_cycle_life_regressor.pkl"

META_COLS    = ["cell_id", "source_file", "batch_date_inferred", "cell_index"]
TARGET_COL   = "cycle_life"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)


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


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _tbl_md(rows: list, headers: list) -> str:
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(model: XGBRegressor, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=kf,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        return_train_score=False,
    )

    n_folds = len(cv_results["test_r2"])
    rows = []
    for i in range(n_folds):
        rows.append({
            "fold":  i + 1,
            "mae":   round(-cv_results["test_neg_mean_absolute_error"][i], 4),
            "rmse":  round(-cv_results["test_neg_root_mean_squared_error"][i], 4),
            "r2":    round(cv_results["test_r2"][i], 4),
        })

    cv_df = pd.DataFrame(rows)

    # Summary rows
    for stat_name, fn in [("mean", np.mean), ("std", np.std)]:
        cv_df = pd.concat([cv_df, pd.DataFrame([{
            "fold": stat_name,
            "mae":  round(fn(-cv_results["test_neg_mean_absolute_error"]), 4),
            "rmse": round(fn(-cv_results["test_neg_root_mean_squared_error"]), 4),
            "r2":   round(fn(cv_results["test_r2"]), 4),
        }])], ignore_index=True)

    return cv_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(y_test, y_pred, feat_imp_df: pd.DataFrame) -> dict:
    saved = {}

    # A. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4, color="#4C72B0", label="Predictions")
    lims = [min(y_test.min(), y_pred.min()) - 50, max(y_test.max(), y_pred.max()) + 50]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction (y=x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Cycle Life")
    ax.set_ylabel("Predicted Cycle Life")
    ax.set_title("XGBoost Regression: Predicted vs Actual Cycle Life")
    ax.legend()
    ax.grid(alpha=0.4)
    saved["pred_vs_actual"] = _savefig("regression_predicted_vs_actual.png")

    # B. Residual plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", linewidths=0.4, color="#DD8452")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted Cycle Life")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("XGBoost Regression: Residual Plot")
    ax.grid(alpha=0.4)
    saved["residual"] = _savefig("regression_residual_plot.png")

    # C. Feature importance (top 15)
    top15 = feat_imp_df.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"], top15["importance"], color="#4C72B0", edgecolor="black")
    ax.set_xlabel("Importance (weight)")
    ax.set_title("XGBoost Regression: Top 15 Feature Importances")
    ax.grid(axis="x", alpha=0.4)
    saved["feat_imp"] = _savefig("regression_feature_importance.png")

    return saved


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_report(
    train_df, test_df, feature_cols,
    cv_df, metrics_df, feat_imp_df,
    fig_saved, tbl_saved,
) -> Path:

    cv_mean = cv_df[cv_df["fold"] == "mean"].iloc[0]
    cv_std  = cv_df[cv_df["fold"] == "std"].iloc[0]

    metrics = dict(zip(metrics_df["metric"], metrics_df["value"].round(4)))

    cv_table = _tbl_md(
        [(r["fold"], r["mae"], r["rmse"], r["r2"])
         for _, r in cv_df.iterrows()],
        ["fold", "MAE", "RMSE", "R²"]
    )

    test_table = _tbl_md(
        [(r["metric"], round(r["value"], 4)) for _, r in metrics_df.iterrows()],
        ["Metric", "Value"]
    )

    feat_rows = [(r["feature"], round(r["importance"], 6), int(r["importance_rank"]))
                 for _, r in feat_imp_df.head(10).iterrows()]
    feat_table = _tbl_md(feat_rows, ["Feature", "Importance", "Rank"])

    params_table = _tbl_md(
        list(XGB_PARAMS.items()),
        ["Parameter", "Value"]
    )

    tbl_paths = "\n".join(f"- `{p}`" for p in tbl_saved.values())
    fig_paths = "\n".join(f"- `{p}`" for p in fig_saved.values())

    report = f"""# Stage 1 XGBoost Regression Summary — Cycle Life Prediction

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
| Training rows | {len(train_df)} |
| Test rows | {len(test_df)} |
| Feature columns | {len(feature_cols)} |
| Target | `cycle_life` (continuous) |

---

## 3. Feature Set

{len(feature_cols)} physical early-cycle features:

{_tbl_md([(i, f) for i, f in enumerate(feature_cols)], ["Index", "Feature"])}

---

## 4. Model Configuration

{params_table}

XGBoost is well-suited for this task because it handles nonlinear feature-target relationships
and is robust on small tabular datasets without requiring feature scaling.

---

## 5. Cross-Validation Results (5-Fold, Training Set Only)

{cv_table}

| Summary | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Mean | {cv_mean["mae"]} | {cv_mean["rmse"]} | {cv_mean["r2"]} |
| Std  | {cv_std["mae"]}  | {cv_std["rmse"]}  | {cv_std["r2"]} |

The test set was kept **untouched** during cross-validation.

---

## 6. Test Set Results

{test_table}

---

## 7. Feature Importance (Top 10)

{feat_table}

---

## 8. Generated Outputs

### Model
- `{MODEL_PATH}`

### Tables
{tbl_paths}

### Figures
{fig_paths}

---

## 9. Key Interpretation

- The model was trained on only {len(train_df)} samples from a controlled laboratory benchmark.
  Performance metrics indicate feasibility, not readiness for field deployment.
- Features most predictive of cycle life ({feat_imp_df.iloc[0]["feature"]},
  {feat_imp_df.iloc[1]["feature"]}) are consistent with the electrochemical degradation
  literature: early capacity retention and internal resistance growth are known
  early-life predictors.
- MAPE (if finite) reflects relative prediction error, useful for comparing across cell
  life ranges.
- The test set contains only {len(test_df)} cells, so test metrics carry high variance.
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
"""

    out_path = RPT_DIR / "stage1_regression_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_xgboost_regression() -> None:
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
        feat_list    = pd.read_csv(FEAT_CSV)
        feature_cols = feat_list["feature_name"].tolist()
    else:
        exclude = set(META_COLS) | {TARGET_COL}
        feature_cols = [c for c in train_df.select_dtypes(include="number").columns
                        if c not in exclude]
        print(f"  [WARN] Feature list CSV not found; inferred {len(feature_cols)} features.")

    # Validate columns exist
    for col in feature_cols + [TARGET_COL]:
        for df_name, df in [("train", train_df), ("test", test_df)]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing from {df_name} CSV.")

    print(f"\nFeatures ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f}")

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[TARGET_COL].values

    # Validate no NaN
    for name, arr in [("X_train", X_train), ("X_test", X_test),
                      ("y_train", y_train), ("y_test", y_test)]:
        if np.isnan(arr).any():
            raise ValueError(f"NaN values found in {name}. Clean the data first.")

    # --- Model ---
    model = XGBRegressor(**XGB_PARAMS)
    print(f"\nModel: XGBRegressor")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")

    # --- Cross-validation ---
    print("\nRunning 5-fold cross-validation on training set...")
    cv_model = XGBRegressor(**XGB_PARAMS)
    cv_df    = run_cv(cv_model, X_train, y_train)
    cv_mean  = cv_df[cv_df["fold"] == "mean"].iloc[0]
    cv_std   = cv_df[cv_df["fold"] == "std"].iloc[0]
    print(f"  CV MAE  : {cv_mean['mae']:.2f} ± {cv_std['mae']:.2f}")
    print(f"  CV RMSE : {cv_mean['rmse']:.2f} ± {cv_std['rmse']:.2f}")
    print(f"  CV R²   : {cv_mean['r2']:.4f} ± {cv_std['r2']:.4f}")
    tbl_saved = {}
    tbl_saved["cv"] = _savetbl(cv_df, "stage1_regression_cv_metrics.csv")

    # --- Fit final model ---
    print("\nFitting final model on full training set...")
    model.fit(X_train, y_train)

    # --- Save model ---
    if USE_JOBLIB:
        joblib.dump(model, MODEL_PATH)
    else:
        with open(MODEL_PATH, "wb") as f:
            import pickle
            pickle.dump(model, f)
    print(f"Model saved  : {MODEL_PATH}")

    # --- Test evaluation ---
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    mape = _mape(y_test, y_pred)

    print(f"\nTest set results:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%" if np.isfinite(mape) else "  MAPE : N/A")

    metrics_df = pd.DataFrame([
        {"metric": "MAE",  "value": round(mae,  4)},
        {"metric": "RMSE", "value": round(rmse, 4)},
        {"metric": "R2",   "value": round(r2,   4)},
        {"metric": "MAPE", "value": round(mape, 4) if np.isfinite(mape) else np.nan},
    ])
    tbl_saved["metrics"] = _savetbl(metrics_df, "stage1_regression_metrics.csv")

    # --- Predictions table ---
    residuals = y_test - y_pred
    abs_err   = np.abs(residuals)
    mask_nz   = y_test != 0
    ape       = np.full(len(y_test), np.nan)
    ape[mask_nz] = np.abs(residuals[mask_nz] / y_test[mask_nz]) * 100

    pred_df = test_df[META_COLS].copy()
    pred_df["actual_cycle_life"]          = y_test
    pred_df["predicted_cycle_life"]       = y_pred.round(2)
    pred_df["residual"]                   = residuals.round(2)
    pred_df["absolute_error"]             = abs_err.round(2)
    pred_df["absolute_percentage_error"]  = np.round(ape, 2)
    tbl_saved["predictions"] = _savetbl(pred_df, "stage1_regression_predictions.csv")

    # --- Feature importance ---
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature":          feature_cols,
        "importance":       importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp_df["importance_rank"] = range(1, len(feat_imp_df) + 1)
    tbl_saved["feat_imp"] = _savetbl(feat_imp_df, "stage1_regression_feature_importance.csv")

    print("\nTop 10 feature importances:")
    for _, row in feat_imp_df.head(10).iterrows():
        print(f"  {int(row['importance_rank']):2d}. {row['feature']:<35} {row['importance']:.6f}")

    # --- Figures ---
    print("\nGenerating figures...")
    fig_saved = make_figures(y_test, y_pred, feat_imp_df)
    for k, p in fig_saved.items():
        print(f"  {p.name}")

    # --- Report ---
    print("\nGenerating markdown report...")
    rpt_path = make_report(
        train_df, test_df, feature_cols,
        cv_df, metrics_df, feat_imp_df,
        fig_saved, tbl_saved,
    )
    print(f"  {rpt_path}")

    print("\n--- Saved tables ---")
    for k, p in tbl_saved.items():
        print(f"  {p}")

    print(f"\nDone.")


if __name__ == "__main__":
    try:
        train_xgboost_regression()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
