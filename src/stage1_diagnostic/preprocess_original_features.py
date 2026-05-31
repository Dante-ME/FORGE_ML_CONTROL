"""
Preprocessing for Stage 1 model-ready datasets.
Input:  data/processed/severson_original_features.csv
Output: data/processed/stage1_{regression,classification}_{train,test}.csv
        outputs/tables/stage1/stage1_preprocessing_*.csv
        reports/stage1_preprocessing_summary.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV    = PROJECT_ROOT / "data" / "processed" / "severson_original_features.csv"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
TBL_DIR      = PROJECT_ROOT / "outputs" / "tables" / "stage1"
RPT_DIR      = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE    = 0.2

META_COLS    = ["cell_id", "source_file", "batch_date_inferred", "cell_index"]
TARGET_REG   = "cycle_life"
TARGET_CLS   = "screening_label"
TARGET_CLS_ENC = "screening_label_encoded"

EXPECTED_FEATURES = [
    "n_cycles_available",
    "q_discharge_cycle_2",
    "q_discharge_cycle_100",
    "q_discharge_mean_2_100",
    "q_charge_mean_2_100",
    "capacity_fade_2_100",
    "capacity_fade_pct_2_100",
    "capacity_slope_2_100",
    "ir_mean_2_100",
    "ir_change_2_100",
    "tavg_mean_2_100",
    "tmax_mean_2_100",
    "tmin_mean_2_100",
    "chargetime_mean_2_100",
    "chargetime_change_2_100",
    "soh_cycle_100_proxy",
]

LABEL_ORDER = ["Recycle", "Conditional Reuse", "Reuse"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(META_COLS) | {
        TARGET_REG, TARGET_CLS, TARGET_CLS_ENC, "cell_index"
    }
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        print(f"  [WARN] Expected feature columns not found: {missing}")

    available_numeric = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude
    ]
    if not available_numeric:
        raise ValueError("No valid feature columns found in the input CSV.")
    return available_numeric


def _label_distribution(df: pd.DataFrame, label_col: str = TARGET_CLS) -> dict:
    counts = df[label_col].value_counts()
    return {lbl: int(counts.get(lbl, 0)) for lbl in LABEL_ORDER}


def _savetbl(df_out: pd.DataFrame, name: str) -> Path:
    path = TBL_DIR / name
    df_out.to_csv(path, index=False)
    return path


def _savecsv(df_out: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def save_metadata_tables(
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_map: dict,
    out_paths: dict,
) -> None:

    # Feature list
    feat_list = pd.DataFrame({
        "feature_name":  feat_cols,
        "feature_index": range(len(feat_cols)),
    })
    _savetbl(feat_list, "stage1_feature_list.csv")

    # Label mapping
    lbl_map_df = pd.DataFrame(
        sorted(label_map.items(), key=lambda x: x[1]),
        columns=[TARGET_CLS, TARGET_CLS_ENC]
    )
    _savetbl(lbl_map_df, "stage1_label_mapping.csv")
    json_path = TBL_DIR / "stage1_label_mapping.json"
    json_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    # Train/test distribution
    def _row(split_name, df_split):
        dist = _label_distribution(df_split)
        total = len(df_split)
        row = {"split": split_name, "rows": total}
        for lbl in LABEL_ORDER:
            row[f"{lbl}_count"]   = dist[lbl]
            row[f"{lbl}_percent"] = round(dist[lbl] / total * 100, 1) if total else 0.0
        return row

    dist_df = pd.DataFrame([_row("train", train_df), _row("test", test_df)])
    _savetbl(dist_df, "stage1_train_test_distribution.csv")

    # Preprocessing summary
    summary = pd.DataFrame([{
        "input_rows":                    len(df_raw),
        "input_columns":                 len(df_raw.columns),
        "rows_dropped_missing_cycle_life": len(df_raw) - len(df_model),
        "model_rows":                    len(df_model),
        "train_rows":                    len(train_df),
        "test_rows":                     len(test_df),
        "test_size":                     TEST_SIZE,
        "random_state":                  RANDOM_STATE,
        "number_of_features":            len(feat_cols),
        "regression_train_path":         str(out_paths["reg_train"]),
        "regression_test_path":          str(out_paths["reg_test"]),
        "classification_train_path":     str(out_paths["cls_train"]),
        "classification_test_path":      str(out_paths["cls_test"]),
    }])
    _savetbl(summary, "stage1_preprocessing_summary.csv")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def save_report(
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_map: dict,
    out_paths: dict,
) -> Path:

    def _tbl(rows: list[tuple], headers: list[str]) -> str:
        sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
        head = "| " + " | ".join(headers) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
        return "\n".join([head, sep, body])

    feat_table = _tbl(
        [(f, i) for i, f in enumerate(feat_cols)],
        ["feature_name", "feature_index"]
    )

    lbl_rows = sorted(label_map.items(), key=lambda x: x[1])
    lbl_table = _tbl(lbl_rows, [TARGET_CLS, TARGET_CLS_ENC])

    def _dist_rows(split_name, df_split):
        dist = _label_distribution(df_split)
        total = len(df_split)
        return (split_name, total,
                dist["Recycle"], dist["Conditional Reuse"], dist["Reuse"],
                f"{dist['Recycle']/total*100:.1f}%",
                f"{dist['Conditional Reuse']/total*100:.1f}%",
                f"{dist['Reuse']/total*100:.1f}%")

    dist_table = _tbl(
        [_dist_rows("train", train_df), _dist_rows("test", test_df)],
        ["split", "rows", "Recycle", "Conditional Reuse", "Reuse",
         "Recycle%", "Cond.Reuse%", "Reuse%"]
    )

    report = f"""# Stage 1 Preprocessing Summary — Original Severson Features

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
| Input shape | {df_raw.shape[0]} rows × {df_raw.shape[1]} columns |
| Source | 3 original Severson batch files (Stanford/MIT LFP benchmark) |

> This dataset is a **public laboratory benchmark** (Severson et al., 2019).
> It should be framed as benchmark validation for the FORGE diagnostic pipeline,
> not as representative of Indonesian EV battery field data.

---

## 3. Cleaning Step

| Action | Count |
| --- | --- |
| Input rows | {len(df_raw)} |
| Rows dropped (missing cycle_life) | {len(df_raw) - len(df_model)} |
| Remaining model rows | {len(df_model)} |

Rows with missing `cycle_life` were removed because supervised learning requires valid targets.

---

## 4. Feature Columns

**{len(feat_cols)} physical features** were selected. Excluded: `cell_index`, `cycle_life`,
`screening_label`, `screening_label_encoded`, and metadata columns.

{feat_table}

---

## 5. Target Definitions

| Task | Target Column | Type |
| --- | --- | --- |
| Regression | `cycle_life` | Continuous float |
| Classification | `screening_label_encoded` | Integer (0–2) |

---

## 6. Label Encoding

`sklearn.preprocessing.LabelEncoder` was applied to `screening_label`.

{lbl_table}

Label mapping saved to:
- `outputs/tables/stage1/stage1_label_mapping.csv`
- `outputs/tables/stage1/stage1_label_mapping.json`

---

## 7. Train-Test Split

| Setting | Value |
| --- | --- |
| Split ratio | 80% train / 20% test |
| Stratified by | `screening_label` |
| Random state | {RANDOM_STATE} |
| Validation set | Not created (dataset is small; use cross-validation in training scripts) |

The **same stratified split** was used for both regression and classification, so both
task datasets contain identical cell_id rows in each partition.

{dist_table}

---

## 8. Generated Outputs

### Model-Ready CSV Files
- `{out_paths['reg_train']}`
- `{out_paths['reg_test']}`
- `{out_paths['cls_train']}`
- `{out_paths['cls_test']}`

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
"""

    out_path = RPT_DIR / "stage1_preprocessing_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_original_features() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Please run:  python src/stage1_diagnostic/extract_original_features.py"
        )

    for d in [PROC_DIR, TBL_DIR, RPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(INPUT_CSV)
    print(f"Input          : {INPUT_CSV}")
    print(f"Input shape    : {df_raw.shape}")

    # Drop missing cycle_life
    df_model = df_raw.dropna(subset=[TARGET_REG]).copy()
    dropped  = len(df_raw) - len(df_model)
    print(f"Rows dropped (missing cycle_life): {dropped}")
    print(f"Model rows     : {len(df_model)}")

    # Resolve feature columns
    feat_cols = _resolve_feature_cols(df_model)
    print(f"\nFeatures ({len(feat_cols)}):")
    for f in feat_cols:
        print(f"  {f}")

    # Label encoding
    le = LabelEncoder()
    df_model[TARGET_CLS_ENC] = le.fit_transform(df_model[TARGET_CLS])
    label_map = {lbl: int(enc) for lbl, enc in zip(le.classes_, le.transform(le.classes_))}
    print(f"\nLabel mapping: {label_map}")

    # Stratified train/test split (shared for both tasks)
    train_df, test_df = train_test_split(
        df_model,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_model[TARGET_CLS],
    )
    print(f"\nTrain rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print("Train distribution:", _label_distribution(train_df))
    print("Test  distribution:", _label_distribution(test_df))

    # Output paths
    out_paths = {
        "reg_train":  PROC_DIR / "stage1_regression_train.csv",
        "reg_test":   PROC_DIR / "stage1_regression_test.csv",
        "cls_train":  PROC_DIR / "stage1_classification_train.csv",
        "cls_test":   PROC_DIR / "stage1_classification_test.csv",
    }

    reg_cols = META_COLS + feat_cols + [TARGET_REG]
    cls_cols = META_COLS + feat_cols + [TARGET_CLS, TARGET_CLS_ENC]

    _savecsv(train_df[reg_cols], out_paths["reg_train"])
    _savecsv(test_df[reg_cols],  out_paths["reg_test"])
    _savecsv(train_df[cls_cols], out_paths["cls_train"])
    _savecsv(test_df[cls_cols],  out_paths["cls_test"])

    # Metadata tables
    save_metadata_tables(df_raw, df_model, train_df, test_df, feat_cols, label_map, out_paths)

    # Markdown report
    rpt_path = save_report(df_raw, df_model, train_df, test_df, feat_cols, label_map, out_paths)

    # Console summary
    print("\n--- Saved output files ---")
    for k, p in out_paths.items():
        print(f"  {p}")
    for name in [
        "stage1_preprocessing_summary.csv",
        "stage1_feature_list.csv",
        "stage1_label_mapping.csv",
        "stage1_train_test_distribution.csv",
    ]:
        print(f"  {TBL_DIR / name}")
    print(f"  {rpt_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    try:
        preprocess_original_features()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
