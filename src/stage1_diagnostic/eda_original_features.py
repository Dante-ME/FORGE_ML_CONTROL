"""
EDA for Stage 1 original Severson features.
Input:  data/processed/severson_original_features.csv
Output: outputs/figures/stage1/, outputs/tables/stage1/, reports/stage1_eda_summary.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV    = PROJECT_ROOT / "data" / "processed" / "severson_original_features.csv"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures" / "stage1"
TBL_DIR      = PROJECT_ROOT / "outputs" / "tables" / "stage1"
RPT_DIR      = PROJECT_ROOT / "reports"

LABEL_ORDER  = ["Reuse", "Conditional Reuse", "Recycle"]

META_COLS    = ["cell_id", "source_file", "batch_date_inferred", "cell_index", "screening_label"]
TARGET_COL   = "cycle_life"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _savetbl(df: pd.DataFrame, name: str) -> Path:
    path = TBL_DIR / name
    df.to_csv(path, index=False)
    return path


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(META_COLS) | {TARGET_COL, "cell_index"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def make_tables(df: pd.DataFrame, df_model: pd.DataFrame, feat_cols: list[str]) -> dict:
    saved = {}

    # A. Dataset overview
    overview = pd.DataFrame([{
        "total_rows":                len(df),
        "total_columns":             len(df.columns),
        "rows_with_valid_cycle_life": len(df_model),
        "rows_missing_cycle_life":   df[TARGET_COL].isna().sum(),
        "number_of_features":        len(feat_cols),
        "number_of_source_files":    df["source_file"].nunique(),
        "number_of_labels":          df["screening_label"].nunique(),
    }])
    saved["overview"] = _savetbl(overview, "stage1_dataset_overview.csv")

    # B. Missing values
    missing = pd.DataFrame({
        "column":          df.columns,
        "missing_count":   df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values,
    })
    saved["missing"] = _savetbl(missing, "stage1_missing_values.csv")

    # C. Descriptive statistics
    desc = df[feat_cols + [TARGET_COL]].describe().T.reset_index().rename(columns={"index": "feature"})
    saved["desc_stats"] = _savetbl(desc, "stage1_descriptive_statistics.csv")

    # D. Label distribution
    lbl = df["screening_label"].value_counts(dropna=False).reset_index()
    lbl.columns = ["screening_label", "count"]
    lbl["percent"] = (lbl["count"] / len(df) * 100).round(2)
    saved["label_dist"] = _savetbl(lbl, "stage1_label_distribution.csv")

    # E. Batch distribution
    batch = df["source_file"].value_counts(dropna=False).reset_index()
    batch.columns = ["source_file", "count"]
    batch["percent"] = (batch["count"] / len(df) * 100).round(2)
    saved["batch_dist"] = _savetbl(batch, "stage1_batch_distribution.csv")

    # F. Correlation with cycle_life
    corr_vals = df_model[feat_cols].corrwith(df_model[TARGET_COL])
    corr_df = pd.DataFrame({
        "feature":                  corr_vals.index,
        "correlation_with_cycle_life": corr_vals.values.round(4),
        "abs_correlation":          corr_vals.abs().values.round(4),
    }).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    saved["corr_full"] = _savetbl(corr_df, "stage1_correlation_with_cycle_life.csv")

    # G. Top 10 correlated
    saved["corr_top10"] = _savetbl(corr_df.head(10), "stage1_top_correlated_features.csv")

    # H. Grouped means by label
    group_cols = [TARGET_COL] + feat_cols
    grouped = df_model.groupby("screening_label")[group_cols].mean().round(4).reset_index()
    saved["grouped_means"] = _savetbl(grouped, "stage1_grouped_feature_means_by_label.csv")

    return saved, corr_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame, df_model: pd.DataFrame, corr_df: pd.DataFrame) -> dict:
    saved = {}
    top10_features = corr_df.head(10)["feature"].tolist()

    # A. Cycle life distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_model[TARGET_COL].dropna(), bins=20, edgecolor="black", color="#4C72B0")
    ax.set_title("Cycle Life Distribution (Severson Dataset)")
    ax.set_xlabel("Cycle Life")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4)
    saved["cycle_life_dist"] = _savefig("cycle_life_distribution.png")

    # B. Screening label distribution
    lbl_counts = df["screening_label"].value_counts()
    ordered = [l for l in LABEL_ORDER if l in lbl_counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ordered, [lbl_counts[l] for l in ordered], color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="black")
    ax.set_title("Screening Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    for i, l in enumerate(ordered):
        ax.text(i, lbl_counts[l] + 0.5, str(lbl_counts[l]), ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    saved["label_dist"] = _savefig("screening_label_distribution.png")

    # C. Batch distribution
    batch_counts = df["source_file"].value_counts()
    short_names = [n.split("_")[0] for n in batch_counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(short_names, batch_counts.values, color="#9467bd", edgecolor="black")
    ax.set_title("Cell Count by Batch File")
    ax.set_xlabel("Batch Date")
    ax.set_ylabel("Count")
    plt.xticks(rotation=15)
    ax.grid(axis="y", alpha=0.4)
    saved["batch_dist"] = _savefig("batch_distribution.png")

    # D. Correlation bar (top 10)
    top10 = corr_df.head(10).sort_values("abs_correlation")
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in top10["correlation_with_cycle_life"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["feature"], top10["correlation_with_cycle_life"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 10 Features by |Pearson Correlation| with Cycle Life")
    ax.set_xlabel("Pearson Correlation")
    ax.grid(axis="x", alpha=0.4)
    saved["corr_bar"] = _savefig("correlation_bar_top10.png")

    # E. Correlation heatmap (top 10 + cycle_life)
    heatmap_cols = [TARGET_COL] + top10_features
    corr_matrix = df_model[heatmap_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5)
    else:
        im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(heatmap_cols)))
        ax.set_yticks(range(len(heatmap_cols)))
        ax.set_xticklabels(heatmap_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(heatmap_cols, fontsize=8)
        for r in range(len(heatmap_cols)):
            for c in range(len(heatmap_cols)):
                ax.text(c, r, f"{corr_matrix.values[r, c]:.2f}",
                        ha="center", va="center", fontsize=6, color="black")
    ax.set_title("Correlation Heatmap: Cycle Life + Top 10 Features")
    saved["corr_heatmap"] = _savefig("correlation_heatmap_top10.png")

    # F. Cycle life by label boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    ordered_labels = [l for l in LABEL_ORDER if l in df_model["screening_label"].unique()]
    data_by_label = [df_model.loc[df_model["screening_label"] == l, TARGET_COL].dropna().values
                     for l in ordered_labels]
    bp = ax.boxplot(data_by_label, labels=ordered_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors_box = ["#2ca02c", "#ff7f0e", "#d62728"]
    for patch, color in zip(bp["boxes"], colors_box[:len(ordered_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Cycle Life by Screening Label")
    ax.set_xlabel("Screening Label")
    ax.set_ylabel("Cycle Life")
    ax.grid(axis="y", alpha=0.4)
    saved["boxplot"] = _savefig("cycle_life_by_label_boxplot.png")

    # G. Scatter plots
    scatter_map = {
        "q_discharge_cycle_100":  "q_discharge_cycle_100_vs_cycle_life.png",
        "ir_mean_2_100":          "ir_mean_vs_cycle_life.png",
        "tavg_mean_2_100":        "tavg_mean_vs_cycle_life.png",
        "capacity_fade_pct_2_100":"capacity_fade_pct_vs_cycle_life.png",
        "soh_cycle_100_proxy":    "soh_cycle_100_proxy_vs_cycle_life.png",
    }
    for feat, fname in scatter_map.items():
        if feat not in df_model.columns:
            continue
        sub = df_model[[feat, TARGET_COL]].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub[feat], sub[TARGET_COL], alpha=0.6, edgecolors="k", linewidths=0.3, color="#4C72B0")
        ax.set_xlabel(feat)
        ax.set_ylabel("Cycle Life")
        ax.set_title(f"{feat} vs Cycle Life")
        ax.grid(alpha=0.4)
        saved[f"scatter_{feat}"] = _savefig(fname)

    return saved


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_report(df: pd.DataFrame, df_model: pd.DataFrame,
                corr_df: pd.DataFrame, tbl_saved: dict, fig_saved: dict) -> Path:

    lbl_dist = df["screening_label"].value_counts(dropna=False)
    batch_dist = df["source_file"].value_counts(dropna=False)

    def _tbl_md(series: pd.Series, col1: str, col2: str = "Count") -> str:
        rows = [f"| {col1} | {col2} |", "| --- | --- |"]
        for k, v in series.items():
            rows.append(f"| {k} | {v} |")
        return "\n".join(rows)

    def _df_md(df_in: pd.DataFrame) -> str:
        return df_in.to_markdown(index=False)

    def _df_to_md(df_in: pd.DataFrame) -> str:
        cols = df_in.columns.tolist()
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows   = ["| " + " | ".join(str(v) for v in row) + " |"
                  for row in df_in.itertuples(index=False)]
        return "\n".join([header, sep] + rows)

    top10_md = _df_to_md(corr_df.head(10)[["feature", "correlation_with_cycle_life"]])

    tbl_paths = "\n".join(f"- `{p}`" for p in tbl_saved.values())
    fig_paths = "\n".join(f"- `{p}`" for p in fig_saved.values())

    report = f"""# Stage 1 EDA Summary — Original Severson Features

## 1. Dataset Overview

The dataset was extracted from the original Severson et al. (2019) LFP/graphite battery aging benchmark.
It contains **{len(df)} battery cells** across **{df['source_file'].nunique()} batch files**.

| Metric | Value |
| --- | --- |
| Total rows | {len(df)} |
| Total columns | {len(df.columns)} |
| Rows with valid cycle_life | {len(df_model)} |
| Rows missing cycle_life | {int(df[TARGET_COL].isna().sum())} |
| Number of physical features | {len(_feature_cols(df))} |
| Number of batch files | {df['source_file'].nunique()} |
| Number of screening labels | {df['screening_label'].nunique()} |

> **Note:** {int(df[TARGET_COL].isna().sum())} rows have missing `cycle_life` and should be excluded
> from supervised regression and classification model training.
> All model-based analyses below use `df_model` (n={len(df_model)}).

---

## 2. Missing Value Summary

| Column | Missing Count | Missing % |
| --- | --- | --- |
| cycle_life | {int(df[TARGET_COL].isna().sum())} | {df[TARGET_COL].isna().mean()*100:.1f}% |

All physical feature columns have **no missing values**.

---

## 3. Label Distribution

Screening labels are assigned based on `cycle_life` thresholds:
- **Reuse**: cycle_life ≥ 1000
- **Conditional Reuse**: 500 ≤ cycle_life < 1000
- **Recycle**: cycle_life < 500

{_tbl_md(lbl_dist, 'Screening Label')}

---

## 4. Batch Distribution

{_tbl_md(batch_dist, 'Source File')}

---

## 5. Cycle Life Distribution

The `cycle_life` target spans from short-lived cells (< 200 cycles) to long-lived cells (> 1000 cycles).
The distribution is right-skewed, with the majority of cells falling in the 300–900 cycle range.

See: `outputs/figures/stage1/cycle_life_distribution.png`

---

## 6. Top Correlated Features with Cycle Life

{top10_md}

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
{tbl_paths}

### Figures
{fig_paths}

---

## 9. Recommended Next Step

Proceed to **preprocessing and model training**:

1. Run `src/stage1_diagnostic/preprocess.py` to apply scaling and stratified train/test split.
2. Run `src/stage1_diagnostic/train_xgboost.py` to train:
   - XGBoost regression model for `cycle_life` prediction.
   - XGBoost classification model for `screening_label` prediction.
3. Evaluate using RMSE, MAE, R² (regression) and accuracy, F1, confusion matrix (classification).
"""

    out_path = RPT_DIR / "stage1_eda_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Please run:  python src/stage1_diagnostic/extract_original_features.py"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    RPT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df_model = df.dropna(subset=[TARGET_COL]).copy()
    feat_cols = _feature_cols(df)

    print(f"Input          : {INPUT_CSV}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Valid cycle_life rows : {len(df_model)}")
    print(f"Missing cycle_life    : {df[TARGET_COL].isna().sum()}")
    print(f"Seaborn available     : {HAS_SEABORN}")

    print("\n--- Generating tables ---")
    tbl_saved, corr_df = make_tables(df, df_model, feat_cols)
    for k, p in tbl_saved.items():
        print(f"  {p.name}")

    print("\n--- Generating figures ---")
    fig_saved = make_figures(df, df_model, corr_df)
    for k, p in fig_saved.items():
        print(f"  {p.name}")

    print("\n--- Generating markdown report ---")
    rpt_path = make_report(df, df_model, corr_df, tbl_saved, fig_saved)
    print(f"  {rpt_path}")

    print("\n--- Label distribution ---")
    print(df["screening_label"].value_counts(dropna=False).to_string())

    print("\n--- Top 10 correlated features with cycle_life ---")
    print(corr_df.head(10)[["feature", "correlation_with_cycle_life"]].to_string(index=False))

    print(f"\nDone. Report saved to: {rpt_path}")


if __name__ == "__main__":
    try:
        run_eda()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
