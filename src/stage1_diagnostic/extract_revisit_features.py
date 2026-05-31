from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw" / "severson_revisit"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "severson_revisit_features.csv"


def get_cell_number(path: Path) -> int:
    """
    Extract numeric cell number from filename.
    Example: cell12.csv -> 12
    """
    match = re.search(r"cell(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Invalid cell filename: {path.name}")
    return int(match.group(1))


def create_screening_label(cycle_life: float) -> str:
    """
    Create FORGE screening label based on cycle life.
    """
    if cycle_life >= 1000:
        return "Reuse"
    elif cycle_life >= 500:
        return "Conditional Reuse"
    else:
        return "Recycle"


def extract_matrix_features(matrix: np.ndarray) -> dict:
    """
    Extract statistical features from one battery cell matrix.

    Expected matrix shape:
    rows    = voltage/capacity curve points
    columns = early cycles
    """
    matrix = np.asarray(matrix, dtype=float)

    column_means = np.mean(matrix, axis=0)
    column_stds = np.std(matrix, axis=0)

    first_10 = matrix[:, :10]
    last_10 = matrix[:, -10:]

    cycle_index = np.arange(matrix.shape[1])

    mean_slope = np.polyfit(cycle_index, column_means, 1)[0]
    std_slope = np.polyfit(cycle_index, column_stds, 1)[0]

    features = {
        "matrix_mean": np.mean(matrix),
        "matrix_std": np.std(matrix),
        "matrix_min": np.min(matrix),
        "matrix_max": np.max(matrix),
        "matrix_abs_mean": np.mean(np.abs(matrix)),
        "matrix_abs_max": np.max(np.abs(matrix)),

        "cycle_mean_start": np.mean(first_10),
        "cycle_mean_end": np.mean(last_10),
        "cycle_mean_delta": np.mean(last_10) - np.mean(first_10),

        "cycle_std_start": np.std(first_10),
        "cycle_std_end": np.std(last_10),
        "cycle_std_delta": np.std(last_10) - np.std(first_10),

        "mean_slope_over_cycles": mean_slope,
        "std_slope_over_cycles": std_slope,

        "q01": np.quantile(matrix, 0.01),
        "q05": np.quantile(matrix, 0.05),
        "q50": np.quantile(matrix, 0.50),
        "q95": np.quantile(matrix, 0.95),
        "q99": np.quantile(matrix, 0.99),
    }

    return features


def process_split(split_name: str) -> list[dict]:
    """
    Process one split: train, test1, or test2.
    """
    split_dir = RAW_DIR / split_name
    cycle_life_path = RAW_DIR / "cycle_lives" / f"{split_name}_cycle_lives.csv"

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    if not cycle_life_path.exists():
        raise FileNotFoundError(f"Missing cycle life file: {cycle_life_path}")

    cell_files = sorted(split_dir.glob("cell*.csv"), key=get_cell_number)
    cycle_lives = pd.read_csv(cycle_life_path, header=None).iloc[:, 0].to_numpy()

    if len(cell_files) != len(cycle_lives):
        raise ValueError(
            f"Mismatch in {split_name}: "
            f"{len(cell_files)} cell files but {len(cycle_lives)} cycle life values."
        )

    rows = []

    for idx, cell_file in enumerate(cell_files):
        cell_number = get_cell_number(cell_file)
        matrix = pd.read_csv(cell_file, header=None).to_numpy(dtype=float)

        cycle_life = float(cycle_lives[idx])

        row = {
            "cell_id": f"{split_name}_cell{cell_number}",
            "split": split_name,
            "cell_number": cell_number,
            "cycle_life": cycle_life,
            "screening_label": create_screening_label(cycle_life),
            "matrix_rows": matrix.shape[0],
            "matrix_cols": matrix.shape[1],
        }

        row.update(extract_matrix_features(matrix))
        rows.append(row)

    return rows


def extract_revisit_features() -> pd.DataFrame:
    """
    Convert Severson Revisit matrix-per-cell CSV files into one tabular dataset.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {RAW_DIR}\n"
            "Please make sure data/raw/severson_revisit exists."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split_name in ["train", "test1", "test2"]:
        print(f"Processing split: {split_name}")
        split_rows = process_split(split_name)
        all_rows.extend(split_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


def main():
    df = extract_revisit_features()

    print("\n" + "=" * 70)
    print("Severson Revisit Feature Extraction Complete")
    print("=" * 70)

    print(f"Output path      : {OUTPUT_PATH}")
    print(f"Dataframe shape  : {df.shape}")
    print(f"Total rows       : {len(df)}")

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nScreening label distribution:")
    print(df["screening_label"].value_counts())

    print("\nFirst five rows:")
    print(df.head())

    print("=" * 70)


if __name__ == "__main__":
    main()