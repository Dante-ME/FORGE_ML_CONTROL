"""
Extract early-cycle summary features from the original Severson battery dataset MATLAB struct files.
Output: data/processed/severson_original_features.csv
"""

import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Resolve project root regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "severson_original"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "severson_original_features.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(dataset) -> np.ndarray:
    """Return a 1-D float64 array from an HDF5 dataset; empty array on failure."""
    try:
        arr = np.array(dataset, dtype=np.float64).ravel()
        return arr
    except Exception:
        return np.array([], dtype=np.float64)


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if arr.size > 0 else np.nan


def _safe_at_cycle(cycle_arr: np.ndarray, value_arr: np.ndarray, target: int) -> float:
    """Return value_arr entry whose cycle is closest to target; prefer exact match."""
    if cycle_arr.size == 0 or value_arr.size == 0:
        return np.nan
    n = min(cycle_arr.size, value_arr.size)
    cycles = cycle_arr[:n]
    values = value_arr[:n]
    exact = np.where(cycles == target)[0]
    if exact.size > 0:
        return float(values[exact[0]])
    idx = np.argmin(np.abs(cycles - target))
    return float(values[idx])


def _safe_slope(cycle_arr: np.ndarray, value_arr: np.ndarray) -> float:
    """Linear slope via np.polyfit; nan if fewer than 2 valid points."""
    n = min(cycle_arr.size, value_arr.size)
    if n < 2:
        return np.nan
    x = cycle_arr[:n]
    y = value_arr[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    try:
        slope, _ = np.polyfit(x[mask], y[mask], 1)
        return float(slope)
    except Exception:
        return np.nan


def _infer_batch_date(filename: str) -> str:
    """Extract YYYY-MM-DD from filename or return 'unknown'."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else "unknown"


def _screening_label(cycle_life) -> str:
    try:
        cl = float(cycle_life)
        if cl >= 1000:
            return "Reuse"
        if cl >= 500:
            return "Conditional Reuse"
        return "Recycle"
    except (TypeError, ValueError):
        return "Unknown"


# ---------------------------------------------------------------------------
# Per-cell extraction
# ---------------------------------------------------------------------------

def _extract_cell(f: h5py.File, batch: h5py.Group, i: int,
                  batch_date: str, source_file: str) -> dict | None:
    """Extract features for cell index i. Returns None if cell must be skipped."""
    cell_id = f"{batch_date}_cell_{i:03d}"

    # Dereference summary
    try:
        summary = f[batch["summary"][i, 0]]
    except Exception as e:
        print(f"  [WARN] {cell_id}: cannot dereference summary — {e}. Skipping.")
        return None

    # Dereference cycle_life
    try:
        cl_raw = f[batch["cycle_life"][i, 0]]
        cycle_life = float(np.array(cl_raw).ravel()[0])
    except Exception as e:
        print(f"  [WARN] {cell_id}: cannot dereference cycle_life — {e}. cycle_life=NaN.")
        cycle_life = np.nan

    # Extract raw 1-D arrays from summary
    def _get(key):
        try:
            return _flatten(summary[key])
        except KeyError:
            print(f"  [WARN] {cell_id}: missing summary key '{key}'. Using empty array.")
            return np.array([], dtype=np.float64)

    cycle      = _get("cycle")
    q_dis      = _get("QDischarge")
    q_chg      = _get("QCharge")
    ir         = _get("IR")
    tavg       = _get("Tavg")
    tmax       = _get("Tmax")
    tmin       = _get("Tmin")
    chargetime = _get("chargetime")

    # Early-cycle window mask (cycle 2–100)
    if cycle.size > 0:
        mask = (cycle >= 2) & (cycle <= 100)
        c_win      = cycle[mask]
        q_dis_win  = q_dis[mask]   if q_dis.size      >= cycle.size else q_dis[:mask.sum()]
        q_chg_win  = q_chg[mask]   if q_chg.size      >= cycle.size else q_chg[:mask.sum()]
        ir_win     = ir[mask]      if ir.size         >= cycle.size else ir[:mask.sum()]
        tavg_win   = tavg[mask]    if tavg.size       >= cycle.size else tavg[:mask.sum()]
        tmax_win   = tmax[mask]    if tmax.size       >= cycle.size else tmax[:mask.sum()]
        tmin_win   = tmin[mask]    if tmin.size       >= cycle.size else tmin[:mask.sum()]
        ct_win     = chargetime[mask] if chargetime.size >= cycle.size else chargetime[:mask.sum()]

        # Align each array to c_win length
        def _align(arr):
            n = c_win.size
            if arr.size >= n:
                return arr[:n]
            # pad with nan if shorter
            out = np.full(n, np.nan)
            out[:arr.size] = arr
            return out

        q_dis_win  = _align(q_dis_win)
        q_chg_win  = _align(q_chg_win)
        ir_win     = _align(ir_win)
        tavg_win   = _align(tavg_win)
        tmax_win   = _align(tmax_win)
        tmin_win   = _align(tmin_win)
        ct_win     = _align(ct_win)
    else:
        c_win = q_dis_win = q_chg_win = ir_win = np.array([])
        tavg_win = tmax_win = tmin_win = ct_win = np.array([])

    n_avail = int(c_win.size)

    # Point features
    q2   = _safe_at_cycle(cycle, q_dis, 2)
    q100 = _safe_at_cycle(c_win, q_dis_win, 100) if c_win.size > 0 else np.nan

    capacity_fade     = (q2 - q100) if np.isfinite(q2) and np.isfinite(q100) else np.nan
    capacity_fade_pct = (capacity_fade / q2 * 100) if np.isfinite(capacity_fade) and np.isfinite(q2) and q2 != 0 else np.nan
    soh_proxy         = (q100 / q2 * 100) if np.isfinite(q100) and np.isfinite(q2) and q2 != 0 else np.nan

    ir_change = (float(ir_win[-1]) - float(ir_win[0])) if ir_win.size >= 2 and np.isfinite(ir_win[0]) and np.isfinite(ir_win[-1]) else np.nan
    ct_change = (float(ct_win[-1]) - float(ct_win[0])) if ct_win.size >= 2 and np.isfinite(ct_win[0]) and np.isfinite(ct_win[-1]) else np.nan

    return {
        "cell_id":               cell_id,
        "source_file":           source_file,
        "batch_date_inferred":   batch_date,
        "cell_index":            i,
        "cycle_life":            cycle_life if np.isfinite(cycle_life) else np.nan,
        "n_cycles_available":    n_avail,
        # physical features
        "q_discharge_cycle_2":   q2,
        "q_discharge_cycle_100": q100,
        "q_discharge_mean_2_100": _safe_mean(q_dis_win),
        "q_charge_mean_2_100":   _safe_mean(q_chg_win),
        "capacity_fade_2_100":   capacity_fade,
        "capacity_fade_pct_2_100": capacity_fade_pct,
        "capacity_slope_2_100":  _safe_slope(c_win, q_dis_win),
        "ir_mean_2_100":         _safe_mean(ir_win),
        "ir_change_2_100":       ir_change,
        "tavg_mean_2_100":       _safe_mean(tavg_win),
        "tmax_mean_2_100":       _safe_mean(tmax_win),
        "tmin_mean_2_100":       _safe_mean(tmin_win),
        "chargetime_mean_2_100": _safe_mean(ct_win),
        "chargetime_change_2_100": ct_change,
        "soh_cycle_100_proxy":   soh_proxy,
        "screening_label":       _screening_label(cycle_life),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_original_features() -> pd.DataFrame:
    mat_files = sorted(DATA_DIR.glob("*.mat"))

    if not mat_files:
        print(f"[ERROR] No .mat files found in {DATA_DIR}")
        return pd.DataFrame()

    print(f"Found {len(mat_files)} .mat file(s):")
    for f in mat_files:
        print(f"  {f.name}")

    records = []

    for mat_path in mat_files:
        source_file  = mat_path.name
        batch_date   = _infer_batch_date(source_file)
        print(f"\nProcessing: {source_file}  (batch_date={batch_date})")

        try:
            with h5py.File(mat_path, "r") as f:
                batch    = f["batch"]
                n_cells  = batch["summary"].shape[0]
                print(f"  Cells found: {n_cells}")

                for i in range(n_cells):
                    rec = _extract_cell(f, batch, i, batch_date, source_file)
                    if rec is not None:
                        records.append(rec)

        except Exception as e:
            print(f"  [ERROR] Failed to open {source_file}: {e}")
            continue

    if not records:
        print("[ERROR] No records extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  .mat files found      : {len(mat_files)}")
    print(f"  Files processed       : {[f.name for f in mat_files]}")
    print(f"  Total rows extracted  : {len(df)}")
    print(f"  DataFrame shape       : {df.shape}")
    print("\n--- First 5 rows ---")
    print(df.head().to_string())
    print("\n--- screening_label distribution ---")
    print(df["screening_label"].value_counts(dropna=False).to_string())
    print("\n--- Missing value summary (columns with any NaN) ---")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if na_counts.empty:
        print("  No missing values.")
    else:
        print(na_counts.to_string())
    print(f"\n  Output CSV saved to   : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    df = extract_original_features()
    if df.empty:
        print("[FATAL] Extraction produced no data. Check warnings above.")
        sys.exit(1)
    _print_summary(df)
