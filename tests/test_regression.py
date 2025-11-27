# test_regression.py
#
# Regression tests for simulation outputs.
# Compare baseline vs cholesky / cholesky+parallelization.
#
# Ignore:
#   - header row (pandas handles)
#   - first column (dimension p)
#
# Conditions:
#   1) MSE(relative error) < 0.05
#   2) max(relative error) < 0.25
#
# Files tested:
#   - single_normal_mean.csv / single_normal_std.csv
#   - single_t_mean.csv / single_t_std.csv
#   - multi_normal_mean.csv / multi_normal_std.csv
#   - multi_t_mean.csv / multi_t_std.csv
#   - convergence_rate_snr*.csv

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------- configuration ----------------------

BASELINE_DIR = Path("results") / "tables" / "baseline"
VARIANT_DIRS = {
    "cholesky": Path("results") / "tables" / "cholesky",
    "cholesky+parallelization": Path("results") / "tables" / "cholesky+parallelization",
}

MSE_TOL = 0.05     # MSE of relative error must be < 5%
MAX_TOL = 0.25     # Max relative error < 25%
ABS_ZERO_TOL = 1e-10


def _list_common_csv_files() -> List[str]:
    """
    List CSV files in baseline/ excluding *_std.csv,
    and verify the same files exist in all variant dirs.
    """
    baseline_files = sorted(
        [p.name for p in BASELINE_DIR.glob("*.csv")]
    )
    if not baseline_files:
        raise RuntimeError("No CSV files found in baseline (excluding *_std.csv).")

    # Verify each variant contains the same files
    for variant_name, vdir in VARIANT_DIRS.items():
        missing = [f for f in baseline_files if not (vdir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Variant '{variant_name}' missing files: {missing}"
            )

    return baseline_files


def _load_numeric_matrix(path: Path) -> np.ndarray:
    """
    Load CSV, drop the first column (dimension p), keep all numeric columns.
    """
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")

    # Drop first column (p)
    df_no_p = df.iloc[:, 1:]

    num_df = df_no_p.select_dtypes(include=[np.number])

    if num_df.empty:
        raise ValueError(f"No numeric columns to compare in {path}")

    return num_df.to_numpy(float)


def _compute_relative_errors(baseline: np.ndarray, variant: np.ndarray) -> np.ndarray:
    """
    Compute element-wise relative errors.
    baseline, variant have same shape.
    Returns array of same shape of relative errors.

    Rules:
      - both ~0 => error = 0
      - baseline ~0, variant not ~0 => catastrophic mismatch => return inf array
      - else regular relative error = |v - b| / |b|
    """
    if baseline.shape != variant.shape:
        raise AssertionError("Shape mismatch.")

    baseline_small = np.abs(baseline) < ABS_ZERO_TOL
    variant_small  = np.abs(variant) < ABS_ZERO_TOL

    rel = np.zeros_like(baseline, float)

    # both zero-like -> ok
    both_small = baseline_small & variant_small
    rel[both_small] = 0.0

    # baseline non-small -> standard relative error
    regular = ~baseline_small
    rel[regular] = np.abs(variant[regular] - baseline[regular]) / np.abs(baseline[regular])

    # baseline small but variant not small => catastrophic mismatch
    bad = baseline_small & ~variant_small
    if np.any(bad):
        return np.full_like(baseline, np.inf, float)

    return rel


def _mse_and_max_error(baseline: np.ndarray, variant: np.ndarray) -> tuple[float, float]:
    """
    Compute:
      - MSE(relative error)
      - max(relative error)
    """
    rel = _compute_relative_errors(baseline, variant)

    if np.isinf(rel).any():
        return float("inf"), float("inf")

    mse = float(np.mean(rel**2))
    max_err = float(np.max(rel))
    return mse, max_err


# -------- parametrization --------
CSV_FILES = _list_common_csv_files()
VARIANT_KEYS = list(VARIANT_DIRS.keys())


@pytest.mark.parametrize("csv_name", CSV_FILES)
@pytest.mark.parametrize("variant_key", VARIANT_KEYS)
def test_outputs_close_to_baseline(csv_name: str, variant_key: str):
    """
    Test that:
      - MSE(relative error) < MSE_TOL
      - max(relative error) < MAX_TOL
    for each CSV file in each variant.
    """
    baseline_path = BASELINE_DIR / csv_name
    variant_path  = VARIANT_DIRS[variant_key] / csv_name

    baseline_vals = _load_numeric_matrix(baseline_path)
    variant_vals  = _load_numeric_matrix(variant_path)

    mse, max_err = _mse_and_max_error(baseline_vals, variant_vals)

    assert mse < MSE_TOL, (
        f"[{csv_name}] variant '{variant_key}' failed MSE check: "
        f"MSE={mse:.6f}, tol={MSE_TOL}"
    )

    assert max_err < MAX_TOL, (
        f"[{csv_name}] variant '{variant_key}' failed max-error check: "
        f"max_err={max_err:.6f}, tol={MAX_TOL}"
    )